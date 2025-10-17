use std::{
    cmp::Ordering,
    collections::hash_map::Entry,
    sync::{Arc, atomic::AtomicU8},
    thread::{self},
    time::Instant,
};

use crossbeam::queue::{ArrayQueue, SegQueue};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, Vec3};
use parking_lot::Mutex;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{CHUNK_SIDE_32, Chunk},
    chunk_view::{ChunkGrid, ChunkPool},
    coords::{
        ChunkVector, LocalVector, NEIGHBOR_OFFSETS, cv_to_wv, vec3_wv_to_cv, wv_to_cv, wv_to_lv,
    },
    generation::VoxelGenerator,
    grid::{ChunkManager, GridView},
    hasher::{VoxelHashMap, VoxelHashSet},
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, compute_priority},
    world::{CHUNKEE_CORE_METRICS, Histograms, ResultQueues, Throughputs},
};

#[derive(Debug, Clone)]
pub struct Deltas(pub VoxelHashMap<VoxelId>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[inline(always)]
pub fn neighbors_of(cv: IVec3) -> [ChunkVector; 6] {
    BLOCK_FACES.map(|face| cv + face.into_normal())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkState {
    None,
    Loading,
    NeedsMesh,
    Meshing,
    Meshed,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PhysicsState {
    None,
    Meshing,
    NeedsMesh,
    Meshed,
}

pub type ChunkVersion = u32;

#[derive(Debug, Clone)]
pub struct WorldChunk {
    pub cv: ChunkVector,
    pub state: ChunkState,
    pub chunk: Chunk,
    pub deltas: Deltas,
    pub is_dirty: bool,
    pub version: ChunkVersion,
    pub uniform_voxel_id: Option<VoxelId>,
    pub physics_state: PhysicsState,
}

impl Default for WorldChunk {
    fn default() -> Self {
        Self {
            cv: ChunkVector::MAX,
            state: ChunkState::None,
            chunk: Chunk::new(),
            deltas: Deltas::default(),
            is_dirty: false,
            version: 0,
            uniform_voxel_id: None,
            physics_state: PhysicsState::None,
        }
    }
}

impl WorldChunk {
    pub fn is_stable(&self) -> bool {
        matches!(
            self.state,
            ChunkState::NeedsMesh | ChunkState::Meshing | ChunkState::Meshed
        )
    }

    pub fn reset(&mut self, cv: ChunkVector) {
        self.cv = cv;
        self.is_dirty = false;
        self.state = ChunkState::Loading;
        self.deltas = Deltas::default();
        self.uniform_voxel_id = None;
        self.physics_state = PhysicsState::None;
        self.version = 0;
    }

    pub fn is_loading(&self) -> bool {
        self.state == ChunkState::Loading
    }

    pub fn is_meshing(&self) -> bool {
        self.state == ChunkState::Meshing
    }
}

pub struct PhysicsEntity {
    pub id: i64,
    pub pos: Vec3,
}

pub enum PipelineMessage {
    ChunkEdits(Vec<(IVec3, VoxelId)>),
    CameraDataUpdate(CameraData),
    PhysicsEntitiesUpdate(Vec<PhysicsEntity>),
    Shutdown,
}

pub enum PipelineResult {
    MeshReady {
        cv: ChunkVector,
        mesh: ChunkMeshGroup,
    },
    ChunksUnloaded {
        cvs: Vec<ChunkVector>,
    },
    PhysicsMeshReady {
        cv: ChunkVector,
        mesh: Vec<Vec3>,
    },
    PhysicsMeshUnload {
        cvs: Vec<ChunkVector>,
    },
    EditsApplied(Vec<(IVec3, VoxelId)>),
}

#[derive(Debug, Clone)]
pub struct WorkerTask {
    pub cv: ChunkVector,
    pub priority: u32,
}

impl PartialEq for WorkerTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for WorkerTask {}
impl PartialOrd for WorkerTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WorkerTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

pub struct WorkerSharedState {
    pub load: ArrayQueue<WorkerTask>,
    pub mesh: ArrayQueue<WorkerTask>,
    pub physics: SegQueue<WorkerTask>,
    edit: SegQueue<WorkerTask>,
    pub camera_update: SegQueue<CameraData>,
    camera_lock: Mutex<()>,
    pub physics_entities: SegQueue<Vec<PhysicsEntity>>,
    physics_lock: Mutex<VoxelHashSet>,
    pub chunk_edits: SegQueue<Vec<(IVec3, VoxelId)>>,
    chunk_edits_lock: Mutex<()>,
}

impl WorkerSharedState {
    pub fn new(cap: usize) -> Self {
        Self {
            load: ArrayQueue::new(cap),
            mesh: ArrayQueue::new(cap),
            physics: SegQueue::new(),
            edit: SegQueue::new(),
            camera_update: SegQueue::new(),
            camera_lock: Mutex::new(()),
            physics_entities: SegQueue::new(),
            physics_lock: Mutex::new(VoxelHashSet::default()),
            chunk_edits: SegQueue::new(),
            chunk_edits_lock: Mutex::new(()),
        }
    }

    fn is_empty(&self) -> bool {
        self.load.is_empty()
            && self.mesh.is_empty()
            && self.physics.is_empty()
            && self.edit.is_empty()
            && self.camera_update.is_empty()
            && self.chunk_edits.is_empty()
            && self.physics_entities.is_empty()
    }

    fn print_len(&self) {
        println!(
            "load={} | mesh={} | physics={} | edit={}",
            self.load.len(),
            self.mesh.len(),
            self.physics.len(),
            self.edit.len()
        )
    }

    fn reprioritze_queues(&self, camera_data: &CameraData, view: &GridView, voxel_size: f32) {
        let time = Instant::now();
        WorkerSharedState::repriotize_queue(&self.load, camera_data, view, voxel_size);
        WorkerSharedState::repriotize_queue(&self.mesh, camera_data, view, voxel_size);
        CHUNKEE_CORE_METRICS
            .histograms
            .get(Histograms::ReprioritizeQueues)
            .record(time.elapsed());
    }

    fn repriotize_queue(
        queue: &ArrayQueue<WorkerTask>,
        camera_data: &CameraData,
        view: &GridView,
        voxel_size: f32,
    ) {
        let mut next_tasks = Vec::with_capacity(queue.len());
        while let Some(task) = queue.pop() {
            if view.cv_to_idx_with_origin(task.cv).is_some() {
                next_tasks.push(WorkerTask {
                    cv: task.cv,
                    priority: compute_priority(task.cv, camera_data, voxel_size),
                });
            }
        }
        next_tasks.sort();
        for task in next_tasks {
            queue.push(task).ok();
        }
    }
}

pub struct WorkerPool {
    pub sender: Sender<()>,
}

impl WorkerPool {
    pub fn new<V: 'static + ChunkeeVoxel>(
        num_threads: usize,
        worker_state: Arc<WorkerSharedState>,
        chunk_manager: Arc<ChunkManager>,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        results: Arc<ResultQueues>,
        voxel_size: f32,
    ) -> Self {
        let (sx, rx) = crossbeam_channel::unbounded();

        for _ in 0..num_threads {
            let worker_state = Arc::clone(&worker_state);
            let chunk_manager = Arc::clone(&chunk_manager);
            let chunk_store = Arc::clone(&chunk_store);
            let generator = Arc::clone(&generator);
            let results = results.clone();

            let thread_sx = sx.clone();
            let thread_rx = rx.clone();
            let voxel_size = voxel_size;

            thread::spawn(move || {
                Self::worker_loop::<V>(
                    &worker_state,
                    chunk_manager,
                    chunk_store,
                    generator,
                    thread_sx,
                    thread_rx,
                    &results,
                    voxel_size,
                );
            });
        }

        Self { sender: sx }
    }

    fn worker_loop<V: 'static + ChunkeeVoxel>(
        worker_state: &WorkerSharedState,
        chunk_manager: Arc<ChunkManager>,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        sx: Sender<()>,
        rx: Receiver<()>,
        results: &ResultQueues,
        voxel_size: f32,
    ) {
        loop {
            while let Ok(_) = rx.recv() {
                let track_task_throughput = || {
                    CHUNKEE_CORE_METRICS
                        .throughputs
                        .get(Throughputs::Tasks)
                        .record();
                };

                if !worker_state.camera_update.is_empty()
                    && let Some(_guard) = worker_state.camera_lock.try_lock()
                {
                    let mut camera_data = worker_state.camera_update.pop().unwrap();
                    while let Some(cam) = worker_state.camera_update.pop() {
                        camera_data = cam
                    }

                    let camera_cv = vec3_wv_to_cv(camera_data.pos, voxel_size);

                    let mut dirty_chunks = Vec::new();
                    let mut new_load_tasks = Vec::new();

                    let current_view = chunk_manager.view.load();
                    if current_view.needs_update(camera_cv) {
                        let grid_update_time = Instant::now();
                        let (new_view, reset_tasks, remesh_cvs) =
                            GridView::compute_new(&current_view, camera_cv);
                        chunk_manager.view.store(Arc::new(new_view));

                        for (chunk_idx, new_cv) in reset_tasks {
                            let mut world_chunk = chunk_manager.pool.flat[chunk_idx].write();

                            if world_chunk.state != ChunkState::None {
                                if world_chunk.state == ChunkState::Meshed {
                                    // results.mesh_unload.push(world_chunk.cv);
                                }
                                if world_chunk.is_dirty {
                                    let persisted_chunk = PersistedChunk {
                                        uniform_voxel_id: world_chunk.uniform_voxel_id.take(),
                                        deltas: world_chunk.deltas.clone(),
                                    };
                                    dirty_chunks.push((world_chunk.cv, persisted_chunk));
                                }
                            }

                            world_chunk.reset(new_cv);
                            new_load_tasks.push(WorkerTask {
                                cv: new_cv,
                                priority: compute_priority(new_cv, &camera_data, voxel_size),
                            });
                            sx.send(()).ok();
                        }

                        for cv in remesh_cvs {
                            chunk_manager.write(cv, |wc| {
                                if wc.is_stable() {
                                    wc.state = ChunkState::Meshing;
                                    wc.version = wc.version.wrapping_add(1);
                                    worker_state
                                        .mesh
                                        .push(WorkerTask {
                                            priority: compute_priority(
                                                cv,
                                                &camera_data,
                                                voxel_size,
                                            ),
                                            cv,
                                        })
                                        .ok();
                                    sx.send(()).ok();
                                }
                            });
                        }

                        new_load_tasks.sort();
                        for item in new_load_tasks {
                            worker_state.load.push(item).ok();
                        }
                        sx.send(()).ok();

                        if !dirty_chunks.is_empty() {
                            // TODO: Spawn a task for this
                            chunk_store.save_chunks(dirty_chunks);
                            // batch_unload_deltas_task(dirty_chunks, &chunk_store, &thread_pool);
                        }

                        CHUNKEE_CORE_METRICS
                            .histograms
                            .get(Histograms::GridUpdate)
                            .record(grid_update_time.elapsed());

                        worker_state.print_len();
                    } else {
                        let view = chunk_manager.view.load();
                        if !worker_state.is_empty() {
                            worker_state.reprioritze_queues(&camera_data, &view, voxel_size);
                            sx.send(()).ok();
                        }
                    }
                }

                if !worker_state.physics_entities.is_empty()
                    && let Some(mut previous_physics_entities) =
                        worker_state.physics_lock.try_lock()
                {
                    if chunk_manager.view.load().initialized {
                        let mut new_entities = worker_state.physics_entities.pop().unwrap();
                        while let Some(entities) = worker_state.physics_entities.pop() {
                            new_entities = entities
                        }
                        let mut current_physics_entities = VoxelHashSet::default();
                        let physics_radius = ((CHUNK_SIDE_32 / 2) as f32) * voxel_size;

                        for entity in &new_entities {
                            for offset in NEIGHBOR_OFFSETS {
                                let scaled_offset = offset.as_vec3() * physics_radius;
                                let neighbor_cv =
                                    vec3_wv_to_cv(entity.pos + scaled_offset, voxel_size);

                                current_physics_entities.insert(neighbor_cv);
                            }
                        }

                        for cv in current_physics_entities.difference(&previous_physics_entities) {
                            chunk_manager.write(*cv, |wc| {
                                if wc.physics_state == PhysicsState::None {
                                    wc.physics_state = PhysicsState::Meshing;
                                    worker_state.physics.push(WorkerTask {
                                        priority: 0,
                                        cv: *cv,
                                    });
                                    sx.send(()).ok();
                                }
                            });
                        }

                        for cv in previous_physics_entities.difference(&current_physics_entities) {
                            chunk_manager.write(*cv, |wc| {
                                if wc.physics_state == PhysicsState::Meshed {
                                    results.physics_unload.push(*cv);
                                }

                                wc.physics_state = PhysicsState::None;
                            });
                        }

                        *previous_physics_entities = current_physics_entities
                    }
                }

                if !worker_state.chunk_edits.is_empty()
                    && let Some(_guard) = worker_state.chunk_edits_lock.try_lock()
                {
                    let mut edits = vec![];
                    while let Some(mut edit) = worker_state.chunk_edits.pop() {
                        edits.extend_from_slice(&mut edit);
                    }

                    let mut voxel_edits: VoxelHashMap<Vec<(LocalVector, VoxelId)>> =
                        VoxelHashMap::default();
                    let mut neighbors_remesh = VoxelHashSet::default();

                    for (wv, voxel_id) in edits {
                        let cv = wv_to_cv(wv);

                        match voxel_edits.entry(cv) {
                            Entry::Occupied(mut occ) => {
                                occ.get_mut().push((wv, voxel_id));
                            }
                            Entry::Vacant(vac) => {
                                vac.insert_entry(vec![(wv, voxel_id)]);
                            }
                        }
                    }

                    let mut edit_work_items = vec![];
                    let mut physics_work_items = vec![];

                    for (cv, edits) in voxel_edits.iter_mut() {
                        chunk_manager.write(*cv, |wc| {
                            if !wc.is_stable() {
                                return;
                            }

                            for (wv, voxel_id) in edits.drain(..) {
                                let lv = wv_to_lv(wv);
                                wc.deltas.0.insert(lv, voxel_id);
                                let old_voxel_id = wc.chunk.set_voxel::<V>(lv, voxel_id);
                                results.edits.push((wv, old_voxel_id));

                                let (neighbors_mask, affected_neighbors) =
                                    wc.chunk.get_voxel_edge_faces(lv);
                                if neighbors_mask.count_ones() > 0 {
                                    for affected in affected_neighbors {
                                        if let Some(face) = affected {
                                            let neighbor_cv = cv + face.into_normal();
                                            neighbors_remesh.insert(neighbor_cv);
                                        }
                                    }
                                }
                            }

                            wc.is_dirty = true;
                            wc.state = ChunkState::Meshing;
                            wc.version = wc.version.wrapping_add(1);

                            edit_work_items.push(WorkerTask {
                                priority: 0,
                                cv: *cv,
                            });

                            if matches!(
                                wc.physics_state,
                                PhysicsState::Meshing | PhysicsState::Meshed
                            ) {
                                wc.physics_state = PhysicsState::Meshing;
                                physics_work_items.push(WorkerTask {
                                    priority: 0,
                                    cv: *cv,
                                });
                            }
                        });
                    }

                    for neighbor_cv in neighbors_remesh {
                        if !voxel_edits.contains_key(&neighbor_cv) {
                            chunk_manager.write(neighbor_cv, |wc| {
                                if !wc.is_stable() {
                                    return;
                                }

                                wc.state = ChunkState::Meshing;
                                wc.version = wc.version.wrapping_add(1);

                                edit_work_items.push(WorkerTask {
                                    priority: 0,
                                    cv: neighbor_cv,
                                });
                            });
                        }
                    }

                    for work_item in edit_work_items {
                        worker_state.edit.push(work_item);
                        sx.send(()).ok();
                    }

                    for work_item in physics_work_items {
                        worker_state.physics.push(work_item);
                        sx.send(()).ok();
                    }
                }

                if let Some(task) = worker_state.physics.pop() {
                    track_task_throughput();
                    let cv = task.cv;

                    let time = Instant::now();
                    match physics_mesh_task::<V>(cv, &chunk_manager, voxel_size) {
                        TaskResult::Ok((version, mesh)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.physics_state == PhysicsState::Meshing
                                    && world_chunk.version == version
                                {
                                    world_chunk.physics_state = PhysicsState::Meshed;

                                    results.physics_load.push((cv, mesh));
                                    sx.send(()).ok();
                                }
                            });
                        }
                        TaskResult::NotReady => {
                            worker_state.physics.push(task);
                        }
                        TaskResult::Invalid => {}
                    }

                    CHUNKEE_CORE_METRICS
                        .histograms
                        .get(Histograms::PhysicsTask)
                        .record(time.elapsed());
                }

                if let Some(task) = worker_state.edit.pop() {
                    track_task_throughput();
                    let cv = task.cv;

                    match edit_mesh_task::<V>(cv, &chunk_manager, voxel_size) {
                        TaskResult::Ok((version, mesh)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.state == ChunkState::Meshing
                                    && world_chunk.version == version
                                {
                                    world_chunk.state = ChunkState::Meshed;

                                    results.mesh_load.push((cv, mesh));
                                    sx.send(()).ok();
                                }
                            });
                        }
                        TaskResult::NotReady => {
                            worker_state.edit.push(task);
                        }
                        TaskResult::Invalid => {}
                    }
                }

                if let Some(task) = worker_state.mesh.pop() {
                    track_task_throughput();
                    let cv = task.cv;

                    let time = Instant::now();
                    match mesh_task::<V>(cv, &chunk_manager, voxel_size) {
                        TaskResult::Ok((version, mesh)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.state == ChunkState::Meshing
                                    && world_chunk.version == version
                                {
                                    world_chunk.state = ChunkState::Meshed;

                                    results.mesh_load.push((cv, mesh));
                                    sx.send(()).ok();
                                }
                            });
                        }
                        TaskResult::NotReady => {
                            worker_state.mesh.push(task).ok();
                        }
                        TaskResult::Invalid => {}
                    }

                    CHUNKEE_CORE_METRICS
                        .histograms
                        .get(Histograms::MeshTask)
                        .record(time.elapsed());
                }

                if let Some(task) = worker_state.load.pop() {
                    let cv = task.cv;

                    let time = Instant::now();
                    match load_task::<V>(cv, &chunk_manager, &chunk_store, &generator, voxel_size) {
                        TaskResult::Ok((mut chunk, deltas, uniform_voxel_id)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.state == ChunkState::Loading {
                                    merge_deltas::<V>(&mut chunk, &deltas);
                                    world_chunk.chunk = chunk;
                                    world_chunk.deltas = deltas;
                                    world_chunk.uniform_voxel_id = uniform_voxel_id;
                                    world_chunk.state = ChunkState::Meshing;

                                    worker_state.mesh.push(task).ok();
                                    sx.send(()).ok();
                                }
                            });
                        }
                        _ => {}
                    }

                    CHUNKEE_CORE_METRICS
                        .histograms
                        .get(Histograms::LoadTask)
                        .record(time.elapsed());
                }

                if !worker_state.is_empty() {
                    sx.send(()).ok();
                }
            }
        }
    }
}

enum TaskResult<T> {
    Ok(T),
    NotReady,
    Invalid,
}

fn load_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_manager: &ChunkManager,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
    voxel_size: f32,
) -> TaskResult<(Chunk, Deltas, Option<VoxelId>)> {
    if !chunk_manager.check(cv, |wc| wc.state == ChunkState::Loading) {
        return TaskResult::Invalid;
    }

    let mut chunk = Chunk::new();

    let time = Instant::now();
    let persisted_chunk = chunk_store.load_chunk(cv);
    CHUNKEE_CORE_METRICS
        .histograms
        .get(Histograms::Load)
        .record(time.elapsed());
    let (deltas, uniform_voxel_id) = persisted_chunk
        .map(|pc| (pc.deltas, pc.uniform_voxel_id))
        .unwrap_or_default();

    if let Some(uniform) = uniform_voxel_id {
        chunk.fill::<V>(uniform);
    } else {
        let time = Instant::now();
        generator.apply(cv_to_wv(cv), &mut chunk, voxel_size);
        CHUNKEE_CORE_METRICS
            .histograms
            .get(Histograms::Generate)
            .record(time.elapsed());
    }

    TaskResult::Ok((chunk, deltas, uniform_voxel_id))
}

fn mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_manager: &ChunkManager,
    voxel_size: f32,
) -> TaskResult<(ChunkVersion, ChunkMeshGroup)> {
    if !chunk_manager.check(cv, |wc| wc.state == ChunkState::Meshing) {
        return TaskResult::Invalid;
    }

    let mut neighbors = Box::new([None; 6]);

    for (idx, neighbor_cv) in neighbors_of(cv).into_iter().enumerate() {
        match chunk_manager.read(neighbor_cv, |wc| wc.is_stable().then_some(wc.chunk.clone())) {
            Some(Some(chunk)) => neighbors[idx] = Some(chunk),
            Some(None) => return TaskResult::NotReady,
            None => {}
        }
    }

    let neighbors = Box::new(neighbors.map(|mut chunk| chunk.take().unwrap_or(Chunk::new())));

    let Some((center, version)) =
        chunk_manager.read(cv, |wc| (Box::new(wc.chunk.clone()), wc.version))
    else {
        return TaskResult::Invalid;
    };

    let mesh = mesh_chunk::<V>(cv, center, neighbors, voxel_size);

    TaskResult::Ok((version, mesh))
}

// Compiler bug, using the mesh_task above causes the app to crash...
fn edit_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_manager: &ChunkManager,
    voxel_size: f32,
) -> TaskResult<(ChunkVersion, ChunkMeshGroup)> {
    if !chunk_manager.check(cv, |wc| wc.state == ChunkState::Meshing) {
        return TaskResult::Invalid;
    }

    let mut neighbors = Box::new([None; 6]);

    for (idx, neighbor_cv) in neighbors_of(cv).into_iter().enumerate() {
        match chunk_manager.read(neighbor_cv, |wc| wc.is_stable().then_some(wc.chunk.clone())) {
            Some(Some(chunk)) => neighbors[idx] = Some(chunk),
            Some(None) => return TaskResult::NotReady,
            None => {}
        }
    }

    let neighbors = Box::new(neighbors.map(|mut chunk| chunk.take().unwrap_or(Chunk::new())));

    let Some((center, version)) =
        chunk_manager.read(cv, |wc| (Box::new(wc.chunk.clone()), wc.version))
    else {
        return TaskResult::Invalid;
    };

    let mesh = mesh_chunk::<V>(cv, center, neighbors, voxel_size);

    TaskResult::Ok((version, mesh))
}

fn physics_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_manager: &ChunkManager,
    voxel_size: f32,
) -> TaskResult<(ChunkVersion, Vec<Vec3>)> {
    let (chunk, version) = match chunk_manager.read(cv, |wc| {
        wc.is_stable()
            .then_some((Box::new(wc.chunk.clone()), wc.version))
    }) {
        Some(Some(res)) => res,
        Some(None) => {
            return TaskResult::NotReady;
        }
        None => {
            return TaskResult::Invalid;
        }
    };

    let mesh = mesh_physics_chunk::<V>(cv, chunk, voxel_size);

    TaskResult::Ok((version, mesh))
}

pub fn merge_deltas<V: ChunkeeVoxel>(chunk: &mut Chunk, deltas: &Deltas) {
    for (lv, voxel_id) in deltas.0.iter() {
        chunk.set_voxel::<V>(*lv, *voxel_id);
    }
}
