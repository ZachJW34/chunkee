use std::{
    cmp::Ordering,
    collections::hash_map::Entry,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, Vec3};

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{CHUNK_SIDE_32, Chunk},
    coords::{
        ChunkVector, LocalVector, NEIGHBOR_OFFSETS, camera_vec3_to_cv, cv_to_wv, wv_to_cv, wv_to_lv,
    },
    define_metrics,
    generation::VoxelGenerator,
    grid::{
        ChunkManager, GridView, check_chunk_p, check_chunk_pv, check_chunk_s, check_chunk_stable,
        check_chunk_sv,
    },
    hasher::{VoxelHashMap, VoxelHashSet},
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    metrics::Metrics,
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, compute_priority},
    world::ResultQueues,
};

#[derive(Debug, Clone)]
pub struct Deltas(pub VoxelHashMap<VoxelId>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub fn neighbors_of(cv: IVec3) -> [ChunkVector; 6] {
    BLOCK_FACES.map(|face| cv + face.into_normal())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkState {
    None,
    Loading,
    Meshing,
    MeshReady,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PhysicsState {
    None,
    Meshing,
    MeshReady,
}

type ChunkVersion = u32;

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
        matches!(self.state, ChunkState::Meshing | ChunkState::MeshReady)
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
}

define_metrics! {
    pub enum PipelineMetrics {
        PipelineLoop => "Pipeline::loop",
        GridUpdate => "Grid::update",
        IoTask => "Task::IO",
        GenerationTask => "Task::Generation",
        MeshTask => "Task::Mesh",
        PhysicsTask => "Task::Physics",
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

pub fn spawn_pipeline_thread<V: 'static + ChunkeeVoxel>(
    chunk_manager: Arc<ChunkManager>,
    chunk_store: Arc<ChunkStore>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    message_receiver: Receiver<PipelineMessage>,
    results: Arc<ResultQueues>,
    radius: u32,
    voxel_size: f32,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut metrics = Metrics::<PipelineMetrics>::new(Duration::from_secs(2));
        let mut camera_data: Option<CameraData> = None;
        let thread_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let mut previous_physics_entities = VoxelHashSet::default();

        // let total_tasks = calc_total_chunks(radius) as usize;
        let work_queues = Arc::new(WorkQueues {
            load: SegQueue::new(),
            mesh: SegQueue::new(),
            physics: SegQueue::new(),
        });

        let worker_pool = WorkerPool::new::<V>(
            12,
            work_queues.clone(),
            chunk_manager.clone(),
            chunk_store.clone(),
            generator.clone(),
            results.clone(),
            voxel_size,
        );

        loop {
            metrics.batch_print();
            let mut camera_moved = false;
            for message in message_receiver.try_iter() {
                match message {
                    PipelineMessage::Shutdown => return,
                    PipelineMessage::CameraDataUpdate(new_camera_data) => {
                        if camera_data.as_ref().map_or(true, |old_camera_data| {
                            old_camera_data.pos != new_camera_data.pos
                        }) {
                            camera_moved = true;
                        }
                        camera_data.replace(new_camera_data);
                    }
                    PipelineMessage::PhysicsEntitiesUpdate(new_entities) => {
                        if chunk_manager.view.load().initialized {
                            let mut current_physics_entities = VoxelHashSet::default();
                            let physics_radius = ((CHUNK_SIDE_32 / 2) as f32) * voxel_size;

                            for entity in &new_entities {
                                for offset in NEIGHBOR_OFFSETS {
                                    let scaled_offset = offset.as_vec3() * physics_radius;
                                    let neighbor_cv =
                                        camera_vec3_to_cv(entity.pos + scaled_offset, voxel_size);

                                    current_physics_entities.insert(neighbor_cv);
                                }
                            }

                            for cv in
                                current_physics_entities.difference(&previous_physics_entities)
                            {
                                chunk_manager.write(*cv, |wc| {
                                    if wc.physics_state == PhysicsState::None {
                                        wc.physics_state = PhysicsState::Meshing;
                                        work_queues.physics.push(WorkerTask {
                                            priority: 0,
                                            cv: *cv,
                                        });
                                        worker_pool.sender.send(()).ok();
                                    }
                                });
                            }

                            for cv in
                                previous_physics_entities.difference(&current_physics_entities)
                            {
                                chunk_manager.write(*cv, |wc| {
                                    if wc.physics_state == PhysicsState::MeshReady {
                                        results.physics_unload.push(*cv);
                                    }

                                    wc.physics_state = PhysicsState::None;
                                });
                            }

                            previous_physics_entities = current_physics_entities;
                        }
                    }
                    PipelineMessage::ChunkEdits(edits) => {
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

                        let mut mesh_work_items = vec![];
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

                                mesh_work_items.push(WorkerTask {
                                    priority: 0,
                                    cv: *cv,
                                });

                                if matches!(
                                    wc.physics_state,
                                    PhysicsState::Meshing | PhysicsState::MeshReady
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

                                    mesh_work_items.push(WorkerTask {
                                        priority: 0,
                                        cv: neighbor_cv,
                                    });
                                });
                            }
                        }

                        for work_item in mesh_work_items {
                            work_queues.mesh.push(work_item);
                            worker_pool.sender.send(()).ok();
                        }

                        for work_item in physics_work_items {
                            work_queues.physics.push(work_item);
                            worker_pool.sender.send(()).ok();
                        }
                    }
                }
            }

            if camera_data.is_none() {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let camera_data = camera_data.as_ref().unwrap();
            let camera_cv = camera_vec3_to_cv(camera_data.pos, voxel_size);

            if camera_moved {
                let mut dirty_chunks = Vec::new();
                let mut new_load_tasks = Vec::new();

                let current_view = chunk_manager.view.load();
                if current_view.needs_update(camera_cv) {
                    let grid_update_time = Instant::now();
                    let (new_view, reset_tasks) = GridView::compute_new(&current_view, camera_cv);
                    chunk_manager.view.store(Arc::new(new_view));

                    for (chunk_idx, new_cv) in reset_tasks {
                        let mut world_chunk = chunk_manager.pool.flat[chunk_idx].write();

                        if world_chunk.state != ChunkState::None {
                            if world_chunk.state == ChunkState::MeshReady {
                                results.mesh_unload.push(world_chunk.cv);
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
                            priority: compute_priority(new_cv, camera_data, voxel_size),
                        });
                        worker_pool.sender.send(()).ok();
                    }

                    if !dirty_chunks.is_empty() {
                        batch_unload_deltas_task(dirty_chunks, &chunk_store, &thread_pool);
                    }

                    new_load_tasks.sort();
                    for item in new_load_tasks {
                        work_queues.load.push(item);
                    }
                    worker_pool.sender.send(()).ok();

                    metrics
                        .get_mut(PipelineMetrics::GridUpdate)
                        .record(grid_update_time.elapsed());

                    work_queues.print_len();
                    thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                }

                let loop_time = Instant::now();
                let view = chunk_manager.view.load();
                if !view.initialized {
                    thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                }

                if !work_queues.is_empty() {
                    work_queues.reprioritze_queues(camera_data, &view, voxel_size);

                    worker_pool.sender.send(()).ok();
                }

                metrics
                    .get_mut(PipelineMetrics::PipelineLoop)
                    .record(loop_time.elapsed());
            }

            if !work_queues.is_empty() {
                work_queues.print_len();
                worker_pool.sender.send(()).ok();
            }

            thread::sleep(Duration::from_millis(5));
        }
    })
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

struct WorkQueues {
    load: SegQueue<WorkerTask>,
    mesh: SegQueue<WorkerTask>,
    physics: SegQueue<WorkerTask>,
}

impl WorkQueues {
    fn is_empty(&self) -> bool {
        self.load.is_empty() && self.mesh.is_empty() && self.physics.is_empty()
    }

    fn print_len(&self) {
        println!(
            "load={} | mesh={} | physics={}",
            self.load.len(),
            self.mesh.len(),
            self.physics.len(),
        )
    }

    fn reprioritze_queues(&self, camera_data: &CameraData, view: &GridView, voxel_size: f32) {
        WorkQueues::repriotize_queue(&self.load, camera_data, view, voxel_size);
        WorkQueues::repriotize_queue(&self.mesh, camera_data, view, voxel_size);
    }

    fn repriotize_queue(
        queue: &SegQueue<WorkerTask>,
        camera_data: &CameraData,
        view: &GridView,
        voxel_size: f32,
    ) {
        let mut next_tasks = Vec::with_capacity(queue.len());
        while let Some(task) = queue.pop() {
            if GridView::cv_to_idx_with_origin(task.cv, view.grid_origin, view.dimensions).is_some()
            {
                next_tasks.push(WorkerTask {
                    cv: task.cv,
                    priority: compute_priority(task.cv, camera_data, voxel_size),
                });
            }
        }
        next_tasks.sort();
        for task in next_tasks {
            queue.push(task);
        }
    }
}

struct WorkerPool {
    pub sender: Sender<()>,
}

impl WorkerPool {
    fn new<V: 'static + ChunkeeVoxel>(
        num_threads: usize,
        work_queues: Arc<WorkQueues>,
        chunk_manager: Arc<ChunkManager>,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        results: Arc<ResultQueues>,
        voxel_size: f32,
    ) -> Self {
        let (sx, rx) = crossbeam_channel::unbounded();

        for _ in 0..num_threads {
            let work_queues = Arc::clone(&work_queues);
            let chunk_manager = Arc::clone(&chunk_manager);
            let chunk_store = Arc::clone(&chunk_store);
            let generator = Arc::clone(&generator);
            let results = results.clone();

            let thread_sx = sx.clone();
            let thread_rx = rx.clone();
            let voxel_size = voxel_size;

            thread::spawn(move || {
                Self::worker_loop::<V>(
                    &work_queues,
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
        work_queues: &WorkQueues,
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
                if let Some(work_item) = work_queues.physics.pop() {
                    let cv = work_item.cv;

                    match physics_mesh_task::<V>(cv, &chunk_manager, voxel_size) {
                        TaskResult::Ok((version, mesh)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.physics_state == PhysicsState::Meshing
                                    && world_chunk.version == version
                                {
                                    world_chunk.physics_state = PhysicsState::MeshReady;

                                    results.physics_load.push((cv, mesh));
                                    sx.send(()).ok();
                                }
                            });
                        }
                        TaskResult::NotReady => {
                            work_queues.physics.push(work_item);
                        }
                        TaskResult::Invalid => {}
                    }
                }

                if let Some(work_item) = work_queues.mesh.pop() {
                    let cv = work_item.cv;

                    match mesh_task::<V>(cv, &chunk_manager, voxel_size) {
                        TaskResult::Ok((version, mesh)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.state == ChunkState::Meshing
                                    && world_chunk.version == version
                                {
                                    world_chunk.state = ChunkState::MeshReady;

                                    results.mesh_load.push((cv, mesh));
                                    sx.send(()).ok();
                                }
                            });
                        }
                        TaskResult::NotReady => {
                            work_queues.mesh.push(work_item);
                        }
                        TaskResult::Invalid => {}
                    }
                }

                if let Some(work_item) = work_queues.load.pop() {
                    let cv = work_item.cv;

                    match load_task::<V>(cv, &chunk_manager, &chunk_store, &generator, voxel_size) {
                        TaskResult::Ok((mut chunk, deltas, uniform_voxel_id)) => {
                            chunk_manager.write(cv, |world_chunk| {
                                if world_chunk.state == ChunkState::Loading {
                                    merge_deltas::<V>(&mut chunk, &deltas);
                                    world_chunk.chunk = chunk;
                                    world_chunk.deltas = deltas;
                                    world_chunk.uniform_voxel_id = uniform_voxel_id;
                                    world_chunk.state = ChunkState::Meshing;

                                    work_queues.mesh.push(work_item);
                                    sx.send(()).ok();
                                }
                            });
                        }
                        _ => {}
                    }
                }

                if !work_queues.is_empty() {
                    // work_queues.print_len();
                    sx.send(()).ok();
                }
            }
        }
    }
}

fn batch_unload_deltas_task(
    chunks_to_save: Vec<(ChunkVector, PersistedChunk)>,
    chunk_store: &Arc<ChunkStore>,
    thread_pool: &rayon::ThreadPool,
) {
    let chunk_store = chunk_store.clone();
    thread_pool.spawn(move || {
        chunk_store.save_chunks(chunks_to_save);
    });
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

    let persisted_chunk = chunk_store.load_chunk(cv);
    let (deltas, uniform_voxel_id) = persisted_chunk
        .map(|pc| (pc.deltas, pc.uniform_voxel_id))
        .unwrap_or_default();

    if let Some(uniform) = uniform_voxel_id {
        chunk.fill::<V>(uniform);
    } else {
        generator.apply(cv_to_wv(cv), &mut chunk, voxel_size);
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
