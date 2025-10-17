use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    chunk::{CHUNK_SIDE_32, Chunk},
    chunk_view::{ChunkGrid, ChunkPool, ChunkView},
    coords::{
        ChunkVector, LocalVector, NEIGHBOR_OFFSETS, WorldVector, cv_to_wv, vec3_wv_to_cv, wv_to_cv,
    },
    generation::VoxelGenerator,
    hasher::{VoxelHashMap, VoxelHashSet},
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    pipeline::{ChunkState, ChunkVersion, Deltas, PhysicsState, neighbors_of},
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, compute_priority},
};
use crossbeam::{queue::SegQueue, select};
use crossbeam_channel::{Receiver, RecvError, Sender};
use glam::{IVec3, Vec3};
use parking_lot::{Condvar, Mutex};
use std::{
    collections::hash_map::Entry,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle, sleep},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub enum ManagerMessage {
    CameraUpdate(CameraData),
    PlayerEdits(Vec<(IVec3, VoxelId)>),
    PhysicsEntitiesUpdate(Vec<WorldVector>),
    Terminate(),
}

pub type PhysicsMesh = Vec<Vec3>;

#[derive(Debug)]
pub enum WorkerResult {
    Load(ChunkVector, Chunk, Deltas, Option<VoxelId>),
    Mesh(ChunkVector, ChunkVersion, ChunkMeshGroup),
    PhysicsMesh(ChunkVector, ChunkVersion, PhysicsMesh),
}

struct MeshPayload {
    version: ChunkVersion,
    chunk: Box<Chunk>,
    neighbors: Box<[Chunk; 6]>,
}

struct PhysicsPayload {
    version: ChunkVersion,
    chunk: Box<Chunk>,
}

struct WorkerJob<T> {
    cv: ChunkVector,
    priority: u32,
    payload: T,
}

type LoadJob = WorkerJob<()>;
type MeshJob = WorkerJob<MeshPayload>;
type PhysicsJob = WorkerJob<PhysicsPayload>;

impl<T> WorkerJob<T> {
    fn new(cv: ChunkVector, priority: u32, payload: T) -> Self {
        Self {
            cv,
            priority,
            payload,
        }
    }
}

impl<T> PartialEq for WorkerJob<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl<T> Eq for WorkerJob<T> {}
impl<T> PartialOrd for WorkerJob<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for WorkerJob<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

struct Queues {
    load: SegQueue<LoadJob>,
    mesh: SegQueue<MeshJob>,
    edits: SegQueue<MeshJob>,
    physics: SegQueue<PhysicsJob>,
}

impl Queues {
    fn new() -> Self {
        Self {
            load: SegQueue::new(),
            mesh: SegQueue::new(),
            physics: SegQueue::new(),
            edits: SegQueue::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.load.is_empty()
    }

    fn print(&self) {
        println!(
            "load={} | mesh={} | physics={} | edit={}",
            self.load.len(),
            self.mesh.len(),
            self.physics.len(),
            self.edits.len()
        )
    }

    fn reprioritze_queues(&self, camera_data: &CameraData, view: &ChunkGrid, voxel_size: f32) {
        Queues::repriotize_queue(&self.load, camera_data, view, voxel_size);
        Queues::repriotize_queue(&self.mesh, camera_data, view, voxel_size);
    }

    fn repriotize_queue<T>(
        queue: &SegQueue<WorkerJob<T>>,
        camera_data: &CameraData,
        view: &ChunkGrid,
        voxel_size: f32,
    ) {
        let mut next_tasks = Vec::with_capacity(queue.len());
        while let Some(job) = queue.pop() {
            if view.cv_to_idx_with_origin(job.cv).is_some() {
                let priority = compute_priority(job.cv, camera_data, voxel_size);
                let new_job = WorkerJob::<T>::new(job.cv, priority, job.payload);
                next_tasks.push(new_job);
            }
        }
        next_tasks.sort();
        for task in next_tasks {
            queue.push(task);
        }
    }
}

struct JobQueue {
    queues: Queues,
    mutex: Mutex<()>,
    cvar: Condvar,
}

impl JobQueue {
    fn new() -> Self {
        Self {
            queues: Queues::new(),
            mutex: Mutex::new(()),
            cvar: Condvar::new(),
        }
    }
}

pub enum Update {
    Mesh(ChunkVector, ChunkMeshGroup),
    MeshUnload(ChunkVector),
    Physics(ChunkVector, PhysicsMesh),
    PhysicsUnload(ChunkVector),
    // Edits()
}

pub struct ChunkeeConfig {
    pub radius: u32,
    pub voxel_size: f32,
    pub thread_count: usize,
    pub generator: Box<dyn VoxelGenerator>,
}

pub struct ChunkeeWorldManager<V: 'static + ChunkeeVoxel> {
    pub updates: Arc<SegQueue<Update>>,
    camera_data: Option<CameraData>,
    jobs: Arc<JobQueue>,
    manager: Option<JoinHandle<()>>,
    workers: Vec<JoinHandle<()>>,
    main_tx: Sender<ManagerMessage>,
    main_rx: Receiver<ManagerMessage>,
    active: Arc<AtomicBool>,
    config: Arc<ChunkeeConfig>,
    chunk_view: Arc<ChunkView>,
    chunk_store: Arc<ChunkStore>,
    _voxel_type: PhantomData<V>,
}

const CAMERA_THRESHOLD_RADIANS: f32 = 1.0;

impl<V: 'static + ChunkeeVoxel> ChunkeeWorldManager<V> {
    pub fn new(config: ChunkeeConfig) -> Self {
        let (main_tx, main_rx) = crossbeam_channel::unbounded();
        let chunk_view = ChunkView::new(config.radius);
        let chunk_store = ChunkStore::new();
        Self {
            config: Arc::new(config),
            jobs: Arc::new(JobQueue::new()),
            manager: None,
            workers: vec![],
            main_tx,
            main_rx,
            active: Arc::new(AtomicBool::new(false)),
            chunk_view: Arc::new(chunk_view),
            chunk_store: Arc::new(chunk_store),
            updates: Arc::new(SegQueue::new()),
            camera_data: None,
            _voxel_type: PhantomData,
        }
    }

    pub fn start(&mut self) {
        if self.active.swap(true, Ordering::Release) {
            // already active
            return;
        }

        assert!(
            self.config.thread_count > 0,
            "Chunkee requires at least one thread"
        );

        let main_rx = self.main_rx.clone();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<WorkerResult>();
        let chunk_view = self.chunk_view.clone();
        let job_queue = self.jobs.clone();
        let updates = self.updates.clone();
        let config = self.config.clone();
        let manager_thread = thread::spawn(move || {
            manager_loop::<V>(main_rx, result_rx, job_queue, updates, chunk_view, config)
        });

        let worker_threads_count = (self.config.thread_count - 1) as usize;
        let mut worker_threads = Vec::with_capacity(worker_threads_count as usize);
        for _ in 0..worker_threads_count {
            let job_queue = self.jobs.clone();
            let result_tx = result_tx.clone();
            let active = self.active.clone();
            let chunk_store = self.chunk_store.clone();
            let config = self.config.clone();
            worker_threads.push(thread::spawn(move || {
                worker_loop::<V>(job_queue, result_tx, active, chunk_store, config)
            }))
        }

        self.manager = Some(manager_thread);
        self.workers = worker_threads;
    }

    pub fn stop(&mut self) {
        if !self.active.swap(false, Ordering::Release) {
            return;
        }

        self.main_tx.send(ManagerMessage::Terminate()).ok();
        if let Some(manager) = self.manager.take() {
            manager.join().ok();
        }

        self.jobs.cvar.notify_all();

        for worker in self.workers.drain(..) {
            worker.join().ok();
        }
    }

    pub fn update_camera(&mut self, camera_data: CameraData) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }

        if self.camera_data.is_none_or(|previous| {
            let current_cv = vec3_wv_to_cv(camera_data.pos, self.config.voxel_size);
            let previous_cv = vec3_wv_to_cv(previous.pos, self.config.voxel_size);

            (current_cv != previous_cv)
                || (camera_data.forward.angle_between(previous.forward) > CAMERA_THRESHOLD_RADIANS)
        }) {
            self.main_tx
                .send(ManagerMessage::CameraUpdate(camera_data))
                .ok();
            self.camera_data = Some(camera_data);
        }
    }

    pub fn set_voxels_at(&self, changes: &[(WorldVector, V)]) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }

        let edits: Vec<(WorldVector, VoxelId)> = changes
            .iter()
            .map(|(wv, voxel)| (*wv, VoxelId::new((*voxel).into(), Rotation::default())))
            .collect();

        self.main_tx.send(ManagerMessage::PlayerEdits(edits)).ok();
    }

    pub fn update_physics_entities(&self, entities: Vec<Vec3>) {
        // if let Some(worker_pool) = self.worker_pool.as_ref() {
        //     self.worker_state.physics_entities.push(entities);
        //     worker_pool.sender.send(()).ok();
        // }
    }
}

// impl<V: 'static + ChunkeeVoxel> Drop for ChunkeeWorldManager<V> {
//     fn drop(&mut self) {
//         self.stop();
//     }
// }

fn manager_loop<V: 'static + ChunkeeVoxel>(
    main_rx: Receiver<ManagerMessage>,
    result_rx: Receiver<WorkerResult>,
    job_queue: Arc<JobQueue>,
    updates: Arc<SegQueue<Update>>,
    chunk_view: Arc<ChunkView>,
    config: Arc<ChunkeeConfig>,
) {
    let mut physics_handler = PhysicsHandler::new();
    let mut camera_data: Option<CameraData> = None;
    loop {
        let time = Instant::now();
        let (results, camera_update, player_edits, new_physics_entities) =
            match drain_channels(&main_rx, &result_rx) {
                Ok(res) => res,
                Err(ManagerChannelError::RecV(e)) => {
                    println!("Channel disconnected");
                    return;
                }
                Err(ManagerChannelError::Terminate) => {
                    return;
                }
            };

        if camera_data.is_none() && camera_update.is_none() {
            continue;
        }

        if let Some(cam) = camera_update {
            camera_data.replace(cam);
        }

        println!(
            "Results={} camera_update={:?}",
            results.len(),
            camera_update
        );

        // Chunkview update
        let cam = camera_data.unwrap();
        let cam_cv = vec3_wv_to_cv(cam.pos, config.voxel_size);
        let mut chunks_to_reset = Vec::new();
        {
            let current_chunk_grid = chunk_view.grid.load();
            if current_chunk_grid.needs_update(cam_cv) {
                let time = Instant::now();
                let (new_view, reset_tasks, remesh_cvs) =
                    ChunkGrid::compute_new(&current_chunk_grid, cam_cv);
                chunk_view.grid.store(Arc::new(new_view));
                chunks_to_reset.extend_from_slice(&reset_tasks);
                println!("ChunkView update: t={:?}", time.elapsed());
            }
        }

        let grid = chunk_view.grid.load();
        let mut chunks_to_check = Vec::new();

        {
            let mut chunk_pool_guard = chunk_view.pool.write();

            // resets
            let mut dirty_chunks = Vec::new();
            for (chunk_idx, new_cv) in chunks_to_reset {
                let wc = &mut chunk_pool_guard.flat[chunk_idx];

                if wc.state != ChunkState::None {
                    if wc.is_dirty {
                        let persisted_chunk = PersistedChunk {
                            uniform_voxel_id: wc.uniform_voxel_id.take(),
                            deltas: std::mem::take(&mut wc.deltas),
                        };
                        dirty_chunks.push((wc.cv, persisted_chunk));
                    }

                    updates.push(Update::MeshUnload(wc.cv));
                }

                wc.reset(new_cv);
                let priority = compute_priority(wc.cv, &cam, config.voxel_size);
                job_queue
                    .queues
                    .load
                    .push(LoadJob::new(wc.cv, priority, ()));
            }

            // results
            let time = Instant::now();
            for result in results {
                match result {
                    WorkerResult::Load(cv, chunk, deltas, uniform) => {
                        if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid)
                            && wc.is_loading()
                        {
                            // println!("Loaded {}", wc.cv);
                            wc.chunk = chunk;
                            wc.deltas = deltas;
                            wc.uniform_voxel_id = uniform;
                            wc.state = ChunkState::NeedsMesh;
                            chunks_to_check.push(wc.cv);

                            for cv in neighbors_of(wc.cv) {
                                chunks_to_check.push(cv);
                            }
                        };
                    }
                    WorkerResult::Mesh(cv, version, chunk_mesh_group) => {
                        if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid)
                            && wc.is_meshing()
                            && wc.version == version
                        {
                            // println!("Meshed {}", wc.cv);
                            wc.state = ChunkState::Meshed;
                            updates.push(Update::Mesh(cv, chunk_mesh_group));
                        }
                    }
                    WorkerResult::PhysicsMesh(cv, version, physics_mesh) => {
                        if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid)
                            && wc.version == version
                        {
                            wc.physics_state = PhysicsState::Meshed;
                            updates.push(Update::Physics(cv, physics_mesh));
                        }
                    }
                }
            }
            println!("Results processing: {:?}", time.elapsed());

            // Physics update
            if let Some(entities) = new_physics_entities
                && !entities.is_empty()
            {
                let (new_pmeshes, stale_pmeshes) =
                    physics_handler.update(entities, config.voxel_size);

                for cv in new_pmeshes {
                    if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid)
                        && wc.physics_state == PhysicsState::None
                    {
                        wc.physics_state = PhysicsState::NeedsMesh;
                    }
                }

                for cv in stale_pmeshes {
                    if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid) {
                        if wc.physics_state == PhysicsState::Meshed {
                            updates.push(Update::PhysicsUnload(wc.cv));
                        }
                        wc.physics_state = PhysicsState::None;
                    }
                }
            }

            // edits
            if !player_edits.is_empty() {
                let mut voxel_edits: VoxelHashMap<Vec<(LocalVector, VoxelId)>> =
                    VoxelHashMap::default();
                let mut neighbors_remesh = VoxelHashSet::default();

                for (wv, voxel_id) in player_edits {
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

                for (cv, edits) in voxel_edits.iter_mut() {
                    // only allow changes if already meshed
                    if let Some(wc) = chunk_pool_guard.get_mut(*cv, &grid)
                        && wc.is_stable()
                    {
                        for (wv, voxel_id) in edits.drain(..) {
                            let lv = wv_to_cv(wv);
                            wc.deltas.0.insert(lv, voxel_id);
                            let old_voxel_id = wc.chunk.set_voxel::<V>(lv, voxel_id);
                            // TODO: handle edits
                            // results.edits.push((wv, old_voxel_id));

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
                        wc.state = ChunkState::NeedsMesh;
                        wc.version = wc.version.wrapping_add(1);
                        chunks_to_check.push(wc.cv);

                        if matches!(
                            wc.physics_state,
                            PhysicsState::Meshing | PhysicsState::Meshed
                        ) {
                            wc.physics_state = PhysicsState::NeedsMesh;
                        }
                    };
                }

                for neighbor_cv in neighbors_remesh {
                    if !voxel_edits.contains_key(&neighbor_cv)
                        && let Some(wc) = chunk_pool_guard.get_mut(neighbor_cv, &grid)
                        && wc.is_stable()
                    {
                        wc.state = ChunkState::NeedsMesh;
                        wc.version = wc.version.wrapping_add(1);
                        chunks_to_check.push(wc.cv);
                    };
                }
            }

            drop(chunk_pool_guard);
            // TODO: save dirty chunks
        }

        let chunks_to_check: Vec<_> = chunks_to_check
            .iter()
            .filter(|cv| chunks_to_check.contains(cv))
            .collect();

        let mut meshing_cvs = Vec::new();
        let mut pmeshing_cvs = Vec::new();
        {
            let chunk_pool_guard = chunk_view.pool.read();

            // println!("chunks_to_check={}", chunks_to_check.len());

            for cv in chunks_to_check {
                if let Some(wc) = chunk_pool_guard.get(*cv, &grid)
                    && wc.state == ChunkState::NeedsMesh
                    && let Some(payload) =
                        create_mesh_payload_if_ready(wc.cv, &chunk_pool_guard, &grid)
                {
                    // println!("Created mesh job for {}", wc.cv);
                    meshing_cvs.push(wc.cv);
                    let priority = compute_priority(wc.cv, &cam, config.voxel_size);
                    job_queue
                        .queues
                        .mesh
                        .push(MeshJob::new(wc.cv, priority, payload));
                }
            }

            for cv in &physics_handler.p_meshes {
                if let Some(wc) = chunk_pool_guard.get(*cv, &grid)
                    && wc.physics_state == PhysicsState::NeedsMesh
                {
                    pmeshing_cvs.push(wc.cv);
                    let priority = compute_priority(wc.cv, &cam, config.voxel_size);

                    job_queue.queues.physics.push(PhysicsJob::new(
                        wc.cv,
                        priority,
                        PhysicsPayload {
                            version: wc.version,
                            chunk: Box::new(wc.chunk.clone()),
                        },
                    ));
                }
            }
        }

        {
            let mut chunk_pool_guard = chunk_view.pool.write();

            for cv in meshing_cvs {
                if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid) {
                    wc.state = ChunkState::Meshing
                }
            }

            for cv in pmeshing_cvs {
                if let Some(wc) = chunk_pool_guard.get_mut(cv, &grid) {
                    wc.physics_state = PhysicsState::Meshing;
                }
            }
        }

        if camera_update.is_some() {
            let time = Instant::now();
            job_queue
                .queues
                .reprioritze_queues(&cam, &grid, config.voxel_size);
            println!("ReprioritizeQueues: t={:?}", time.elapsed());
        }

        if !job_queue.queues.is_empty() {
            // job_queue.queues.print();
            job_queue.cvar.notify_all();
        }

        println!("Loop time: {:?}", time.elapsed());
        sleep(Duration::from_millis(10));
    }
}

fn worker_loop<V: 'static + ChunkeeVoxel>(
    job_queue: Arc<JobQueue>,
    result_tx: Sender<WorkerResult>,
    active: Arc<AtomicBool>,
    chunk_store: Arc<ChunkStore>,
    config: Arc<ChunkeeConfig>,
) {
    loop {
        let mut guard = job_queue.mutex.lock();

        job_queue.cvar.wait_while(&mut guard, |_| {
            job_queue.queues.is_empty() && active.load(Ordering::Acquire)
        });

        if !active.load(Ordering::Acquire) && job_queue.queues.is_empty() {
            break;
        }

        while !job_queue.queues.is_empty() && active.load(Ordering::Acquire) {
            if let Some(job) = job_queue.queues.physics.pop() {
                let mesh = mesh_physics_chunk::<V>(job.cv, job.payload.chunk, config.voxel_size);
                result_tx
                    .send(WorkerResult::PhysicsMesh(job.cv, job.payload.version, mesh))
                    .ok();
            }

            if let Some(job) = job_queue.queues.edits.pop() {
                let mesh_group = mesh_chunk::<V>(
                    job.cv,
                    job.payload.chunk,
                    job.payload.neighbors,
                    config.voxel_size,
                );
                result_tx
                    .send(WorkerResult::Mesh(job.cv, job.payload.version, mesh_group))
                    .ok();
            }

            if let Some(job) = job_queue.queues.mesh.pop() {
                let mesh_group = mesh_chunk::<V>(
                    job.cv,
                    job.payload.chunk,
                    job.payload.neighbors,
                    config.voxel_size,
                );
                result_tx
                    .send(WorkerResult::Mesh(job.cv, job.payload.version, mesh_group))
                    .ok();
            }

            if let Some(job) = job_queue.queues.load.pop() {
                let (chunk, deltas, uniform) =
                    load_chunk::<V>(job.cv, &chunk_store, &config.generator, config.voxel_size);
                result_tx
                    .send(WorkerResult::Load(job.cv, chunk, deltas, uniform))
                    .ok();
            }
        }

        // if let Some(cv) = queue.load.pop_front() {
        //     drop(queue);

        //     todo!();
        // }
    }
}

fn create_mesh_payload_if_ready(
    cv: ChunkVector,
    chunk_pool: &ChunkPool,
    chunk_grid: &ChunkGrid,
) -> Option<MeshPayload> {
    if let Some(wc) = chunk_pool.get(cv, chunk_grid)
        && wc.is_stable()
    {
        let neighbors = if let Some(neighbors) = collect_neighbors(cv, chunk_pool, chunk_grid) {
            neighbors
        } else {
            return None;
        };

        let chunk = Box::new(wc.chunk.clone());
        let version = wc.version;

        Some(MeshPayload {
            version,
            chunk,
            neighbors,
        })
    } else {
        None
    }
}

type DrainedChannels = (
    Vec<WorkerResult>,
    Option<CameraData>,
    Vec<(IVec3, VoxelId)>,
    Option<VoxelHashSet>,
);

enum ManagerChannelError {
    RecV(RecvError),
    Terminate,
}

fn drain_channels(
    main_rx: &Receiver<ManagerMessage>,
    result_rx: &Receiver<WorkerResult>,
) -> Result<DrainedChannels, ManagerChannelError> {
    let mut messages = Vec::new();
    let mut results = Vec::new();
    let mut camera_update = None;
    let mut player_edits = Vec::new();
    let mut new_physics_entities = None;

    select! {
      recv(main_rx) -> msg => {
        match msg {
            Ok(msg) => {
              messages.push(msg);
            },
            Err(e) => {
              return Err(ManagerChannelError::RecV(e));
            }
        }

      },
      recv(result_rx) -> res => {
        match res {
            Ok(res) => {
              results.push(res);
            },
            Err(e) => {
              return Err(ManagerChannelError::RecV(e));
            }
        }
      },
    }

    while let Ok(msg) = main_rx.try_recv() {
        messages.push(msg)
    }

    while let Ok(res) = result_rx.try_recv() {
        results.push(res);
    }

    for msg in messages {
        match msg {
            ManagerMessage::Terminate() => return Err(ManagerChannelError::Terminate),
            ManagerMessage::CameraUpdate(camera_data) => {
                camera_update.replace(camera_data);
            }
            ManagerMessage::PlayerEdits(edits) => {
                player_edits.extend_from_slice(&edits);
            }
            ManagerMessage::PhysicsEntitiesUpdate(entities) => {
                if new_physics_entities.is_none() {
                    new_physics_entities = Some(VoxelHashSet::default());
                }

                if let Some(nep) = new_physics_entities.as_mut() {
                    nep.extend(entities.into_iter());
                }
            }
        }
    }

    Ok((results, camera_update, player_edits, new_physics_entities))
}

fn collect_neighbors(
    cv: ChunkVector,
    chunk_pool: &ChunkPool,
    chunk_grid: &ChunkGrid,
) -> Option<Box<[Chunk; 6]>> {
    let neighbor_cvs = neighbors_of(cv);
    let mut neighbor_chunks = [None; 6];

    for (idx, neighbor_cv) in neighbor_cvs.iter().enumerate() {
        if let Some(neighbor_chunk) = chunk_pool.get(*neighbor_cv, chunk_grid) {
            if neighbor_chunk.is_stable() {
                neighbor_chunks[idx] = Some(neighbor_chunk);
            } else {
                return None;
            }
        }
    }

    let mut neighbors = Box::new([None; 6]);
    for (idx, maybe_neighbor) in neighbor_chunks.into_iter().enumerate() {
        if let Some(neighbor_chunk) = maybe_neighbor {
            neighbors[idx] = Some(neighbor_chunk.chunk.clone());
        } else {
            neighbors[idx] = Some(Chunk::new());
        }
    }

    let neighbors = Box::new(neighbors.map(|mut chunk| chunk.take().unwrap_or(Chunk::new())));

    Some(neighbors)
}

fn load_chunk<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
    voxel_size: f32,
) -> (Chunk, Deltas, Option<VoxelId>) {
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

    (chunk, deltas, uniform_voxel_id)
}

struct PhysicsHandler {
    p_meshes: VoxelHashSet,
}

impl PhysicsHandler {
    fn new() -> Self {
        Self {
            p_meshes: VoxelHashSet::default(),
        }
    }

    fn update(
        &mut self,
        entities: VoxelHashSet,
        voxel_size: f32,
    ) -> (Vec<ChunkVector>, Vec<ChunkVector>) {
        let mut required_pmeshes = VoxelHashSet::default();
        let physics_radius = ((CHUNK_SIDE_32 / 2) as f32) * voxel_size;
        for entity in entities {
            for offset in NEIGHBOR_OFFSETS {
                let scaled_offset = offset.as_vec3() * physics_radius;
                let neighbor_cv = vec3_wv_to_cv(entity.as_vec3() + scaled_offset, voxel_size);

                required_pmeshes.insert(neighbor_cv);
            }
        }

        let new_pmeshes: Vec<_> = required_pmeshes
            .difference(&self.p_meshes)
            .copied()
            .collect();
        let stale_pmeshes: Vec<_> = self
            .p_meshes
            .difference(&required_pmeshes)
            .copied()
            .collect();

        self.p_meshes = required_pmeshes;

        (new_pmeshes, stale_pmeshes)
    }
}
