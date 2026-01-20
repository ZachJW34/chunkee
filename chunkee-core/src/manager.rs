use std::{
    collections::{HashSet, hash_map::Entry},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use fxhash::{FxHashMap, FxHashSet};
use glam::{IVec3, Vec3};
use parking_lot::{Condvar, Mutex};

use crate::{
    block::{ChunkeeVoxel, VoxelId},
    chunk::{Chunk, get_voxel_edge_faces},
    clipmap::{ChunkKey, ClipMap, LODConfig},
    coords::{CHUNK_SIZE, NEIGHBOR_OFFSETS, wp_to_wv, wv_to_lv},
    generation::VoxelGenerator,
    meshing::{ChunkMeshGroup, PhysicsMesh, mesh_chunk, mesh_physics_chunk},
    storage::ChunkStore,
    streaming::{CameraData, compute_priority},
    sv64::{Interner, SV64Tree},
    traversal::DDAState,
};

pub enum VoxelRaycast<V: ChunkeeVoxel> {
    Hit((IVec3, V)),
    Miss,
    None,
}

#[derive(Debug, Clone)]
pub struct Deltas(pub FxHashMap<IVec3, VoxelId>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
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

pub type ChunkVersion = u32;

#[derive(Debug, Clone)]
pub struct WorldChunk {
    pub key: ChunkKey,
    pub state: ChunkState,
    pub tree: SV64Tree,
    pub deltas: Deltas,
    pub is_dirty: bool,
    pub version: ChunkVersion,
    pub physics_state: PhysicsState,
}

impl Default for WorldChunk {
    fn default() -> Self {
        Self {
            key: ChunkKey::MAX,
            state: ChunkState::None,
            tree: Default::default(),
            deltas: Deltas::default(),
            is_dirty: false,
            version: 0,
            physics_state: PhysicsState::None,
        }
    }
}

impl WorldChunk {
    #[inline(always)]
    pub fn is_stable(&self) -> bool {
        matches!(self.state, ChunkState::Meshing | ChunkState::MeshReady)
    }

    #[inline(always)]
    pub fn is_editable(&self) -> bool {
        self.is_stable() && self.key.lod == 0
    }

    #[inline(always)]
    pub fn reuse(&mut self, key: ChunkKey) {
        self.key = key;
        self.tree = Default::default();
        self.is_dirty = false;
        self.state = ChunkState::Loading;
        self.deltas = Deltas::default();
        self.physics_state = PhysicsState::None;
        self.version = 0;
    }
}

pub struct ChunkManager {
    pool: Vec<WorldChunk>,
    map: FxHashMap<ChunkKey, WorldChunk>,
}

impl ChunkManager {
    fn new(num_chunks: u32) -> Self {
        let pool: Vec<WorldChunk> = vec![WorldChunk::default(); num_chunks as usize];
        let map = FxHashMap::with_capacity_and_hasher(num_chunks as usize, Default::default());

        Self { pool, map }
    }

    fn insert_chunk(&mut self, key: ChunkKey) {
        let mut new_chunk = self.pool.pop().unwrap_or_else(|| WorldChunk::default());
        new_chunk.reuse(key);
        self.map.insert(key, new_chunk);
    }

    fn remove_chunk<F>(&mut self, key: ChunkKey, mut cb: F)
    where
        F: FnMut(&mut WorldChunk),
    {
        if let Some(mut world_chunk) = self.map.remove(&key) {
            cb(&mut world_chunk);

            self.pool.push(world_chunk);
        }
    }

    fn get(&self, key: ChunkKey) -> Option<&WorldChunk> {
        self.map.get(&key)
    }

    fn get_mut(&mut self, key: ChunkKey) -> Option<&mut WorldChunk> {
        self.map.get_mut(&key)
    }
}

#[derive(Debug)]
pub enum WorkerResult {
    Load(ChunkKey, SV64Tree, Deltas, Box<ChunkMeshGroup>),
    Mesh(ChunkKey, ChunkVersion, Box<ChunkMeshGroup>),
    PhysicsMesh(ChunkKey, ChunkVersion, PhysicsMesh),
    Intern(ChunkKey, ChunkVersion, SV64Tree),
}

struct MeshPayload {
    version: ChunkVersion,
    tree: SV64Tree,
}

struct PhysicsPayload {
    version: ChunkVersion,
    tree: SV64Tree,
}

struct WorkerJob<T> {
    key: ChunkKey,
    priority: u32,
    payload: T,
}

type LoadJob = WorkerJob<()>;
type MeshJob = WorkerJob<MeshPayload>;
type PhysicsJob = WorkerJob<PhysicsPayload>;
type SaveJob = Vec<(ChunkKey, Deltas)>;

impl<T> WorkerJob<T> {
    fn new(key: ChunkKey, priority: u32, payload: T) -> Self {
        Self {
            key,
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
    intern: SegQueue<(ChunkKey, ChunkVersion, SV64Tree)>,
    edits: SegQueue<MeshJob>,
    physics: SegQueue<PhysicsJob>,
    saves: SegQueue<SaveJob>,
}

impl Queues {
    fn new() -> Self {
        Self {
            load: SegQueue::new(),
            mesh: SegQueue::new(),
            intern: SegQueue::new(),
            physics: SegQueue::new(),
            edits: SegQueue::new(),
            saves: SegQueue::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.load.is_empty()
            && self.mesh.is_empty()
            && self.physics.is_empty()
            && self.edits.is_empty()
            && self.saves.is_empty()
            && self.intern.is_empty()
    }

    fn print(&self) {
        println!(
            "l={} | i={} | m={} | p={} | e={} | s={}",
            self.load.len(),
            self.intern.len(),
            self.mesh.len(),
            self.physics.len(),
            self.edits.len(),
            self.saves.len(),
        )
    }

    fn reprioritze_queues(&self, camera_data: &CameraData, clipmap: &ClipMap, voxel_size: f32) {
        Queues::repriotize_queue(&self.load, camera_data, clipmap, voxel_size);
        Queues::repriotize_queue(&self.mesh, camera_data, clipmap, voxel_size);
    }

    fn repriotize_queue<T>(
        queue: &SegQueue<WorkerJob<T>>,
        camera_data: &CameraData,
        clipmap: &ClipMap,
        voxel_size: f32,
    ) {
        let mut next_tasks = Vec::with_capacity(queue.len());
        while let Some(job) = queue.pop() {
            if clipmap.is_chunk_valid(job.key) {
                let priority = compute_priority(job.key, camera_data, voxel_size);
                let new_job = WorkerJob::<T>::new(job.key, priority, job.payload);
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
    Mesh(ChunkKey, Box<ChunkMeshGroup>),
    MeshUnload(ChunkKey),
    Physics(ChunkKey, PhysicsMesh),
    PhysicsUnload(ChunkKey),
}

pub struct ChunkeeConfig {
    pub lod_config: LODConfig,
    pub voxel_size: f32,
    pub thread_count: usize,
    pub generator: Box<dyn VoxelGenerator>,
}

const CAMERA_THRESHOLD_RADIANS: f32 = 0.5;

pub struct ChunkeeManager<V: 'static + ChunkeeVoxel> {
    pub updates: Vec<Update>,
    pub freeze_clipmap: bool,
    clipmap: ClipMap,
    active: Arc<AtomicBool>,
    config: Arc<ChunkeeConfig>,
    chunk_manager: ChunkManager,
    camera: Option<CameraData>,
    chunk_store: Arc<ChunkStore>,
    workers: Vec<JoinHandle<()>>,
    job_queue: Arc<JobQueue>,
    result_tx: Sender<WorkerResult>,
    result_rx: Receiver<WorkerResult>,
    physics_manager: PhysicsManager,
    interner: Arc<Mutex<Interner>>,
    last_camera_wv: Option<IVec3>,
    pending_unloads: FxHashSet<ChunkKey>,
    pending_saves: FxHashSet<ChunkKey>,
    last_save: Instant,
    _voxel_type: PhantomData<V>,
}

impl<V: 'static + ChunkeeVoxel> ChunkeeManager<V> {
    pub fn new(config: ChunkeeConfig) -> Self {
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<WorkerResult>();
        let num_chunks = config.lod_config.num_chunks();

        Self {
            updates: Vec::new(),
            freeze_clipmap: false,
            clipmap: ClipMap::new(config.lod_config.clone()),
            active: Arc::new(AtomicBool::new(false)),
            chunk_manager: ChunkManager::new(num_chunks),
            config: Arc::new(config),
            camera: None,
            chunk_store: Arc::new(ChunkStore::new()),
            workers: Vec::new(),
            job_queue: Arc::new(JobQueue::new()),
            result_tx,
            result_rx,
            physics_manager: PhysicsManager::new(),
            interner: Arc::new(Mutex::new(Interner::new(num_chunks))),
            last_camera_wv: None,
            pending_unloads: Default::default(),
            pending_saves: Default::default(),
            last_save: Instant::now(),
            _voxel_type: PhantomData,
        }
    }

    pub fn start(&mut self) {
        if self.active.swap(true, Ordering::Relaxed) {
            return;
        }

        assert!(
            self.config.thread_count > 0,
            "Chunkee requires at least one thread",
        );

        let mut worker_threads = Vec::with_capacity(self.config.thread_count);
        for _ in 0..self.config.thread_count {
            let job_queue = self.job_queue.clone();
            let result_tx = self.result_tx.clone();
            let active = self.active.clone();
            let chunk_store = self.chunk_store.clone();
            let interner = self.interner.clone();
            let config = self.config.clone();
            worker_threads.push(thread::spawn(move || {
                worker_loop::<V>(job_queue, result_tx, active, chunk_store, interner, config)
            }))
        }

        self.workers = worker_threads;
        self.last_save = Instant::now();
    }

    pub fn stop(&mut self) {
        if !self.active.swap(false, Ordering::Relaxed) {
            return;
        }

        self.job_queue.cvar.notify_all();

        for worker in self.workers.drain(..) {
            worker.join().ok();
        }
    }

    pub fn raycast(
        &mut self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_steps: u32,
    ) -> VoxelRaycast<V> {
        let mut dda = DDAState::from_pos_and_dir(
            (ray_origin / self.config.voxel_size).into(),
            ray_direction.into(),
        );

        for _ in 0..(max_steps as usize) {
            let wv = dda.next_voxelpos;

            if let Some(key) = self.clipmap.wv_to_chunk_key(wv)
                && key.lod == 0
                && let Some(world_chunk) = self.chunk_manager.get(key)
            {
                if !world_chunk.is_stable() {
                    return VoxelRaycast::None;
                }

                let lv = wv_to_lv(wv);
                let voxel_id = world_chunk.tree.get_voxel(lv);
                let voxel = V::from(voxel_id.type_id());
                if voxel.visibilty() != VoxelVisibility::Empty {
                    return VoxelRaycast::Hit((wv, voxel));
                }

                dda.step_mut();
            }
        }

        VoxelRaycast::Miss
    }

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    pub fn update(
        &mut self,
        camera: CameraData,
        physics_entities: Vec<Vec3>,
        edits: &[(IVec3, V)],
    ) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        let significant_drift = self.update_camera(camera);
        if !self.freeze_clipmap || self.clipmap.anchor == IVec3::MAX {
            self.update_clipmap(&camera);
        }
        self.handle_physics(physics_entities, &camera);
        self.process_edits(edits);
        self.process_worker_results();
        // TODO: Need to do this every tick?
        self.reap_obsolete_chunks();

        if significant_drift {
            self.job_queue.queues.reprioritze_queues(
                &camera,
                &self.clipmap,
                self.config.voxel_size,
            );
        }

        if !self.job_queue.queues.is_empty() {
            self.job_queue.queues.print();
            self.job_queue.cvar.notify_all();
        }

        self.save_edits();
    }

    fn process_edits(&mut self, edits: &[(IVec3, V)]) {
        let mut voxel_edits: FxHashMap<ChunkKey, Vec<(IVec3, VoxelId)>> =
            FxHashMap::with_hasher(Default::default());
        let mut chunks_to_mesh: FxHashMap<ChunkKey, bool> =
            FxHashMap::with_hasher(Default::default());

        for (wv, voxel) in edits {
            let Some(key) = self.clipmap.wv_to_chunk_key(*wv) else {
                continue;
            };
            let lv = wv_to_lv(*wv);
            let voxel_id = VoxelId::new((*voxel).into());
            match voxel_edits.entry(key) {
                Entry::Occupied(mut occ) => {
                    occ.get_mut().push((lv, voxel_id));
                }
                Entry::Vacant(vac) => {
                    vac.insert_entry(vec![(lv, voxel_id)]);
                }
            }
        }

        for (key, edits) in voxel_edits.iter_mut() {
            if let Some(world_chunk) = self.chunk_manager.get_mut(*key)
                && world_chunk.is_editable()
            {
                world_chunk.tree.set_voxels(edits);
                world_chunk.is_dirty = true;
                world_chunk.version = world_chunk.version.wrapping_add(1);

                for (lv, voxel_id) in edits.drain(..) {
                    world_chunk.deltas.0.insert(lv, voxel_id);
                    let (neighbors_mask, affected_neighbors) = get_voxel_edge_faces(lv);

                    if neighbors_mask.count_ones() > 0 {
                        for affected in affected_neighbors {
                            if let Some(face) = affected {
                                let neighbor_wv = key.to_wv() + face.into_normal();
                                if let Some(neighbor_key) =
                                    self.clipmap.wv_to_chunk_key(neighbor_wv)
                                    && key.lod == 0
                                {
                                    chunks_to_mesh.entry(neighbor_key).or_insert(false);
                                }
                            }
                        }
                    }
                }

                chunks_to_mesh.insert(*key, true);

                self.job_queue.queues.intern.push((
                    world_chunk.key,
                    world_chunk.version,
                    world_chunk.tree.clone(),
                ));
                self.pending_saves.insert(*key);
            }
        }

        for (key, edited) in chunks_to_mesh {
            if let Some(world_chunk) = self.chunk_manager.get_mut(key)
                && world_chunk.is_stable()
            {
                if edited && self.physics_manager.pchunks.contains(&key) {
                    world_chunk.physics_state = PhysicsState::Meshing;
                    self.job_queue.queues.physics.push(PhysicsJob::new(
                        key,
                        0,
                        PhysicsPayload {
                            version: world_chunk.version,
                            tree: world_chunk.tree.clone(),
                        },
                    ));
                }

                world_chunk.state = ChunkState::Meshing;
                self.job_queue.queues.edits.push(MeshJob::new(
                    key,
                    0,
                    MeshPayload {
                        version: world_chunk.version,
                        tree: world_chunk.tree.clone(),
                    },
                ));
            }
        }
    }

    fn update_camera(&mut self, current: CameraData) -> bool {
        let current_wv = wp_to_wv(current.pos, self.config.voxel_size);
        self.last_camera_wv = Some(current_wv);

        if let Some(previous) = self.camera {
            let current_cv = wp_to_wv(current.pos, self.config.voxel_size);
            let previous_cv = wp_to_wv(previous.pos, self.config.voxel_size);
            let significant = (current_cv != previous_cv)
                || (current.forward.angle_between(previous.forward) > CAMERA_THRESHOLD_RADIANS);

            if significant {
                self.camera = Some(current);
            }

            significant
        } else {
            self.camera = Some(current);
            true
        }
    }

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn update_clipmap(&mut self, camera: &CameraData) {
        let player_wv = wp_to_wv(camera.pos, self.config.voxel_size);
        let events = self.clipmap.update(player_wv);
        let mut chunks_to_persist = vec![];

        if events.anchor_shifted {
            for key in events.to_unload {
                self.chunk_manager.remove_chunk(key, |world_chunk| {
                    if world_chunk.is_dirty && world_chunk.key.lod == 0 {
                        chunks_to_persist
                            .push((world_chunk.key, std::mem::take(&mut world_chunk.deltas)));
                    }

                    if world_chunk.state == ChunkState::MeshReady {
                        if self.clipmap.contains(key.to_wv()) {
                            self.pending_unloads.insert(key);
                        } else {
                            self.updates.push(Update::MeshUnload(key));
                        }
                    }
                });
            }
        }

        if !chunks_to_persist.is_empty() {
            self.job_queue.queues.saves.push(chunks_to_persist);
        }

        for key in events.to_load {
            self.chunk_manager.insert_chunk(key);

            let priority = compute_priority(key, &camera, self.config.voxel_size);
            self.job_queue
                .queues
                .load
                .push(LoadJob::new(key, priority, ()));
        }
    }

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn process_worker_results(&mut self) {
        while let Ok(message) = self.result_rx.try_recv() {
            match message {
                WorkerResult::Load(key, tree, deltas, mesh_group) => {
                    if let Some(world_chunk) = self.chunk_manager.get_mut(key)
                        && world_chunk.state == ChunkState::Loading
                    {
                        world_chunk.tree = tree;
                        world_chunk.deltas = deltas;

                        self.job_queue.queues.intern.push((
                            world_chunk.key,
                            world_chunk.version,
                            world_chunk.tree.clone(),
                        ));

                        world_chunk.state = ChunkState::MeshReady;
                        self.updates.push(Update::Mesh(key, mesh_group));
                    }
                }
                WorkerResult::Mesh(cv, version, mesh_group) => {
                    if let Some(world_chunk) = self.chunk_manager.get_mut(cv)
                        && world_chunk.version == version
                        && world_chunk.state == ChunkState::Meshing
                    {
                        world_chunk.state = ChunkState::MeshReady;
                        self.updates.push(Update::Mesh(cv, mesh_group));
                    }
                }
                WorkerResult::PhysicsMesh(cv, version, pmesh) => {
                    if let Some(world_chunk) = self.chunk_manager.get_mut(cv)
                        && world_chunk.version == version
                        && world_chunk.physics_state == PhysicsState::Meshing
                    {
                        world_chunk.physics_state = PhysicsState::MeshReady;
                        self.updates.push(Update::Physics(cv, pmesh));
                    }
                }
                WorkerResult::Intern(cv, version, tree) => {
                    if let Some(world_chunk) = self.chunk_manager.get_mut(cv)
                        && world_chunk.version == version
                    {
                        world_chunk.tree = tree;
                    }
                }
            }
        }
    }

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn handle_physics(&mut self, physics_entities: Vec<Vec3>, camera: &CameraData) {
        let stale_pchunks =
            self.physics_manager
                .update(physics_entities, &self.clipmap, self.config.voxel_size);

        for cv in stale_pchunks {
            if let Some(world_chunk) = self.chunk_manager.get_mut(cv) {
                if world_chunk.physics_state == PhysicsState::MeshReady {
                    self.updates.push(Update::PhysicsUnload(cv));
                }

                world_chunk.physics_state = PhysicsState::None;
            }
        }

        for cv in &self.physics_manager.pchunks {
            if let Some(world_chunk) = self.chunk_manager.get_mut(*cv)
                && world_chunk.physics_state == PhysicsState::None
                && world_chunk.is_stable()
            {
                world_chunk.physics_state = PhysicsState::Meshing;
                let priority = compute_priority(*cv, &camera, self.config.voxel_size);
                self.job_queue.queues.physics.push(PhysicsJob::new(
                    *cv,
                    priority,
                    PhysicsPayload {
                        tree: world_chunk.tree.clone(),
                        version: world_chunk.version,
                    },
                ));
            }
        }
    }

    fn save_edits(&mut self) {
        let now = Instant::now();

        if now.duration_since(self.last_save).as_secs() > 5 {
            self.last_save = now;
            let mut chunks_to_persist = vec![];

            for key in self.pending_saves.drain() {
                if let Some(world_chunk) = self.chunk_manager.map.get_mut(&key) {
                    if world_chunk.key.lod == 0 && world_chunk.is_dirty {
                        chunks_to_persist.push((key, world_chunk.deltas.clone()));

                        world_chunk.is_dirty = false;
                    }
                }
            }

            if !chunks_to_persist.is_empty() {
                self.job_queue.queues.saves.push(chunks_to_persist);
            }
        }
    }

    fn reap_obsolete_chunks(&mut self) {
        self.pending_unloads.retain(|key| {
            let wv = key.to_wv();
            let Some(current) = self.clipmap.wv_to_chunk_key(wv) else {
                self.updates.push(Update::MeshUnload(*key));
                return false;
            };

            if current.lod < key.lod {
                //// Refining -> Check to see if children of this LOD can replace it
                if let Some(children) = key.children() {
                    for child in children.iter().flat_map(|key| self.chunk_manager.get(*key)) {
                        if child.state != ChunkState::MeshReady {
                            return true;
                        }
                    }
                }

                self.updates.push(Update::MeshUnload(*key));
                return false;
            }
            // Coarsening -> Check if
            else {
                // Coarsening -> Check if
                if let Some(parent_key) = key.parent(self.config.lod_config.max_lod) {
                    if let Some(parent_chunk) = self.chunk_manager.get(parent_key) {
                        if parent_chunk.state != ChunkState::MeshReady {
                            return true;
                        }
                    }
                }

                self.updates.push(Update::MeshUnload(*key));
                return false;
            };
        });
    }
}

fn worker_loop<V: 'static + ChunkeeVoxel>(
    job_queue: Arc<JobQueue>,
    result_tx: Sender<WorkerResult>,
    active: Arc<AtomicBool>,
    chunk_store: Arc<ChunkStore>,
    interner: Arc<Mutex<Interner>>,
    config: Arc<ChunkeeConfig>,
) {
    loop {
        let mut guard = job_queue.mutex.lock();

        job_queue.cvar.wait_while(&mut guard, |_| {
            job_queue.queues.is_empty() && active.load(Ordering::Relaxed)
        });

        drop(guard);

        if !active.load(Ordering::Relaxed) {
            return;
        }

        #[cfg(feature = "profile")]
        let _work_span = tracing::info_span!("worker_process_batch").entered();

        while !job_queue.queues.is_empty() {
            if !active.load(Ordering::Relaxed) {
                return;
            }

            if let Some(job) = job_queue.queues.physics.pop() {
                let chunk = job.payload.tree.to_chunk();
                let mesh = mesh_physics_chunk::<V>(job.key, chunk, config.voxel_size);
                result_tx
                    .send(WorkerResult::PhysicsMesh(
                        job.key,
                        job.payload.version,
                        mesh,
                    ))
                    .ok();

                continue;
            }

            while let Some(job) = job_queue.queues.saves.pop() {
                chunk_store.save_chunks(job, &config.lod_config);
                continue;
            }

            if let Some(job) = job_queue
                .queues
                .edits
                .pop()
                .or_else(|| job_queue.queues.mesh.pop())
            {
                let chunk = job.payload.tree.to_chunk();
                let mesh_group = mesh_chunk::<V>(job.key, chunk, config.voxel_size);
                result_tx
                    .send(WorkerResult::Mesh(
                        job.key,
                        job.payload.version,
                        Box::new(mesh_group),
                    ))
                    .ok();

                continue;
            }

            if let Some(job) = job_queue.queues.load.pop() {
                // TODO: Don't need to create 32x32x32 chunk if uniform
                let (chunk, deltas) =
                    load_chunk(job.key, &chunk_store, &config.generator, config.voxel_size);
                let tree = SV64Tree::from_chunk(&chunk);
                let mesh_group = if tree.is_air() {
                    ChunkMeshGroup::default()
                } else {
                    mesh_chunk::<V>(job.key, chunk, config.voxel_size)
                };

                result_tx
                    .send(WorkerResult::Load(
                        job.key,
                        tree,
                        deltas,
                        Box::new(mesh_group),
                    ))
                    .ok();
            }

            if !job_queue.queues.intern.is_empty() {
                if let Some(mut interner) = interner.try_lock() {
                    while let Some((cv, version, tree)) = job_queue.queues.intern.pop() {
                        let interned = interner.intern_tree(tree);
                        result_tx
                            .send(WorkerResult::Intern(cv, version, interned))
                            .ok();
                    }

                    continue;
                }
            }
        }
    }
}

#[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
fn load_chunk(
    key: ChunkKey,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
    voxel_size: f32,
) -> (Chunk, Deltas) {
    let mut chunk = Chunk::new();
    let deltas = chunk_store.load_chunk(key).unwrap_or_default();

    generator.apply(key, &mut chunk, voxel_size);
    merge_deltas(&mut chunk, &deltas);

    (chunk, deltas)
}

struct PhysicsManager {
    pub pchunks: FxHashSet<ChunkKey>,
}

impl PhysicsManager {
    const PHYSICS_RADIUS: i32 = CHUNK_SIZE / 2;

    fn new() -> Self {
        Self {
            pchunks: HashSet::default(),
        }
    }

    fn update(&mut self, entities: Vec<Vec3>, clipmap: &ClipMap, voxel_size: f32) -> Vec<ChunkKey> {
        let mut required_pchunks = HashSet::default();
        for entity in entities {
            for offset in NEIGHBOR_OFFSETS {
                let entity_wv = wp_to_wv(entity, voxel_size);
                let radius = offset * Self::PHYSICS_RADIUS;
                let neighbor_wv = entity_wv + radius;
                if let Some(key) = clipmap.wv_to_chunk_key(neighbor_wv) {
                    required_pchunks.insert(key);
                }
            }
        }

        let stale_pchunks: Vec<_> = self
            .pchunks
            .difference(&required_pchunks)
            .copied()
            .collect();

        self.pchunks = required_pchunks;

        stale_pchunks
    }
}

pub fn merge_deltas(chunk: &mut Chunk, deltas: &Deltas) {
    for (lv, voxel_id) in deltas.0.iter() {
        chunk.set_voxel(*lv, *voxel_id);
    }
}
