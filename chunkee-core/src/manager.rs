use std::{
    collections::{HashSet, hash_map::Entry},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use fxhash::FxHashMap;
use glam::{IVec3, Vec3};
use parking_lot::{Condvar, Mutex};

use crate::{
    aabb::AABB,
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{CHUNK_SIDE_32, Chunk, get_voxel_edge_faces},
    coords::{
        ChunkVector, LocalVector, NEIGHBOR_OFFSETS, WorldVector, cv_to_wv, wp_to_cv, wp_to_wv,
        wv_to_cv, wv_to_lv,
    },
    generation::VoxelGenerator,
    hasher::VoxelHashMap,
    meshing::{ChunkMeshGroup, PhysicsMesh, mesh_chunk, mesh_physics_chunk},
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, ChunkRadius, compute_priority},
    sv64::{Interner, SV64Tree},
    traversal::DDAState,
};

pub enum VoxelRaycast<V: ChunkeeVoxel> {
    Hit((WorldVector, V)),
    Miss,
    None,
}

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

pub type ChunkVersion = u32;

#[derive(Debug, Clone)]
pub struct WorldChunk {
    pub cv: ChunkVector,
    pub state: ChunkState,
    pub tree: SV64Tree,
    pub deltas: Deltas,
    pub is_dirty: bool,
    pub version: ChunkVersion,
    pub uniform: Option<VoxelId>,
    pub physics_state: PhysicsState,
}

impl Default for WorldChunk {
    fn default() -> Self {
        Self {
            cv: ChunkVector::MAX,
            state: ChunkState::None,
            tree: Default::default(),
            deltas: Deltas::default(),
            is_dirty: false,
            version: 0,
            uniform: None,
            physics_state: PhysicsState::None,
        }
    }
}

impl WorldChunk {
    pub fn is_stable(&self) -> bool {
        matches!(self.state, ChunkState::Meshing | ChunkState::MeshReady)
    }

    pub fn reuse(&mut self, cv: ChunkVector) {
        self.tree = Default::default();
        self.cv = cv;
        self.is_dirty = false;
        self.state = ChunkState::Loading;
        self.deltas = Deltas::default();
        self.uniform = None;
        self.physics_state = PhysicsState::None;
        self.version = 0;
    }
}

fn floor_div(a: i32, b: i32) -> i32 {
    let d = a / b;
    let r = a % b;
    if r != 0 && (a < 0) != (b < 0) {
        d - 1
    } else {
        d
    }
}

enum ClipmapUpdate {
    Initialized,
    Shift(ClipMap),
}

#[derive(Debug, Clone)]
struct ClipMap {
    grid_anchor: IVec3,
    dimensions: IVec3,
    // num_lods: u32,
    snap_distance: i32,
    initialized: bool,
}

impl ClipMap {
    fn new(radius: ChunkRadius, num_lods: u32) -> Self {
        let dimensions = radius.span().as_ivec3();
        let snap_distance = if num_lods <= 1 {
            1
        } else {
            1 << (num_lods - 1) // 2^(num_lods - 1)
        };

        Self {
            grid_anchor: IVec3::MAX,
            dimensions,
            snap_distance,
            // num_lods,
            initialized: false,
        }
    }

    fn lod_snap(&self, camera_cv: ChunkVector) -> ChunkVector {
        let snap = self.snap_distance;

        if snap == 1 {
            return camera_cv;
        };

        let half_snap = snap / 2;

        let snapped_x = floor_div(camera_cv.x + half_snap, snap) * snap;
        let snapped_y = floor_div(camera_cv.y + half_snap, snap) * snap;
        let snapped_z = floor_div(camera_cv.z + half_snap, snap) * snap;

        ChunkVector::new(snapped_x, snapped_y, snapped_z)
    }

    pub fn update(&mut self, camera_cv: ChunkVector) -> Option<ClipmapUpdate> {
        if !self.initialized {
            self.grid_anchor = self.lod_snap(camera_cv);
            self.initialized = true;

            return Some(ClipmapUpdate::Initialized);
        }

        let target_anchor = self.lod_snap(camera_cv);
        let shift = target_anchor - self.grid_anchor;

        if shift != IVec3::ZERO {
            let previous = self.clone();
            self.grid_anchor = target_anchor;
            Some(ClipmapUpdate::Shift(previous))
        } else {
            None
        }
    }

    pub fn get_grid_origin(&self) -> IVec3 {
        self.grid_anchor - (self.dimensions / 2)
    }

    pub fn to_aabb(&self) -> AABB {
        let grid_origin = self.get_grid_origin();

        AABB::new(grid_origin, grid_origin + self.dimensions)
    }

    pub fn cv_in_range(&self, cv: ChunkVector) -> bool {
        let min = self.get_grid_origin();
        let max = min + self.dimensions;

        cv.x >= min.x
            && cv.x < max.x
            && cv.y >= min.y
            && cv.y < max.y
            && cv.z >= min.z
            && cv.z < max.z
    }
}

pub struct ChunkManager {
    pool: Vec<WorldChunk>,
    map: FxHashMap<ChunkVector, WorldChunk>,
}

impl ChunkManager {
    fn new(radius: ChunkRadius) -> Self {
        let total_chunks = radius.chunk_count() as usize;

        let pool: Vec<WorldChunk> = vec![WorldChunk::default(); total_chunks];
        let map = FxHashMap::with_capacity_and_hasher(total_chunks, Default::default());

        Self { pool, map }
    }

    fn insert_chunk(&mut self, cv: ChunkVector) {
        let mut new_chunk = self.pool.pop().unwrap_or_else(|| WorldChunk::default());
        new_chunk.reuse(cv);
        self.map.insert(cv, new_chunk);
    }

    fn remove_chunk<F: FnMut(&mut WorldChunk)>(&mut self, cv: ChunkVector, mut cb: F) {
        if let Some(mut world_chunk) = self.map.remove(&cv) {
            cb(&mut world_chunk);

            self.pool.push(world_chunk);
        }
    }

    fn get(&self, cv: ChunkVector) -> Option<&WorldChunk> {
        self.map.get(&cv)
    }

    fn get_mut(&mut self, cv: ChunkVector) -> Option<&mut WorldChunk> {
        self.map.get_mut(&cv)
    }
}

#[derive(Debug)]
pub enum WorkerResult {
    Load(ChunkVector, SV64Tree, Deltas, VoxelUniform),
    Mesh(ChunkVector, ChunkVersion, Box<ChunkMeshGroup>),
    PhysicsMesh(ChunkVector, ChunkVersion, PhysicsMesh),
    Intern(ChunkVector, ChunkVersion, SV64Tree),
}

struct MeshPayload {
    version: ChunkVersion,
    tree: SV64Tree,
    neighbors: [Option<SV64Tree>; 6],
}

struct PhysicsPayload {
    version: ChunkVersion,
    tree: SV64Tree,
}

struct WorkerJob<T> {
    cv: ChunkVector,
    priority: u32,
    payload: T,
}

type LoadJob = WorkerJob<()>;
type MeshJob = WorkerJob<MeshPayload>;
type PhysicsJob = WorkerJob<PhysicsPayload>;
type SaveJob = Vec<(ChunkVector, PersistedChunk)>;

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
    intern: SegQueue<(ChunkVector, ChunkVersion, SV64Tree)>,
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
            if clipmap.cv_in_range(job.cv) {
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
    Mesh(ChunkVector, Box<ChunkMeshGroup>),
    MeshUnload(ChunkVector),
    Physics(ChunkVector, PhysicsMesh),
    PhysicsUnload(ChunkVector),
}

pub struct ChunkeeConfig {
    pub radius: ChunkRadius,
    pub voxel_size: f32,
    pub thread_count: usize,
    pub generator: Box<dyn VoxelGenerator>,
}

const CAMERA_THRESHOLD_RADIANS: f32 = 0.5;

pub struct ChunkeeManager<V: 'static + ChunkeeVoxel> {
    pub updates: Vec<Update>,
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
    last_camera_wv: Option<WorldVector>,
    _voxel_type: PhantomData<V>,
}

impl<V: 'static + ChunkeeVoxel> ChunkeeManager<V> {
    pub fn new(config: ChunkeeConfig) -> Self {
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<WorkerResult>();
        let chunk_count = config.radius.chunk_count();

        Self {
            updates: Vec::new(),
            clipmap: ClipMap::new(config.radius, 3),
            active: Arc::new(AtomicBool::new(false)),
            chunk_manager: ChunkManager::new(config.radius),
            config: Arc::new(config),
            camera: None,
            chunk_store: Arc::new(ChunkStore::new()),
            workers: Vec::new(),
            job_queue: Arc::new(JobQueue::new()),
            result_tx,
            result_rx,
            physics_manager: PhysicsManager::new(),
            interner: Arc::new(Mutex::new(Interner::new(chunk_count))),
            last_camera_wv: None,
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
            let pos = dda.next_voxelpos;
            let cv = wv_to_cv(pos);

            if let Some(world_chunk) = self.chunk_manager.get(cv) {
                if !world_chunk.is_stable() {
                    return VoxelRaycast::None;
                }

                let lv = wv_to_lv(pos);
                let voxel_id = world_chunk.tree.get_voxel(lv);
                let voxel = V::from(voxel_id.type_id());
                if voxel.visibilty() != VoxelVisibility::Empty {
                    return VoxelRaycast::Hit((pos, voxel));
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
        edits: &[(WorldVector, V)],
    ) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        let significant_drift = self.update_camera(camera);
        self.update_clipmap(&camera);
        self.handle_physics(physics_entities, &camera);
        self.process_edits(edits);
        self.process_worker_results(&camera);

        if significant_drift {
            self.job_queue.queues.reprioritze_queues(
                &camera,
                &self.clipmap,
                self.config.voxel_size,
            );
        }

        if !self.job_queue.queues.is_empty() {
            // self.job_queue.queues.print();
            self.job_queue.cvar.notify_all();
        }
    }

    fn process_edits(&mut self, edits: &[(WorldVector, V)]) {
        let mut voxel_edits: FxHashMap<ChunkVector, Vec<(LocalVector, VoxelId)>> =
            FxHashMap::with_hasher(Default::default());
        let mut chunks_to_mesh: FxHashMap<ChunkVector, bool> =
            FxHashMap::with_hasher(Default::default());

        for (wv, voxel) in edits {
            let (cv, lv) = (wv_to_cv(*wv), wv_to_lv(*wv));
            let voxel_id = VoxelId::new((*voxel).into());
            match voxel_edits.entry(cv) {
                Entry::Occupied(mut occ) => {
                    occ.get_mut().push((lv, voxel_id));
                }
                Entry::Vacant(vac) => {
                    vac.insert_entry(vec![(lv, voxel_id)]);
                }
            }
        }

        for (cv, edits) in voxel_edits.iter_mut() {
            if let Some(world_chunk) = self.chunk_manager.get_mut(*cv)
                && world_chunk.is_stable()
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
                                let neighbor_cv = cv + face.into_normal();
                                chunks_to_mesh.entry(neighbor_cv).or_insert(false);
                            }
                        }
                    }
                }

                chunks_to_mesh.insert(*cv, true);

                self.job_queue.queues.intern.push((
                    world_chunk.cv,
                    world_chunk.version,
                    world_chunk.tree.clone(),
                ));
            }
        }

        for (cv, edited) in chunks_to_mesh {
            if let Some(world_chunk) = self.chunk_manager.get_mut(cv)
                && world_chunk.is_stable()
            {
                if edited && self.physics_manager.pchunks.contains(&cv) {
                    world_chunk.physics_state = PhysicsState::Meshing;
                    self.job_queue.queues.physics.push(PhysicsJob::new(
                        cv,
                        0,
                        PhysicsPayload {
                            version: world_chunk.version,
                            tree: world_chunk.tree.clone(),
                        },
                    ));
                }

                world_chunk.state = ChunkState::Meshing;

                if let MeshableResult::Ready(payload) = self.meshable(cv) {
                    self.job_queue
                        .queues
                        .edits
                        .push(MeshJob::new(cv, 0, payload));
                }
            }
        }
    }

    fn update_camera(&mut self, current: CameraData) -> bool {
        let current_wv = wp_to_wv(current.pos, self.config.voxel_size);
        // if let Some(previous_wv) = self.last_camera_wv
        //     && previous_wv != current_wv
        // {
        //     println!("camera: {previous_wv}=>{current_wv}");
        // } else if self.last_camera_wv.is_none() {
        //     println!("camera: {current_wv}");
        // }
        self.last_camera_wv = Some(current_wv);

        if let Some(previous) = self.camera {
            let current_cv = wp_to_cv(current.pos, self.config.voxel_size);
            let previous_cv = wp_to_cv(previous.pos, self.config.voxel_size);
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
        let camera_cv = wp_to_cv(camera.pos, self.config.voxel_size);
        if let Some(update) = self.clipmap.update(camera_cv) {
            let (added, removed): (Vec<AABB>, Vec<AABB>) = match update {
                ClipmapUpdate::Initialized => (vec![self.clipmap.to_aabb()], vec![]),
                ClipmapUpdate::Shift(previous) => {
                    self.clipmap.to_aabb().difference(&previous.to_aabb())
                }
            };

            let mut chunks_to_persist = Vec::new();
            for cv in removed.iter().flat_map(|aabb| aabb.iter()) {
                self.chunk_manager.remove_chunk(cv, |world_chunk| {
                    if world_chunk.is_dirty {
                        chunks_to_persist.push((
                            world_chunk.cv,
                            PersistedChunk {
                                deltas: std::mem::take(&mut world_chunk.deltas),
                                uniform_voxel_id: world_chunk.uniform.take(),
                            },
                        ))
                    }

                    if world_chunk.state == ChunkState::MeshReady {
                        self.updates.push(Update::MeshUnload(world_chunk.cv));
                    }
                });
            }

            if !chunks_to_persist.is_empty() {
                self.job_queue.queues.saves.push(chunks_to_persist);
            }

            for cv in added.iter().flat_map(|aabb| aabb.iter()) {
                self.chunk_manager.insert_chunk(cv);

                let priority = compute_priority(cv, &camera, self.config.voxel_size);
                self.job_queue
                    .queues
                    .load
                    .push(LoadJob::new(cv, priority, ()));
            }
        }
    }

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn process_worker_results(&mut self, camera: &CameraData) {
        while let Ok(message) = self.result_rx.try_recv() {
            match message {
                WorkerResult::Load(cv, tree, deltas, voxel_uniform) => {
                    let mut meshable_chunk_candidates = vec![];
                    if let Some(world_chunk) = self.chunk_manager.get_mut(cv)
                        && world_chunk.state == ChunkState::Loading
                    {
                        world_chunk.tree = tree;
                        world_chunk.deltas = deltas;
                        match voxel_uniform {
                            VoxelUniform::None => world_chunk.uniform = None,
                            VoxelUniform::Loaded(voxel_id) => world_chunk.uniform = Some(voxel_id),
                            VoxelUniform::Generated(voxel_id) => {
                                world_chunk.uniform = Some(voxel_id);
                                world_chunk.is_dirty = true
                            }
                        };

                        self.job_queue.queues.intern.push((
                            world_chunk.cv,
                            world_chunk.version,
                            world_chunk.tree.clone(),
                        ));

                        if let Some(uniform) = world_chunk.uniform
                            && uniform == VoxelId::AIR
                        {
                            world_chunk.state = ChunkState::MeshReady;
                        } else {
                            world_chunk.state = ChunkState::Meshing;
                            meshable_chunk_candidates.push(cv);
                        }

                        meshable_chunk_candidates.extend_from_slice(&neighbors_of(cv));
                    }

                    for cv in meshable_chunk_candidates {
                        if let MeshableResult::Ready(payload) = self.meshable(cv) {
                            let priority = compute_priority(cv, &camera, self.config.voxel_size);

                            self.job_queue
                                .queues
                                .mesh
                                .push(MeshJob::new(cv, priority, payload))
                        }
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
        let stale_pchunks = self
            .physics_manager
            .update(physics_entities, self.config.voxel_size);

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

    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn meshable(&self, cv: ChunkVector) -> MeshableResult {
        if let Some(world_chunk) = self.chunk_manager.get(cv)
            && world_chunk.is_stable()
        {
            let neighbor_cvs = neighbors_of(cv);
            let mut neighbors = [const { None }; 6];

            for (idx, neighbor_cv) in neighbor_cvs.iter().enumerate() {
                if let Some(neighbor_chunk) = self.chunk_manager.get(*neighbor_cv) {
                    if neighbor_chunk.is_stable() {
                        let chunk = neighbor_chunk.tree.clone();
                        neighbors[idx] = Some(chunk);
                    } else {
                        return MeshableResult::NotReady;
                    }
                }
            }

            MeshableResult::Ready(MeshPayload {
                version: world_chunk.version,
                tree: world_chunk.tree.clone(),
                neighbors: neighbors,
            })
        } else {
            MeshableResult::NotReady
        }
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

            if let Some(job) = job_queue.queues.saves.pop() {
                chunk_store.save_chunks(job);
                continue;
            }

            if let Some(job) = job_queue.queues.physics.pop() {
                let chunk = job.payload.tree.to_chunk();
                let mesh = mesh_physics_chunk::<V>(job.cv, chunk, config.voxel_size);
                result_tx
                    .send(WorkerResult::PhysicsMesh(job.cv, job.payload.version, mesh))
                    .ok();

                continue;
            }

            if let Some(job) = job_queue
                .queues
                .edits
                .pop()
                .or_else(|| job_queue.queues.mesh.pop())
            {
                let chunk = job.payload.tree.to_chunk();
                let neighbors = job
                    .payload
                    .neighbors
                    .map(|m_tree| m_tree.and_then(|tree| Some(tree.to_chunk())));
                let mesh_group = mesh_chunk::<V>(job.cv, chunk, neighbors, config.voxel_size);
                result_tx
                    .send(WorkerResult::Mesh(
                        job.cv,
                        job.payload.version,
                        Box::new(mesh_group),
                    ))
                    .ok();

                continue;
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

            if let Some(job) = job_queue.queues.load.pop() {
                // TODO: Don't need to create 32x32x32 chunk if uniform
                let (chunk, deltas, uniform) =
                    load_chunk(job.cv, &chunk_store, &config.generator, config.voxel_size);
                let tree = SV64Tree::from_chunk(&chunk);

                result_tx
                    .send(WorkerResult::Load(job.cv, tree, deltas, uniform))
                    .ok();
            }
        }
    }
}

enum MeshableResult {
    NotReady,
    Ready(MeshPayload),
}

#[derive(Debug)]
pub enum VoxelUniform {
    None,
    Loaded(VoxelId),
    Generated(VoxelId),
}

impl Default for VoxelUniform {
    fn default() -> Self {
        VoxelUniform::None
    }
}

#[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
fn load_chunk(
    cv: ChunkVector,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
    voxel_size: f32,
) -> (Chunk, Deltas, VoxelUniform) {
    let mut chunk = Chunk::new();
    let mut deltas = Deltas::default();
    let mut voxel_uniform = VoxelUniform::None;

    if let Some(persisted_chunk) = chunk_store.load_chunk(cv) {
        deltas = persisted_chunk.deltas;

        if let Some(uniform) = persisted_chunk.uniform_voxel_id {
            voxel_uniform = VoxelUniform::Loaded(uniform)
        }
    }

    if let VoxelUniform::Loaded(uniform) = voxel_uniform {
        chunk.fill(uniform);
    } else {
        generator.apply(cv_to_wv(cv), &mut chunk, voxel_size);
        let first = &chunk.voxels[0];
        if chunk.voxels.iter().all(|v| v == first) {
            voxel_uniform = VoxelUniform::Generated(*first);
        }
    }

    merge_deltas(&mut chunk, &deltas);

    (chunk, deltas, voxel_uniform)
}

struct PhysicsManager {
    pub pchunks: HashSet<ChunkVector>,
}

impl PhysicsManager {
    const PHYSICS_RADIUS: i32 = CHUNK_SIDE_32 / 2;

    fn new() -> Self {
        Self {
            pchunks: HashSet::default(),
        }
    }

    fn update(&mut self, entities: Vec<Vec3>, voxel_size: f32) -> Vec<ChunkVector> {
        let mut required_pchunks = HashSet::default();
        for entity in entities {
            for offset in NEIGHBOR_OFFSETS {
                let entity_wv = wp_to_wv(entity, voxel_size);
                let radius = offset * Self::PHYSICS_RADIUS;
                let neighbor_wv = entity_wv + radius;
                let neighbor_cv = wv_to_cv(neighbor_wv);

                required_pchunks.insert(neighbor_cv);
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
