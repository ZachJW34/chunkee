use std::{
    cmp::Ordering,
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
    coords::{ChunkVector, NEIGHBOR_OFFSETS, camera_vec3_to_cv, cv_to_wv, wv_to_cv, wv_to_lv},
    define_metrics,
    generation::VoxelGenerator,
    grid::{ChunkGrid, ChunkState, Deltas, GridOp, PhysicsMeshState, WorldChunk, neighbors_of},
    hasher::VoxelHashSet,
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    metrics::Metrics,
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, compute_priority},
    world::{ResultQueues, WorldGrid},
};

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
    world_grid: WorldGrid,
    chunk_store: Arc<ChunkStore>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    message_receiver: Receiver<PipelineMessage>,
    results: Arc<ResultQueues>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut metrics = Metrics::<PipelineMetrics>::new(Duration::from_secs(2));
        let mut camera_data: Option<CameraData> = None;
        let thread_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let mut previous_physics_entities: Vec<PhysicsEntity> = vec![];
        let mut grid_info: Option<(IVec3, IVec3)> = None;

        let work_queues = Arc::new(WorkQueues {
            load: SegQueue::new(),
            mesh: SegQueue::new(),
            physics: SegQueue::new(),
        });

        let worker_pool = WorkerPool::new::<V>(
            6,
            work_queues.clone(),
            world_grid.clone(),
            chunk_store.clone(),
            generator.clone(),
            results.clone(),
        );

        loop {
            metrics.batch_print();
            let loop_time = Instant::now();
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
                        let mut grid = world_grid.write().unwrap();
                        let mut visited_dependents = VoxelHashSet::default();
                        for entity in &previous_physics_entities {
                            for offset in NEIGHBOR_OFFSETS {
                                let scaled_offset = offset * CHUNK_SIDE_32 / 2;
                                let neighbor_cv =
                                    camera_vec3_to_cv(entity.pos + scaled_offset.as_vec3());
                                if !visited_dependents.contains(&neighbor_cv)
                                    && let Some(chunk) = grid.get_mut(neighbor_cv)
                                {
                                    chunk.physics_dependents =
                                        chunk.physics_dependents.saturating_sub(1);
                                    visited_dependents.insert(neighbor_cv);
                                }
                            }
                        }

                        visited_dependents.clear();
                        for entity in &new_entities {
                            for offset in NEIGHBOR_OFFSETS {
                                let scaled_offset = offset * CHUNK_SIDE_32 / 2;
                                let neighbor_cv =
                                    camera_vec3_to_cv(entity.pos + scaled_offset.as_vec3());
                                if !visited_dependents.contains(&neighbor_cv)
                                    && let Some(chunk) = grid.get_mut(neighbor_cv)
                                {
                                    chunk.physics_dependents += 1;
                                    visited_dependents.insert(neighbor_cv);
                                }
                            }
                        }

                        previous_physics_entities = new_entities;
                    }
                    PipelineMessage::ChunkEdits(edits) => {
                        let mut grid = world_grid.write().unwrap();
                        for (wv, voxel_id) in edits {
                            let cv = wv_to_cv(wv);
                            if let Some(world_chunk) = grid.get_mut(cv) {
                                if !world_chunk.is_stable() {
                                    continue;
                                }

                                let lv = wv_to_lv(wv);
                                world_chunk.deltas.0.insert(lv, voxel_id);
                                let old_voxel_id = world_chunk.chunk.set_voxel::<V>(lv, voxel_id);
                                world_chunk.is_dirty = true;
                                world_chunk.state = ChunkState::MeshNeeded;
                                world_chunk.physics_state = PhysicsMeshState::MeshNeeded;
                                world_chunk.version = world_chunk.version.wrapping_add(1);

                                if world_chunk.chunk.is_voxel_on_edge(lv) {
                                    for neighbor_cv in neighbors_of(cv) {
                                        if let Some(neighbor) = grid.get_mut(neighbor_cv) {
                                            neighbor.state = ChunkState::MeshNeeded;
                                            neighbor.physics_state = PhysicsMeshState::MeshNeeded;
                                            neighbor.version = neighbor.version.wrapping_add(1);
                                        }
                                    }
                                }
                                results.edits.push((wv, old_voxel_id));
                            }
                        }
                    }
                }
            }

            if camera_data.is_none() {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let camera_data = camera_data.as_ref().unwrap();

            if camera_moved {
                let mut dirty_chunks = Vec::new();
                let mut grid = world_grid.write().unwrap();
                if let Some(new_grid_info) = grid.update(
                    camera_data,
                    |op| match op {
                        GridOp::Recycle(world_chunk, cv) => {
                            if world_chunk.state != ChunkState::None {
                                results.mesh_unload.push(world_chunk.cv);
                                if world_chunk.is_dirty {
                                    let persisted_chunk = PersistedChunk {
                                        uniform_voxel_id: world_chunk.uniform_voxel_id.take(),
                                        deltas: world_chunk.deltas.clone(),
                                    };
                                    dirty_chunks.push((world_chunk.cv, persisted_chunk));
                                }

                                if world_chunk.physics_dependents > 0 {
                                    results.physics_unload.push(world_chunk.cv);
                                }
                            }

                            world_chunk.reset(cv);
                        }
                    },
                    &mut metrics,
                ) {
                    grid_info = Some(new_grid_info);
                }

                drop(grid);

                if !dirty_chunks.is_empty() {
                    batch_unload_deltas_task(dirty_chunks, &chunk_store, &thread_pool);
                }
            }

            let (origin, dims) = if let Some(info) = grid_info {
                (info.0, info.1)
            } else {
                thread::sleep(std::time::Duration::from_millis(5));
                continue;
            };

            let (new_load_tasks, new_mesh_tasks, new_physics_tasks) = {
                let mut grid = world_grid.write().unwrap();
                let mut load_tasks = Vec::new();
                let mut mesh_tasks = Vec::new();
                let mut physics_loads = Vec::new();
                for world_chunk in grid.flat.iter_mut() {
                    match world_chunk.state {
                        ChunkState::LoadNeeded => {
                            load_tasks.push(WorkItem {
                                priority: compute_priority(world_chunk.cv, camera_data),
                                cv: world_chunk.cv,
                                version: 0,
                            });
                            world_chunk.state = ChunkState::Loading;
                        }
                        ChunkState::MeshNeeded => {
                            mesh_tasks.push(WorkItem {
                                priority: compute_priority(world_chunk.cv, camera_data),
                                cv: world_chunk.cv,
                                version: world_chunk.version,
                            });
                            world_chunk.state = ChunkState::Meshing;
                        }
                        _ => {}
                    }

                    if world_chunk.physics_dependents > 0 {
                        if world_chunk.physics_state == PhysicsMeshState::MeshNeeded
                            && world_chunk.is_stable()
                        {
                            physics_loads.push(WorkItem {
                                priority: 0,
                                cv: world_chunk.cv,
                                version: 0,
                            });
                            world_chunk.physics_state = PhysicsMeshState::Meshing;
                        }
                    } else {
                        if world_chunk.physics_state == PhysicsMeshState::MeshReady {
                            results.physics_unload.push(world_chunk.cv);
                            world_chunk.physics_state = PhysicsMeshState::MeshNeeded;
                        }
                    }
                }
                (load_tasks, mesh_tasks, physics_loads)
            };

            let has_work_to_do = !new_load_tasks.is_empty()
                || !new_mesh_tasks.is_empty()
                || !new_physics_tasks.is_empty();

            // TODO: Filter tasks that are not in range anymore
            if has_work_to_do {
                let mut next_load_tasks =
                    Vec::with_capacity(work_queues.load.len() + new_load_tasks.len());
                while let Some(task) = work_queues.load.pop() {
                    if ChunkGrid::cv_to_idx_with_origin(task.cv, origin, dims).is_some() {
                        next_load_tasks.push(task);
                    }
                }
                next_load_tasks.extend_from_slice(&new_load_tasks);
                next_load_tasks.sort();
                for task in next_load_tasks {
                    work_queues.load.push(task);
                }

                let mut next_mesh_tasks =
                    Vec::with_capacity(work_queues.mesh.len() + new_mesh_tasks.len());
                while let Some(task) = work_queues.mesh.pop() {
                    if ChunkGrid::cv_to_idx_with_origin(task.cv, origin, dims).is_some() {
                        next_mesh_tasks.push(task);
                    }
                }
                next_mesh_tasks.extend_from_slice(&new_mesh_tasks);
                next_mesh_tasks.sort();
                for task in next_mesh_tasks {
                    work_queues.mesh.push(task);
                }

                let mut next_physics_tasks =
                    Vec::with_capacity(work_queues.physics.len() + new_physics_tasks.len());
                while let Some(task) = work_queues.physics.pop() {
                    if ChunkGrid::cv_to_idx_with_origin(task.cv, origin, dims).is_some() {
                        next_physics_tasks.push(task);
                    }
                }
                next_physics_tasks.extend_from_slice(&new_physics_tasks);
                next_physics_tasks.sort();
                for task in next_physics_tasks {
                    work_queues.physics.push(task);
                }
                worker_pool.sender.send(()).ok();

                println!(
                    "load_tasks_len={} | mesh_tasks_len={} | physics_tasks_len={}",
                    work_queues.load.len(),
                    work_queues.mesh.len(),
                    work_queues.physics.len()
                );
            }

            if !has_work_to_do {
                thread::sleep(std::time::Duration::from_millis(5));
            } else {
                metrics
                    .get_mut(PipelineMetrics::PipelineLoop)
                    .record(loop_time.elapsed());
            }
        }
    })
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub priority: u32,
    pub cv: ChunkVector,
    pub version: u32,
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for WorkItem {}
impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

struct WorkQueues {
    load: SegQueue<WorkItem>,
    mesh: SegQueue<WorkItem>,
    physics: SegQueue<WorkItem>,
}

struct WorkerPool {
    pub sender: Sender<()>,
}

impl WorkerPool {
    fn new<V: 'static + ChunkeeVoxel>(
        num_threads: usize,
        work_queues: Arc<WorkQueues>,
        world_grid: WorldGrid,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        results: Arc<ResultQueues>,
    ) -> Self {
        let (sx, rx) = crossbeam_channel::unbounded();

        for _ in 0..num_threads {
            let work_queues = Arc::clone(&work_queues);
            let world_grid = Arc::clone(&world_grid);
            let chunk_store = Arc::clone(&chunk_store);
            let generator = Arc::clone(&generator);
            let results = results.clone();

            let thread_sx = sx.clone();
            let thread_rx = rx.clone();

            thread::spawn(move || {
                Self::worker_loop::<V>(
                    &work_queues,
                    world_grid,
                    chunk_store,
                    generator,
                    thread_sx,
                    thread_rx,
                    &results,
                );
            });
        }

        Self { sender: sx }
    }

    fn worker_loop<V: 'static + ChunkeeVoxel>(
        work_queues: &WorkQueues,
        world_grid: WorldGrid,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        sx: Sender<()>,
        rx: Receiver<()>,
        results: &ResultQueues,
    ) {
        loop {
            while let Ok(_) = rx.recv() {
                if let Some(work_item) = work_queues.physics.pop() {
                    let cv = work_item.cv;

                    match physics_mesh_task::<V>(cv, &world_grid) {
                        TaskResult::Ok(mesh) => {
                            if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                                && world_chunk.physics_state == PhysicsMeshState::Meshing
                            {
                                results.physics_load.push((cv, mesh));
                                world_chunk.physics_state = PhysicsMeshState::MeshReady;
                            }

                            sx.send(()).ok();
                            continue;
                        }
                        TaskResult::NotReady => {
                            if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                                && world_chunk.physics_state == PhysicsMeshState::Meshing
                            {
                                world_chunk.physics_state = PhysicsMeshState::MeshNeeded;
                            }
                        }
                        TaskResult::Invalid => {}
                    }
                }

                if let Some(work_item) = work_queues.mesh.pop() {
                    let cv = work_item.cv;

                    match mesh_task::<V>(cv, &world_grid) {
                        TaskResult::Ok(mesh) => {
                            if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                                && world_chunk.state == ChunkState::Meshing
                                && world_chunk.version == work_item.version
                            {
                                results.mesh_load.push((cv, mesh));
                                world_chunk.state = ChunkState::MeshReady;
                            }

                            sx.send(()).ok();
                            continue;
                        }
                        TaskResult::NotReady => {
                            if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                                && world_chunk.state == ChunkState::Meshing
                                && world_chunk.version == work_item.version
                            {
                                world_chunk.state = ChunkState::MeshNeeded;
                            }
                        }
                        TaskResult::Invalid => {}
                    }
                }

                if let Some(work_item) = work_queues.load.pop() {
                    let cv = work_item.cv;

                    match load_task::<V>(cv, &world_grid, &chunk_store, &generator) {
                        TaskResult::Ok((chunk, deltas, uniform_voxel_id)) => {
                            if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                                && world_chunk.state == ChunkState::Loading
                            {
                                world_chunk.chunk = chunk;
                                world_chunk.deltas = deltas;
                                world_chunk.uniform_voxel_id = uniform_voxel_id;
                                world_chunk.merge_deltas::<V>();
                                world_chunk.state = ChunkState::MeshNeeded;
                            }
                        }
                        _ => {}
                    }
                }

                if !work_queues.load.is_empty()
                    || !work_queues.mesh.is_empty()
                    || !work_queues.physics.is_empty()
                {
                    sx.send(()).ok();
                    continue;
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
    world_grid: &WorldGrid,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
) -> TaskResult<(Chunk, Deltas, Option<VoxelId>)> {
    if world_grid
        .read()
        .unwrap()
        .get(cv)
        .is_none_or(|world_chunk| world_chunk.state != ChunkState::Loading)
    {
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
        generator.apply(cv_to_wv(cv), &mut chunk);
    }

    TaskResult::Ok((chunk, deltas, uniform_voxel_id))
}

fn mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    world_grid: &WorldGrid,
) -> TaskResult<ChunkMeshGroup> {
    let grid = world_grid.read().unwrap();

    let world_chunk = if let Some(world_chunk) = grid.get(cv) {
        world_chunk
    } else {
        return TaskResult::Invalid;
    };

    let neighbors: [Option<&WorldChunk>; 6] =
        std::array::from_fn(|idx| grid.get(cv + BLOCK_FACES[idx].into_normal()));

    let not_all_stable = neighbors
        .iter()
        .any(|op| op.is_some_and(|wc| !wc.is_stable()));

    if not_all_stable {
        return TaskResult::NotReady;
    }

    let complete_neighbors: Box<[Chunk; 6]> = Box::new(std::array::from_fn(|i| {
        if let Some(n) = neighbors[i] {
            return n.chunk.clone();
        }

        Chunk::new()
    }));

    let center = Box::new(world_chunk.chunk.clone());

    drop(grid);

    let mesh = mesh_chunk::<V>(cv, center, complete_neighbors);

    TaskResult::Ok(mesh)
}

fn physics_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    world_grid: &WorldGrid,
) -> TaskResult<Vec<Vec3>> {
    let chunk = if let Some(world_chunk) = world_grid.read().unwrap().get(cv) {
        if !world_chunk.is_stable() {
            return TaskResult::NotReady;
        }
        Box::new(world_chunk.chunk.clone())
    } else {
        return TaskResult::Invalid;
    };

    let mesh = mesh_physics_chunk::<V>(cv, chunk);

    TaskResult::Ok(mesh)
}
