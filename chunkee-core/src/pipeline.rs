use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, Vec3};

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::Chunk,
    coords::{ChunkVector, NEIGHBOR_OFFSETS, camera_vec3_to_cv, cv_to_wv, wv_to_cv, wv_to_lv},
    define_metrics,
    generation::VoxelGenerator,
    grid::{Deltas, GridOp, PhysicsMeshState, PipelineState, WorldChunk, neighbors_of},
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    metrics::Metrics,
    storage::{ChunkStore, PersistedChunk},
    streaming::{CameraData, calculate_chunk_priority},
    world::WorldGrid,
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
    ChunkUnloaded {
        cv: ChunkVector,
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
pub struct WorkItem {
    pub priority: u32,
    pub cv: ChunkVector,
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
        other.priority.cmp(&self.priority)
    }
}

struct WorkQueues {
    load: BinaryHeap<WorkItem>,
    mesh: BinaryHeap<WorkItem>,
    physics: BinaryHeap<WorkItem>,
}

struct WorkerPool;

impl WorkerPool {
    fn new<V: 'static + ChunkeeVoxel>(
        num_threads: usize,
        work_queues: Arc<Mutex<WorkQueues>>,
        cvar: Arc<Condvar>,
        world_grid: WorldGrid,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        result_sender: Sender<PipelineResult>,
    ) {
        for _ in 0..num_threads {
            let work_queues = Arc::clone(&work_queues);
            let cvar = Arc::clone(&cvar);
            let world_grid = Arc::clone(&world_grid);
            let chunk_store = Arc::clone(&chunk_store);
            let generator = Arc::clone(&generator);
            let result_sender = result_sender.clone();

            thread::spawn(move || {
                Self::worker_loop::<V>(
                    work_queues,
                    cvar,
                    world_grid,
                    chunk_store,
                    generator,
                    result_sender,
                );
            });
        }
    }

    fn worker_loop<V: 'static + ChunkeeVoxel>(
        work_queues: Arc<Mutex<WorkQueues>>,
        cvar: Arc<Condvar>,
        world_grid: WorldGrid,
        chunk_store: Arc<ChunkStore>,
        generator: Arc<Box<dyn VoxelGenerator>>,
        result_sender: Sender<PipelineResult>,
    ) {
        loop {
            let next_physics_mesh = work_queues.lock().unwrap().physics.pop();
            if let Some(work_item) = next_physics_mesh {
                let cv = work_item.cv;

                if let Some(mesh) = physics_mesh_task::<V>(cv, &world_grid) {
                    if !mesh.is_empty() {
                        result_sender
                            .send(PipelineResult::PhysicsMeshReady { cv: cv, mesh })
                            .ok();
                    }

                    cvar.notify_one();
                    continue;
                } else {
                    if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                        && world_chunk.physics_state == PhysicsMeshState::Meshing
                    {
                        world_chunk.physics_state = PhysicsMeshState::MeshNeeded;
                    }
                }
            }

            let next_mesh_item = work_queues.lock().unwrap().mesh.pop();
            if let Some(work_item) = next_mesh_item {
                let cv = work_item.cv;

                if let Some(mesh) = mesh_task::<V>(cv, &world_grid) {
                    if !mesh.opaque.indices.is_empty() || !mesh.translucent.indices.is_empty() {
                        result_sender
                            .send(PipelineResult::MeshReady { cv: cv, mesh })
                            .ok();
                    }

                    cvar.notify_one();
                    continue;
                } else {
                    if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
                        && world_chunk.state == PipelineState::Meshing
                    {
                        world_chunk.state = PipelineState::MeshNeeded;
                    }
                }
            }

            let next_load_item = work_queues.lock().unwrap().load.pop();
            if let Some(work_item) = next_load_item {
                let cv = work_item.cv;
                if let Some(_) = load_task::<V>(cv, &world_grid, &chunk_store, &generator) {
                    cvar.notify_one();
                    continue;
                }
            } else {
                cvar.wait(work_queues.lock().unwrap()).ok();
            }
        }
    }
}

pub fn spawn_pipeline_thread<V: 'static + ChunkeeVoxel>(
    world_grid: WorldGrid,
    chunk_store: Arc<ChunkStore>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    message_receiver: Receiver<PipelineMessage>,
    result_sender: Sender<PipelineResult>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut metrics = Metrics::<PipelineMetrics>::new(Duration::from_secs(2));
        let mut camera_data: Option<CameraData> = None;
        let thread_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let mut previous_physics_entities: Vec<PhysicsEntity> = vec![];

        let cvar = Arc::new(Condvar::new());
        let work_queues = Arc::new(Mutex::new(WorkQueues {
            load: BinaryHeap::new(),
            mesh: BinaryHeap::new(),
            physics: BinaryHeap::new(),
        }));

        WorkerPool::new::<V>(
            12,
            work_queues.clone(),
            cvar.clone(),
            world_grid.clone(),
            chunk_store.clone(),
            generator.clone(),
            result_sender.clone(),
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
                        for entity in &previous_physics_entities {
                            let cv = camera_vec3_to_cv(entity.pos);
                            for offset in NEIGHBOR_OFFSETS {
                                let neighbor_cv = offset + cv;
                                if let Some(chunk) = grid.get_mut(neighbor_cv) {
                                    chunk.physics_dependents =
                                        chunk.physics_dependents.saturating_sub(1);
                                }
                            }
                        }

                        for entity in &new_entities {
                            let cv = camera_vec3_to_cv(entity.pos);
                            for offset in NEIGHBOR_OFFSETS {
                                let neighbor_cv = offset + cv;
                                if let Some(chunk) = grid.get_mut(neighbor_cv) {
                                    chunk.physics_dependents += 1;
                                }
                            }
                        }

                        previous_physics_entities = new_entities;
                    }
                    PipelineMessage::ChunkEdits(edits) => {
                        let mut applied_edits = Vec::with_capacity(edits.len());
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
                                world_chunk.state = PipelineState::MeshNeeded;
                                world_chunk.physics_state = PhysicsMeshState::MeshNeeded;

                                if world_chunk.chunk.is_voxel_on_edge(lv) {
                                    for neighbor_cv in neighbors_of(cv) {
                                        if let Some(neighbor) = grid.get_mut(neighbor_cv) {
                                            neighbor.state = PipelineState::MeshNeeded;
                                            neighbor.physics_state = PhysicsMeshState::MeshNeeded;
                                        }
                                    }
                                }
                                applied_edits.push((wv, old_voxel_id));
                            }
                        }

                        result_sender
                            .send(PipelineResult::EditsApplied(applied_edits))
                            .ok();
                    }
                }
            }

            if camera_data.is_none() {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let camera_data = camera_data.as_ref().unwrap();

            if camera_moved {
                let mut chunks_to_save = Vec::new();
                let mut physics_unloads = Vec::new();

                let mut grid = world_grid.write().unwrap();
                grid.update(
                    camera_data,
                    |op| match op {
                        GridOp::Recycle(world_chunk, cv) => {
                            if world_chunk.is_dirty {
                                let persisted_chunk = PersistedChunk {
                                    uniform_voxel_id: world_chunk.uniform_voxel_id.take(),
                                    deltas: world_chunk.deltas.clone(),
                                };
                                chunks_to_save.push((world_chunk.cv, persisted_chunk));
                            }

                            if world_chunk.physics_dependents > 0 {
                                physics_unloads.push(world_chunk.cv);
                            }

                            world_chunk.reset(cv);
                        } // GridOp::Keep(world_chunk) => {
                          //     world_chunk.priority =
                          //         calculate_chunk_priority(world_chunk.cv, camera_data);
                          // }
                    },
                    &mut metrics,
                );

                drop(grid);

                if !chunks_to_save.is_empty() {
                    batch_unload_deltas_task(chunks_to_save, &chunk_store, &thread_pool);
                }

                if !physics_unloads.is_empty() {
                    result_sender
                        .send(PipelineResult::PhysicsMeshUnload {
                            cvs: physics_unloads,
                        })
                        .ok();
                }
            }

            let (load_tasks, mesh_tasks, physics_loads, physics_unloads) = {
                let mut grid = world_grid.write().unwrap();
                let mut load_tasks = Vec::new();
                let mut mesh_tasks = Vec::new();
                let mut physics_loads = Vec::new();
                let mut physics_unloads = Vec::new();
                for world_chunk in grid.flat.iter_mut() {
                    match world_chunk.state {
                        PipelineState::LoadNeeded => {
                            load_tasks.push(WorkItem {
                                priority: calculate_chunk_priority(world_chunk.cv, camera_data),
                                cv: world_chunk.cv,
                            });
                            world_chunk.state = PipelineState::Loading;
                        }
                        PipelineState::MeshNeeded => {
                            mesh_tasks.push(WorkItem {
                                priority: calculate_chunk_priority(world_chunk.cv, camera_data),
                                cv: world_chunk.cv,
                            });
                            world_chunk.state = PipelineState::Meshing;
                        }
                        _ => {}
                    }

                    if world_chunk.physics_dependents > 0 {
                        if world_chunk.physics_state == PhysicsMeshState::MeshNeeded
                            && world_chunk.is_stable()
                        {
                            physics_loads.push(WorkItem {
                                priority: 0, // physics has high priority
                                cv: world_chunk.cv,
                            });
                            world_chunk.physics_state = PhysicsMeshState::Meshing;
                        }
                    } else {
                        if world_chunk.physics_state == PhysicsMeshState::MeshReady {
                            physics_unloads.push(world_chunk.cv);
                            world_chunk.physics_state = PhysicsMeshState::MeshNeeded;
                        }
                    }
                }
                (load_tasks, mesh_tasks, physics_loads, physics_unloads)
            };

            if !physics_unloads.is_empty() {
                result_sender
                    .send(PipelineResult::PhysicsMeshUnload {
                        cvs: physics_unloads,
                    })
                    .ok();
            }

            let has_work_to_do =
                !load_tasks.is_empty() || !mesh_tasks.is_empty() || !physics_loads.is_empty();

            if has_work_to_do {
                let mut queues = work_queues.lock().unwrap();
                queues.load.extend(load_tasks);
                queues.mesh.extend(mesh_tasks);
                queues.physics.extend(physics_loads);
                cvar.notify_all();
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

fn load_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    world_grid: &WorldGrid,
    chunk_store: &ChunkStore,
    generator: &Box<dyn VoxelGenerator>,
) -> Option<()> {
    if let Some(world_chunk) = world_grid.read().unwrap().get(cv)
        && world_chunk.state != PipelineState::Loading
    {
        return None;
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

    if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
        && world_chunk.state == PipelineState::Loading
    {
        world_chunk.deltas = deltas;
        world_chunk.uniform_voxel_id = uniform_voxel_id;
        world_chunk.chunk = chunk;
        world_chunk.merge_deltas::<V>();
        world_chunk.state = PipelineState::MeshNeeded;

        Some(())
    } else {
        None
    }
}

fn mesh_task<V: ChunkeeVoxel>(cv: ChunkVector, world_grid: &WorldGrid) -> Option<ChunkMeshGroup> {
    let grid = world_grid.read().unwrap();

    let world_chunk = if let Some(world_chunk) = grid.get(cv) {
        world_chunk
    } else {
        return None;
    };

    let neighbors: [Option<&WorldChunk>; 6] =
        std::array::from_fn(|idx| grid.get(cv + BLOCK_FACES[idx].into_normal()));

    let not_all_stable = neighbors
        .iter()
        .any(|op| op.is_some_and(|wc| !wc.is_stable()));

    if not_all_stable {
        return None;
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

    if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv)
        && world_chunk.state == PipelineState::Meshing
    {
        world_chunk.state = PipelineState::MeshReady;
        Some(mesh)
    } else {
        None
    }
}

fn physics_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    world_grid: &WorldGrid,
) -> Option<Vec<Vec3>> {
    let chunk = if let Some(world_chunk) = world_grid.read().unwrap().get(cv) {
        if !world_chunk.is_stable() {
            return None;
        }
        Box::new(world_chunk.chunk.clone())
    } else {
        return None;
    };

    let mesh = mesh_physics_chunk::<V>(cv, chunk);

    if let Some(world_chunk) = world_grid.write().unwrap().get_mut(cv) {
        if world_chunk.physics_state == PhysicsMeshState::Meshing {
            world_chunk.physics_state = PhysicsMeshState::MeshReady;
        }
    }

    Some(mesh)
}
