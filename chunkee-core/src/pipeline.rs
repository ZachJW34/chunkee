use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    num::NonZero,
    sync::{Arc, Condvar, Mutex, Weak},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use glam::{IVec3, Vec3};
use lru::LruCache;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{Chunk, Chunk32, Chunk64},
    coords::{ChunkVector, NEIGHBOR_OFFSETS, camera_vec3_to_cv, cv_to_wv, wv_to_cv, wv_to_lv},
    dag::TaskGraph,
    define_metrics,
    generation::VoxelGenerator,
    grid::{Deltas, GridOp, PipelineState, WorldChunk, neighbors_of},
    hasher::{BuildMortonHasher, VoxelHashMap},
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    metrics::Metrics,
    storage::{BatchedPersistedChunkMap, ChunkStore, PersistedChunk},
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

#[derive(PartialEq)]
pub enum PhysicsMeshState {
    MeshNeeded,
    Meshing,
    MeshReady,
}

pub struct PhysicsMesh {
    dependents: u32,
    state: PhysicsMeshState,
}

impl PhysicsMesh {
    fn new() -> Self {
        Self {
            dependents: 0,
            state: PhysicsMeshState::MeshNeeded,
        }
    }
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

enum TaskResult {
    // IoComplete {
    //     persisted_chunks: BatchedPersistedChunkMap,
    //     duration: Duration,
    // },
    LoadComplete {
        cv: ChunkVector,
        chunk: Chunk,
        deltas: Option<Deltas>,
        uniform_voxel_id: Option<VoxelId>,
        duration: Duration,
        generated: bool,
    },
    MeshComplete {
        cv: ChunkVector,
        mesh: ChunkMeshGroup,
        duration: Duration,
    },
    PhysicsMeshComplete {
        cv: ChunkVector,
        mesh: Vec<Vec3>,
        duration: Duration,
    },
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
        self.priority.cmp(&other.priority)
    }
}

// pub type WorkQueue = BinaryHeap<WorkItem>;

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

        let (t_sx, t_rx) = crossbeam_channel::unbounded::<TaskResult>();

        let thread_pool = rayon::ThreadPoolBuilder::new().build().unwrap();

        let mut previous_physics_entities: Vec<PhysicsEntity> = vec![];
        let mut physics_meshes: Arc<Mutex<VoxelHashMap<PhysicsMesh>>> =
            Arc::new(Mutex::new(Default::default()));

        // struct LOD0Asset(Chunk32);
        // struct LOD0 {
        //     cv: ChunkVector,
        //     state: PipelineState,
        //     data: Weak<LOD0Asset>,
        // }

        // struct LOD1Asset(Chunk64);
        // struct LOD1 {
        //     cv: ChunkVector,
        //     state: PipelineState,
        //     dependents: [Option<Arc<LOD1Asset>>; 7],
        //     data: Weak<LOD1Asset>,
        // }

        // let capacity = (17 * 17 * 17) as usize;
        // let world_chunk_tracker: DashMap<ChunkVector, WorldChunk, BuildMortonHasher> =
        //     DashMap::with_capacity_and_hasher(capacity, BuildMortonHasher::new());

        // let lod0_lru: LruCache<ChunkVector, Arc<Chunk32>, BuildMortonHasher> =
        //     LruCache::with_hasher(NonZero::new(1000).unwrap(), BuildMortonHasher::new());
        // let lod0_state_map: VoxelHashMap<LOD0> = Default::default();
        // let lod1_lru: LruCache<ChunkVector, Arc<LOD1Asset>, BuildMortonHasher> =
        //     LruCache::with_hasher(NonZero::new(1000).unwrap(), BuildMortonHasher::new());
        // let lod1_state_map: VoxelHashMap<LOD1> = Default::default();

        let cvar = Arc::new(Condvar::new());
        let load_queue: VecDeque<WorkItem> = VecDeque::new();
        let mesh_queue: VecDeque<WorkItem> = VecDeque::new();
        let physics_queue: VecDeque<WorkItem> = VecDeque::new();
        let queue_lock = Arc::new(Mutex::new((load_queue, mesh_queue, physics_queue)));

        let num_threads = 12;

        enum Job {
            Load(WorkItem),
            Mesh(WorkItem),
            PhysicsMesh(WorkItem),
        }

        for i in 0..num_threads {
            let cvar = cvar.clone();
            let chunk_store = chunk_store.clone();
            let queue_lock = queue_lock.clone();
            let generator = generator.clone();
            let grid = world_grid.clone();
            let result_sender = result_sender.clone();
            let physics_meshes = physics_meshes.clone();

            thread::spawn(move || {
                println!("[Worker {}]: Started.", i);

                loop {
                    let mut guard = queue_lock.lock().unwrap();

                    let job: Job = if let Some(item) = guard.2.pop_front() {
                        Job::PhysicsMesh(item)
                    } else if let Some(item) = guard.1.pop_front() {
                        Job::Mesh(item)
                    } else if let Some(item) = guard.0.pop_front() {
                        Job::Load(item)
                    } else {
                        cvar.wait(guard).ok();
                        continue;
                    };

                    drop(guard);

                    match job {
                        Job::Load(item) => {
                            let cv = item.cv;
                            if let Some(world_chunk) = grid.write().unwrap().get_mut(cv)
                                && world_chunk.state == PipelineState::LoadNeeded
                            {
                                world_chunk.state = PipelineState::Loading
                            }

                            let mut chunk = Chunk::new();

                            let persisted_chunk = chunk_store.load_chunk(cv);
                            let (deltas, uniform_voxel_id) = persisted_chunk
                                .map(|pc| (pc.deltas, pc.uniform_voxel_id))
                                .unwrap_or_default();

                            if let Some(uniform) = uniform_voxel_id {
                                chunk.fill::<V>(uniform);

                                if let Some(world_chunk) = grid.write().unwrap().get_mut(cv)
                                    && world_chunk.state == PipelineState::Loading
                                {
                                    world_chunk.state = PipelineState::MeshNeeded;
                                    world_chunk.deltas = deltas;
                                    world_chunk.uniform_voxel_id = Some(uniform);
                                };
                            } else {
                                generator.apply(cv_to_wv(cv), &mut chunk);

                                if let Some(world_chunk) = grid.write().unwrap().get_mut(cv)
                                    && world_chunk.state == PipelineState::Loading
                                {
                                    world_chunk.state = PipelineState::MeshNeeded;
                                    world_chunk.deltas = deltas;
                                    world_chunk.uniform_voxel_id = uniform_voxel_id;
                                };
                            }

                            cvar.notify_one();
                        }
                        Job::Mesh(item) => {
                            let cv = item.cv;
                            let mut grid_lock = grid.write().unwrap();

                            let world_chunk = if let Some(world_chunk) = grid_lock.get(cv) {
                                world_chunk
                            } else {
                                continue;
                            };

                            let neighbors: [Option<&WorldChunk>; 6] = std::array::from_fn(|idx| {
                                grid_lock.get(cv + BLOCK_FACES[idx].into_normal())
                            });

                            let not_all_stable = neighbors
                                .iter()
                                .any(|op| op.is_some_and(|wc| !wc.is_stable()));

                            if not_all_stable {
                                // TODO: consider below
                                // let mut queue_guard = queue_lock.lock().unwrap();
                                // queue_guard.mesh.push_back(item);
                                // // Don't notify, as we didn't add "new" valuable work
                                // continue; // Go back to the top to wait or find other work
                                continue;
                            }

                            let complete_neighbors: Box<[Chunk; 6]> =
                                Box::new(std::array::from_fn(|i| {
                                    if let Some(n) = neighbors[i] {
                                        return n.chunk.clone();
                                    }

                                    Chunk::new()
                                }));

                            let center = Box::new(world_chunk.chunk.clone());

                            grid_lock.get_mut(cv).unwrap().state = PipelineState::Meshing;

                            drop(grid_lock);

                            let mesh = mesh_chunk::<V>(cv, center, complete_neighbors);

                            if let Some(world_chunk) = grid.write().unwrap().get_mut(cv)
                                && world_chunk.state == PipelineState::Meshing
                            {
                                world_chunk.state = PipelineState::MeshReady;
                            }

                            result_sender
                                .send(PipelineResult::MeshReady { cv, mesh })
                                .ok();

                            cvar.notify_one();
                        }
                        Job::PhysicsMesh(item) => {
                            let cv = item.cv;
                            let chunk = if let Some(world_chunk) = grid.read().unwrap().get(cv)
                                && world_chunk.is_stable()
                            {
                                Box::new(world_chunk.chunk.clone())
                            } else {
                                continue;
                            };

                            if let Some(p_mesh) = physics_meshes.lock().unwrap().get_mut(&cv) {
                                p_mesh.state = PhysicsMeshState::Meshing
                            } else {
                                continue;
                            }

                            let mesh = mesh_physics_chunk::<V>(cv, chunk);

                            result_sender
                                .send(PipelineResult::PhysicsMeshReady { cv, mesh })
                                .ok();

                            if let Some(p_mesh) = physics_meshes.lock().unwrap().get_mut(&cv) {
                                p_mesh.state = PhysicsMeshState::MeshReady
                            } else {
                                continue;
                            }
                        }
                    }
                }
            });
        }
        loop {
            metrics.batch_print();
            let loop_time = Instant::now();
            let mut camera_moved = false;
            for message in message_receiver.try_iter() {
                match message {
                    PipelineMessage::Shutdown => return,
                    PipelineMessage::CameraDataUpdate(cd) => {
                        if camera_data
                            .as_ref()
                            .map_or(true, |old_cd| old_cd.pos != cd.pos)
                        {
                            camera_moved = true;
                        }
                        camera_data.replace(cd);
                    }
                    PipelineMessage::PhysicsEntitiesUpdate(new_entities) => {
                        let mut physics_meshes = physics_meshes.lock().unwrap();
                        for entity in &previous_physics_entities {
                            let cv = camera_vec3_to_cv(entity.pos);
                            for offset in NEIGHBOR_OFFSETS {
                                let neighbor_cv = offset + cv;
                                if let Some(p_mesh) = physics_meshes.get_mut(&neighbor_cv) {
                                    p_mesh.dependents = p_mesh.dependents.saturating_sub(1);
                                }
                            }
                        }

                        for entity in &new_entities {
                            let cv = camera_vec3_to_cv(entity.pos);
                            for offset in NEIGHBOR_OFFSETS {
                                let neighbor_cv = offset + cv;
                                let p_mesh = physics_meshes
                                    .entry(neighbor_cv)
                                    .or_insert(PhysicsMesh::new());
                                p_mesh.dependents += 1;
                            }
                        }

                        previous_physics_entities = new_entities;
                    }
                    PipelineMessage::ChunkEdits(edits) => {
                        let mut grid_lock = world_grid.write().unwrap();
                        let mut applied_edits = Vec::with_capacity(edits.len());
                        for (wv, voxel_id) in edits {
                            let cv = wv_to_cv(wv);
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.is_stable()
                            {
                                let lv = wv_to_lv(wv);
                                world_chunk.deltas.0.insert(lv, voxel_id);
                                let old_voxel_id = world_chunk.chunk.set_voxel::<V>(lv, voxel_id);
                                world_chunk.is_dirty = true;
                                world_chunk.state = PipelineState::MeshNeeded;
                                world_chunk.priority = 0;

                                let mut physics_meshes = physics_meshes.lock().unwrap();

                                if let Some(p_mesh) = physics_meshes.get_mut(&cv)
                                    && p_mesh.state != PhysicsMeshState::MeshNeeded
                                {
                                    p_mesh.state = PhysicsMeshState::MeshNeeded;
                                }

                                if world_chunk.chunk.is_voxel_on_edge(lv) {
                                    for neighbor_cv in neighbors_of(cv) {
                                        if let Some(neighbor) = grid_lock.get_mut(neighbor_cv) {
                                            neighbor.state = PipelineState::MeshNeeded;
                                            neighbor.priority = 0;

                                            if let Some(p_mesh) =
                                                physics_meshes.get_mut(&neighbor_cv)
                                                && p_mesh.state != PhysicsMeshState::MeshNeeded
                                            {
                                                p_mesh.state = PhysicsMeshState::MeshNeeded;
                                            }
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
                let mut chunks_to_load = Vec::new();
                let mut chunks_to_save = Vec::new();

                let mut grid_lock = world_grid.write().unwrap();
                grid_lock.update(
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

                            let priority = calculate_chunk_priority(cv, camera_data);

                            world_chunk.cv = cv;
                            world_chunk.priority = priority;
                            world_chunk.is_dirty = false;
                            world_chunk.state = PipelineState::LoadNeeded;
                            world_chunk.deltas = Deltas::default();
                            world_chunk.uniform_voxel_id = None;

                            chunks_to_load.push(cv);
                        }
                        GridOp::Keep(world_chunk) => {
                            world_chunk.priority =
                                calculate_chunk_priority(world_chunk.cv, camera_data);
                        }
                    },
                    &mut metrics,
                );

                drop(grid_lock);

                if !chunks_to_save.is_empty() {
                    batch_unload_deltas_task(chunks_to_save, &chunk_store, &thread_pool);
                }

                // if !chunks_to_load.is_empty() {
                //     let mut grid_lock = world_grid.write().unwrap();
                //     for cv in chunks_to_load {
                //         if let Some(world_chunk) = grid_lock.get_mut(cv) {
                //             world_chunk.state = PipelineState::DeltasLoading;
                //         }
                //     }
                // }
            }

            let results: Vec<_> = t_rx.try_iter().collect();

            let (mut generate_tasks, mut mesh_tasks) = {
                let grid_lock = world_grid.read().unwrap();
                // let mut io_tasks = Vec::new();
                let mut load_tasks = Vec::new();
                let mut mesh_tasks = Vec::new();
                for world_chunk in grid_lock.flat.iter() {
                    match world_chunk.state {
                        // PipelineState::DeltasNeeded => io_tasks.push(WorkItem {
                        //     priority: world_chunk.priority,
                        //     cv: world_chunk.cv,
                        // }),
                        PipelineState::LoadNeeded => load_tasks.push(WorkItem {
                            priority: calculate_chunk_priority(world_chunk.cv, camera_data),
                            cv: world_chunk.cv,
                        }),
                        PipelineState::MeshNeeded => mesh_tasks.push(WorkItem {
                            priority: calculate_chunk_priority(world_chunk.cv, camera_data),
                            cv: world_chunk.cv,
                        }),
                        _ => {}
                    }
                }
                (load_tasks, mesh_tasks)
            };

            // io_tasks.sort();
            generate_tasks.sort();
            mesh_tasks.sort();

            // for item in generate_tasks.iter().take(10) {
            //     load_queue.push(item.clone());
            // }

            let mut physics_unloads = vec![];
            let mut physics_loads = vec![];

            physics_meshes.lock().unwrap().retain(|cv, p_mesh| {
                if p_mesh.dependents == 0 {
                    physics_unloads.push(*cv);
                    return false;
                }

                if p_mesh.state == PhysicsMeshState::MeshNeeded {
                    physics_loads.push(*cv);
                }

                true
            });

            result_sender
                .send(PipelineResult::PhysicsMeshUnload {
                    cvs: physics_unloads,
                })
                .ok();

            let work_to_do = !results.is_empty()
                || !generate_tasks.is_empty()
                || !mesh_tasks.is_empty()
                || !physics_loads.is_empty();

            if work_to_do {
                println!(
                    "max threads: {} | current threads: {}",
                    rayon::max_num_threads(),
                    rayon::current_num_threads()
                );
                // for _ in 0..threads_to_spin_up {

                //     let load_queue = load_queue.clone();
                //     rayon::spawn(move|| {
                //         if let Some(item) = load_queue.pop() {
                //             println!("Working on item: {item:?}");
                //         }

                //     });
                // }
            }

            if work_to_do {
                let mut grid_lock = world_grid.write().unwrap();

                for result in results {
                    match result {
                        // TaskResult::IoComplete {
                        //     persisted_chunks,
                        //     duration,
                        // } => {
                        //     metrics.get_mut(PipelineMetrics::IoTask).record(duration);
                        //     for (cv, persisted_chunk) in persisted_chunks {
                        //         if let Some(world_chunk) = grid_lock.get_mut(cv) {
                        //             if world_chunk.state == PipelineState::DeltasLoading {
                        //                 if let Some(pc) = persisted_chunk {
                        //                     world_chunk.deltas = pc.deltas;
                        //                     world_chunk.uniform_voxel_id = pc.uniform_voxel_id;
                        //                 };
                        //                 if let Some(uniform) = world_chunk.uniform_voxel_id {
                        //                     world_chunk.chunk.fill::<V>(uniform);
                        //                     world_chunk.merge_deltas::<V>();
                        //                     world_chunk.state = PipelineState::MeshNeeded;
                        //                 } else {
                        //                     world_chunk.state = PipelineState::GenerationNeeded;
                        //                 }
                        //             }
                        //         }
                        //     }
                        // }
                        // TODO: Maybe an enum here for an io load vs gen load
                        TaskResult::LoadComplete {
                            cv,
                            chunk,
                            deltas,
                            uniform_voxel_id,
                            duration,
                            generated,
                        } => {
                            metrics
                                .get_mut(PipelineMetrics::GenerationTask)
                                .record(duration);

                            let mut should_notify_neighbors = false;
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.state == PipelineState::Loading
                            {
                                should_notify_neighbors = true;

                                if let Some(del) = deltas {
                                    world_chunk.deltas = del;
                                }
                                if uniform_voxel_id.is_some() {
                                    world_chunk.uniform_voxel_id = uniform_voxel_id;
                                    world_chunk.is_dirty = generated;
                                }
                                world_chunk.chunk = chunk;
                                world_chunk.merge_deltas::<V>();
                                world_chunk.state = PipelineState::MeshNeeded;
                            }

                            if should_notify_neighbors {
                                for neighbor_cv in neighbors_of(cv) {
                                    if let Some(neighbor) = grid_lock.get_mut(neighbor_cv)
                                        && matches!(
                                            neighbor.state,
                                            PipelineState::MeshNeeded
                                                | PipelineState::Meshing
                                                | PipelineState::MeshReady
                                        )
                                    {
                                        neighbor.state = PipelineState::MeshNeeded;
                                    }
                                }
                            }
                        }
                        TaskResult::MeshComplete { cv, mesh, duration } => {
                            metrics.get_mut(PipelineMetrics::MeshTask).record(duration);
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.state == PipelineState::Meshing
                            {
                                world_chunk.state = PipelineState::MeshReady;
                                result_sender
                                    .send(PipelineResult::MeshReady { cv, mesh })
                                    .ok();
                            }
                        }
                        TaskResult::PhysicsMeshComplete { cv, mesh, duration } => {
                            metrics
                                .get_mut(PipelineMetrics::PhysicsTask)
                                .record(duration);
                            if let Some(p_mesh) = physics_meshes.lock().unwrap().get_mut(&cv)
                                && p_mesh.state == PhysicsMeshState::Meshing
                            {
                                p_mesh.state = PhysicsMeshState::MeshReady;
                                result_sender
                                    .send(PipelineResult::PhysicsMeshReady { cv, mesh })
                                    .ok();
                            }
                        }
                    }
                }

                // while let Some(cv) = physics_loads.pop()
                //     && let Some(p_mesh) = physics_meshes.get_mut(&cv)
                //     && p_mesh.state == PhysicsMeshState::MeshNeeded
                //     && let Some(world_chunk) = grid_lock.get(cv)
                //     && world_chunk.is_stable()
                // {
                //     p_mesh.state = PhysicsMeshState::Meshing;
                //     let chunk = Box::new(world_chunk.chunk.clone());
                //     physics_mesh_task::<V>(cv, chunk, &t_sx, &thread_pool);
                // }

                let drain_limit = 50;
                for _ in 0..drain_limit {
                    if generate_tasks.is_empty()
                        && mesh_tasks.is_empty()
                        && physics_loads.is_empty()
                    {
                        break;
                    }

                    if let Some(task) = generate_tasks.pop()
                        && let Some(world_chunk) = grid_lock.get_mut(task.cv)
                    {
                        world_chunk.state = PipelineState::Loading;
                        load_task::<V>(task.cv, &chunk_store, &generator, &t_sx, &thread_pool);
                    }

                    while let Some(task) = mesh_tasks.pop() {
                        let neighbors: [Option<&WorldChunk>; 6] = std::array::from_fn(|idx| {
                            grid_lock.get(task.cv + BLOCK_FACES[idx].into_normal())
                        });

                        let not_all_stable = neighbors
                            .iter()
                            .any(|op| op.is_some_and(|wc| !wc.is_stable()));

                        if not_all_stable {
                            continue;
                        }

                        let mut neighbor_chunks: Box<[Option<Chunk>; 6]> = Box::new([None; 6]);
                        for (i, maybe_chunk) in neighbors.iter().enumerate() {
                            if let Some(world_chunk) = maybe_chunk {
                                neighbor_chunks[i] = Some(world_chunk.chunk.clone())
                            }
                        }

                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.state = PipelineState::Meshing;
                            let boxed_chunk = Box::new(world_chunk.chunk.clone());
                            mesh_task::<V>(
                                task.cv,
                                boxed_chunk,
                                neighbor_chunks,
                                &t_sx,
                                &thread_pool,
                            );

                            break;
                        }
                    }

                    // if let Some(task) = io_tasks.pop()
                    //     && let Some(world_chunk) = grid_lock.get_mut(task.cv)
                    // {
                    //     world_chunk.state = PipelineState::DeltasLoading;
                    //     io_task(vec![task.cv], &chunk_store, &t_sx, &io_pool);
                    // }
                }
            }

            if !work_to_do && t_rx.is_empty() {
                thread::sleep(std::time::Duration::from_millis(5));
                // cold loop
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
    io_pool: &rayon::ThreadPool,
) {
    let chunk_store = chunk_store.clone();
    io_pool.spawn(move || {
        chunk_store.save_chunks(chunks_to_save);
    });
}

fn load_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk_store: &Arc<ChunkStore>,
    generator: &Arc<Box<dyn VoxelGenerator>>,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
) {
    let chunk_store = chunk_store.clone();
    let generator = generator.clone();
    let t_sx = t_sx.clone();
    compute_pool.spawn(move || {
        let time = Instant::now();

        let mut chunk = Chunk::new();

        let persisted_chunk = chunk_store.load_chunk(cv);

        if let Some(pc) = &persisted_chunk
            && let Some(uniform) = pc.uniform_voxel_id
        {
            chunk.fill::<V>(uniform);
            t_sx.send(TaskResult::LoadComplete {
                cv,
                chunk,
                uniform_voxel_id: pc.uniform_voxel_id,
                deltas: persisted_chunk.map(|pc| pc.deltas).take(),
                duration: time.elapsed(),
                generated: false,
            })
            .ok();

            return;
        }

        generator.apply(cv_to_wv(cv), &mut chunk);

        t_sx.send(TaskResult::LoadComplete {
            cv,
            chunk,
            uniform_voxel_id: chunk.is_uniform(),
            deltas: persisted_chunk.map(|pc| pc.deltas).take(),
            duration: time.elapsed(),
            generated: true,
        })
        .ok();
    });
}

fn mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: Box<Chunk>,
    neighbors: Box<[Option<Chunk>; 6]>,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
) {
    let t_sx = t_sx.clone();
    compute_pool.spawn(move || {
        let time = Instant::now();

        let complete_neighbors = Box::new(std::array::from_fn(|i| {
            if let Some(n) = neighbors[i] {
                return n;
            }

            Chunk::new()
        }));
        let mesh = mesh_chunk::<V>(cv, chunk, complete_neighbors);
        t_sx.send(TaskResult::MeshComplete {
            cv,
            mesh,
            duration: time.elapsed(),
        })
        .ok();
    });
}

fn physics_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: Box<Chunk>,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
) {
    let t_sx = t_sx.clone();
    compute_pool.spawn(move || {
        let time = Instant::now();

        let mesh = mesh_physics_chunk::<V>(cv, chunk);
        if !mesh.is_empty() {
            t_sx.send(TaskResult::PhysicsMeshComplete {
                cv,
                mesh,
                duration: time.elapsed(),
            })
            .ok();
        }
    });
}
