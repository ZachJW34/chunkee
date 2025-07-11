use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::Arc,
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender};

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel},
    chunk::{ChunkLOD, Deltas, LOD, neighbors_of},
    coords::{ChunkVector, cv_to_wv},
    generation::VoxelGenerator,
    grid::{ComputeState, IOState},
    meshing::{ChunkMeshGroup, mesh_chunk},
    storage::ChunkStore,
    streaming::{CameraData, calc_lod, calculate_chunk_priority},
    world::WorldGrid,
};

pub enum PipelineMessage {
    ChunkEdit(ChunkVector),
    CameraDataUpdate(CameraData),
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
}

enum TaskResult {
    DeltasLoaded {
        cv: ChunkVector,
        deltas: Option<Deltas>,
    },
    GenerationComplete {
        cv: ChunkVector,
        chunk_lod: ChunkLOD,
    },
    MeshComplete {
        cv: ChunkVector,
        mesh: ChunkMeshGroup,
    },
    // UnloadComplete {
    //     cv: ChunkVector,
    // },
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

pub type WorkQueue = BinaryHeap<WorkItem>;

pub fn spawn_pipeline_thread<V: 'static + ChunkeeVoxel>(
    world_grid: WorldGrid,
    chunk_store: Arc<ChunkStore>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    message_receiver: Receiver<PipelineMessage>,
    result_sender: Sender<PipelineResult>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut camera_data: Option<CameraData> = None;

        let (t_sx, t_rx) = crossbeam_channel::unbounded::<TaskResult>();

        let io_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let compute_pool = rayon::ThreadPoolBuilder::new().build().unwrap();

        loop {
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
                    _ => {}
                }
            }

            if camera_data.is_none() {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let camera_data = camera_data.as_ref().unwrap();

            if camera_moved {
                let mut grid_lock = world_grid.write().unwrap();
                grid_lock.shift_origin_with(camera_data, |cv, world_chunk| {
                    let lod = calc_lod(cv, camera_data.pos);
                    let priority = calculate_chunk_priority(cv, camera_data);

                    if world_chunk.cv != cv {
                        if world_chunk.is_dirty {
                            println!("{cv} unloading: dirty");
                            unload_deltas_task(
                                world_chunk.cv,
                                world_chunk.deltas.clone(),
                                &chunk_store,
                                &io_pool,
                            )
                        }

                        world_chunk.cv = cv;
                        world_chunk.priority = priority;
                        world_chunk.is_dirty = false;
                        world_chunk.io_state = IOState::DeltasNeeded;
                        world_chunk.compute_state = ComputeState::GenerationNeeded;
                        world_chunk.chunk_lod = ChunkLOD::new(lod);
                        world_chunk.deltas = Deltas::default();
                    } else {
                        world_chunk.priority = priority;
                        if world_chunk.chunk_lod.lod_level() != lod {
                            world_chunk.chunk_lod = ChunkLOD::new(lod);
                            world_chunk.compute_state = ComputeState::GenerationNeeded;
                        }
                    }
                });
            }

            let results: Vec<_> = t_rx.try_iter().collect();

            let (mut io_tasks, mut generate_tasks, mut mesh_tasks) = {
                let grid_lock = world_grid.read().unwrap();
                let mut io_tasks = Vec::new();
                let mut gen_tasks = Vec::new();
                let mut mesh_tasks = Vec::new();
                for world_chunk in grid_lock.chunks.iter() {
                    match world_chunk.io_state {
                        IOState::DeltasNeeded => {
                            io_tasks.push(WorkItem {
                                priority: world_chunk.priority,
                                cv: world_chunk.cv,
                            });
                        }
                        _ => {}
                    };

                    match world_chunk.compute_state {
                        ComputeState::GenerationNeeded => gen_tasks.push(WorkItem {
                            priority: world_chunk.priority,
                            cv: world_chunk.cv,
                        }),
                        ComputeState::MeshNeeded => mesh_tasks.push(WorkItem {
                            priority: world_chunk.priority,
                            cv: world_chunk.cv,
                        }),
                        _ => {}
                    }
                }
                (io_tasks, gen_tasks, mesh_tasks)
            };

            io_tasks.sort();
            generate_tasks.sort();
            mesh_tasks.sort();

            let work_to_do = !results.is_empty()
                || !io_tasks.is_empty()
                || !generate_tasks.is_empty()
                || !mesh_tasks.is_empty();

            if work_to_do {
                let mut grid_lock = world_grid.write().unwrap();

                for result in results {
                    match result {
                        TaskResult::DeltasLoaded { cv, deltas } => {
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.io_state == IOState::DeltasLoading
                            {
                                world_chunk.io_state = IOState::DeltasLoaded;
                                if let Some(deltas) = deltas {
                                    println!(
                                        "{cv} loaded deltas: remeshing. compute_state={:?}",
                                        world_chunk.compute_state
                                    );
                                    world_chunk.deltas = deltas;
                                    if world_chunk.is_stable() {
                                        world_chunk.merge_deltas::<V>();
                                        world_chunk.compute_state = ComputeState::MeshNeeded;
                                    }
                                }
                            }
                        }
                        TaskResult::GenerationComplete { cv, chunk_lod } => {
                            let mut should_notify_neighbors = false;
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.compute_state == ComputeState::Generating
                            {
                                should_notify_neighbors = true;
                                world_chunk.chunk_lod = chunk_lod;
                                world_chunk.merge_deltas::<V>();
                                world_chunk.compute_state = ComputeState::MeshNeeded;
                            }

                            if should_notify_neighbors {
                                for neighbor_cv in neighbors_of(cv) {
                                    if let Some(neighbor) = grid_lock.get_mut(neighbor_cv)
                                        && matches!(
                                            neighbor.compute_state,
                                            ComputeState::MeshNeeded
                                                | ComputeState::Meshing
                                                | ComputeState::MeshReady
                                        )
                                    {
                                        neighbor.compute_state = ComputeState::MeshNeeded;
                                    }
                                }
                            }
                        }
                        TaskResult::MeshComplete { cv, mesh } => {
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.compute_state == ComputeState::Meshing
                            {
                                world_chunk.compute_state = ComputeState::MeshReady;
                                result_sender
                                    .send(PipelineResult::MeshReady { cv, mesh })
                                    .ok();
                            }
                        } // TaskResult::UnloadComplete { cv: _ } => {}
                    }
                }

                let drain_limit = 100;
                for _ in 0..drain_limit {
                    if io_tasks.is_empty() && generate_tasks.is_empty() && mesh_tasks.is_empty() {
                        break;
                    }

                    if let Some(task) = io_tasks.pop() {
                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.io_state = IOState::DeltasLoading;
                            load_deltas_task(task.cv, &chunk_store, &t_sx, &io_pool);
                        }
                    }

                    if let Some(task) = generate_tasks.pop() {
                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.compute_state = ComputeState::Generating;
                            let lod = world_chunk.chunk_lod.lod_level();
                            generate_task(task.cv, &generator, &t_sx, &compute_pool, lod);
                        }
                    }

                    while let Some(task) = mesh_tasks.pop() {
                        let mut neighbors: Box<[Option<ChunkLOD>; 6]> =
                            Box::new(Default::default());
                        let mut all_neighbors_ready = true;
                        for (i, face) in BLOCK_FACES.iter().enumerate() {
                            let neighbor_cv = task.cv + face.into_normal();
                            if let Some(neighbor) = grid_lock.get(neighbor_cv) {
                                if neighbor.is_stable() {
                                    neighbors[i] = Some(neighbor.chunk_lod.clone());
                                } else {
                                    all_neighbors_ready = false;
                                    break;
                                }
                            } else {
                                neighbors[i] = None;
                            }
                        }

                        if !all_neighbors_ready {
                            continue;
                        }

                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.compute_state = ComputeState::Meshing;
                            let boxed_chunk_lod = Box::new(world_chunk.chunk_lod.clone());
                            mesh_task::<V>(
                                task.cv,
                                boxed_chunk_lod,
                                neighbors,
                                camera_data,
                                &t_sx,
                                &compute_pool,
                            );
                        }
                    }
                }
            }

            if !work_to_do && t_rx.is_empty() {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    })
}

fn load_deltas_task(
    cv: ChunkVector,
    chunk_store: &Arc<ChunkStore>,
    t_sx: &Sender<TaskResult>,
    io_pool: &rayon::ThreadPool,
) {
    let t_sx: Sender<TaskResult> = t_sx.clone();
    let chunk_store = chunk_store.clone();
    io_pool.spawn(move || {
        let deltas = chunk_store.load_delta(cv);
        t_sx.send(TaskResult::DeltasLoaded { cv, deltas }).ok();
    });
}

fn unload_deltas_task(
    cv: ChunkVector,
    deltas: Deltas,
    chunk_store: &Arc<ChunkStore>,
    // t_sx: &Sender<TaskResult>,
    io_pool: &rayon::ThreadPool,
) {
    let chunk_store = chunk_store.clone();

    // let t_sx: Sender<TaskResult> = t_sx.clone();
    io_pool.spawn(move || {
        chunk_store.save_delta(cv, &deltas);
        // t_sx.send(TaskResult::UnloadComplete { cv }).ok();
    });
}

fn generate_task(
    cv: ChunkVector,
    generator: &Arc<Box<dyn VoxelGenerator>>,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
    lod: LOD,
) {
    let generator = generator.clone();
    let t_sx = t_sx.clone();
    compute_pool.spawn(move || {
        let mut chunk = ChunkLOD::new(lod);
        generator.apply(cv_to_wv(cv), &mut chunk);
        t_sx.send(TaskResult::GenerationComplete {
            cv,
            chunk_lod: chunk,
        })
        .ok();
    });
}

fn mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: Box<ChunkLOD>,
    neighbors: Box<[Option<ChunkLOD>; 6]>,
    camera_data: &CameraData,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
) {
    let t_sx = t_sx.clone();
    let camera_pos = camera_data.pos;
    compute_pool.spawn(move || {
        let complete_neighbors = Box::new(std::array::from_fn(|i| {
            if let Some(n) = neighbors[i] {
                return n;
            }

            let neighbor_cv = cv + BLOCK_FACES[i].into_normal();
            let lod = calc_lod(neighbor_cv, camera_pos);
            ChunkLOD::new(lod)
        }));
        let mesh = mesh_chunk::<V>(cv, chunk, complete_neighbors);
        t_sx.send(TaskResult::MeshComplete { cv, mesh }).ok();
    });
}

// fn debug_cv(cv: ChunkVector, stage: &str) {
//     let cvs_to_watch = [IVec3::new(0, 0, 0)];
//     if cvs_to_watch.contains(&cv) {
//         println!("{cv}: {stage}");
//     }
// }
