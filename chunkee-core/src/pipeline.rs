use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    vec,
};

use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel},
    chunk::{Chunk, Deltas, neighbors_of},
    coords::{ChunkVector, cv_to_wv},
    generation::VoxelGenerator,
    meshing::{ChunkMeshData, mesh_chunk},
    storage::ChunkStore,
    streaming::{CameraData, ChunkStreamer, calculate_chunk_priority},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    NeedsLoad,
    LoadingDeltas,
    NeedsGeneration,
    Generating,
    NeedsMesh,
    Meshing,
    MeshReady,
    NeedsUnload,
    Unloading,
}

pub struct WorldChunk {
    pub state: PipelineState,
    pub chunk: Chunk,
    pub deltas: Deltas,
    // pub neighbors_mask: NeighborsMask,
    pub is_dirty: bool,
}

impl Default for WorldChunk {
    fn default() -> Self {
        Self {
            state: PipelineState::NeedsLoad,
            is_dirty: false,
            chunk: Chunk::default(),
            deltas: Deltas::default(),
            // neighbors_mask: 0,
        }
    }
}

impl WorldChunk {
    // pub fn assign_neighbor(&mut self, face: &BlockFace) {
    //     self.neighbors_mask |= 1 << face.opposite() as u8;
    // }

    // pub fn neighbors_full(&self) -> bool {
    //     self.neighbors_mask.count_ones() == 6
    // }

    pub fn merge_deltas<V: ChunkeeVoxel>(&mut self) {
        for (lv, voxel_id) in self.deltas.0.iter() {
            self.chunk.set_voxel::<V>(*lv, *voxel_id);
        }
    }
}

pub type WorldChunks = DashMap<ChunkVector, WorldChunk>;

pub enum PipelineMessage {
    ChunkEdit(ChunkVector),
    CameraDataUpdate(CameraData),
    Shutdown,
}

pub enum PipelineResult {
    MeshReady {
        cv: ChunkVector,
        mesh: ChunkMeshData,
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
        chunk: Chunk,
    },
    MeshComplete {
        cv: ChunkVector,
        mesh: ChunkMeshData,
    },
    UnloadComplete {
        cv: ChunkVector,
    },
}

#[derive(Debug)]
pub struct PrioritizedWorkItem {
    pub priority: u32,
    pub cv: ChunkVector,
}

impl PartialEq for PrioritizedWorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for PrioritizedWorkItem {}
impl PartialOrd for PrioritizedWorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PrioritizedWorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

pub type WorkQueue = Mutex<BinaryHeap<PrioritizedWorkItem>>;

pub fn spawn_pipeline_thread<V: 'static + ChunkeeVoxel>(
    world_chunks: Arc<WorldChunks>,
    chunk_store: Arc<ChunkStore>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    radius: u32,
    message_receiver: Receiver<PipelineMessage>,
    result_sender: Sender<PipelineResult>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut streamer = ChunkStreamer::new();
        let mut camera_data: Option<CameraData> = None;

        let (t_sx, t_rx) = crossbeam_channel::unbounded::<TaskResult>();

        let load_queue: Arc<WorkQueue> = Arc::new(Mutex::new(BinaryHeap::new()));
        let load_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let generate_queue: Arc<WorkQueue> = Arc::new(Mutex::new(BinaryHeap::new()));
        let mesh_queue: Arc<WorkQueue> = Arc::new(Mutex::new(BinaryHeap::new()));
        let unload_queue: Arc<WorkQueue> = Arc::new(Mutex::new(BinaryHeap::new()));

        loop {
            for message in message_receiver.try_iter() {
                match message {
                    PipelineMessage::Shutdown => return,
                    PipelineMessage::ChunkEdit(cv) => {
                        let mut mesh_queue = mesh_queue.lock().unwrap();
                        mesh_queue.push(PrioritizedWorkItem { priority: 0, cv });
                    }
                    PipelineMessage::CameraDataUpdate(cd) => {
                        camera_data.replace(cd);
                    }
                }
            }

            if camera_data.is_none() {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let camera_data = camera_data.as_ref().unwrap();

            streamer.preprocess_chunks(
                camera_data,
                radius,
                &world_chunks,
                &load_queue,
                &unload_queue,
            );

            for result in t_rx.try_iter() {
                match result {
                    TaskResult::DeltasLoaded { cv, deltas } => {
                        if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                            && world_chunk.state == PipelineState::LoadingDeltas
                        {
                            if let Some(deltas) = deltas {
                                world_chunk.deltas = deltas;
                            }
                            world_chunk.state = PipelineState::NeedsGeneration;
                            let mut generate_queue = generate_queue.lock().unwrap();
                            generate_queue.push(PrioritizedWorkItem {
                                priority: calculate_chunk_priority(cv, camera_data),
                                cv,
                            });
                        }
                    }
                    TaskResult::GenerationComplete { cv, chunk } => {
                        let mut should_notify_neighbors = false;
                        if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                            && world_chunk.state == PipelineState::Generating
                        {
                            should_notify_neighbors = true;
                            world_chunk.chunk = chunk;
                            world_chunk.merge_deltas::<V>();

                            world_chunk.state = PipelineState::NeedsMesh;
                            let mut mesh_queue = mesh_queue.lock().unwrap();
                            mesh_queue.push(PrioritizedWorkItem {
                                priority: calculate_chunk_priority(cv, camera_data),
                                cv,
                            });
                        }

                        if should_notify_neighbors {
                            let mut neighbors = vec![];
                            for neighbor_cv in neighbors_of(cv) {
                                if let Some(mut neighbor) = world_chunks.get_mut(&neighbor_cv)
                                    && matches!(
                                        neighbor.state,
                                        PipelineState::Meshing | PipelineState::MeshReady
                                    )
                                {
                                    neighbor.state = PipelineState::NeedsMesh;
                                    neighbors.push(neighbor_cv);
                                }
                            }
                            for neighbor_cv in neighbors {
                                let mut mesh_queue = mesh_queue.lock().unwrap();
                                mesh_queue.push(PrioritizedWorkItem {
                                    priority: calculate_chunk_priority(neighbor_cv, camera_data),
                                    cv: neighbor_cv,
                                });
                            }
                        }
                    }
                    TaskResult::MeshComplete { cv, mesh } => {
                        if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                            && matches!(world_chunk.state, PipelineState::Meshing)
                        {
                            world_chunk.state = match world_chunk.state {
                                PipelineState::Meshing => PipelineState::MeshReady,
                                _ => unreachable!(),
                            };

                            result_sender
                                .send(PipelineResult::MeshReady { cv, mesh })
                                .ok();
                        }
                    }
                    TaskResult::UnloadComplete { cv } => {
                        if world_chunks
                            .remove_if(&cv, |_, chunk| chunk.state == PipelineState::Unloading)
                            .is_some()
                        {
                            result_sender
                                .send(PipelineResult::ChunkUnloaded { cv })
                                .ok();
                        }
                    }
                }
            }

            // Consider only draining some "x" amount to allow culling of no longer needed io operations.
            // There is already an early bail out, but it reduce the amount of time spent spawning threads with stale data
            while let Some(load_item) = load_queue.lock().unwrap().pop() {
                let cv = load_item.cv;
                if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                    && world_chunk.state == PipelineState::NeedsLoad
                {
                    world_chunk.state = PipelineState::LoadingDeltas;

                    let t_sx = t_sx.clone();
                    let chunk_store = chunk_store.clone();
                    load_pool.spawn(move || {
                        let deltas = chunk_store.load_delta(cv);
                        t_sx.send(TaskResult::DeltasLoaded { cv, deltas }).ok();
                    });
                }
            }

            while let Some(generate_item) = generate_queue.lock().unwrap().pop() {
                let cv = generate_item.cv;
                if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                    && world_chunk.state == PipelineState::NeedsGeneration
                {
                    world_chunk.state = PipelineState::Generating;

                    let t_sx = t_sx.clone();
                    let generator = generator.clone();
                    rayon::spawn(move || {
                        let mut chunk = Chunk::default();
                        generator.apply(cv_to_wv(cv), &mut chunk);
                        t_sx.send(TaskResult::GenerationComplete { cv, chunk }).ok();
                    });
                }
            }

            while let Some(mesh_item) = mesh_queue.lock().unwrap().pop() {
                let cv = mesh_item.cv;
                let chunk = if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                    && world_chunk.state == PipelineState::NeedsMesh
                {
                    world_chunk.state = PipelineState::Meshing;

                    world_chunk.chunk.clone()
                } else {
                    continue;
                };

                if chunk.is_empty() {
                    let mesh = ChunkMeshData::default();
                    t_sx.send(TaskResult::MeshComplete { cv, mesh: mesh }).ok();
                    continue;
                };

                let neighbors = realize_neighbors(cv, &world_chunks);

                if neighbors.iter().all(|n| n.is_some_and(|n| n.is_solid())) {
                    let mesh = ChunkMeshData::default();
                    t_sx.send(TaskResult::MeshComplete { cv, mesh: mesh }).ok();
                    continue;
                }

                let t_sx = t_sx.clone();
                let chunk = Box::new(chunk);
                rayon::spawn(move || {
                    let mesh = mesh_chunk::<V>(cv, &chunk, &neighbors);
                    t_sx.send(TaskResult::MeshComplete { cv, mesh }).ok();
                });
            }

            while let Some(unload_item) = unload_queue.lock().unwrap().pop() {
                let cv = unload_item.cv;
                if let Some(mut world_chunk) = world_chunks.get_mut(&cv)
                    && world_chunk.state == PipelineState::NeedsUnload
                {
                    world_chunk.state = PipelineState::Unloading;
                    let t_sx = t_sx.clone();

                    if world_chunk.is_dirty {
                        let chunk_store = chunk_store.clone();
                        let deltas = world_chunk.deltas.clone();

                        rayon::spawn(move || {
                            chunk_store.save_delta(cv, &deltas);
                            t_sx.send(TaskResult::UnloadComplete { cv }).ok();
                        });
                    } else {
                        t_sx.send(TaskResult::UnloadComplete { cv }).ok();
                    }
                }
            }

            thread::sleep(std::time::Duration::from_millis(10));
        }
    })
}

fn realize_neighbors(cv: ChunkVector, world_chunks: &WorldChunks) -> Box<[Option<Chunk>; 6]> {
    let mut neighbors: Box<[Option<Chunk>; 6]> = Box::new(Default::default());

    for (i, face) in BLOCK_FACES.iter().enumerate() {
        if let Some(chunk_state) = world_chunks.get(&(cv + face.into_normal())) {
            neighbors[i] = Some(chunk_state.chunk.clone());
        }
    }

    neighbors
}

pub fn is_stable(state: PipelineState) -> bool {
    matches!(
        state,
        PipelineState::NeedsMesh | PipelineState::Meshing | PipelineState::MeshReady
    )
}
