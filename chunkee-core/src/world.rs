use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    chunk::neighbors_of,
    coords::{ChunkVector, WorldVector, wv_to_cv, wv_to_lv},
    generation::VoxelGenerator,
    grid::{ChunkGrid, ComputeState},
    meshing::ChunkMeshGroup,
    pipeline::{PipelineMessage, PipelineResult, spawn_pipeline_thread},
    storage::ChunkStore,
    streaming::CameraData,
    traversal::DDAState,
};
use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use glam::Vec3;
use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
    thread::JoinHandle,
};

pub struct ChunkeeWorldConfig {
    pub radius_xz: u32,
    pub radius_y: u32,
    pub generator: Box<dyn VoxelGenerator>,
}

pub type MeshQueue = SegQueue<(ChunkVector, ChunkMeshGroup)>;
pub type UnloadQueue = SegQueue<ChunkVector>;
pub type WorldGrid = Arc<RwLock<ChunkGrid>>;

pub struct ChunkeeWorld<V: ChunkeeVoxel> {
    pub mesh_queue: MeshQueue,
    pub radius_xz: u32,
    pub radius_y: u32,
    grid: WorldGrid,
    camera_pos: Vec3,
    chunk_store: Arc<ChunkStore>,
    pipeline: Option<Pipeline>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    _voxel_type: PhantomData<V>,
}

impl<V: 'static + ChunkeeVoxel> ChunkeeWorld<V> {
    pub fn new(config: ChunkeeWorldConfig) -> Self {
        let ChunkeeWorldConfig {
            radius_xz,
            radius_y,
            generator,
        } = config;
        let chunk_store = Arc::new(ChunkStore::new());
        let grid = Arc::new(RwLock::new(ChunkGrid::new(radius_xz, radius_y, radius_xz)));
        let generator = Arc::new(generator);

        Self {
            grid,
            chunk_store,
            radius_xz,
            radius_y,
            generator,
            camera_pos: Vec3::NAN,
            pipeline: None,
            mesh_queue: SegQueue::new(),
            _voxel_type: PhantomData,
        }
    }

    pub fn enable_pipeline(&mut self) {
        if self.pipeline.is_none() {
            let (pipeline_sender, pipeline_receiver) = crossbeam_channel::unbounded();
            let (mesh_sender, mesh_receiver) = crossbeam_channel::unbounded();

            let pipeline_handle = spawn_pipeline_thread::<V>(
                self.grid.clone(),
                self.chunk_store.clone(),
                self.generator.clone(),
                pipeline_receiver,
                mesh_sender,
            );

            self.pipeline = Some(Pipeline {
                handle: pipeline_handle,
                sender: pipeline_sender,
                receiver: mesh_receiver,
            })
        }
    }

    pub fn disable_pipeline(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            pipeline.sender.send(PipelineMessage::Shutdown).ok();
            pipeline.handle.join().ok();
        };
    }

    pub fn update(&mut self, camera_data: CameraData) {
        if self.pipeline.is_none() {
            return;
        }

        let pipeline = self.pipeline.as_ref().unwrap();

        if self.camera_pos != camera_data.pos {
            self.camera_pos = camera_data.pos;
            pipeline
                .sender
                .send(PipelineMessage::CameraDataUpdate(camera_data))
                .ok();
        }

        for result in pipeline.receiver.try_iter() {
            match result {
                PipelineResult::MeshReady { cv, mesh } => self.mesh_queue.push((cv, mesh)),
                PipelineResult::ChunkUnloaded { cv: _cv } => {}
            }
        }
    }

    pub fn set_voxel_at(&mut self, voxel: V, wv: WorldVector) -> Option<V> {
        if self.pipeline.is_none() {
            return None;
        }

        let cv = wv_to_cv(wv);
        let mut grid_lock = self.grid.write().unwrap();
        if let Some(world_chunk) = grid_lock.get_mut(cv)
            && world_chunk.is_stable()
            // Does it make sense to allow editing of chunks that are not max resolution??
            && world_chunk.chunk_lod.lod_level() == 1
        {
            let lv = wv_to_lv(wv);

            // TODO: apply rotation based on cameras normal
            let new_voxel_id = VoxelId::new(voxel.into(), Rotation::default());
            world_chunk.deltas.0.insert(lv, new_voxel_id);
            let old_voxel_id = world_chunk.chunk_lod.set_voxel::<V>(lv, new_voxel_id);
            world_chunk.is_dirty = true;
            world_chunk.compute_state = ComputeState::MeshNeeded;
            world_chunk.priority = 0;

            if world_chunk.chunk_lod.is_voxel_on_edge(lv) {
                for neighbor_cv in neighbors_of(cv) {
                    if let Some(neighbor) = grid_lock.get_mut(neighbor_cv) {
                        neighbor.compute_state = ComputeState::MeshNeeded;
                        neighbor.priority = 0;
                    }
                }
            }

            Some(old_voxel_id.type_id().into())
        } else {
            None
        }
    }

    pub fn raycast_hit(
        &mut self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_steps: u32,
    ) -> VoxelRaycast<V> {
        let mut dda = DDAState::from_pos_and_dir(ray_origin.into(), ray_direction.into());

        for _ in 0..(max_steps as usize) {
            let pos = dda.next_voxelpos;
            let cv = wv_to_cv(pos);
            if let Ok(lock) = self.grid.try_read()
                && let Some(world_chunk) = lock.get(cv)
                && world_chunk.is_stable()
                && world_chunk.chunk_lod.lod_level() == 1
            {
                let lv = wv_to_lv(pos);
                let voxel_id = world_chunk.chunk_lod.get_voxel(lv);
                let voxel = V::from(voxel_id.type_id());
                if voxel.visibilty() != VoxelVisibility::Empty {
                    return VoxelRaycast::Hit((pos, voxel));
                }

                dda.step_mut();
            } else {
                return VoxelRaycast::None;
            }
        }

        VoxelRaycast::Miss
    }
}

pub enum VoxelRaycast<V: ChunkeeVoxel> {
    Hit((WorldVector, V)),
    Miss,
    None,
}

struct Pipeline {
    pub handle: JoinHandle<()>,
    pub sender: Sender<PipelineMessage>,
    pub receiver: Receiver<PipelineResult>,
}
