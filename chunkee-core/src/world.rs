use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    coords::{ChunkVector, WorldVector, wv_to_cv, wv_to_lv},
    generation::VoxelGenerator,
    grid::ChunkGrid,
    meshing::ChunkMeshGroup,
    pipeline::{PhysicsEntity, PipelineMessage, spawn_pipeline_thread},
    storage::ChunkStore,
    streaming::CameraData,
    traversal::DDAState,
};
use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use crossbeam_channel::Sender;
use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use std::{marker::PhantomData, sync::Arc, thread::JoinHandle};

pub struct ChunkeeWorldConfig {
    pub radius: u32,
    pub generator: Box<dyn VoxelGenerator>,
    pub voxel_size: f32,
}

pub type MeshQueue = SegQueue<(ChunkVector, ChunkMeshGroup)>;
pub type PhysicsMeshQueue = SegQueue<(ChunkVector, Vec<Vec3>)>;
pub type UnloadQueue = SegQueue<ChunkVector>;
pub type EditsAppliedQueue = SegQueue<(IVec3, VoxelId)>;
pub type WorldGrid = Arc<RwLock<ChunkGrid>>;

pub struct ResultQueues {
    pub mesh_load: SegQueue<(ChunkVector, ChunkMeshGroup)>,
    pub mesh_unload: SegQueue<ChunkVector>,
    pub physics_load: SegQueue<(ChunkVector, Vec<Vec3>)>,
    pub physics_unload: SegQueue<ChunkVector>,
    pub edits: SegQueue<(WorldVector, VoxelId)>,
}

pub struct ChunkeeWorld<V: ChunkeeVoxel> {
    pub results: Arc<ResultQueues>,
    pub radius: u32,
    grid: WorldGrid,
    camera_pos: Vec3,
    chunk_store: Arc<ChunkStore>,
    pipeline: Option<Pipeline>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    pub voxel_size: f32,
    _voxel_type: PhantomData<V>,
}

impl<V: 'static + ChunkeeVoxel> ChunkeeWorld<V> {
    pub fn new(config: ChunkeeWorldConfig) -> Self {
        let ChunkeeWorldConfig {
            radius,
            generator,
            voxel_size,
        } = config;

        let chunk_store = Arc::new(ChunkStore::new());
        let grid = Arc::new(RwLock::new(ChunkGrid::new(radius)));
        let generator = Arc::new(generator);
        let results = Arc::new(ResultQueues {
            mesh_load: SegQueue::new(),
            mesh_unload: SegQueue::new(),
            physics_load: SegQueue::new(),
            physics_unload: SegQueue::new(),
            edits: SegQueue::new(),
        });

        Self {
            results,
            grid,
            chunk_store,
            radius,
            generator,
            camera_pos: Vec3::NAN,
            pipeline: None,
            voxel_size,
            _voxel_type: PhantomData,
        }
    }

    pub fn enable_pipeline(&mut self) {
        if self.pipeline.is_none() {
            let (pipeline_sender, pipeline_receiver) = crossbeam_channel::unbounded();
            let pipeline_handle = spawn_pipeline_thread::<V>(
                self.grid.clone(),
                self.chunk_store.clone(),
                self.generator.clone(),
                pipeline_receiver,
                self.results.clone(),
                self.radius,
                self.voxel_size,
            );
            self.pipeline = Some(Pipeline {
                handle: pipeline_handle,
                sender: pipeline_sender,
                // receiver: mesh_receiver,
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
        let pipeline = if let Some(pipeline) = self.pipeline.as_ref() {
            pipeline
        } else {
            return;
        };

        if self.camera_pos != camera_data.pos {
            self.camera_pos = camera_data.pos;
            pipeline
                .sender
                .send(PipelineMessage::CameraDataUpdate(camera_data))
                .ok();
        }

        // for result in pipeline.receiver.try_iter() {
        //     match result {
        //         PipelineResult::MeshReady { cv, mesh } => self.mesh_queue.push((cv, mesh)),
        //         PipelineResult::PhysicsMeshReady { cv, mesh } => {
        //             self.physics_mesh_queue.push((cv, mesh))
        //         }
        //         PipelineResult::PhysicsMeshUnload { cvs } => {
        //             for cv in cvs {
        //                 self.physics_mesh_unload_queue.push(cv);
        //             }
        //         }
        //         PipelineResult::EditsApplied(edits) => {
        //             for edit in edits {
        //                 self.edits_applied_queue.push(edit);
        //             }
        //         }
        //         PipelineResult::ChunkUnloaded { cv: _cv } => {}
        //     }
        // }
    }

    pub fn set_voxels_at(&self, changes: &[(WorldVector, V)]) {
        let pipeline = if let Some(pipeline) = self.pipeline.as_ref() {
            pipeline
        } else {
            return;
        };

        let edits: Vec<(WorldVector, VoxelId)> = changes
            .iter()
            .map(|(wv, voxel)| (*wv, VoxelId::new((*voxel).into(), Rotation::default())))
            .collect();

        if !edits.is_empty() {
            pipeline
                .sender
                .send(PipelineMessage::ChunkEdits(edits))
                .ok();
        }
    }

    pub fn try_raycast(
        &mut self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_steps: u32,
    ) -> VoxelRaycast<V> {
        let mut dda =
            DDAState::from_pos_and_dir((ray_origin / self.voxel_size).into(), ray_direction.into());

        for _ in 0..(max_steps as usize) {
            let pos = dda.next_voxelpos;
            let cv = wv_to_cv(pos);
            let maybe_rw = self
                .grid
                .try_read()
                .and_then(|grid_r| grid_r.get(cv).cloned());
            if let Some(rw) = maybe_rw
                && let Some(world_chunk) = rw.try_read()
                && world_chunk.is_stable()
            {
                let lv = wv_to_lv(pos);
                let voxel_id = world_chunk.chunk.get_voxel(lv);
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

    // pub fn get_voxels_for_aabb(&self, aabb: AABB, padding: i32) -> Vec<(WorldVector, VoxelId)> {
    //     let padded_min = aabb.min - padding;
    //     let padded_max = aabb.max + padding;
    //     let mut voxels = vec![];
    //     let grid_lock = self.grid.read();

    //     for x in padded_min.x..padded_max.x {
    //         for y in padded_min.y..padded_max.y {
    //             for z in padded_min.z..padded_max.z {
    //                 let wv = IVec3::new(x, y, z);
    //                 let cv = wv_to_cv(wv);
    //                 let lv = wv_to_lv(wv);
    //                 if let Some(world_chunk) = grid_lock.get(cv)
    //                     && world_chunk.is_stable()
    //                 {
    //                     voxels.push((wv, world_chunk.chunk.get_voxel(lv)));
    //                 } else {
    //                     voxels.push((wv, VoxelId::AIR));
    //                 }
    //             }
    //         }
    //     }

    //     voxels
    // }

    pub fn update_physics_entities(&self, entities: Vec<PhysicsEntity>) {
        let pipeline = if let Some(pipeline) = self.pipeline.as_ref() {
            pipeline
        } else {
            return;
        };

        pipeline
            .sender
            .send(PipelineMessage::PhysicsEntitiesUpdate(entities))
            .ok();
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
    // pub receiver: Receiver<PipelineResult>,
}
