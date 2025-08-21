use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    coords::{ChunkVector, WorldVector, wv_to_cv, wv_to_lv},
    generation::VoxelGenerator,
    grid::ChunkManager,
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
    chunk_manager: Arc<ChunkManager>,
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
        let chunk_manager = Arc::new(ChunkManager::new(radius));
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
            chunk_manager,
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
                self.chunk_manager.clone(),
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
            if let Some(res) = self.chunk_manager.try_read(cv, |wc| {
                if !wc.is_stable() {
                    return Some(VoxelRaycast::None);
                }

                let lv = wv_to_lv(pos);
                let voxel_id = wc.chunk.get_voxel(lv);
                let voxel = V::from(voxel_id.type_id());
                if voxel.visibilty() != VoxelVisibility::Empty {
                    return Some(VoxelRaycast::Hit((pos, voxel)));
                }

                None
            }) {
                if let Some(vr) = res {
                    return vr;
                } else {
                    dda.step_mut();
                }
            } else {
                return VoxelRaycast::None;
            }
        }

        VoxelRaycast::Miss
    }

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

    pub fn chunk_in_range(&self, cv: ChunkVector) -> bool {
        self.chunk_manager
            .view
            .load()
            .cv_to_idx_with_origin(cv)
            .is_some()
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
}
