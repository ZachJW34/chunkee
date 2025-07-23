use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    coords::{AABB, ChunkVector, WorldVector, wv_to_cv, wv_to_lv},
    generation::VoxelGenerator,
    grid::ChunkGrid,
    meshing::ChunkMeshGroup,
    pipeline::{PhysicsEntity, PipelineMessage, PipelineResult, spawn_pipeline_thread},
    storage::ChunkStore,
    streaming::CameraData,
    traversal::DDAState,
};
use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, Vec3};
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
pub type PhysicsMeshQueue = SegQueue<(ChunkVector, Vec<Vec3>)>;
pub type UnloadQueue = SegQueue<ChunkVector>;
pub type EditsAppliedQueue = SegQueue<(IVec3, VoxelId)>;
pub type WorldGrid = Arc<RwLock<ChunkGrid>>;

pub struct ChunkeeWorld<V: ChunkeeVoxel> {
    pub mesh_queue: MeshQueue,
    pub physics_mesh_queue: PhysicsMeshQueue,
    pub physics_mesh_unload_queue: UnloadQueue,
    pub edits_applied_queue: EditsAppliedQueue,
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
            physics_mesh_queue: SegQueue::new(),
            physics_mesh_unload_queue: SegQueue::new(),
            edits_applied_queue: SegQueue::new(),
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
                PipelineResult::PhysicsMeshReady { cv, mesh } => {
                    self.physics_mesh_queue.push((cv, mesh))
                }
                PipelineResult::PhysicsMeshUnload { cvs } => {
                    for cv in cvs {
                        self.physics_mesh_unload_queue.push(cv);
                    }
                }
                PipelineResult::EditsApplied(edits) => {
                    for edit in edits {
                        self.edits_applied_queue.push(edit);
                    }
                }
                PipelineResult::ChunkUnloaded { cv: _cv } => {}
            }
        }
    }

    pub fn set_voxels_at(&self, changes: &[(WorldVector, V)]) {
        if self.pipeline.is_none() {
            return;
        }

        let edits: Vec<(WorldVector, VoxelId)> = changes
            .iter()
            .map(|(wv, voxel)| (*wv, VoxelId::new((*voxel).into(), Rotation::default())))
            .collect();

        if !edits.is_empty() {
            self.pipeline
                .as_ref()
                .unwrap()
                .sender
                .send(PipelineMessage::ChunkEdits(edits))
                .ok();
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

    pub fn get_voxels_for_aabb(&self, aabb: AABB, padding: i32) -> Vec<(WorldVector, VoxelId)> {
        let padded_min = aabb.min - padding;
        let padded_max = aabb.max + padding;

        let mut voxels = vec![];
        let grid_lock = self.grid.read().unwrap();
        for x in padded_min.x..padded_max.x {
            for y in padded_min.y..padded_max.y {
                for z in padded_min.z..padded_max.z {
                    let wv = IVec3::new(x, y, z);
                    let cv = wv_to_cv(wv);
                    let lv = wv_to_lv(wv);
                    if let Some(world_chunk) = grid_lock.get(cv)
                        && world_chunk.chunk_lod.lod_level() == 1
                    {
                        voxels.push((wv, world_chunk.chunk_lod.get_voxel(lv)));
                    } else {
                        voxels.push((wv, VoxelId::AIR));
                    }
                }
            }
        }

        voxels
    }

    pub fn update_physics_entities(&self, entities: Vec<PhysicsEntity>) {
        if self.pipeline.is_none() {
            return;
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        pipeline
            .sender
            .send(PipelineMessage::PhysicsEntitiesUpdate(entities))
            .ok();
    }

    // pub fn get_collision_mesh_for_aabb(&self, aabb: AABB, padding: i32) -> Vec<[Vec3; 3]> {
    //     let voxels = self.get_voxels_for_aabb(aabb, padding);
    //     let triangles = gener
    //     todo!();
    // }
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
