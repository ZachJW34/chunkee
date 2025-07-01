use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, Rotation, VoxelId},
    coords::{ChunkVector, WorldVector, wv_to_cv, wv_to_lv},
    generation::VoxelGenerator,
    meshing::ChunkMeshData,
    pipeline::{
        PipelineMessage, PipelineResult, PipelineState, WorldChunks, is_stable,
        spawn_pipeline_thread,
    },
    storage::ChunkStore,
    streaming::CameraData,
};
use crossbeam::queue::SegQueue;
use crossbeam_channel::{Receiver, Sender};
use glam::Vec3;
use std::{marker::PhantomData, sync::Arc, thread::JoinHandle};

pub struct ChunkeeWorldConfig {
    pub radius: u32,
    pub generator: Box<dyn VoxelGenerator>,
}

pub type MeshQueue = SegQueue<(ChunkVector, ChunkMeshData)>;
pub type UnloadQueue = SegQueue<ChunkVector>;

pub struct ChunkeeWorld<V: ChunkeeVoxel> {
    pub mesh_queue: SegQueue<(ChunkVector, ChunkMeshData)>,
    pub unload_queue: SegQueue<ChunkVector>,
    pub radius: u32,
    chunks: Arc<WorldChunks>,
    camera_pos: Vec3,
    chunk_store: Arc<ChunkStore>,
    pipeline: Option<Pipeline>,
    generator: Arc<Box<dyn VoxelGenerator>>,
    _voxel_type: PhantomData<V>,
}

impl<V: 'static + ChunkeeVoxel> ChunkeeWorld<V> {
    pub fn new(config: ChunkeeWorldConfig) -> Self {
        let chunk_store = Arc::new(ChunkStore::new());
        let chunks = Arc::new(WorldChunks::default());
        let generator = Arc::new(config.generator);

        Self {
            chunks,
            chunk_store,
            generator,
            radius: config.radius,
            camera_pos: Vec3::NAN,
            pipeline: None,
            mesh_queue: SegQueue::new(),
            unload_queue: SegQueue::new(),
            _voxel_type: PhantomData,
        }
    }

    pub fn enable_pipeline(&mut self) {
        if self.pipeline.is_none() {
            let (pipeline_sender, pipeline_receiver) = crossbeam_channel::unbounded();
            let (mesh_sender, mesh_receiver) = crossbeam_channel::unbounded();

            let pipeline_handle = spawn_pipeline_thread::<V>(
                self.chunks.clone(),
                self.chunk_store.clone(),
                self.generator.clone(),
                self.radius,
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
                PipelineResult::ChunkUnloaded { cv } => self.unload_queue.push(cv),
            }
        }
    }

    pub fn set_voxel_at(&mut self, voxel: V, wv: WorldVector) -> Option<V> {
        if self.pipeline.is_none() {
            return None;
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let cv = wv_to_cv(wv);
        if let Some(mut world_chunk) = self.chunks.get_mut(&cv)
            && is_stable(world_chunk.state)
        {
            let lv = wv_to_lv(wv);
            // TODO: apply rotation based on cameras normal
            let new_voxel_id = VoxelId::new(voxel.into(), Rotation::default());
            world_chunk.deltas.0.insert(lv, new_voxel_id);
            let old_voxel_id = world_chunk.chunk.set_voxel::<V>(lv, new_voxel_id);

            world_chunk.is_dirty = true;
            world_chunk.state = PipelineState::NeedsMesh;
            pipeline.sender.send(PipelineMessage::ChunkEdit(cv)).ok();

            drop(world_chunk);

            for face in BLOCK_FACES {
                let neighbor_cv = cv + face.into_normal();
                if neighbor_cv != cv
                    && let Some(mut neighbor) = self.chunks.get_mut(&neighbor_cv)
                {
                    neighbor.state = PipelineState::NeedsMesh;
                    pipeline
                        .sender
                        .send(PipelineMessage::ChunkEdit(neighbor_cv))
                        .ok();
                }
            }

            Some(old_voxel_id.type_id().into())
        } else {
            None
        }
    }
}

struct Pipeline {
    pub handle: JoinHandle<()>,
    pub sender: Sender<PipelineMessage>,
    pub receiver: Receiver<PipelineResult>,
}
