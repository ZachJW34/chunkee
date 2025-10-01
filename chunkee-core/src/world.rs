use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId},
    coords::{ChunkVector, WorldVector, vec3_wv_to_cv, wv_to_cv, wv_to_lv},
    define_metrics,
    generation::VoxelGenerator,
    grid::ChunkManager,
    meshing::ChunkMeshGroup,
    metrics::{HistogramMetrics, MetricsRegistry, ThroughputMetrics},
    pipeline::{PhysicsEntity, WorkerPool, WorkerSharedState},
    storage::ChunkStore,
    streaming::{CameraData, calc_total_chunks},
    traversal::DDAState,
};
use block_mesh::VoxelVisibility;
use crossbeam::queue::SegQueue;
use glam::{IVec3, Vec3};
use once_cell::sync::Lazy;
use std::{marker::PhantomData, sync::Arc, time::Instant};

define_metrics! {
    pub enum Histograms {
        ReprioritizeQueues => "ChunkeeCore::RepriortizeQueues",
        GridUpdate => "ChunkeeCore::GridUpdate",
        Load => "ChunkeeCore::Load",
        Generate => "ChunkeeCore::Generate",
        LoadTask => "ChunkeeCore::LoadTask",
        MeshTask => "ChunkeeCore::MeshTask",
        PhysicsTask => "ChunkeeCore::PhysicsTask",
        TryRaycast => "ChunkeeCore::TryRaycast",
    }
}

define_metrics! {
    pub enum Throughputs {
        Tasks => "ChunkeeCore::Task"
    }
}

pub struct ChunkeeCoreMetrics {
    pub histograms: MetricsRegistry<Histograms, HistogramMetrics>,
    pub throughputs: MetricsRegistry<Throughputs, ThroughputMetrics>,
}

pub static CHUNKEE_CORE_METRICS: Lazy<ChunkeeCoreMetrics> = Lazy::new(|| ChunkeeCoreMetrics {
    histograms: MetricsRegistry::new(),
    throughputs: MetricsRegistry::new(),
});

const CAMERA_THRESHOLD_RADIANS: f32 = 1.0;

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
    // pub mesh_unload: SegQueue<ChunkVector>,
    pub physics_load: SegQueue<(ChunkVector, Vec<Vec3>)>,
    pub physics_unload: SegQueue<ChunkVector>,
    pub edits: SegQueue<(WorldVector, VoxelId)>,
}

pub struct ChunkeeWorld<V: ChunkeeVoxel> {
    pub results: Arc<ResultQueues>,
    pub radius: u32,
    chunk_manager: Arc<ChunkManager>,
    camera_data: Option<CameraData>,
    chunk_store: Arc<ChunkStore>,
    worker_state: Arc<WorkerSharedState>,
    worker_pool: Option<WorkerPool>,
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
            // mesh_unload: SegQueue::new(),
            physics_load: SegQueue::new(),
            physics_unload: SegQueue::new(),
            edits: SegQueue::new(),
        });
        let total_tasks = calc_total_chunks(radius) as usize;
        let worker_state = Arc::new(WorkerSharedState::new(total_tasks));

        Self {
            results,
            chunk_manager,
            chunk_store,
            radius,
            generator,
            camera_data: None,
            worker_state,
            worker_pool: None,
            voxel_size,
            _voxel_type: PhantomData,
        }
    }

    pub fn enable_pipeline(&mut self) {
        if self.worker_pool.is_none() {
            let worker_pool = WorkerPool::new::<V>(
                4,
                self.worker_state.clone(),
                self.chunk_manager.clone(),
                self.chunk_store.clone(),
                self.generator.clone(),
                self.results.clone(),
                self.voxel_size,
            );

            self.worker_pool = Some(worker_pool)
        }
    }

    pub fn disable_pipeline(&mut self) {
        if let Some(_worker_pool) = self.worker_pool.take() {};
    }

    pub fn update(&mut self, camera_data: CameraData) {
        if let Some(worker_pool) = self.worker_pool.as_ref() {
            // metrics
            if !self.worker_state.load.is_empty()
                || !self.worker_state.mesh.is_empty()
                || !self.worker_state.physics.is_empty()
            {
                CHUNKEE_CORE_METRICS
                    .throughputs
                    .get(Throughputs::Tasks)
                    .start();
            } else {
                CHUNKEE_CORE_METRICS
                    .throughputs
                    .get(Throughputs::Tasks)
                    .end();
            }
            if self.camera_data.is_none_or(|previous| {
                let current_cv = vec3_wv_to_cv(camera_data.pos, self.voxel_size);
                let previous_cv = vec3_wv_to_cv(previous.pos, self.voxel_size);

                (current_cv != previous_cv)
                    || (camera_data.forward.angle_between(previous.forward)
                        > CAMERA_THRESHOLD_RADIANS)
            }) {
                self.worker_state.camera_update.push(camera_data);
                worker_pool.sender.send(()).ok();
                self.camera_data = Some(camera_data);
            }
        }
    }

    pub fn set_voxels_at(&self, changes: &[(WorldVector, V)]) {
        if let Some(worker_pool) = self.worker_pool.as_ref() {
            let edits: Vec<(WorldVector, VoxelId)> = changes
                .iter()
                .map(|(wv, voxel)| (*wv, VoxelId::new((*voxel).into(), Rotation::default())))
                .collect();

            if !edits.is_empty() {
                self.worker_state.chunk_edits.push(edits);
                worker_pool.sender.send(()).ok();
            }
        }
    }

    pub fn try_raycast(
        &mut self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_steps: u32,
    ) -> VoxelRaycast<V> {
        let time = Instant::now();
        let mut dda =
            DDAState::from_pos_and_dir((ray_origin / self.voxel_size).into(), ray_direction.into());

        let result = {
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
        };

        CHUNKEE_CORE_METRICS
            .histograms
            .get(Histograms::TryRaycast)
            .record(time.elapsed());
        result
    }

    pub fn update_physics_entities(&self, entities: Vec<PhysicsEntity>) {
        if let Some(worker_pool) = self.worker_pool.as_ref() {
            self.worker_state.physics_entities.push(entities);
            worker_pool.sender.send(()).ok();
        }
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
