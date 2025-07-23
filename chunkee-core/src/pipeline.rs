use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    sync::Arc,
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender};
use glam::{IVec3, Vec3};

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{Chunk, ChunkLOD, LOD},
    coords::{
        AABB, ChunkVector, NEIGHBOR_OFFSETS, camera_vec3_to_cv, cv_to_wv, wv_to_cv, wv_to_lv,
    },
    generation::VoxelGenerator,
    grid::{Deltas, GridOp, PipelineState, WorldChunk, neighbors_of},
    hasher::VoxelHashMap,
    meshing::{ChunkMeshGroup, mesh_chunk, mesh_physics_chunk},
    storage::{BatchedPersistedChunkMap, ChunkStore, PersistedChunk},
    streaming::{CameraData, calc_lod, calculate_chunk_priority},
    world::WorldGrid,
};

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
    PersistedChunksLoaded {
        persisted_chunks: BatchedPersistedChunkMap,
    },
    GenerationComplete {
        cv: ChunkVector,
        chunk_lod: ChunkLOD,
        uniform_voxel_id: Option<VoxelId>,
    },
    MeshComplete {
        cv: ChunkVector,
        mesh: ChunkMeshGroup,
    },
    PhysicsMeshComplete {
        cv: ChunkVector,
        mesh: Vec<Vec3>,
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

        let io_pool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let compute_pool = rayon::ThreadPoolBuilder::new().build().unwrap();

        let mut previous_physics_entities: Vec<PhysicsEntity> = vec![];
        let mut physics_meshes: VoxelHashMap<PhysicsMesh> = Default::default();

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
                    PipelineMessage::PhysicsEntitiesUpdate(new_entities) => {
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
                                && world_chunk.chunk_lod.lod_level() == 1
                            {
                                let lv = wv_to_lv(wv);
                                world_chunk.deltas.0.insert(lv, voxel_id);
                                let old_voxel_id =
                                    world_chunk.chunk_lod.set_voxel::<V>(lv, voxel_id);
                                world_chunk.is_dirty = true;
                                world_chunk.state = PipelineState::MeshNeeded;
                                world_chunk.priority = 0; // High priority

                                if let Some(p_mesh) = physics_meshes.get_mut(&cv)
                                    && p_mesh.state != PhysicsMeshState::MeshNeeded
                                {
                                    p_mesh.state = PhysicsMeshState::MeshNeeded;
                                }

                                if world_chunk.chunk_lod.is_voxel_on_edge(lv) {
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
                let mut chunks_to_downsample = vec![];

                let mut grid_lock = world_grid.write().unwrap();
                grid_lock.shift_and_remap(camera_data, |op| match op {
                    GridOp::Recycle(world_chunk, cv) => {
                        if world_chunk.is_dirty {
                            let persisted_chunk = PersistedChunk {
                                uniform_voxel_id: world_chunk.uniform_voxel_id.take(),
                                deltas: world_chunk.deltas.clone(),
                            };
                            chunks_to_save.push((world_chunk.cv, persisted_chunk));
                        }

                        let lod = calc_lod(cv, camera_data.pos);
                        let priority = calculate_chunk_priority(cv, camera_data);

                        world_chunk.cv = cv;
                        world_chunk.priority = priority;
                        world_chunk.is_dirty = false;
                        world_chunk.state = PipelineState::DeltasLoading;
                        world_chunk.chunk_lod = ChunkLOD::new(lod);
                        world_chunk.deltas = Deltas::default();
                        world_chunk.uniform_voxel_id = None;

                        chunks_to_load.push(cv);
                    }
                    GridOp::Keep(world_chunk) => {
                        let cv = world_chunk.cv;
                        let lod = calc_lod(cv, camera_data.pos);
                        world_chunk.priority = calculate_chunk_priority(cv, camera_data);

                        if world_chunk.is_stable() && world_chunk.chunk_lod.lod_level() < lod {
                            chunks_to_downsample.push((cv, lod));
                            return;
                        }

                        if world_chunk.chunk_lod.lod_level() != lod {
                            world_chunk.chunk_lod = ChunkLOD::new(lod);
                            if let Some(uniform) = world_chunk.uniform_voxel_id {
                                world_chunk.chunk_lod = ChunkLOD::new_uniform(lod, uniform);
                                world_chunk.merge_deltas::<V>();
                                world_chunk.state = PipelineState::MeshNeeded;
                            } else {
                                world_chunk.state = PipelineState::GenerationNeeded;
                            }
                        }
                    }
                });

                drop(grid_lock);

                if !chunks_to_save.is_empty() {
                    batch_unload_deltas_task(chunks_to_save, &chunk_store, &io_pool);
                }

                if !chunks_to_load.is_empty() {
                    let aabbs = group_chunks_into_aabbs(chunks_to_load);
                    batch_load_deltas_task(aabbs, &chunk_store, &t_sx, &io_pool);
                }

                let mut grid_lock = world_grid.write().unwrap();
                for (cv, lod) in chunks_to_downsample {
                    if let Some(world_chunk) = grid_lock.get_mut(cv) {
                        let mut new_chunk_lod = ChunkLOD::new(lod);
                        let size = new_chunk_lod.size();
                        let step = (new_chunk_lod.lod_scale_factor()
                            / world_chunk.chunk_lod.lod_scale_factor())
                            as i32;

                        for x in 0..size {
                            for y in 0..size {
                                for z in 0..size {
                                    let lv_small = IVec3::new(x, y, z);
                                    let lv_large = lv_small * step;

                                    let sample = world_chunk.chunk_lod.get_voxel(lv_large);
                                    new_chunk_lod.set_voxel::<V>(lv_small, sample);
                                }
                            }
                        }

                        world_chunk.chunk_lod = new_chunk_lod;
                        world_chunk.state = PipelineState::MeshNeeded;
                    }
                }
            }

            let results: Vec<_> = t_rx.try_iter().collect();

            let (mut generate_tasks, mut mesh_tasks) = {
                let grid_lock = world_grid.read().unwrap();
                let mut gen_tasks = Vec::new();
                let mut mesh_tasks = Vec::new();
                for world_chunk in grid_lock.chunks.iter() {
                    match world_chunk.state {
                        PipelineState::GenerationNeeded => gen_tasks.push(WorkItem {
                            priority: world_chunk.priority,
                            cv: world_chunk.cv,
                        }),
                        PipelineState::MeshNeeded => mesh_tasks.push(WorkItem {
                            priority: world_chunk.priority,
                            cv: world_chunk.cv,
                        }),
                        _ => {}
                    }
                }
                (gen_tasks, mesh_tasks)
            };

            generate_tasks.sort();
            mesh_tasks.sort();

            let mut physics_unloads = vec![];
            let mut physics_loads = vec![];

            physics_meshes.retain(|cv, p_mesh| {
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
                let mut grid_lock = world_grid.write().unwrap();

                for result in results {
                    match result {
                        TaskResult::PersistedChunksLoaded { persisted_chunks } => {
                            for (cv, persisted_chunk) in persisted_chunks {
                                if let Some(world_chunk) = grid_lock.get_mut(cv) {
                                    if world_chunk.state == PipelineState::DeltasLoading {
                                        if let Some(pc) = persisted_chunk {
                                            world_chunk.deltas = pc.deltas;
                                            world_chunk.uniform_voxel_id = pc.uniform_voxel_id;
                                        };
                                        if let Some(uniform) = world_chunk.uniform_voxel_id {
                                            let lod: u8 = world_chunk.chunk_lod.lod_level();
                                            world_chunk.chunk_lod =
                                                ChunkLOD::new_uniform(lod, uniform);
                                            world_chunk.merge_deltas::<V>();
                                            world_chunk.state = PipelineState::MeshNeeded;
                                        } else {
                                            world_chunk.state = PipelineState::GenerationNeeded;
                                        }
                                    }
                                }
                            }
                        }
                        TaskResult::GenerationComplete {
                            cv,
                            chunk_lod,
                            uniform_voxel_id,
                        } => {
                            let mut should_notify_neighbors = false;
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.state == PipelineState::Generating
                            {
                                should_notify_neighbors = true;
                                world_chunk.chunk_lod = chunk_lod;
                                if uniform_voxel_id.is_some() {
                                    world_chunk.uniform_voxel_id = uniform_voxel_id;
                                    world_chunk.is_dirty = true;
                                }
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
                        TaskResult::MeshComplete { cv, mesh } => {
                            if let Some(world_chunk) = grid_lock.get_mut(cv)
                                && world_chunk.state == PipelineState::Meshing
                            {
                                world_chunk.state = PipelineState::MeshReady;
                                result_sender
                                    .send(PipelineResult::MeshReady { cv, mesh })
                                    .ok();
                            }
                        }
                        TaskResult::PhysicsMeshComplete { cv, mesh } => {
                            if let Some(p_mesh) = physics_meshes.get_mut(&cv)
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

                let drain_limit = 100;
                for _ in 0..drain_limit {
                    if generate_tasks.is_empty()
                        && mesh_tasks.is_empty()
                        && physics_loads.is_empty()
                    {
                        break;
                    }

                    if let Some(task) = generate_tasks.pop() {
                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.state = PipelineState::Generating;
                            let lod = world_chunk.chunk_lod.lod_level();
                            generate_task(task.cv, &generator, &t_sx, &compute_pool, lod);
                        }
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

                        let mut neighbor_lods: Box<[Option<ChunkLOD>; 6]> = Box::new([None; 6]);
                        for (i, maybe_chunk) in neighbors.iter().enumerate() {
                            if let Some(world_chunk) = maybe_chunk {
                                neighbor_lods[i] = Some(world_chunk.chunk_lod.clone())
                            }
                        }

                        if let Some(world_chunk) = grid_lock.get_mut(task.cv) {
                            world_chunk.state = PipelineState::Meshing;
                            let boxed_chunk_lod = Box::new(world_chunk.chunk_lod.clone());
                            mesh_task::<V>(
                                task.cv,
                                boxed_chunk_lod,
                                neighbor_lods,
                                camera_data,
                                &t_sx,
                                &compute_pool,
                            );

                            break;
                        }
                    }

                    while let Some(cv) = physics_loads.pop()
                        && let Some(p_mesh) = physics_meshes.get_mut(&cv)
                        && p_mesh.state == PhysicsMeshState::MeshNeeded
                        && let Some(world_chunk) = grid_lock.get(cv)
                        && world_chunk.is_stable()
                        && world_chunk.chunk_lod.lod_level() == 1
                    {
                        println!("{cv} Generating physics mesh...");
                        p_mesh.state = PhysicsMeshState::Meshing;
                        let chunk_lod = Box::new(world_chunk.chunk_lod.clone());
                        physics_mesh_task::<V>(cv, chunk_lod, &t_sx, &compute_pool);
                        break;
                    }
                }
            }

            if !work_to_do && t_rx.is_empty() {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    })
}

fn batch_load_deltas_task(
    aabbs: Vec<AABB>,
    chunk_store: &Arc<ChunkStore>,
    t_sx: &Sender<TaskResult>,
    io_pool: &rayon::ThreadPool,
) {
    let t_sx = t_sx.clone();
    let chunk_store = chunk_store.clone();
    io_pool.spawn(move || {
        let mut persisted_chunks: BatchedPersistedChunkMap = Default::default();
        for aabb in aabbs {
            chunk_store.load_chunks_in_aabb(aabb.min, aabb.max, &mut persisted_chunks);
        }
        t_sx.send(TaskResult::PersistedChunksLoaded { persisted_chunks })
            .ok();
    });
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
        let mut chunk_lod = ChunkLOD::new(lod);
        generator.apply(cv_to_wv(cv), &mut chunk_lod);

        let uniform_voxel_id = if let ChunkLOD::LOD1(Chunk::Uniform(uniform)) = chunk_lod {
            Some(uniform)
        } else {
            None
        };

        t_sx.send(TaskResult::GenerationComplete {
            cv,
            chunk_lod,
            uniform_voxel_id,
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

fn physics_mesh_task<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: Box<ChunkLOD>,
    t_sx: &Sender<TaskResult>,
    compute_pool: &rayon::ThreadPool,
) {
    let t_sx = t_sx.clone();
    compute_pool.spawn(move || {
        let mesh = mesh_physics_chunk::<V>(cv, chunk);
        if !mesh.is_empty() {
            t_sx.send(TaskResult::PhysicsMeshComplete { cv, mesh }).ok();
        }
    });
}

pub fn group_chunks_into_aabbs(chunks_to_load: Vec<IVec3>) -> Vec<AABB> {
    let mut chunks = chunks_to_load.into_iter().collect::<HashSet<_>>();
    let mut aabbs = Vec::new();

    while !chunks.is_empty() {
        let seed = *chunks.iter().next().unwrap();
        chunks.remove(&seed);

        let mut aabb = AABB {
            min: seed,
            max: seed,
        };

        let mut expanded_in_iteration = true;
        while expanded_in_iteration {
            expanded_in_iteration = false;

            // -X
            let mut can_expand = true;
            for y in aabb.min.y..=aabb.max.y {
                for z in aabb.min.z..=aabb.max.z {
                    if !chunks.contains(&IVec3::new(aabb.min.x - 1, y, z)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for y in aabb.min.y..=aabb.max.y {
                    for z in aabb.min.z..=aabb.max.z {
                        chunks.remove(&IVec3::new(aabb.min.x - 1, y, z));
                    }
                }
                aabb.min.x -= 1;
                expanded_in_iteration = true;
            }

            // +X
            let mut can_expand = true;
            for y in aabb.min.y..=aabb.max.y {
                for z in aabb.min.z..=aabb.max.z {
                    if !chunks.contains(&IVec3::new(aabb.max.x + 1, y, z)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for y in aabb.min.y..=aabb.max.y {
                    for z in aabb.min.z..=aabb.max.z {
                        chunks.remove(&IVec3::new(aabb.max.x + 1, y, z));
                    }
                }
                aabb.max.x += 1;
                expanded_in_iteration = true;
            }

            // -Y
            let mut can_expand = true;
            for x in aabb.min.x..=aabb.max.x {
                for z in aabb.min.z..=aabb.max.z {
                    if !chunks.contains(&IVec3::new(x, aabb.min.y - 1, z)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for x in aabb.min.x..=aabb.max.x {
                    for z in aabb.min.z..=aabb.max.z {
                        chunks.remove(&IVec3::new(x, aabb.min.y - 1, z));
                    }
                }
                aabb.min.y -= 1;
                expanded_in_iteration = true;
            }

            // +Y
            let mut can_expand = true;
            for x in aabb.min.x..=aabb.max.x {
                for z in aabb.min.z..=aabb.max.z {
                    if !chunks.contains(&IVec3::new(x, aabb.max.y + 1, z)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for x in aabb.min.x..=aabb.max.x {
                    for z in aabb.min.z..=aabb.max.z {
                        chunks.remove(&IVec3::new(x, aabb.max.y + 1, z));
                    }
                }
                aabb.max.y += 1;
                expanded_in_iteration = true;
            }

            // -Z
            let mut can_expand = true;
            for x in aabb.min.x..=aabb.max.x {
                for y in aabb.min.y..=aabb.max.y {
                    if !chunks.contains(&IVec3::new(x, y, aabb.min.z - 1)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for x in aabb.min.x..=aabb.max.x {
                    for y in aabb.min.y..=aabb.max.y {
                        chunks.remove(&IVec3::new(x, y, aabb.min.z - 1));
                    }
                }
                aabb.min.z -= 1;
                expanded_in_iteration = true;
            }

            // +Z
            let mut can_expand = true;
            for x in aabb.min.x..=aabb.max.x {
                for y in aabb.min.y..=aabb.max.y {
                    if !chunks.contains(&IVec3::new(x, y, aabb.max.z + 1)) {
                        can_expand = false;
                        break;
                    }
                }
                if !can_expand {
                    break;
                }
            }
            if can_expand {
                for x in aabb.min.x..=aabb.max.x {
                    for y in aabb.min.y..=aabb.max.y {
                        chunks.remove(&IVec3::new(x, y, aabb.max.z + 1));
                    }
                }
                aabb.max.z += 1;
                expanded_in_iteration = true;
            }
        }

        aabbs.push(aabb);
    }

    aabbs
}
