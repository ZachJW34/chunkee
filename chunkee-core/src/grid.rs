use std::{collections::HashMap, time::Instant};

use glam::IVec3;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, VoxelId},
    chunk::{ChunkLOD, LOD},
    coords::{ChunkVector, camera_vec3_to_cv},
    hasher::BuildMortonHasher,
    streaming::CameraData,
};

#[derive(Debug, Clone)]
pub struct Deltas(pub HashMap<ChunkVector, VoxelId, BuildMortonHasher>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub fn neighbors_of(cv: IVec3) -> [ChunkVector; 6] {
    BLOCK_FACES.map(|face| cv + face.into_normal())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    DeltasNeeded,
    DeltasLoading,
    GenerationNeeded,
    Generating,
    MeshNeeded,
    Meshing,
    MeshReady,
}

#[derive(Debug, Clone)]
pub struct WorldChunk {
    pub cv: ChunkVector,
    // pub io_state: IOState,
    pub state: PipelineState,
    pub chunk_lod: ChunkLOD,
    pub deltas: Deltas,
    pub is_dirty: bool,
    pub priority: u32,
    // Some if the generator produced a chunk composed entirely of one voxel type
    pub uniform_voxel_id: Option<VoxelId>,
}

impl Default for WorldChunk {
    fn default() -> Self {
        Self {
            cv: ChunkVector::MAX,
            // io_state: IOState::DeltasNeeded,
            state: PipelineState::DeltasNeeded,
            chunk_lod: ChunkLOD::new(3),
            deltas: Deltas::default(),
            is_dirty: false,
            priority: 0,
            uniform_voxel_id: None,
        }
    }
}

impl WorldChunk {
    pub fn new(cv: ChunkVector, lod: LOD, priority: u32) -> Self {
        Self {
            cv,
            // io_state: IOState::DeltasNeeded,
            state: PipelineState::DeltasNeeded,
            chunk_lod: ChunkLOD::new(lod),
            deltas: Deltas::default(),
            is_dirty: false,
            priority,
            uniform_voxel_id: None,
        }
    }

    pub fn merge_deltas<V: ChunkeeVoxel>(&mut self) {
        let lod_scale_factor = self.chunk_lod.lod_scale_factor() as i32;

        for (lv, voxel_id) in self.deltas.0.iter() {
            let lv_scaled = *lv / lod_scale_factor;
            self.chunk_lod.set_voxel::<V>(lv_scaled, *voxel_id);
        }
    }

    pub fn is_stable(&self) -> bool {
        matches!(
            self.state,
            PipelineState::MeshNeeded | PipelineState::Meshing | PipelineState::MeshReady
        )
    }
}

pub enum GridOp<'a> {
    Recycle(&'a mut WorldChunk, ChunkVector),
    Keep(&'a mut WorldChunk),
}

pub struct ChunkGrid {
    pub chunks: Vec<WorldChunk>,
    indices: Vec<usize>,
    pub dimensions: IVec3,
    pub grid_origin: IVec3,
}

impl ChunkGrid {
    pub fn new(radius_x: u32, radius_y: u32, radius_z: u32) -> Self {
        let dims = IVec3::new(
            2 * radius_x as i32 + 1,
            2 * radius_y as i32 + 1,
            2 * radius_z as i32 + 1,
        );
        let capacity = (dims.x * dims.y * dims.z) as usize;

        Self {
            chunks: vec![WorldChunk::default(); capacity],
            indices: (0..capacity).collect(),
            dimensions: dims,
            grid_origin: IVec3::MAX,
        }
    }

    pub fn get(&self, cv: IVec3) -> Option<&WorldChunk> {
        let grid_idx = Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)?;
        let chunk_idx = self.indices.get(grid_idx)?;
        let chunk = self.chunks.get(*chunk_idx)?;

        (chunk.cv == cv).then_some(chunk)
    }

    pub fn get_mut(&mut self, cv: IVec3) -> Option<&mut WorldChunk> {
        let grid_idx = Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)?;
        let chunk_idx = self.indices.get(grid_idx)?;
        let chunk = self.chunks.get_mut(*chunk_idx)?;

        (chunk.cv == cv).then_some(chunk)
    }

    fn cv_to_idx_with_origin(cv: IVec3, origin: IVec3, dims: IVec3) -> Option<usize> {
        let local = cv - origin;
        if (0..dims.x).contains(&local.x)
            && (0..dims.y).contains(&local.y)
            && (0..dims.z).contains(&local.z)
        {
            Some((local.x + local.y * dims.x + local.z * dims.x * dims.y) as usize)
        } else {
            None
        }
    }

    fn idx_to_local(idx: usize, dims: IVec3) -> IVec3 {
        let x = (idx as i32 % (dims.x * dims.y)) % dims.x;
        let y = (idx as i32 % (dims.x * dims.y)) / dims.x;
        let z = idx as i32 / (dims.x * dims.y);
        IVec3::new(x, y, z)
    }

    pub fn shift_and_remap<F>(&mut self, camera_data: &CameraData, mut on_remap: F) -> bool
    where
        F: FnMut(GridOp),
    {
        let camera_cv = camera_vec3_to_cv(camera_data.pos);
        let dims = self.dimensions;
        let new_origin = camera_cv - (dims / 2);

        if new_origin == self.grid_origin {
            return false;
        }

        let t = Instant::now();

        if self.grid_origin == IVec3::MAX {
            self.grid_origin = new_origin;
            for i in 0..self.chunks.len() {
                let local = Self::idx_to_local(i, dims);
                let new_cv = new_origin + local;
                on_remap(GridOp::Recycle(&mut self.chunks[i], new_cv));
                self.indices[i] = i;
            }
            println!("{camera_cv} [shift_and_remap] init: {:?}", t.elapsed());
            return true;
        }

        self.grid_origin = new_origin;

        let mut new_indices = vec![usize::MAX; self.chunks.len()];
        let mut used_chunks = vec![false; self.chunks.len()];

        // Pass 1: Preserve chunks that are still visible and issue `Keep` callbacks.
        for old_grid_idx in 0..self.indices.len() {
            let chunk_idx = self.indices[old_grid_idx];
            let cv = self.chunks[chunk_idx].cv;

            if let Some(new_grid_idx) = Self::cv_to_idx_with_origin(cv, new_origin, dims) {
                new_indices[new_grid_idx] = chunk_idx;
                used_chunks[chunk_idx] = true;
                on_remap(GridOp::Keep(&mut self.chunks[chunk_idx]));
            }
        }

        // Pass 2: Collect indices of chunks that are no longer used.
        let mut recycled_indices: Vec<_> = (0..self.chunks.len())
            .filter(|&i| !used_chunks[i])
            .collect();

        // Pass 3: Fill empty slots in the new grid with recycled chunks and issue `Recycle` callbacks.
        for new_grid_idx in 0..new_indices.len() {
            if new_indices[new_grid_idx] == usize::MAX {
                let recycled_chunk_idx =
                    recycled_indices.pop().expect("Not enough recycled chunks");
                new_indices[new_grid_idx] = recycled_chunk_idx;

                let local = Self::idx_to_local(new_grid_idx, dims);
                let new_cv = new_origin + local;

                on_remap(GridOp::Recycle(
                    &mut self.chunks[recycled_chunk_idx],
                    new_cv,
                ));
            }
        }

        self.indices = new_indices;

        println!("{} [shift_and_remap]: {:?}", camera_cv, t.elapsed());
        true
    }
}
