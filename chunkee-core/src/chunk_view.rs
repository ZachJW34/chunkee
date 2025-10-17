use arc_swap::ArcSwap;
use glam::IVec3;
use parking_lot::RwLock;

use crate::{block::BLOCK_FACES, coords::ChunkVector, pipeline::WorldChunk};

pub struct ChunkPool {
    pub flat: Vec<WorldChunk>,
}

impl ChunkPool {
    pub fn new(radius: u32) -> Self {
        let max_length = 2 * radius + 1;
        let capacity = max_length.pow(3);

        Self {
            flat: (0..capacity).map(|_| WorldChunk::default()).collect(),
        }
    }

    pub fn get(&self, cv: ChunkVector, grid: &ChunkGrid) -> Option<&WorldChunk> {
        let grid_idx = grid.cv_to_idx_with_origin(cv)?;
        let chunk_idx = *grid.indices.get(grid_idx)?;
        self.flat.get(chunk_idx)
    }

    pub fn get_mut(&mut self, cv: ChunkVector, grid: &ChunkGrid) -> Option<&mut WorldChunk> {
        let grid_idx = grid.cv_to_idx_with_origin(cv)?;
        let chunk_idx = *grid.indices.get(grid_idx)?;
        self.flat.get_mut(chunk_idx)
    }
}

pub const FREE_GRID_IDX: usize = usize::MAX;

#[derive(Clone)]
pub struct ChunkGrid {
    pub indices: Vec<usize>,
    pub dimensions: IVec3,
    pub grid_origin: IVec3,
    pub grid_anchor: IVec3,
    pub initialized: bool,
}

pub type ChunkResetTask = (usize, ChunkVector);

impl ChunkGrid {
    const SNAP_DISTANCE: i32 = 4;
    const TRIGGER_RADIUS: i32 = 4;

    pub fn new(radius: u32) -> Self {
        let max_length = 2 * radius + 1;
        let capacity = max_length.pow(3) as usize;

        Self {
            indices: (0..capacity).collect(),
            dimensions: IVec3::splat(max_length as i32),
            grid_origin: IVec3::MAX,
            grid_anchor: IVec3::MAX,
            initialized: false,
        }
    }

    pub fn needs_update(&self, camera_cv: ChunkVector) -> bool {
        if !self.initialized {
            return true;
        }

        let dist_from_anchor = (camera_cv - self.grid_anchor).abs().max_element();

        dist_from_anchor > Self::TRIGGER_RADIUS
    }

    pub fn is_frontier(&self, cv: IVec3) -> bool {
        for face in BLOCK_FACES {
            let neighbor_cv = cv + face.into_normal();
            if self.cv_to_idx_with_origin(neighbor_cv).is_none() {
                return true;
            }
        }
        false
    }

    pub fn compute_new(
        old_grid: &ChunkGrid,
        camera_cv: ChunkVector,
    ) -> (Self, Vec<ChunkResetTask>, Vec<ChunkVector>) {
        let mut new_grid = old_grid.clone();

        if !new_grid.initialized {
            let snap_block_origin = IVec3::new(
                (camera_cv.x >> 2) << 2,
                (camera_cv.y >> 2) << 2,
                (camera_cv.z >> 2) << 2,
            );
            new_grid.grid_anchor = snap_block_origin + IVec3::splat(Self::SNAP_DISTANCE / 2);
            new_grid.grid_origin = new_grid.grid_anchor - new_grid.dimensions / 2;
            let new_indices: Vec<_> = (0..old_grid.indices.len()).collect();
            new_grid.indices = new_indices.clone();
            new_grid.initialized = true;

            let reset_tasks = (0..new_grid.indices.len())
                .map(|idx| {
                    let new_cv =
                        new_grid.grid_origin + Self::idx_to_local(idx, new_grid.dimensions);
                    (idx, new_cv)
                })
                .collect();

            return (new_grid, reset_tasks, Vec::new());
        }

        let direction = (camera_cv - new_grid.grid_anchor).signum();
        new_grid.grid_anchor += direction * Self::SNAP_DISTANCE;
        new_grid.grid_origin = new_grid.grid_anchor - new_grid.dimensions / 2;

        let mut new_indices = vec![FREE_GRID_IDX; old_grid.indices.len()];
        let mut used_chunks = vec![false; old_grid.indices.len()];
        let mut remesh_cvs = Vec::new();

        for old_grid_idx in 0..old_grid.indices.len() {
            let chunk_idx = old_grid.indices[old_grid_idx];

            let local_pos = Self::idx_to_local(old_grid_idx, old_grid.dimensions);
            let cv = old_grid.grid_origin + local_pos;

            if let Some(new_grid_idx) = new_grid.cv_to_idx_with_origin(cv) {
                new_indices[new_grid_idx] = chunk_idx;
                used_chunks[chunk_idx] = true;

                if old_grid.is_frontier(cv) && !new_grid.is_frontier(cv) {
                    remesh_cvs.push(cv);
                }
            }
        }

        let mut recycled_chunk_indices: Vec<_> = (0..old_grid.indices.len())
            .filter(|&i| !used_chunks[i])
            .collect();

        let mut reset_tasks = Vec::new();

        for new_grid_idx in 0..new_indices.len() {
            if new_indices[new_grid_idx] == FREE_GRID_IDX {
                let recycled_chunk_idx = recycled_chunk_indices.pop().expect("Mismatch");
                new_indices[new_grid_idx] = recycled_chunk_idx;

                let new_cv =
                    new_grid.grid_origin + Self::idx_to_local(new_grid_idx, new_grid.dimensions);

                reset_tasks.push((recycled_chunk_idx, new_cv));
            }
        }

        new_grid.indices = new_indices;

        (new_grid, reset_tasks, remesh_cvs)
    }

    #[inline(always)]
    pub fn cv_to_idx_with_origin(&self, cv: IVec3) -> Option<usize> {
        let origin = self.grid_origin;
        let dims = self.dimensions;
        let local = cv - origin;

        if local.x >= 0
            && local.x < dims.x
            && local.y >= 0
            && local.y < dims.y
            && local.z >= 0
            && local.z < dims.z
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
}

pub struct ChunkView {
    pub pool: RwLock<ChunkPool>,
    pub grid: ArcSwap<ChunkGrid>,
}

impl ChunkView {
    pub fn new(radius: u32) -> Self {
        Self {
            pool: RwLock::new(ChunkPool::new(radius)),
            grid: ArcSwap::new(ChunkGrid::new(radius).into()),
        }
    }

    // fn get(&self, cv: ChunkVector) -> Option<&WorldChunk> {
    //     let view = self.view.load();
    //     let grid_idx = view.cv_to_idx_with_origin(cv)?;
    //     let chunk_idx = *view.indices.get(grid_idx)?;
    //     self.pool.flat.get(chunk_idx)
    // }
}
