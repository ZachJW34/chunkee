use std::sync::Arc;

use arc_swap::ArcSwap;
use glam::IVec3;
use parking_lot::RwLock;

use crate::{
    coords::ChunkVector,
    pipeline::{ChunkState, PhysicsState, WorldChunk},
};

pub struct ChunkPool {
    pub flat: Vec<Arc<RwLock<WorldChunk>>>,
}

impl ChunkPool {
    pub fn new(radius: u32) -> Self {
        let max_length = 2 * radius + 1;
        let capacity = max_length.pow(3);

        Self {
            flat: (0..capacity)
                .map(|_| Arc::new(RwLock::new(WorldChunk::default())))
                .collect(),
        }
    }
}

pub const FREE_GRID_IDX: usize = usize::MAX;

#[derive(Clone)]
pub struct GridView {
    pub indices: Vec<usize>,
    pub dimensions: IVec3,
    pub grid_origin: IVec3,
    pub grid_anchor: IVec3,
    pub initialized: bool,
}

pub type ChunkResetTask = (usize, ChunkVector);

impl GridView {
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

    pub fn compute_new(old_view: &GridView, camera_cv: ChunkVector) -> (Self, Vec<ChunkResetTask>) {
        let mut new_view = old_view.clone();

        if !new_view.initialized {
            let snap_block_origin = IVec3::new(
                (camera_cv.x >> 2) << 2,
                (camera_cv.y >> 2) << 2,
                (camera_cv.z >> 2) << 2,
            );
            new_view.grid_anchor = snap_block_origin + IVec3::splat(Self::SNAP_DISTANCE / 2);
            new_view.grid_origin = new_view.grid_anchor - new_view.dimensions / 2;
            let new_indices: Vec<_> = (0..old_view.indices.len()).collect();
            new_view.indices = new_indices.clone();
            new_view.initialized = true;

            let reset_tasks = (0..new_view.indices.len())
                .map(|idx| {
                    let new_cv =
                        new_view.grid_origin + Self::idx_to_local(idx, new_view.dimensions);
                    (idx, new_cv)
                })
                .collect();

            return (new_view, reset_tasks);
        }

        let direction = (camera_cv - new_view.grid_anchor).signum();
        new_view.grid_anchor += direction * Self::SNAP_DISTANCE;
        new_view.grid_origin = new_view.grid_anchor - new_view.dimensions / 2;

        let mut new_indices = vec![FREE_GRID_IDX; old_view.indices.len()];
        let mut used_chunks = vec![false; old_view.indices.len()];

        for old_grid_idx in 0..old_view.indices.len() {
            let chunk_idx = old_view.indices[old_grid_idx];

            let local_pos = Self::idx_to_local(old_grid_idx, old_view.dimensions);
            let cv = old_view.grid_origin + local_pos;

            if let Some(new_grid_idx) =
                Self::cv_to_idx_with_origin(cv, new_view.grid_origin, new_view.dimensions)
            {
                new_indices[new_grid_idx] = chunk_idx;
                used_chunks[chunk_idx] = true;
            }
        }

        let mut recycled_chunk_indices: Vec<_> = (0..old_view.indices.len())
            .filter(|&i| !used_chunks[i])
            .collect();

        let mut reset_tasks = Vec::new();

        for new_grid_idx in 0..new_indices.len() {
            if new_indices[new_grid_idx] == FREE_GRID_IDX {
                let recycled_chunk_idx = recycled_chunk_indices.pop().expect("Mismatch");
                new_indices[new_grid_idx] = recycled_chunk_idx;

                let new_cv =
                    new_view.grid_origin + Self::idx_to_local(new_grid_idx, new_view.dimensions);

                reset_tasks.push((recycled_chunk_idx, new_cv));
            }
        }

        new_view.indices = new_indices;

        (new_view, reset_tasks)
    }

    pub fn cv_to_idx_with_origin(cv: IVec3, origin: IVec3, dims: IVec3) -> Option<usize> {
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

pub struct ChunkManager {
    pub pool: ChunkPool,
    pub view: ArcSwap<GridView>,
}

impl ChunkManager {
    pub fn new(radius: u32) -> Self {
        Self {
            pool: ChunkPool::new(radius),
            view: ArcSwap::new(GridView::new(radius).into()),
        }
    }

    pub fn read<F, R>(&self, cv: ChunkVector, f: F) -> Option<R>
    where
        F: FnOnce(&WorldChunk) -> R,
    {
        let world_chunk = self.get_lock(cv)?.read();

        if world_chunk.cv == cv {
            Some(f(&world_chunk))
        } else {
            None
        }
    }

    pub fn try_read<F, R>(&self, cv: ChunkVector, f: F) -> Option<R>
    where
        F: FnOnce(&WorldChunk) -> R,
    {
        let world_chunk = self.get_lock(cv)?.try_read()?;

        if world_chunk.cv == cv {
            Some(f(&world_chunk))
        } else {
            None
        }
    }

    pub fn write<F, R>(&self, cv: ChunkVector, f: F) -> Option<R>
    where
        F: FnOnce(&mut WorldChunk) -> R,
    {
        let mut world_chunk = self.get_lock(cv)?.write();

        if world_chunk.cv == cv {
            Some(f(&mut world_chunk))
        } else {
            None
        }
    }

    pub fn check<F>(&self, cv: ChunkVector, f: F) -> bool
    where
        F: FnOnce(&WorldChunk) -> bool,
    {
        if let Some(rw) = self.get_lock(cv) {
            let world_chunk = rw.read();

            if world_chunk.cv == cv {
                return f(&world_chunk);
            }
        }

        false
    }

    fn get_lock(&self, cv: ChunkVector) -> Option<&Arc<RwLock<WorldChunk>>> {
        let view = self.view.load();
        let grid_idx = GridView::cv_to_idx_with_origin(cv, view.grid_origin, view.dimensions)?;
        let chunk_idx = *view.indices.get(grid_idx)?;
        self.pool.flat.get(chunk_idx)
    }

    pub fn get_state_matched(
        &self,
        cv: IVec3,
        state: ChunkState,
    ) -> Option<Arc<RwLock<WorldChunk>>> {
        let view = self.view.load();
        let grid_idx = GridView::cv_to_idx_with_origin(cv, view.grid_origin, view.dimensions)?;
        let chunk_idx = *view.indices.get(grid_idx)?;
        let chunk_arc = self.pool.flat.get(chunk_idx)?;
        let world_chunk = chunk_arc.read();

        if world_chunk.cv == cv && world_chunk.state == state {
            Some(chunk_arc.clone())
        } else {
            None
        }
    }
}

pub fn check_chunk_stable(world_chunk: &WorldChunk, cv: ChunkVector) -> bool {
    world_chunk.cv == cv && world_chunk.is_stable()
}

pub fn check_chunk_p(world_chunk: &WorldChunk, cv: ChunkVector, state: PhysicsState) -> bool {
    world_chunk.cv == cv && world_chunk.physics_state == state
}

pub fn check_chunk_s(world_chunk: &WorldChunk, cv: ChunkVector, state: ChunkState) -> bool {
    world_chunk.cv == cv && world_chunk.state == state
}

pub fn check_chunk_sv(
    world_chunk: &WorldChunk,
    cv: ChunkVector,
    state: ChunkState,
    version: u32,
) -> bool {
    world_chunk.cv == cv && world_chunk.state == state && world_chunk.version == version
}

pub fn check_chunk_pv(
    world_chunk: &WorldChunk,
    cv: ChunkVector,
    state: PhysicsState,
    version: u32,
) -> bool {
    world_chunk.cv == cv && world_chunk.physics_state == state && world_chunk.version == version
}
