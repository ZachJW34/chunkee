use std::{sync::Arc, time::Instant};

use glam::IVec3;
use log::info;
use parking_lot::RwLock;

use crate::{
    coords::ChunkVector,
    metrics::Metrics,
    pipeline::{PipelineMetrics, WorldChunk},
};

pub enum GridOp<'a> {
    Recycle(&'a mut WorldChunk, ChunkVector),
    // Keep(&'a mut WorldChunk),
}

pub struct ChunkGrid {
    pub flat: Vec<Arc<RwLock<WorldChunk>>>,
    indices: Vec<usize>,
    pub previous_cv: ChunkVector,
    pub dimensions: IVec3,
    pub grid_origin: IVec3,
    pub grid_anchor: IVec3,
}

impl ChunkGrid {
    pub fn new(radius: u32) -> Self {
        let max_length = 2 * radius + 1;
        let dims = IVec3::splat(max_length as i32);
        let capacity = (dims.x * dims.y * dims.z) as usize;

        Self {
            flat: (0..capacity)
                .map(|_| Arc::new(RwLock::new(WorldChunk::default())))
                .collect(),
            indices: (0..capacity).collect(),
            dimensions: dims,
            previous_cv: IVec3::MAX,
            grid_origin: IVec3::MAX,
            grid_anchor: IVec3::MAX,
        }
    }

    pub fn get(&self, cv: IVec3) -> Option<&Arc<RwLock<WorldChunk>>> {
        let grid_idx = Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)?;
        let chunk_idx = self.indices.get(grid_idx)?;
        self.flat.get(*chunk_idx)
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

    pub fn update<F>(
        &mut self,
        camera_cv: ChunkVector,
        mut on_remap: F,
        metrics: &mut Metrics<PipelineMetrics>,
    ) -> Option<(IVec3, IVec3)>
    where
        F: FnMut(GridOp),
    {
        const SNAP_DISTANCE: i32 = 4;
        const TRIGGER_RADIUS: i32 = 4;

        if camera_cv != self.previous_cv {
            println!("{} -> {}", self.previous_cv, camera_cv);
            self.previous_cv = camera_cv;
        }

        let is_first_run = self.grid_anchor == IVec3::MAX;

        if is_first_run {
            let snap_block_origin = IVec3::new(
                (camera_cv.x >> 2) << 2,
                (camera_cv.y >> 2) << 2,
                (camera_cv.z >> 2) << 2,
            );
            self.grid_anchor = snap_block_origin + IVec3::splat(SNAP_DISTANCE / 2);
            self.previous_cv = camera_cv;
        }

        let dist_from_anchor = (camera_cv - self.grid_anchor).abs().max_element();

        if dist_from_anchor <= TRIGGER_RADIUS && !is_first_run {
            return None;
        }

        if !is_first_run {
            let direction = (camera_cv - self.grid_anchor).signum();
            self.grid_anchor += direction * SNAP_DISTANCE;
        }

        let new_origin = self.grid_anchor - self.dimensions / 2;

        info!("Grid snapping to new origin {new_origin} from {camera_cv}");
        let time = Instant::now();

        self.grid_origin = new_origin;

        let mut new_indices = vec![usize::MAX; self.flat.len()];
        let mut used_chunks = vec![false; self.flat.len()];

        if !is_first_run {
            for old_grid_idx in 0..self.indices.len() {
                let chunk_idx = self.indices[old_grid_idx];
                let cv = self.flat[chunk_idx].read().cv;

                if let Some(new_grid_idx) =
                    Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)
                {
                    new_indices[new_grid_idx] = chunk_idx;
                    used_chunks[chunk_idx] = true;
                    // on_remap(GridOp::Keep(&mut self.flat[chunk_idx]));
                }
            }
        }

        let mut recycled_indices: Vec<_> =
            (0..self.flat.len()).filter(|&i| !used_chunks[i]).collect();
        for new_grid_idx in 0..new_indices.len() {
            if new_indices[new_grid_idx] == usize::MAX {
                let recycled_chunk_idx =
                    recycled_indices.pop().expect("Not enough recycled chunks");
                new_indices[new_grid_idx] = recycled_chunk_idx;

                let local = Self::idx_to_local(new_grid_idx, self.dimensions);
                let new_cv = self.grid_origin + local;

                on_remap(GridOp::Recycle(
                    &mut self.flat[recycled_chunk_idx].write(),
                    new_cv,
                ));
            }
        }

        self.indices = new_indices;

        metrics
            .get_mut(PipelineMetrics::GridUpdate)
            .record(time.elapsed());

        Some((self.grid_origin, self.dimensions))
    }
}

// use std::{collections::HashMap, time::Instant};

// use glam::IVec3;
// use log::info;

// use crate::{
//     coords::ChunkVector,
//     metrics::Metrics,
//     pipeline::{PipelineMetrics, WorldChunk},
// };

// pub enum GridOp<'a> {
//     Recycle(&'a mut WorldChunk, ChunkVector),
//     // Keep(&'a mut WorldChunk),
// }

// pub struct ChunkGrid {
//     pub flat: Vec<WorldChunk>,
//     pub previous_cv: ChunkVector,
//     pub dimensions: IVec3,
//     pub grid_origin: IVec3,
//     pub grid_anchor: IVec3,
// }

// impl ChunkGrid {
//     pub fn new(radius: u32) -> Self {
//         let max_length = 2 * radius + 1;
//         let dims = IVec3::splat(max_length as i32);
//         let capacity = (dims.x * dims.y * dims.z) as usize;

//         Self {
//             flat: vec![WorldChunk::default(); capacity],
//             dimensions: dims,
//             previous_cv: IVec3::MAX,
//             grid_origin: IVec3::MAX,
//             grid_anchor: IVec3::MAX,
//         }
//     }

//     pub fn get(&self, cv: IVec3) -> Option<&WorldChunk> {
//         let grid_idx = Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)?;
//         let chunk = self.flat.get(grid_idx)?;
//         (chunk.cv == cv).then_some(chunk)
//     }

//     pub fn get_mut(&mut self, cv: IVec3) -> Option<&mut WorldChunk> {
//         let grid_idx = Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)?;
//         let chunk = self.flat.get_mut(grid_idx)?;
//         (chunk.cv == cv).then_some(chunk)
//     }

//     pub fn cv_to_idx_with_origin(cv: IVec3, origin: IVec3, dims: IVec3) -> Option<usize> {
//         let local = cv - origin;
//         if local.x >= 0
//             && local.x < dims.x
//             && local.y >= 0
//             && local.y < dims.y
//             && local.z >= 0
//             && local.z < dims.z
//         {
//             Some((local.x + local.y * dims.x + local.z * dims.x * dims.y) as usize)
//         } else {
//             None
//         }
//     }

//     fn idx_to_local(idx: usize, dims: IVec3) -> IVec3 {
//         let x = (idx as i32 % (dims.x * dims.y)) % dims.x;
//         let y = (idx as i32 % (dims.x * dims.y)) / dims.x;
//         let z = idx as i32 / (dims.x * dims.y);
//         IVec3::new(x, y, z)
//     }

//     pub fn update<F>(
//         &mut self,
//         camera_cv: ChunkVector,
//         mut on_remap: F,
//         metrics: &mut Metrics<PipelineMetrics>,
//     ) -> Option<(IVec3, IVec3)>
//     where
//         F: FnMut(GridOp),
//     {
//         const SNAP_DISTANCE: i32 = 4;
//         const TRIGGER_RADIUS: i32 = 4;

//         if camera_cv != self.previous_cv {
//             println!("{} -> {}", self.previous_cv, camera_cv);
//             self.previous_cv = camera_cv;
//         }

//         let is_first_run = self.grid_anchor == IVec3::MAX;

//         if is_first_run {
//             let snap_block_origin = IVec3::new(
//                 (camera_cv.x >> 2) << 2,
//                 (camera_cv.y >> 2) << 2,
//                 (camera_cv.z >> 2) << 2,
//             );
//             self.grid_anchor = snap_block_origin + IVec3::splat(SNAP_DISTANCE / 2);
//             self.previous_cv = camera_cv;
//         }

//         let dist_from_anchor = (camera_cv - self.grid_anchor).abs().max_element();

//         if dist_from_anchor <= TRIGGER_RADIUS && !is_first_run {
//             return None;
//         }

//         if !is_first_run {
//             let direction = (camera_cv - self.grid_anchor).signum();
//             self.grid_anchor += direction * SNAP_DISTANCE;
//         }

//         self.grid_origin = self.grid_anchor - self.dimensions / 2;

//         info!(
//             "Grid snapping to new origin {} from {}",
//             self.grid_origin, camera_cv
//         );
//         let time = Instant::now();

//         let mut target_map = vec![usize::MAX; self.flat.len()];
//         let mut used_chunks = vec![false; self.flat.len()];
//         let mut old_cvs = self.flat.iter().map(|c| c.cv).collect::<Vec<_>>();

//         if !is_first_run {
//             for old_grid_idx in 0..self.flat.len() {
//                 let cv = old_cvs[old_grid_idx];
//                 if let Some(new_grid_idx) =
//                     Self::cv_to_idx_with_origin(cv, self.grid_origin, self.dimensions)
//                 {
//                     target_map[old_grid_idx] = new_grid_idx;
//                     used_chunks[new_grid_idx] = true;
//                 }
//             }
//         }

//         let mut recycled_indices: Vec<_> = (0..self.flat.len())
//             .filter(|&i| target_map[i] == usize::MAX)
//             .collect();

//         for new_grid_idx in 0..self.flat.len() {
//             if !used_chunks[new_grid_idx] {
//                 let recycled_chunk_idx =
//                     recycled_indices.pop().expect("Not enough recycled chunks");
//                 target_map[recycled_chunk_idx] = new_grid_idx;

//                 let local = Self::idx_to_local(new_grid_idx, self.dimensions);
//                 let new_cv = self.grid_origin + local;
//                 on_remap(GridOp::Recycle(&mut self.flat[recycled_chunk_idx], new_cv));
//             }
//         }

//         for i in 0..self.flat.len() {
//             while target_map[i] != i {
//                 let target_idx = target_map[i];
//                 self.flat.swap(i, target_idx);
//                 target_map.swap(i, target_idx);
//             }
//         }

//         metrics
//             .get_mut(PipelineMetrics::GridUpdate)
//             .record(time.elapsed());

//         Some((self.grid_origin, self.dimensions))
//     }
// }
