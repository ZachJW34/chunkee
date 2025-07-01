use std::collections::HashMap;

use glam::IVec3;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, Rotation, VoxelId},
    coords::{CHUNK_VOLUME, ChunkVector, LocalVector, lv_to_idx},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(with = "BigArray")]
    pub(crate) voxels: [VoxelId; CHUNK_VOLUME as usize],
    pub(crate) solid_count: u32,
    #[serde(skip)]
    pub(crate) is_dirty: bool,
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            voxels: [VoxelId::AIR; CHUNK_VOLUME as usize],
            solid_count: 0,
            is_dirty: false,
        }
    }
}

impl Chunk {
    pub fn get_voxel(&self, lv: IVec3) -> VoxelId {
        let idx = lv_to_idx(lv);

        self.voxels[idx]
    }

    pub fn set_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector, new_id: VoxelId) -> VoxelId {
        let idx = lv_to_idx(lv);

        let new_voxel = V::from(new_id.type_id());

        let old_voxel_id = self.voxels[idx];
        let old_voxel = V::from(old_voxel_id.type_id());

        self.voxels[idx] = new_id;
        self.solid_count -= old_voxel.is_solid() as u32;
        self.solid_count += new_voxel.is_solid() as u32;
        self.is_dirty = true;

        old_voxel_id
    }

    pub fn remove_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector) -> VoxelId {
        let idx = lv_to_idx(lv);

        let old_voxel_id = self.voxels[idx];
        let old_voxel = V::from(old_voxel_id.type_id());

        self.voxels[idx] = VoxelId::AIR;
        self.solid_count -= old_voxel.is_solid() as u32;
        self.is_dirty = true;

        old_voxel_id
    }

    pub fn set_block<V: ChunkeeVoxel>(&mut self, lv: LocalVector, block: V) -> VoxelId {
        let new_id = VoxelId::new(block.into(), Rotation::default());
        self.set_voxel::<V>(lv, new_id)
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }

    pub fn is_solid(&self) -> bool {
        self.solid_count == CHUNK_VOLUME as u32
    }
}

#[derive(Clone)]
pub struct Deltas(pub HashMap<ChunkVector, VoxelId>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub fn neighbors_of(cv: IVec3) -> [ChunkVector; 6] {
    BLOCK_FACES.map(|face| cv + face.into_normal())
}
