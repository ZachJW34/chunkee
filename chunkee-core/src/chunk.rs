use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use glam::IVec3;

use crate::{
    block::{
        BlockFace, ChunkeeVoxel, NeighborsMask, Rotation, VoxelId, VoxelVisibility,
        neighbors_mask_to_faces,
    },
    coords::LocalVector,
};

pub const CHUNK_SIDE_32: i32 = 32;
pub const CHUNK_VOLUME_32: usize = (CHUNK_SIDE_32 * CHUNK_SIDE_32 * CHUNK_SIDE_32) as usize;
type Shape32 = ConstShape3u32<32, 32, 32>;

#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub(crate) voxels: [VoxelId; CHUNK_VOLUME_32],
    pub(crate) solid_count: u32,
}

impl Chunk {
    pub const VOL: usize = CHUNK_VOLUME_32;
    pub const SIDE: i32 = CHUNK_SIDE_32;

    pub fn new() -> Self {
        Self {
            voxels: [VoxelId::AIR; CHUNK_VOLUME_32],
            solid_count: 0,
        }
    }

    fn linearize(&self, lv: IVec3) -> usize {
        Shape32::linearize(lv.as_uvec3().to_array()) as usize
    }

    pub fn get_voxel(&self, lv: IVec3) -> VoxelId {
        self.voxels[self.linearize(lv)]
    }

    pub fn set_voxel<V: ChunkeeVoxel>(
        &mut self,
        lv: LocalVector,
        new_voxel_id: VoxelId,
    ) -> VoxelId {
        let idx = self.linearize(lv);
        let new_voxel = new_voxel_id.to_voxel::<V>();
        let old_voxel_id = self.voxels[idx];
        let old_voxel = old_voxel_id.to_voxel::<V>();

        self.voxels[idx] = new_voxel_id;
        self.solid_count -= voxel_solid_value::<V>(old_voxel);
        self.solid_count += voxel_solid_value::<V>(new_voxel);

        old_voxel_id
    }

    pub fn fill<V: ChunkeeVoxel>(&mut self, new_id: VoxelId) {
        let new_voxel = new_id.to_voxel::<V>();
        self.voxels.fill(new_id);
        self.solid_count = voxel_solid_value::<V>(new_voxel) * (self.voxels.len() as u32);
    }

    pub fn set_block<V: ChunkeeVoxel>(&mut self, lv: LocalVector, block: V) -> VoxelId {
        let new_id = VoxelId::new(block.into(), Rotation::default());
        self.set_voxel::<V>(lv, new_id)
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }

    pub fn is_solid(&self) -> bool {
        self.solid_count as usize == CHUNK_VOLUME_32
    }

    pub fn is_voxel_on_edge(&self, lv: IVec3) -> bool {
        let min = IVec3::ZERO;
        let max = IVec3::splat(Self::SIDE - 1);

        (lv.cmpeq(min) | lv.cmpeq(max)).any()
    }

    fn get_voxel_edge_faces_mask(&self, lv: IVec3) -> NeighborsMask {
        let mut mask: NeighborsMask = 0;
        let max = Self::SIDE - 1;

        if lv.x == 0 {
            mask |= 1 << (BlockFace::Left as u8);
        }
        if lv.x == max {
            mask |= 1 << (BlockFace::Right as u8);
        }

        if lv.y == 0 {
            mask |= 1 << (BlockFace::Bottom as u8);
        }
        if lv.y == max {
            mask |= 1 << (BlockFace::Top as u8);
        }

        if lv.z == 0 {
            mask |= 1 << (BlockFace::Front as u8);
        }
        if lv.z == max {
            mask |= 1 << (BlockFace::Back as u8);
        }

        mask
    }

    pub fn get_voxel_edge_faces(&self, lv: IVec3) -> (NeighborsMask, [Option<BlockFace>; 6]) {
        let mask = self.get_voxel_edge_faces_mask(lv);

        (mask, neighbors_mask_to_faces(mask))
    }

    pub fn is_uniform(&self) -> Option<VoxelId> {
        let first = self.voxels[0];
        self.voxels
            .iter()
            .all(|voxel_id| *voxel_id == first)
            .then_some(first)
    }
}

pub fn voxel_solid_value<V: ChunkeeVoxel>(voxel: V) -> u32 {
    if voxel.visibilty() == VoxelVisibility::Empty {
        0
    } else {
        1
    }
}

pub type Chunk32 = Chunk;

// pub const CHUNK_SIDE_64: i32 = 64;
// pub const CHUNK_VOLUME_64: usize = (CHUNK_SIDE_64 * CHUNK_SIDE_64 * CHUNK_SIDE_64) as usize;
// type Shape64 = ConstShape3u32<64, 64, 64>;

// pub struct Chunk64 {
//     voxels: [VoxelId; CHUNK_VOLUME_64],
//     is_solid: bool,
// }
