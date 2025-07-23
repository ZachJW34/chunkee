use chunkee_core::{
    block::{Block, BlockTypeId, ChunkeeVoxel, TextureMapping, VoxelCollision},
    block_mesh::VoxelVisibility,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MyVoxels {
    Air = 0,
    Grass,
    Dirt,
    Stone,
    Snow,
    Water,
    Sand,
}

impl From<MyVoxels> for BlockTypeId {
    fn from(voxel: MyVoxels) -> Self {
        voxel as BlockTypeId
    }
}

impl From<BlockTypeId> for MyVoxels {
    fn from(id: BlockTypeId) -> Self {
        match id {
            0 => MyVoxels::Air,
            1 => MyVoxels::Grass,
            2 => MyVoxels::Dirt,
            3 => MyVoxels::Stone,
            4 => MyVoxels::Snow,
            5 => MyVoxels::Water,
            6 => MyVoxels::Sand,
            // Fallback
            _ => MyVoxels::Air,
        }
    }
}

impl Default for MyVoxels {
    fn default() -> Self {
        MyVoxels::Air
    }
}

impl Block for MyVoxels {
    fn name(&self) -> &'static str {
        match self {
            MyVoxels::Air => "Air",
            MyVoxels::Grass => "Grass",
            MyVoxels::Dirt => "Dirt",
            MyVoxels::Stone => "Stone",
            MyVoxels::Snow => "Snow",
            MyVoxels::Water => "Water",
            MyVoxels::Sand => "Sand",
        }
    }

    fn visibilty(&self) -> VoxelVisibility {
        match self {
            MyVoxels::Air => VoxelVisibility::Empty,
            MyVoxels::Water => VoxelVisibility::Translucent,
            _ => VoxelVisibility::Opaque,
        }
    }

    fn texture_mapping(&self) -> TextureMapping {
        match self {
            MyVoxels::Grass => TextureMapping::All(0),
            MyVoxels::Dirt => TextureMapping::All(1),
            MyVoxels::Stone => TextureMapping::All(2),
            MyVoxels::Snow => TextureMapping::All(3),
            MyVoxels::Water => TextureMapping::All(4),
            MyVoxels::Sand => TextureMapping::All(5),
            _ => TextureMapping::None,
        }
    }

    fn collision(&self) -> chunkee_core::block::VoxelCollision {
        match self {
            MyVoxels::Air | MyVoxels::Water => VoxelCollision::None,
            _ => VoxelCollision::Solid,
        }
    }
}

impl ChunkeeVoxel for MyVoxels {}
