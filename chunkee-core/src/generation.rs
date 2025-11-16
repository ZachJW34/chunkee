use crate::{chunk::Chunk, coords::WorldVector};

pub trait VoxelGenerator: Send + Sync {
    fn apply(&self, chunk_start: WorldVector, chunk: &mut Chunk, voxel_size: f32);
}
