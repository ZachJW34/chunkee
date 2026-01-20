use crate::{chunk::Chunk, clipmap::ChunkKey};

pub trait VoxelGenerator: Send + Sync {
    fn apply(&self, chunk_key: ChunkKey, chunk: &mut Chunk, voxel_size: f32);
}
