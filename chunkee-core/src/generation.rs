use crate::{chunk::Chunk32, coords::WorldVector};

pub trait VoxelGenerator: Send + Sync {
    fn apply(&self, chunk_start: WorldVector, chunk: &mut Chunk32);
}
