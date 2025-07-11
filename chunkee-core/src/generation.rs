use crate::{chunk::ChunkLOD, coords::WorldVector};

pub trait VoxelGenerator: Send + Sync {
    fn apply(&self, chunk_start: WorldVector, chunk: &mut ChunkLOD);
}
