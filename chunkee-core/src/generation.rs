use crate::{chunk::Chunk, coords::WorldVector};

pub trait VoxelGenerator: Send + Sync {
    fn apply(&self, origin_wv: WorldVector, chunk: &mut Chunk);
}
