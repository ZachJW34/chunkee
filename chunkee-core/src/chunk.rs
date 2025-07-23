use std::marker::PhantomData;

use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use glam::IVec3;

use crate::{
    block::{ChunkeeVoxel, Rotation, VoxelId, VoxelVisibility},
    coords::LocalVector,
};

type Shape32 = ConstShape3u32<32, 32, 32>;
type Shape16 = ConstShape3u32<16, 16, 16>;
type Shape8 = ConstShape3u32<8, 8, 8>;
type Shape4 = ConstShape3u32<4, 4, 4>;

#[derive(Debug, Clone, Copy)]
pub struct SizedChunk<const VOLUME: usize, S: ConstShape<3, Coord = u32>> {
    pub(crate) voxels: [VoxelId; VOLUME],
    pub(crate) solid_count: u32,
    _shape: PhantomData<S>,
}

impl<const VOLUME: usize, S: ConstShape<3, Coord = u32>> SizedChunk<VOLUME, S> {
    pub const VOL: usize = VOLUME;

    pub fn new() -> Self {
        Self {
            voxels: [VoxelId::AIR; VOLUME],
            solid_count: 0,
            _shape: PhantomData,
        }
    }

    fn linearize(&self, lv: IVec3) -> usize {
        S::linearize(lv.as_uvec3().to_array()) as usize
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
        self.solid_count as usize == VOLUME
    }
}

pub fn voxel_solid_value<V: ChunkeeVoxel>(voxel: V) -> u32 {
    if voxel.visibilty() == VoxelVisibility::Empty {
        0
    } else {
        1
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Chunk<const VOLUME: usize, S: ConstShape<3, Coord = u32>> {
    Uniform(VoxelId),
    Sized(SizedChunk<VOLUME, S>),
}

impl<const VOLUME: usize, S: ConstShape<3, Coord = u32>> Chunk<VOLUME, S> {
    pub const VOL: usize = VOLUME;

    pub fn new() -> Self {
        Chunk::Uniform(VoxelId::AIR)
    }

    pub fn new_uniform(uniform: VoxelId) -> Self {
        Chunk::Uniform(uniform)
    }

    pub fn get_voxel(&self, lv: LocalVector) -> VoxelId {
        match self {
            Chunk::Uniform(voxel_id) => *voxel_id,
            Chunk::Sized(sized_chunk) => sized_chunk.get_voxel(lv),
        }
    }

    pub fn set_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector, new_id: VoxelId) -> VoxelId {
        match self {
            Chunk::Uniform(uniform) => {
                if *uniform == new_id {
                    return new_id;
                }

                let mut sized_chunk = SizedChunk {
                    voxels: [*uniform; VOLUME],
                    solid_count: 0,
                    _shape: PhantomData,
                };
                sized_chunk.solid_count =
                    voxel_solid_value(uniform.to_voxel::<V>()) * (VOLUME as u32);
                let old_voxel = sized_chunk.set_voxel::<V>(lv, new_id);
                *self = Chunk::Sized(sized_chunk);

                old_voxel
            }
            Chunk::Sized(sized_chunk) => sized_chunk.set_voxel::<V>(lv, new_id),
        }
    }

    pub fn is_empty<V: ChunkeeVoxel>(&self) -> bool {
        match self {
            Chunk::Uniform(uniform) => {
                uniform.to_voxel::<V>().visibilty() == VoxelVisibility::Empty
            }
            Chunk::Sized(sized_chunk) => sized_chunk.is_empty(),
        }
    }

    pub fn is_solid<V: ChunkeeVoxel>(&self) -> bool {
        match self {
            Chunk::Uniform(uniform) => {
                uniform.to_voxel::<V>().visibilty() != VoxelVisibility::Empty
            }
            Chunk::Sized(sized_chunk) => sized_chunk.is_solid(),
        }
    }
}

pub type Chunk32 = Chunk<{ 32 * 32 * 32 }, Shape32>;
pub type Chunk16 = Chunk<{ 16 * 16 * 16 }, Shape16>;
pub type Chunk8 = Chunk<{ 8 * 8 * 8 }, Shape8>;
pub type Chunk4 = Chunk<{ 4 * 4 * 4 }, Shape4>;

pub type LOD = u8;

#[derive(Debug, Clone, Copy)]
pub enum ChunkLOD {
    LOD1(Chunk32),
    LOD2(Chunk16),
    LOD3(Chunk8),
    LOD4(Chunk4),
}

impl ChunkLOD {
    pub fn new(lod: LOD) -> Self {
        match lod {
            1 => Self::LOD1(Chunk32::new()),
            2 => Self::LOD2(Chunk16::new()),
            3 => Self::LOD3(Chunk8::new()),
            4 => Self::LOD4(Chunk4::new()),
            _ => panic!("Unsupported LOD level"),
        }
    }

    pub fn new_uniform(lod: LOD, uniform: VoxelId) -> Self {
        match lod {
            1 => Self::LOD1(Chunk32::new_uniform(uniform)),
            2 => Self::LOD2(Chunk16::new_uniform(uniform)),
            3 => Self::LOD3(Chunk8::new_uniform(uniform)),
            4 => Self::LOD4(Chunk4::new_uniform(uniform)),
            _ => panic!("Unsupported LOD level"),
        }
    }

    pub fn get_voxel(&self, lv: LocalVector) -> VoxelId {
        match self {
            ChunkLOD::LOD1(chunk) => chunk.get_voxel(lv),
            ChunkLOD::LOD2(chunk) => chunk.get_voxel(lv),
            ChunkLOD::LOD3(chunk) => chunk.get_voxel(lv),
            ChunkLOD::LOD4(chunk) => chunk.get_voxel(lv),
        }
    }

    pub fn set_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector, new_id: VoxelId) -> VoxelId {
        match self {
            Self::LOD1(c) => c.set_voxel::<V>(lv, new_id),
            Self::LOD2(c) => c.set_voxel::<V>(lv, new_id),
            Self::LOD3(c) => c.set_voxel::<V>(lv, new_id),
            Self::LOD4(c) => c.set_voxel::<V>(lv, new_id),
        }
    }

    pub fn set_block<V: ChunkeeVoxel>(&mut self, lv: LocalVector, block: V) -> VoxelId {
        let new_id = VoxelId::new(block.into(), Rotation::default());
        self.set_voxel::<V>(lv, new_id)
    }

    pub fn size(&self) -> i32 {
        match self {
            Self::LOD1(_) => 32,
            Self::LOD2(_) => 16,
            Self::LOD3(_) => 8,
            Self::LOD4(_) => 4,
        }
    }

    pub fn is_empty<V: ChunkeeVoxel>(&self) -> bool {
        match self {
            Self::LOD1(c) => c.is_empty::<V>(),
            Self::LOD2(c) => c.is_empty::<V>(),
            Self::LOD3(c) => c.is_empty::<V>(),
            Self::LOD4(c) => c.is_empty::<V>(),
        }
    }

    pub fn is_solid<V: ChunkeeVoxel>(&self) -> bool {
        match self {
            Self::LOD1(c) => c.is_solid::<V>(),
            Self::LOD2(c) => c.is_solid::<V>(),
            Self::LOD3(c) => c.is_solid::<V>(),
            Self::LOD4(c) => c.is_solid::<V>(),
        }
    }

    pub fn lod_level(&self) -> LOD {
        match self {
            ChunkLOD::LOD1(_) => 1,
            ChunkLOD::LOD2(_) => 2,
            ChunkLOD::LOD3(_) => 3,
            ChunkLOD::LOD4(_) => 4,
        }
    }

    pub fn is_voxel_on_edge(&self, lv: IVec3) -> bool {
        let min = IVec3::ZERO;
        let max = IVec3::splat(self.size() - 1);

        (lv.cmpeq(min) | lv.cmpeq(max)).any()
    }

    pub fn lod_scale_factor(&self) -> f32 {
        2.0f32.powi(self.lod_level() as i32 - 1)
    }
}
