use std::{collections::HashMap, marker::PhantomData};

use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use glam::IVec3;

use crate::{
    block::{BLOCK_FACES, ChunkeeVoxel, Rotation, VoxelId, VoxelVisibility},
    coords::{ChunkVector, LocalVector},
};

type Shape32 = ConstShape3u32<32, 32, 32>;
type Shape16 = ConstShape3u32<16, 16, 16>;
type Shape8 = ConstShape3u32<8, 8, 8>;

#[derive(Debug, Clone, Copy)]
pub struct SizedChunk<const VOLUME: usize, S: ConstShape<3, Coord = u32>> {
    pub(crate) voxels: [VoxelId; VOLUME],
    pub(crate) solid_count: u32,
    pub(crate) is_dirty: bool,
    _shape: PhantomData<S>,
}

impl<const VOLUME: usize, S: ConstShape<3, Coord = u32>> SizedChunk<VOLUME, S> {
    pub fn new() -> Self {
        Self {
            voxels: [VoxelId::AIR; VOLUME],
            solid_count: 0,
            is_dirty: false,
            _shape: PhantomData,
        }
    }

    fn linearize(&self, lv: IVec3) -> usize {
        S::linearize(lv.as_uvec3().to_array()) as usize
    }

    pub fn get_voxel(&self, lv: IVec3) -> VoxelId {
        self.voxels[self.linearize(lv)]
    }

    pub fn set_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector, new_id: VoxelId) -> VoxelId {
        let idx = self.linearize(lv);
        let new_voxel = V::from(new_id.type_id());
        let old_voxel_id = self.voxels[idx];
        let old_voxel = V::from(old_voxel_id.type_id());

        self.voxels[idx] = new_id;
        self.solid_count -= Self::solid_value(old_voxel);
        self.solid_count += Self::solid_value(new_voxel);
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
        self.solid_count as usize == VOLUME
    }

    // Don't think this makes sense for translucent values
    fn solid_value<V: ChunkeeVoxel>(voxel: V) -> u32 {
        if voxel.visibilty() == VoxelVisibility::Empty {
            0
        } else {
            1
        }
    }
}

pub type Chunk32 = SizedChunk<{ 32 * 32 * 32 }, Shape32>;
pub type Chunk16 = SizedChunk<{ 16 * 16 * 16 }, Shape16>;
pub type Chunk8 = SizedChunk<{ 8 * 8 * 8 }, Shape8>;

pub type LOD = u8;

#[derive(Debug, Clone, Copy)]
pub enum ChunkLOD {
    LOD1(Chunk32),
    LOD2(Chunk16),
    LOD3(Chunk8),
}

impl ChunkLOD {
    pub fn new(lod: u8) -> Self {
        match lod {
            1 => Self::LOD1(Chunk32::new()),
            2 => Self::LOD2(Chunk16::new()),
            3 => Self::LOD3(Chunk8::new()),
            _ => panic!("Unsupported LOD level"),
        }
    }

    pub fn get_voxel(&self, lv: IVec3) -> VoxelId {
        match self {
            Self::LOD1(c) => c.get_voxel(lv),
            Self::LOD2(c) => c.get_voxel(lv),
            Self::LOD3(c) => c.get_voxel(lv),
        }
    }

    pub fn set_voxel<V: ChunkeeVoxel>(&mut self, lv: LocalVector, new_id: VoxelId) -> VoxelId {
        match self {
            Self::LOD1(c) => c.set_voxel::<V>(lv, new_id),
            Self::LOD2(c) => c.set_voxel::<V>(lv, new_id),
            Self::LOD3(c) => c.set_voxel::<V>(lv, new_id),
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
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::LOD1(c) => c.is_empty(),
            Self::LOD2(c) => c.is_empty(),
            Self::LOD3(c) => c.is_empty(),
        }
    }

    pub fn is_solid(&self) -> bool {
        match self {
            Self::LOD1(c) => c.is_solid(),
            Self::LOD2(c) => c.is_solid(),
            Self::LOD3(c) => c.is_solid(),
        }
    }

    pub fn lod_level(&self) -> u8 {
        match self {
            ChunkLOD::LOD1(_) => 1,
            ChunkLOD::LOD2(_) => 2,
            ChunkLOD::LOD3(_) => 3,
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

#[derive(Debug, Clone)]
pub struct Deltas(pub HashMap<ChunkVector, VoxelId>);

impl Default for Deltas {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub fn neighbors_of(cv: IVec3) -> [ChunkVector; 6] {
    BLOCK_FACES.map(|face| cv + face.into_normal())
}
