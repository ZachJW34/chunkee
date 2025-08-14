use glam::{IVec3, Vec3};

// pub const WORLD_CHUNK_HEIGHT: i32 = 128;
pub const CHUNK_SIZE: i32 = 32;
pub const CHUNK_VOLUME: i32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const CHUNK_SIZE_VEC: IVec3 = IVec3::splat(CHUNK_SIZE);

pub type WorldVector = IVec3;
pub type ChunkVector = IVec3;
pub type LocalVector = IVec3;

#[inline(always)]
pub fn wv_to_cv(wv: WorldVector) -> ChunkVector {
    wv.div_euclid(CHUNK_SIZE_VEC)
}

#[inline(always)]
pub fn cv_to_wv(cv: ChunkVector) -> WorldVector {
    cv * CHUNK_SIZE_VEC
}

#[inline(always)]
pub fn wv_to_lv(wv: WorldVector) -> LocalVector {
    wv.rem_euclid(CHUNK_SIZE_VEC)
}

#[inline(always)]
pub fn cv_lv_to_wv(cv: ChunkVector, lv: LocalVector) -> WorldVector {
    (cv * CHUNK_SIZE_VEC) + lv
}

#[inline(always)]
pub fn lv_to_idx(lv: LocalVector) -> usize {
    (lv.x + lv.y * CHUNK_SIZE + lv.z * CHUNK_SIZE * CHUNK_SIZE) as usize
}

#[inline(always)]
pub fn idx_to_lv(idx: usize) -> IVec3 {
    let idx = idx as i32;
    IVec3 {
        x: (idx % CHUNK_SIZE),
        y: (idx / CHUNK_SIZE) % CHUNK_SIZE,
        z: idx / (CHUNK_SIZE * CHUNK_SIZE),
    }
}

#[inline(always)]
pub fn vec3_wv_to_cv(pos: Vec3, voxel_size: f32) -> ChunkVector {
    let world_voxel_pos = (pos / voxel_size).floor().as_ivec3();
    wv_to_cv(world_voxel_pos)
}

pub const NEIGHBOR_OFFSETS: [IVec3; 27] = [
    IVec3::new(-1, -1, -1),
    IVec3::new(-1, -1, 0),
    IVec3::new(-1, -1, 1),
    IVec3::new(-1, 0, -1),
    IVec3::new(-1, 0, 0),
    IVec3::new(-1, 0, 1),
    IVec3::new(-1, 1, -1),
    IVec3::new(-1, 1, 0),
    IVec3::new(-1, 1, 1),
    IVec3::new(0, -1, -1),
    IVec3::new(0, -1, 0),
    IVec3::new(0, -1, 1),
    IVec3::new(0, 0, -1),
    IVec3::new(0, 0, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 1, -1),
    IVec3::new(0, 1, 0),
    IVec3::new(0, 1, 1),
    IVec3::new(1, -1, -1),
    IVec3::new(1, -1, 0),
    IVec3::new(1, -1, 1),
    IVec3::new(1, 0, -1),
    IVec3::new(1, 0, 0),
    IVec3::new(1, 0, 1),
    IVec3::new(1, 1, -1),
    IVec3::new(1, 1, 0),
    IVec3::new(1, 1, 1),
];

// #[inline(always)]
// pub fn cv_to_col_idx(cv: IVec3) -> usize {
//     (cv.y + WORLD_CHUNK_HEIGHT / 2) as usize
// }

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: IVec3,
    pub max: IVec3,
}
