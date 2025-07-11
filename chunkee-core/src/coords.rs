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
pub fn camera_vec3_to_cv(pos: Vec3) -> ChunkVector {
    wv_to_cv(pos.floor().as_ivec3())
}

// #[inline(always)]
// pub fn cv_to_col_idx(cv: IVec3) -> usize {
//     (cv.y + WORLD_CHUNK_HEIGHT / 2) as usize
// }
