use glam::{IVec3, Vec3};

use crate::coords::{CHUNK_SIZE, cv_to_wv, vec3_wv_to_cv};

// const LOD1_DIST: f32 = 8.0 * CHUNK_SIZE as f32;
// const LOD2_DIST: f32 = 16.0 * CHUNK_SIZE as f32;
// const LOD3_DIST: f32 = 64.0 * CHUNK_SIZE as f32;

// const LOD1_DIST_SQ: f32 = LOD1_DIST * LOD1_DIST; // e.g., 128*128 = 16384
// const LOD2_DIST_SQ: f32 = LOD2_DIST * LOD2_DIST; // e.g., 256*256 = 65536
// const LOD3_DIST_SQ: f32 = LOD3_DIST * LOD3_DIST; // e.g., 256*256 = 65536

// pub fn calc_lod(cv: ChunkVector, camera_pos: Vec3) -> LOD {
//     let distance_sq = cv_camera_distance_sq(cv, camera_pos);

//     if distance_sq < LOD1_DIST_SQ {
//         1
//     } else if distance_sq < LOD2_DIST_SQ {
//         2
//     } else {
//         3
//     }
// }

// pub fn calc_lod(cv: ChunkVector, camera_pos: Vec3) -> u8 {
// let distance_sq = cv_camera_distance_sq(cv, camera_pos);

// if distance_sq < LOD1_DIST_SQ {
//     1
// } else if distance_sq < LOD2_DIST_SQ {
//     2
// } else if distance_sq < LOD3_DIST_SQ {
//     3
// } else {
//     4
// }
//
// 1
// }

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    #[inline]
    pub fn distance_to(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }

    pub fn normalize(mut self) -> Self {
        let magnitude = self.normal.length();
        self.normal /= magnitude;
        self.d /= magnitude;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

#[derive(Debug, Clone, Copy)]
pub struct CameraData {
    pub pos: Vec3,
    pub frustum: Frustum,
    pub forward: Vec3,
}

impl Frustum {
    // TODO: Investigate frustum check when moving backwards, something is off
    pub fn is_chunk_in_frustum(&self, cv: IVec3, voxel_size: f32) -> bool {
        let min_corner = cv_to_wv(cv).as_vec3() * voxel_size;
        let max_corner = min_corner + Vec3::splat(CHUNK_SIZE as f32 * voxel_size);

        for plane in &self.planes {
            let p_vertex = Vec3::new(
                if plane.normal.x > 0.0 {
                    max_corner.x
                } else {
                    min_corner.x
                },
                if plane.normal.y > 0.0 {
                    max_corner.y
                } else {
                    min_corner.y
                },
                if plane.normal.z > 0.0 {
                    max_corner.z
                } else {
                    min_corner.z
                },
            );

            if plane.distance_to(p_vertex) < 0.0 {
                return false;
            }
        }

        true
    }
}

pub fn cv_camera_distance_sq(cv: IVec3, camera_pos: Vec3, voxel_size: f32) -> f32 {
    (cv_to_wv(cv).as_vec3() * voxel_size).distance_squared(camera_pos)
}

pub fn compute_priority(cv: IVec3, camera_data: &CameraData, voxel_size: f32) -> u32 {
    let camera_cv = vec3_wv_to_cv(camera_data.pos, voxel_size);
    let delta = camera_cv - cv;
    if delta.x.abs() <= 2 && delta.y.abs() <= 2 && delta.z.abs() <= 2 {
        return 0;
    }

    let distance_sq = cv.distance_squared(camera_cv);
    let not_in_frustum = !camera_data.frustum.is_chunk_in_frustum(cv, voxel_size);

    (distance_sq as u32) + (not_in_frustum as u32) * 100000
}

pub fn calc_total_chunks(radius: u32) -> u32 {
    ((radius * 2) + 1).pow(3)
}
