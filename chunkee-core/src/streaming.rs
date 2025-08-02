use glam::{IVec3, Vec3};

use crate::coords::{CHUNK_SIZE, ChunkVector, camera_vec3_to_cv, cv_to_wv, wv_to_cv};

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
}

impl Frustum {
    // TODO: Investigate frustum check when moving backwards, something is off
    pub fn is_chunk_in_frustum(&self, cv: ChunkVector) -> bool {
        let min_corner = cv_to_wv(cv).as_vec3();
        let max_corner = min_corner + Vec3::splat(CHUNK_SIZE as f32);

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

// pub struct ChunkStreamResult {
//     pub cv: ChunkVector,
//     pub lod: LOD,
//     pub priority: u32,
// }

// pub struct ChunkStreamer {
//     previous_camera_cv: IVec3,
//     radius_xz: u32,
//     radius_y: u32,
// }

// impl ChunkStreamer {
//     pub fn new(radius_xz: u32, radius_y: u32) -> Self {
//         Self {
//             previous_camera_cv: IVec3::MAX,
//             radius_xz,
//             radius_y,
//         }
//     }

//     pub fn stream_chunks(&mut self, camera_data: &CameraData) -> Option<Vec<ChunkStreamResult>> {
//         let curr_cam_cv = wv_to_cv((camera_data.pos.floor()).as_ivec3());
//         if self.previous_camera_cv == curr_cam_cv {
//             return None;
//         }
//         println!("Entered curr_cam_cv: {curr_cam_cv:?}");

//         self.previous_camera_cv = curr_cam_cv;
//         let chunks_in_range = self.get_chunks_in_range(camera_data);

//         Some(chunks_in_range)
//     }

//     fn get_chunks_in_range(&self, camera_data: &CameraData) -> Vec<ChunkStreamResult> {
//         let camera_cv = wv_to_cv(camera_data.pos.as_ivec3());

//         let mut chunks_in_range =
//             Vec::with_capacity(calc_total_chunks(self.radius_xz, self.radius_y) as usize);
//         let radius_xz_i32 = self.radius_xz as i32;
//         let radius_i32_y = self.radius_y as i32;

//         for y in -radius_i32_y..=radius_i32_y {
//             for z in -radius_xz_i32..=radius_xz_i32 {
//                 for x in -radius_xz_i32..=radius_xz_i32 {
//                     let cv = camera_cv + IVec3::new(x, y, z);
//                     let lod = calc_lod(cv, camera_data.pos);
//                     let priority = calculate_chunk_priority(cv, camera_data);
//                     chunks_in_range.push(ChunkStreamResult { cv, lod, priority });
//                 }
//             }
//         }

//         chunks_in_range
//     }
// }

pub fn should_unload(cv: IVec3, camera_pos: Vec3, radius: u32) -> bool {
    let camera_cv = wv_to_cv(camera_pos.as_ivec3());

    let diff = (cv - camera_cv).abs();
    let radius_buffered = (radius + 1) as i32;
    
    diff.x > radius_buffered || diff.y > radius_buffered || diff.z > radius_buffered
}

pub fn cv_camera_distance_sq(cv: IVec3, camera_pos: Vec3) -> f32 {
    cv_to_wv(cv).as_vec3().distance_squared(camera_pos)
}

pub fn calculate_chunk_priority(cv: ChunkVector, camera_data: &CameraData) -> u32 {
    let camera_cv = camera_vec3_to_cv(camera_data.pos);
    let delta = camera_cv - cv;
    if delta.x.abs() <= 1 && delta.y.abs() <= 1 && delta.z.abs() <= 1 {
        return 0;
    }

    let distance_sq = cv_camera_distance_sq(cv, camera_data.pos);
    let not_in_frustum = !camera_data.frustum.is_chunk_in_frustum(cv);

    (distance_sq as u32) + (not_in_frustum as u32) * 100000
}

pub fn calc_total_chunks(radius_xz: u32, radius_y: u32) -> u32 {
    let size_xz = 2 * radius_xz + 1;
    let size_y = 2 * radius_y + 1;
    size_xz * size_xz * size_y
}
