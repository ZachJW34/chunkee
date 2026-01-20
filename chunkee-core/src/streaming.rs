use glam::Vec3;

use crate::{
    clipmap::ChunkKey,
    coords::{CHUNK_SIZE, wv_to_wp},
};

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
    pub fn is_chunk_in_frustum(&self, min_corner: Vec3, chunk_world_size: f32) -> bool {
        for plane in &self.planes {
            let p_vertex = Vec3::new(
                if plane.normal.x > 0.0 {
                    min_corner.x + chunk_world_size
                } else {
                    min_corner.x
                },
                if plane.normal.y > 0.0 {
                    min_corner.y + chunk_world_size
                } else {
                    min_corner.y
                },
                if plane.normal.z > 0.0 {
                    min_corner.z + chunk_world_size
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

pub fn compute_priority(key: ChunkKey, camera_data: &CameraData, voxel_size: f32) -> u32 {
    let wp = wv_to_wp(key.to_wv(), voxel_size);

    let chunk_world_size = (CHUNK_SIZE as f32) * (key.lod_scale() as f32) * voxel_size;

    let delta = camera_data.pos - wp;
    let safety_margin = chunk_world_size * 1.5;

    if delta.x.abs() <= safety_margin
        && delta.y.abs() <= safety_margin
        && delta.z.abs() <= safety_margin
    {
        return 0;
    }

    let not_in_frustum = !camera_data
        .frustum
        .is_chunk_in_frustum(wp, chunk_world_size);

    let distance_sq = wp.distance_squared(camera_data.pos);

    (distance_sq as u32) + (not_in_frustum as u32) * 1000000
}

pub fn calc_total_chunks(radius: u32) -> u32 {
    ((radius * 2) + 1).pow(3)
}
