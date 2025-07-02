use glam::{IVec3, Vec3};

use crate::{
    coords::{CHUNK_SIZE, ChunkVector, cv_to_wv, wv_to_cv},
    pipeline::{PipelineState, PrioritizedWorkItem, WorkQueue, WorldChunk, WorldChunks},
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

// const UPDATE_GRID_CELL_SIZE: f32 = 16.0;

pub struct ChunkStreamer {
    prev_cam_cv: Option<IVec3>,
}

impl ChunkStreamer {
    pub fn new() -> Self {
        Self { prev_cam_cv: None }
    }

    pub fn preprocess_chunks(
        &mut self,
        camera_data: &CameraData,
        radius: u32,
        world_chunks: &WorldChunks,
        load_queue: &WorkQueue,
        unload_queue: &WorkQueue,
    ) {
        let curr_cam_cv = wv_to_cv((camera_data.pos.floor()).as_ivec3());
        if let Some(previous_cell) = self.prev_cam_cv {
            if previous_cell == curr_cam_cv {
                return;
            }
        }

        println!("Entered curr_cam_cv: {curr_cam_cv:?}");

        let chunkvs_to_unload = world_chunks
            .iter()
            .filter_map(|e| {
                let cv = *e.key();
                if !is_chunk_in_range(cv, camera_data.pos, radius)
                    && e.state != PipelineState::NeedsUnload
                {
                    return Some(cv);
                }
                None
            })
            .collect::<Vec<_>>();

        let mut unload_queue: std::sync::MutexGuard<
            '_,
            std::collections::BinaryHeap<PrioritizedWorkItem>,
        > = unload_queue.lock().unwrap();
        for cv in chunkvs_to_unload {
            world_chunks.alter(&cv, |_, mut cs| {
                cs.state = PipelineState::NeedsUnload;
                cs
            });
            let priority = calculate_chunk_priority(cv, camera_data);
            unload_queue.push(PrioritizedWorkItem { priority, cv: cv });
        }

        let chunkvs_to_load = get_chunks_in_range(camera_data.pos, radius);

        let mut load_queue = load_queue.lock().unwrap();
        for cv in chunkvs_to_load {
            if !world_chunks.contains_key(&cv) {
                world_chunks.insert(cv, WorldChunk::default());
                let priority = calculate_chunk_priority(cv, camera_data);
                load_queue.push(PrioritizedWorkItem { priority, cv: cv });
            }
        }

        self.prev_cam_cv = Some(curr_cam_cv);
    }
}

pub fn get_chunks_in_range(camera_pos: Vec3, radius: u32) -> Vec<IVec3> {
    let camera_pos_i = camera_pos.as_ivec3();
    let camera_cv = wv_to_cv(camera_pos_i);

    let radius = radius as i32;
    let mut chunk_vs = Vec::with_capacity(((radius * 2 + 1).pow(3)) as usize);

    for y in -radius..=radius {
        for z in -radius..=radius {
            for x in -radius..=radius {
                let chunk_v = camera_cv + IVec3::new(x, y, z);
                chunk_vs.push(chunk_v);
            }
        }
    }

    chunk_vs.sort_by_key(|cv| (camera_cv - cv).length_squared());
    chunk_vs
}

pub fn is_chunk_in_range(cv: IVec3, camera_pos: Vec3, radius: u32) -> bool {
    let camera_cv = wv_to_cv(camera_pos.as_ivec3());

    let dist_x = (cv.x - camera_cv.x).abs();
    let dist_y = (cv.y - camera_cv.y).abs();
    let dist_z = (cv.z - camera_cv.z).abs();

    let radius_i32 = radius as i32;

    dist_x <= radius_i32 && dist_y <= radius_i32 && dist_z <= radius_i32
}

pub fn calculate_chunk_priority(cv: ChunkVector, camera_data: &CameraData) -> u32 {
    let distance_sq = cv_to_wv(cv).as_vec3().distance_squared(camera_data.pos);
    let not_in_frustum = !camera_data.frustum.is_chunk_in_frustum(cv);

    (distance_sq as u32) + (not_in_frustum as u32) * 10000
}
