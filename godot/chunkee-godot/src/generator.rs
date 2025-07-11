use std::sync::Arc;

use crate::voxels::MyVoxels;
use chunkee_core::{
    chunk::ChunkLOD,
    coords::{ChunkVector, LocalVector, WorldVector, cv_to_wv, wv_to_cv},
    generation::VoxelGenerator,
    glam::IVec2,
};
use dashmap::DashMap;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

const SEA_LEVEL: i32 = 0;
const BEACH_HEIGHT: i32 = 2;

#[derive(Clone)]
struct ChunkColumnData {
    height_map: [f64; 32 * 32],
    humidity_map: [f64; 32 * 32],
}

pub struct WorldGenerator {
    continent_noise: Fbm<Perlin>,
    terrain_noise: Fbm<Perlin>,
    humidity_noise: Fbm<Perlin>,
    cave_noise_1: Fbm<Perlin>,
    cave_noise_2: Fbm<Perlin>,
    data_cache: DashMap<IVec2, Arc<ChunkColumnData>>,
}

impl WorldGenerator {
    pub fn new() -> Self {
        let continent_noise = Fbm::new(1000).set_frequency(0.001).set_octaves(6);

        let terrain_noise = Fbm::new(2000)
            .set_frequency(0.003)
            .set_octaves(7)
            .set_lacunarity(2.2)
            .set_persistence(0.5);

        let humidity_noise = Fbm::new(4000).set_frequency(0.005).set_octaves(4);

        // More aggressive frequency for larger cave systems
        let cave_noise_1 = Fbm::new(3000).set_frequency(0.007).set_octaves(5);
        let cave_noise_2 = Fbm::new(3001).set_frequency(0.007).set_octaves(5);

        Self {
            continent_noise,
            terrain_noise,
            humidity_noise,
            cave_noise_1,
            cave_noise_2,
            data_cache: DashMap::new(),
        }
    }

    fn get_or_compute_column_data(&self, cv: ChunkVector) -> Arc<ChunkColumnData> {
        let column_cv = IVec2::new(cv.x, cv.z);
        if let Some(data) = self.data_cache.get(&column_cv) {
            return data.clone();
        }

        let mut height_map = [0.0; 32 * 32];
        let mut humidity_map = [0.0; 32 * 32];
        let chunk_world_pos = cv_to_wv(cv);

        for x in 0..32 {
            for z in 0..32 {
                let world_x = (chunk_world_pos.x + x) as f64;
                let world_z = (chunk_world_pos.z + z) as f64;
                let idx = x as usize + z as usize * 32;

                let humidity = self.humidity_noise.get([world_x, world_z]);
                humidity_map[idx] = humidity;

                let cont_val = self.continent_noise.get([world_x, world_z]);
                let continent_mask = ((cont_val + 1.0) / 2.0).powf(0.9).max(0.0).min(1.0);
                let terrain_val = self.terrain_noise.get([world_x, world_z]);
                let elevation = (terrain_val + 1.0) / 2.0;
                let final_terrain = elevation.powf(2.5);
                let terrain_height_scaler = 350.0;
                let base_height = (SEA_LEVEL as f64) - 30.0;
                let land_height = 45.0 * continent_mask;
                let mountain_height = final_terrain * terrain_height_scaler * continent_mask;
                let terrain_height = base_height + land_height + mountain_height;
                height_map[idx] = terrain_height;
            }
        }

        let data = Arc::new(ChunkColumnData {
            height_map,
            humidity_map,
        });
        self.data_cache.insert(column_cv, data.clone());
        data
    }
}

impl VoxelGenerator for WorldGenerator {
    fn apply(&self, chunk_start: WorldVector, chunk: &mut ChunkLOD) {
        let side = chunk.size();
        let lod_scale_factor = chunk.lod_scale_factor() as i32;

        let column_data = self.get_or_compute_column_data(wv_to_cv(chunk_start));

        for x in 0..side {
            for z in 0..side {
                let cache_x = (x * lod_scale_factor) as usize;
                let cache_z = (z * lod_scale_factor) as usize;
                let cache_idx = cache_x + cache_z * 32;

                let terrain_height_i32 = column_data.height_map[cache_idx].floor() as i32;
                let humidity = column_data.humidity_map[cache_idx];

                for y in 0..side {
                    let lv = LocalVector::new(x, y, z);

                    let wv_y_bottom = chunk_start.y + y * lod_scale_factor;
                    let wv_y_top = wv_y_bottom + lod_scale_factor - 1;

                    if wv_y_bottom > terrain_height_i32 && wv_y_bottom > SEA_LEVEL {
                        continue;
                    }

                    let mut block_to_place = MyVoxels::Air;

                    if wv_y_bottom > SEA_LEVEL {
                        if terrain_height_i32 >= wv_y_bottom && terrain_height_i32 <= wv_y_top {
                            if terrain_height_i32 <= SEA_LEVEL + BEACH_HEIGHT {
                                block_to_place = MyVoxels::Sand;
                            } else if humidity < -0.3 {
                                block_to_place = MyVoxels::Sand;
                            } else if terrain_height_i32 > 160 {
                                block_to_place = MyVoxels::Snow;
                            } else if terrain_height_i32 > 90 {
                                block_to_place = MyVoxels::Stone;
                            } else {
                                block_to_place = MyVoxels::Grass;
                            }
                        } else if wv_y_top < terrain_height_i32 {
                            if wv_y_top > terrain_height_i32 - (3 * lod_scale_factor) {
                                block_to_place = MyVoxels::Dirt;
                            } else {
                                block_to_place = MyVoxels::Stone;
                            }
                        }
                    } else {
                        if wv_y_top < terrain_height_i32 {
                            block_to_place = MyVoxels::Sand;
                        } else {
                            block_to_place = MyVoxels::Water;
                        }
                    }

                    // We apply carving to any block that isn't water or air.
                    if block_to_place != MyVoxels::Water && block_to_place != MyVoxels::Air {
                        let wv = WorldVector::new(
                            chunk_start.x + x * lod_scale_factor,
                            chunk_start.y + y * lod_scale_factor,
                            chunk_start.z + z * lod_scale_factor,
                        );

                        // Combine noise values. Multiplying them creates more varied, organic shapes.
                        let noise_val_1 = self.cave_noise_1.get([wv.x as f64, wv.y as f64]);
                        let noise_val_2 = self.cave_noise_2.get([wv.y as f64, wv.z as f64]);
                        let cave_density = noise_val_1.abs() * noise_val_2.abs();

                        // Dynamic threshold: caves are bigger deep down, and smaller near the surface.
                        let depth = (terrain_height_i32 - wv.y).max(0) as f64;
                        // The '40.0' controls how quickly caves shrink as they rise.
                        // The '0.1' is the base 'openness' of caves deep down.
                        let threshold = 0.1 + (1.0 - (depth / 40.0)).clamp(0.0, 1.0).powi(2) * 0.3;

                        if cave_density > threshold {
                            block_to_place = MyVoxels::Air;
                        }
                    }

                    if block_to_place != MyVoxels::Air {
                        chunk.set_block::<MyVoxels>(lv, block_to_place);
                    }
                }
            }
        }
    }
}
