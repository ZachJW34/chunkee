use std::sync::Arc;

use crate::voxels::MyVoxels;
use chunkee_core::{
    chunk::Chunk32,
    coords::{ChunkVector, LocalVector, WorldVector, cv_to_wv, wv_to_cv},
    generation::VoxelGenerator,
    glam::IVec2,
};
use dashmap::DashMap;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

const SEA_LEVEL: f32 = 3.0;
const BEACH_HEIGHT: f32 = 2.0;

#[derive(Clone)]
struct ChunkColumnData {
    height_map: [f64; 32 * 32],
    humidity_map: [f64; 32 * 32],
    biome_map: [f64; 32 * 32],
}

pub struct WorldGenerator {
    continent_noise: Fbm<Perlin>,
    terrain_noise: Fbm<Perlin>,
    humidity_noise: Fbm<Perlin>,
    cave_noise_1: Fbm<Perlin>,
    cave_noise_2: Fbm<Perlin>,
    biome_noise: Fbm<Perlin>,
    dune_noise: Fbm<Perlin>,
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
        let biome_noise = Fbm::new(5000).set_frequency(0.0005).set_octaves(3);
        let dune_noise = Fbm::new(6000)
            .set_frequency(0.008)
            .set_octaves(5)
            .set_persistence(0.4);
        let cave_noise_1 = Fbm::new(3000).set_frequency(0.007).set_octaves(5);
        let cave_noise_2 = Fbm::new(3001).set_frequency(0.007).set_octaves(5);

        Self {
            continent_noise,
            terrain_noise,
            humidity_noise,
            cave_noise_1,
            cave_noise_2,
            biome_noise,
            dune_noise,
            data_cache: DashMap::new(),
        }
    }

    // In `impl WorldGenerator`

    fn get_or_compute_column_data(&self, cv: ChunkVector, voxel_size: f32) -> Arc<ChunkColumnData> {
        let column_cv = IVec2::new(cv.x, cv.z);
        if let Some(data) = self.data_cache.get(&column_cv) {
            return data.clone();
        }

        let mut height_map = [0.0; 32 * 32];
        let mut humidity_map = [0.0; 32 * 32];
        let mut biome_map = [0.0; 32 * 32];
        let chunk_world_pos = cv_to_wv(cv);

        const DESERT_TRANSITION_START: f64 = 0.2;
        const DESERT_TRANSITION_END: f64 = 0.4;
        const TRANSITION_RANGE: f64 = DESERT_TRANSITION_END - DESERT_TRANSITION_START;

        for x in 0..32 {
            for z in 0..32 {
                let world_x = ((chunk_world_pos.x + x) as f32 * voxel_size) as f64;
                let world_z = ((chunk_world_pos.z + z) as f32 * voxel_size) as f64;
                let idx = x as usize + z as usize * 32;

                let biome_val = self.biome_noise.get([world_x, world_z]);
                biome_map[idx] = biome_val;

                // First, calculate the blend factor to see where we are.
                let blend_factor =
                    ((biome_val - DESERT_TRANSITION_START) / TRANSITION_RANGE).clamp(0.0, 1.0);

                if blend_factor == 0.0 {
                    // --- PURE MOUNTAIN BIOME ---
                    // We are fully in a mountain biome, so only do this calculation.
                    humidity_map[idx] = self.humidity_noise.get([world_x, world_z]);
                    let cont_val = self.continent_noise.get([world_x, world_z]);
                    let continent_mask = ((cont_val + 1.0) / 2.0).powf(0.9).max(0.0).min(1.0);
                    let terrain_val = self.terrain_noise.get([world_x, world_z]);
                    let elevation = (terrain_val + 1.0) / 2.0;
                    let final_terrain = elevation.powf(2.5);
                    let terrain_height_scaler = 350.0;
                    let base_height = (SEA_LEVEL as f64) - 30.0;
                    let land_height = 45.0 * continent_mask;
                    let mountain_height = final_terrain * terrain_height_scaler * continent_mask;
                    height_map[idx] = base_height + land_height + mountain_height;
                } else if blend_factor == 1.0 {
                    // --- PURE DESERT BIOME ---
                    // We are fully in a desert, so only do this calculation.
                    humidity_map[idx] = -0.5;
                    let dune_val = self.dune_noise.get([world_x, world_z]);
                    let normalized_dune = (dune_val + 1.0) / 2.0;
                    height_map[idx] = (SEA_LEVEL as f64) + normalized_dune * 25.0;
                } else {
                    // --- TRANSITION ZONE ---
                    // Only here, in the seam between biomes, do we calculate both and blend.
                    let mountain_humidity = 0.2;
                    let cont_val = self.continent_noise.get([world_x, world_z]);
                    let continent_mask = ((cont_val + 1.0) / 2.0).powf(0.9).max(0.0).min(1.0);
                    let terrain_val = self.terrain_noise.get([world_x, world_z]);
                    let elevation = (terrain_val + 1.0) / 2.0;
                    let final_terrain = elevation.powf(2.5);
                    let terrain_height_scaler = 350.0;
                    let base_height = (SEA_LEVEL as f64) - 30.0;
                    let land_height = 45.0 * continent_mask;
                    let mountain_height = final_terrain * terrain_height_scaler * continent_mask;
                    let mountain_terrain_height = base_height + land_height + mountain_height;

                    let desert_humidity = -0.5;
                    let dune_val = self.dune_noise.get([world_x, world_z]);
                    let normalized_dune = (dune_val + 1.0) / 2.0;
                    let desert_terrain_height = (SEA_LEVEL as f64) + normalized_dune * 25.0;

                    height_map[idx] = mountain_terrain_height * (1.0 - blend_factor)
                        + desert_terrain_height * blend_factor;
                    humidity_map[idx] =
                        mountain_humidity * (1.0 - blend_factor) + desert_humidity * blend_factor;
                }
            }
        }

        let data = Arc::new(ChunkColumnData {
            height_map,
            humidity_map,
            biome_map,
        });
        self.data_cache.insert(column_cv, data.clone());
        data
    }
}

impl VoxelGenerator for WorldGenerator {
    fn apply(&self, chunk_start: WorldVector, chunk: &mut Chunk32, voxel_size: f32) {
        let side = 32;
        let column_data = self.get_or_compute_column_data(wv_to_cv(chunk_start), voxel_size);

        for x in 0..side {
            for z in 0..side {
                let cache_idx = x as usize + z as usize * 32;
                let terrain_height = column_data.height_map[cache_idx] as f32;
                let humidity = column_data.humidity_map[cache_idx];

                for y in 0..side {
                    let lv = LocalVector::new(x, y, z);
                    let voxel_world_pos = (chunk_start + lv).as_vec3() * voxel_size;

                    if voxel_world_pos.y > terrain_height && voxel_world_pos.y > SEA_LEVEL {
                        continue;
                    }

                    let mut block_to_place = MyVoxels::Air;

                    let is_underground = voxel_world_pos.y <= terrain_height;

                    if is_underground {
                        let is_surface = (voxel_world_pos.y + voxel_size) > terrain_height;

                        if is_surface {
                            // --- SURFACE LOGIC ---
                            if voxel_world_pos.y > SEA_LEVEL {
                                if terrain_height <= SEA_LEVEL + BEACH_HEIGHT {
                                    block_to_place = MyVoxels::Sand;
                                } else if humidity < -0.3 {
                                    block_to_place = MyVoxels::Sand;
                                } else if terrain_height > 160.0 {
                                    block_to_place = MyVoxels::Snow;
                                } else if terrain_height > 90.0 {
                                    block_to_place = MyVoxels::Stone;
                                } else {
                                    block_to_place = MyVoxels::Grass;
                                }
                            } else {
                                block_to_place = MyVoxels::Sand;
                            }
                        } else {
                            // --- SUBSURFACE LOGIC ---
                            if voxel_world_pos.y > terrain_height - (3.0 * voxel_size) {
                                block_to_place = MyVoxels::Dirt;
                            } else {
                                block_to_place = MyVoxels::Stone;
                            }
                        }
                    } else {
                        // This block is above ground but at/below sea level, so it must be water.
                        if voxel_world_pos.y <= SEA_LEVEL {
                            block_to_place = MyVoxels::Water;
                        }
                    }

                    // --- CAVE CARVING ---
                    if block_to_place != MyVoxels::Water && block_to_place != MyVoxels::Air {
                        let noise_val_1 = self
                            .cave_noise_1
                            .get([voxel_world_pos.x as f64, voxel_world_pos.y as f64]);
                        let noise_val_2 = self
                            .cave_noise_2
                            .get([voxel_world_pos.y as f64, voxel_world_pos.z as f64]);
                        let cave_density = noise_val_1.abs() * noise_val_2.abs();
                        let depth = (terrain_height - voxel_world_pos.y).max(0.0) as f64;
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
