use std::sync::Arc;

use crate::voxels::MyVoxels;
use chunkee_core::{
    chunk::Chunk,
    clipmap::ChunkKey,
    coords::CHUNK_SIZE,
    generation::VoxelGenerator,
    glam::{IVec2, IVec3},
};
use dashmap::DashMap;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

const SEA_LEVEL: f32 = 4.0;
const BEACH_HEIGHT: f32 = 2.0;

struct ChunkColumnData {
    height_map: [f32; (CHUNK_SIZE * CHUNK_SIZE) as usize],
    biome_jitter_map: [f32; (CHUNK_SIZE * CHUNK_SIZE) as usize],
}

pub struct WorldGenerator {
    continent_noise: Fbm<Perlin>,
    terrain_noise: Fbm<Perlin>,
    cave_noise_1: Fbm<Perlin>,
    cave_noise_2: Fbm<Perlin>,
    // TODO: cache management, currently can grow forever. LRU?
    data_cache: DashMap<(IVec2, u8), Arc<ChunkColumnData>>,
    biome_noise: Fbm<Perlin>,
}

impl WorldGenerator {
    pub fn new() -> Self {
        let continent_noise = Fbm::new(1000).set_frequency(0.001).set_octaves(6);

        let terrain_noise = Fbm::new(2000)
            .set_frequency(0.003)
            .set_octaves(7)
            .set_lacunarity(2.2)
            .set_persistence(0.5);

        let cave_noise_1 = Fbm::new(3000).set_frequency(0.007).set_octaves(5);
        let cave_noise_2 = Fbm::new(3001).set_frequency(0.007).set_octaves(5);

        let biome_noise = Fbm::new(4000).set_frequency(0.02).set_octaves(2);

        Self {
            continent_noise,
            terrain_noise,
            cave_noise_1,
            cave_noise_2,
            biome_noise,
            data_cache: DashMap::new(),
        }
    }
    fn get_or_compute_column_data(
        &self,
        chunk_key: ChunkKey,
        voxel_size: f32,
    ) -> Arc<ChunkColumnData> {
        const STRIDE: i32 = CHUNK_SIZE;
        let column_cv = (IVec2::new(chunk_key.v.x, chunk_key.v.z), chunk_key.lod);
        if let Some(data) = self.data_cache.get(&column_cv) {
            return data.clone();
        }

        let mut height_map = [0.0; (CHUNK_SIZE * CHUNK_SIZE) as usize];
        let mut biome_jitter_map = [0.0; (CHUNK_SIZE * CHUNK_SIZE) as usize];

        let wv = chunk_key.to_wv();
        let lod_scale = chunk_key.lod_scale();

        for x in 0..STRIDE {
            for z in 0..STRIDE {
                let world_x = ((wv.x + x * lod_scale) as f32 * voxel_size) as f64;
                let world_z = ((wv.z + z * lod_scale) as f32 * voxel_size) as f64;
                let idx = (x + z * STRIDE) as usize;

                let cont_val = self.continent_noise.get([world_x, world_z]);
                let continent_mask = ((cont_val + 1.0) / 2.0).powf(0.9).max(0.0).min(1.0);
                let terrain_val = self.terrain_noise.get([world_x, world_z]);
                let elevation = (terrain_val + 1.0) / 2.0;
                let final_terrain = elevation.powf(2.5);
                let terrain_height_scaler = 350.0;
                let base_height = (SEA_LEVEL as f64) - 30.0;
                let land_height = 45.0 * continent_mask;
                let mountain_height = final_terrain * terrain_height_scaler * continent_mask;

                height_map[idx] = (base_height + land_height + mountain_height) as f32;

                let jitter = self.biome_noise.get([world_x, world_z]) as f32 * 10.0;
                biome_jitter_map[idx] = jitter;
            }
        }

        let data = Arc::new(ChunkColumnData {
            height_map,
            biome_jitter_map,
        });
        self.data_cache.insert(column_cv, data.clone());
        data
    }
}

impl VoxelGenerator for WorldGenerator {
    fn apply(&self, chunk_key: ChunkKey, chunk: &mut Chunk, voxel_size: f32) {
        const STRIDE: i32 = CHUNK_SIZE;
        let wv = chunk_key.to_wv();
        let lod_scale = chunk_key.lod_scale();
        let voxel_height = voxel_size * lod_scale as f32;
        let wp_top = wv.as_vec3() * voxel_size + voxel_height;
        if wp_top.y > 200.0 {
            return;
        }

        let column_data = self.get_or_compute_column_data(chunk_key, voxel_size);

        // Pre-calculate voxel height for the "Ceiling Check" fix from before

        let mut cave_noise_xy = [0.0; (STRIDE * STRIDE) as usize];
        let mut cave_noise_yz = [0.0; (STRIDE * STRIDE) as usize];

        for i in 0..STRIDE {
            for j in 0..STRIDE {
                let wx_1 = ((wv.x + i * lod_scale) as f32 * voxel_size) as f64;
                let wy_1 = ((wv.y + j * lod_scale) as f32 * voxel_size) as f64;
                cave_noise_xy[(i + j * STRIDE) as usize] = self.cave_noise_1.get([wx_1, wy_1]);

                let wy_2 = ((wv.y + i * lod_scale) as f32 * voxel_size) as f64;
                let wz_2 = ((wv.z + j * lod_scale) as f32 * voxel_size) as f64;
                cave_noise_yz[(i + j * STRIDE) as usize] = self.cave_noise_2.get([wy_2, wz_2]);
            }
        }

        for x in 0..STRIDE {
            for z in 0..STRIDE {
                let col_idx = (x + z * STRIDE) as usize;
                let terrain_height = column_data.height_map[col_idx] as f32;
                let jitter = column_data.biome_jitter_map[col_idx];

                for y in 0..STRIDE {
                    let lv = IVec3::new(x, y, z);
                    let voxel_world_pos = (wv + lv * lod_scale).as_vec3() * voxel_size;
                    let voxel_top = voxel_world_pos.y + voxel_height;

                    // Optimization
                    if voxel_world_pos.y > terrain_height && voxel_world_pos.y > SEA_LEVEL {
                        continue;
                    }

                    let mut block_to_place = MyVoxels::Air;
                    let is_underground = voxel_world_pos.y <= terrain_height;

                    if is_underground {
                        let is_surface = voxel_top > terrain_height;

                        if is_surface {
                            if voxel_world_pos.y > SEA_LEVEL {
                                if terrain_height > 160.0 + jitter {
                                    block_to_place = MyVoxels::Snow;
                                } else if terrain_height > 90.0 + jitter {
                                    block_to_place = MyVoxels::Stone;
                                } else {
                                    block_to_place = MyVoxels::Grass;
                                }

                                // Optional: Prevent sand from getting jittered into the ocean
                                // or allow it if you want varied beach lines
                                if terrain_height <= SEA_LEVEL + BEACH_HEIGHT + (jitter * 0.2) {
                                    block_to_place = MyVoxels::Sand;
                                }
                            } else {
                                block_to_place = MyVoxels::Sand;
                            }
                        } else {
                            // Subsurface logic...
                            if voxel_world_pos.y > terrain_height - (3.0 * voxel_size) {
                                block_to_place = MyVoxels::Dirt;
                            } else {
                                block_to_place = MyVoxels::Stone;
                            }
                        }
                    } else if voxel_top <= SEA_LEVEL {
                        block_to_place = MyVoxels::Water;
                    }

                    if block_to_place != MyVoxels::Water && block_to_place != MyVoxels::Air {
                        let n1 = cave_noise_xy[(x + y * STRIDE) as usize];
                        let n2 = cave_noise_yz[(y + z * STRIDE) as usize];

                        let cave_density = n1.abs() * n2.abs();

                        // Scale cave generation slightly for LODs?
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
