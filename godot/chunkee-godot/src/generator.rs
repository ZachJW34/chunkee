use chunkee_core::{
    chunk::Chunk,
    coords::{CHUNK_SIZE, ChunkVector, LocalVector, cv_to_wv, wv_to_cv},
    generation::VoxelGenerator,
    glam::IVec2,
};
use dashmap::DashMap;
use noise::{HybridMulti, NoiseFn, Perlin};

use crate::voxels::MyVoxels;

const CHUNK_SIZE_USIZE: usize = CHUNK_SIZE as usize;

pub struct WorldGenerator {
    noise: HybridMulti<Perlin>,

    noise_cache: DashMap<IVec2, [f64; (CHUNK_SIZE * CHUNK_SIZE) as usize]>,
}

impl WorldGenerator {
    pub fn new() -> Self {
        let mut noise = HybridMulti::<Perlin>::new(9000);
        noise.octaves = 5;
        noise.frequency = 0.002;
        noise.lacunarity = 2.2;
        noise.persistence = 0.4;

        Self {
            noise,
            noise_cache: DashMap::new(),
        }
    }

    fn get_height_map(&self, cv: ChunkVector) -> [f64; (CHUNK_SIZE * CHUNK_SIZE) as usize] {
        self.noise_cache
            .entry(IVec2::new(cv.x, cv.z))
            .or_insert_with(|| {
                let mut grid = [0.0; (CHUNK_SIZE * CHUNK_SIZE) as usize];
                let chunk_world_pos = cv_to_wv(cv);

                for x in 0..CHUNK_SIZE_USIZE {
                    for z in 0..CHUNK_SIZE_USIZE {
                        let world_x = (chunk_world_pos.x + x as i32) as f64;
                        let world_z = (chunk_world_pos.z + z as i32) as f64;
                        grid[x + z * CHUNK_SIZE_USIZE] = self.noise.get([world_x, world_z])
                    }
                }

                grid
            })
            .clone()
    }
}

impl VoxelGenerator for WorldGenerator {
    fn apply(&self, origin_wv: chunkee_core::coords::WorldVector, chunk: &mut Chunk) {
        let origin_cv = wv_to_cv(origin_wv);
        let height_map = self.get_height_map(origin_cv);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let height_val = height_map[(x as usize) + (z as usize) * CHUNK_SIZE_USIZE];
                let terrain_height = height_val * 64.0;

                for y in 0..CHUNK_SIZE {
                    let lv = LocalVector::new(x, y, z);
                    let wv = origin_wv + lv;

                    if (wv.y as f64) < terrain_height {
                        if wv.y >= 120 {
                            chunk.set_block::<MyVoxels>(lv, MyVoxels::Snow);
                        } else if wv.y >= 60 {
                            chunk.set_block::<MyVoxels>(lv, MyVoxels::Stone);
                        } else if wv.y == (terrain_height.floor() as i32) {
                            chunk.set_block::<MyVoxels>(lv, MyVoxels::Grass);
                        } else {
                            chunk.set_block::<MyVoxels>(lv, MyVoxels::Dirt);
                        }
                    }
                }
            }
        }
    }
}
