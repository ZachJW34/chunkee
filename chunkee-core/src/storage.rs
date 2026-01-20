use bincode::config::BigEndian;
use fxhash::FxHashMap;
use glam::IVec3;
use lz4::block::{compress, decompress};
use serde::{Deserialize, Serialize};
use sled::IVec;

use crate::{
    block::VoxelId,
    clipmap::{ChunkKey, LODConfig},
    coords::{CHUNK_SIZE, idx_to_lv, lv_to_idx},
    manager::Deltas,
};

pub struct ChunkStore {
    // db: sled::Db,
    deltas_tree: sled::Tree,
    // chunk_tree: sled::Tree,
    bincode_config: bincode::config::Configuration<BigEndian>,
}

impl ChunkStore {
    pub fn new() -> Self {
        let path = "./debug_saves";
        std::fs::create_dir_all(path).expect("Failed to create debug save directory");

        let db = sled::open(path).expect("Failed to open sled db");
        let deltas_tree = db
            .open_tree("deltas")
            .expect("Failed to create 'deltas' tree");
        // let chunk_tree = db
        //     .open_tree("chunk")
        //     .expect("Failed to create 'chunk' tree");

        ChunkStore {
            // db,
            deltas_tree,
            // chunk_tree,
            bincode_config: bincode::config::standard().with_big_endian(),
        }
    }

    pub fn save_chunks(&self, mut chunks_arr: Vec<(ChunkKey, Deltas)>, lod_config: &LODConfig) {
        let mut batch = sled::Batch::default();
        let mut current_lod = 0;

        while !chunks_arr.is_empty() && current_lod <= lod_config.max_lod {
            let mut next_lod_edits: FxHashMap<ChunkKey, Deltas> = FxHashMap::default();

            for (key, deltas) in chunks_arr.drain(..) {
                let db_key = chunk_key_to_db_key(key);
                let compressed = self.encode_compress(&deltas);
                batch.insert(&db_key, compressed);

                if current_lod < lod_config.max_lod {
                    if let Some(parent_key) = key.parent(lod_config.max_lod) {
                        let offset = IVec3::new(key.v.x & 1, key.v.y & 1, key.v.z & 1);
                        let parent_deltas = next_lod_edits.entry(parent_key).or_default();

                        for (child_lv, voxel_id) in deltas.0 {
                            let local_half = child_lv >> 1;
                            let parent_lv = local_half + (offset * (CHUNK_SIZE / 2));
                            parent_deltas.0.insert(parent_lv, voxel_id);
                        }
                    }
                }
            }

            let mut next_batch = Vec::new();
            for (p_key, new_deltas) in next_lod_edits {
                let mut existing_deltas = self.load_chunk(p_key).unwrap_or_default();

                for (k, v) in new_deltas.0 {
                    existing_deltas.0.insert(k, v);
                }

                next_batch.push((p_key, existing_deltas));
            }

            chunks_arr = next_batch;
            current_lod += 1;
        }

        self.deltas_tree
            .apply_batch(batch)
            .expect("Failed to apply batch save");
    }

    pub fn load_chunk(&self, chunk_key: ChunkKey) -> Option<Deltas> {
        let key = chunk_key_to_db_key(chunk_key);
        self.deltas_tree
            .get(key)
            .expect("Failed to read deltas in KV store")
            .map(|compressed| self.decompress_decode(compressed))
    }

    fn encode_compress(&self, deltas: &Deltas) -> Vec<u8> {
        let deltas_packed: DeltasPacked = deltas.into();
        let mut encoded = bincode::serde::encode_to_vec(deltas_packed, self.bincode_config)
            .expect("Failed to serialize deltas.");
        let compressed = compress(&mut encoded, None, true).expect("Failed to compress chunk.");

        compressed
    }

    fn decompress_decode(&self, compressed: IVec) -> Deltas {
        let encoded = decompress(&compressed, None).expect("Failed to decompress chunk.");
        let (deltas_packed, _): (DeltasPacked, _) =
            bincode::serde::decode_from_slice(&encoded, self.bincode_config)
                .expect("Failed to decode deltas");

        deltas_packed.into()
    }
}

// fn key_to_ivec3(key: &[u8]) -> IVec3 {
//     const SIGN_FLIP: u32 = 1 << 31;
//     let x = (u32::from_be_bytes(key[0..4].try_into().unwrap()) ^ SIGN_FLIP) as i32;
//     let y = (u32::from_be_bytes(key[4..8].try_into().unwrap()) ^ SIGN_FLIP) as i32;
//     let z = (u32::from_be_bytes(key[8..12].try_into().unwrap()) ^ SIGN_FLIP) as i32;
//     IVec3::new(x, y, z)
// }

fn chunk_key_to_db_key(key: ChunkKey) -> [u8; 13] {
    const SIGN_FLIP: u32 = 1 << 31;
    let mut db_key = [0u8; 13];

    // Byte 0: LOD Level (Vital for avoiding collisions)
    db_key[0] = key.lod;

    // Bytes 1-13: Coordinates
    db_key[1..5].copy_from_slice(&((key.v.x as u32 ^ SIGN_FLIP).to_be_bytes()));
    db_key[5..9].copy_from_slice(&((key.v.y as u32 ^ SIGN_FLIP).to_be_bytes()));
    db_key[9..13].copy_from_slice(&((key.v.z as u32 ^ SIGN_FLIP).to_be_bytes()));

    db_key
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltasPacked(Vec<(u16, VoxelId)>);

impl From<&Deltas> for DeltasPacked {
    fn from(value: &Deltas) -> Self {
        let mut deltas_packed = Vec::with_capacity(value.0.len());
        for (lv, voxel_id) in value.0.iter() {
            let idx = lv_to_idx(*lv);
            deltas_packed.push((idx as u16, *voxel_id));
        }

        DeltasPacked(deltas_packed)
    }
}

impl Into<Deltas> for DeltasPacked {
    fn into(self) -> Deltas {
        let mut deltas = Deltas::default();
        for (idx, voxel_id) in self.0 {
            let lv = idx_to_lv(idx as usize);
            deltas.0.insert(lv, voxel_id);
        }

        deltas
    }
}
