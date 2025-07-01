use std::collections::HashMap;

use bincode::config::BigEndian;
use glam::IVec3;
use lz4::block::{compress, decompress};
use serde::{Deserialize, Serialize};
use sled::IVec;

use crate::{
    block::VoxelId,
    chunk::Deltas,
    coords::{idx_to_lv, lv_to_idx},
};

pub struct ChunkStore {
    db: sled::Db,
    bincode_config: bincode::config::Configuration<BigEndian>,
}

impl ChunkStore {
    pub fn new() -> Self {
        let path = "./debug_saves";
        std::fs::create_dir_all(path).expect("Failed to create debug save directory");

        ChunkStore {
            db: sled::open(path).expect("Failed to open sled db"),
            bincode_config: bincode::config::standard().with_big_endian(),
        }
    }

    pub fn save_deltas(&self, deltas_arr: &[(IVec3, &Deltas)]) {
        let mut batch = sled::Batch::default();

        for (cv, deltas) in deltas_arr {
            let key = ivec3_to_key(*cv);
            let compressed = self.encode_compress(deltas);
            batch.insert(&key, compressed);
        }
        self.db
            .apply_batch(batch)
            .expect("Failed to apply batch save");

        // TODO: Configure default flushing timer or flush here
        // self.db.flush_async();
    }

    pub fn save_delta(&self, cv: IVec3, deltas: &Deltas) {
        self.save_deltas(&[(cv, deltas)]);

        // TODO: Configure default flushing timer or flush here
        // self.db.flush_async();
    }

    pub fn load_delta(&self, cv: IVec3) -> Option<Deltas> {
        let key = ivec3_to_key(cv);
        self.db
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

fn ivec3_to_key(v: IVec3) -> [u8; 12] {
    let mut key = [0u8; 12];
    key[0..4].copy_from_slice(&v.x.to_be_bytes());
    key[4..8].copy_from_slice(&v.y.to_be_bytes());
    key[8..12].copy_from_slice(&v.z.to_be_bytes());
    key
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltasPacked(Vec<(u16, VoxelId)>);

impl From<&Deltas> for DeltasPacked {
    fn from(value: &Deltas) -> Self {
        let mut packed = Vec::with_capacity(value.0.len());
        for (lv, voxel_id) in value.0.iter() {
            let idx = lv_to_idx(*lv);
            packed.push((idx as u16, *voxel_id));
        }

        DeltasPacked(packed)
    }
}

impl Into<Deltas> for DeltasPacked {
    fn into(self) -> Deltas {
        let mut map = HashMap::new();
        for (idx, voxel_id) in self.0 {
            let lv = idx_to_lv(idx as usize);
            map.insert(lv, voxel_id);
        }

        Deltas(map)
    }
}
