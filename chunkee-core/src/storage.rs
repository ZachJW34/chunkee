use bincode::config::BigEndian;
use glam::IVec3;
use lz4::block::{compress, decompress};
use serde::{Deserialize, Serialize};
use sled::IVec;

use crate::{
    block::VoxelId,
    coords::{ChunkVector, idx_to_lv, lv_to_idx},
    grid::Deltas,
    hasher::VoxelHashMap,
};

pub type BatchedPersistedChunkMap = VoxelHashMap<Option<PersistedChunk>>;

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

    pub fn save_chunks(&self, chunks_arr: Vec<(ChunkVector, PersistedChunk)>) {
        let mut batch = sled::Batch::default();

        for (cv, chunk) in chunks_arr {
            let key = ivec3_to_key(cv);
            let compressed = self.encode_compress(chunk);
            batch.insert(&key, compressed);
        }
        self.db
            .apply_batch(batch)
            .expect("Failed to apply batch save");
    }

    pub fn save_chunk(&self, cv: IVec3, persisted_chunk: PersistedChunk) {
        self.save_chunks(vec![(cv, persisted_chunk)]);
    }

    pub fn load_chunk(&self, cv: IVec3) -> Option<PersistedChunk> {
        let key = ivec3_to_key(cv);
        self.db
            .get(key)
            .expect("Failed to read deltas in KV store")
            .map(|compressed| self.decompress_decode(compressed))
    }

    pub fn load_chunks_in_aabb(
        &self,
        min_corner: IVec3,
        max_corner: IVec3,
        dst: &mut BatchedPersistedChunkMap,
    ) {
        println!("min_corner: {min_corner} max_corner: {max_corner}");
        for x in min_corner.x..=max_corner.x {
            for y in min_corner.y..=max_corner.y {
                let start_key = ivec3_to_key(IVec3::new(x, y, min_corner.z));
                let end_key = ivec3_to_key(IVec3::new(x, y, max_corner.z));
                let mut deltas_in_range: VoxelHashMap<PersistedChunk> = self
                    .db
                    .range(start_key..=end_key)
                    .map(|result| {
                        let (key, value) = result.expect("Failed to read from DB range scan");
                        let coord = key_to_ivec3(&key);
                        let persisted_chunk = self.decompress_decode(value);
                        (coord, persisted_chunk)
                    })
                    .collect();

                for z in min_corner.z..=max_corner.z {
                    let cv = IVec3::new(x, y, z);
                    dst.insert(cv, deltas_in_range.remove(&cv));
                }
            }
        }
    }

    fn encode_compress(&self, persisted_chunk: PersistedChunk) -> Vec<u8> {
        let persisted_chunk_packed: PersitedChunkPacked = persisted_chunk.into();
        let mut encoded =
            bincode::serde::encode_to_vec(persisted_chunk_packed, self.bincode_config)
                .expect("Failed to serialize deltas.");
        let compressed = compress(&mut encoded, None, true).expect("Failed to compress chunk.");

        compressed
    }

    fn decompress_decode(&self, compressed: IVec) -> PersistedChunk {
        let encoded = decompress(&compressed, None).expect("Failed to decompress chunk.");
        let (persisted_chunk_packed, _): (PersitedChunkPacked, _) =
            bincode::serde::decode_from_slice(&encoded, self.bincode_config)
                .expect("Failed to decode deltas");

        persisted_chunk_packed.into()
    }
}

fn key_to_ivec3(key: &[u8]) -> IVec3 {
    const SIGN_FLIP: u32 = 1 << 31;
    let x = (u32::from_be_bytes(key[0..4].try_into().unwrap()) ^ SIGN_FLIP) as i32;
    let y = (u32::from_be_bytes(key[4..8].try_into().unwrap()) ^ SIGN_FLIP) as i32;
    let z = (u32::from_be_bytes(key[8..12].try_into().unwrap()) ^ SIGN_FLIP) as i32;
    IVec3::new(x, y, z)
}

fn ivec3_to_key(v: IVec3) -> [u8; 12] {
    const SIGN_FLIP: u32 = 1 << 31;
    let mut key = [0u8; 12];
    key[0..4].copy_from_slice(&((v.x as u32 ^ SIGN_FLIP).to_be_bytes()));
    key[4..8].copy_from_slice(&((v.y as u32 ^ SIGN_FLIP).to_be_bytes()));
    key[8..12].copy_from_slice(&((v.z as u32 ^ SIGN_FLIP).to_be_bytes()));
    key
}

#[derive(Debug, Clone)]
pub struct PersistedChunk {
    pub uniform_voxel_id: Option<VoxelId>,
    pub deltas: Deltas,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersitedChunkPacked {
    uniform_voxel_id: Option<VoxelId>,
    deltas_packed: DeltasPacked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltasPacked(Vec<(u16, VoxelId)>);

impl From<PersistedChunk> for PersitedChunkPacked {
    fn from(value: PersistedChunk) -> Self {
        let uniform_voxel_id = value.uniform_voxel_id;
        let mut deltas_packed = Vec::with_capacity(value.deltas.0.len());
        for (lv, voxel_id) in value.deltas.0.into_iter() {
            let idx = lv_to_idx(lv);
            deltas_packed.push((idx as u16, voxel_id));
        }

        PersitedChunkPacked {
            uniform_voxel_id,
            deltas_packed: DeltasPacked(deltas_packed),
        }
    }
}

impl Into<PersistedChunk> for PersitedChunkPacked {
    fn into(self) -> PersistedChunk {
        let uniform_voxel_id = self.uniform_voxel_id;
        let mut deltas = Deltas::default();
        for (idx, voxel_id) in self.deltas_packed.0 {
            let lv = idx_to_lv(idx as usize);
            deltas.0.insert(lv, voxel_id);
        }

        PersistedChunk {
            uniform_voxel_id,
            deltas,
        }
    }
}

// impl From<&Deltas> for DeltasPacked {
//     fn from(value: &Deltas) -> Self {
//         let mut packed = Vec::with_capacity(value.0.len());
//         for (lv, voxel_id) in value.0.iter() {
//             let idx = lv_to_idx(*lv);
//             packed.push((idx as u16, *voxel_id));
//         }

//         DeltasPacked(packed)
//     }
// }

// impl Into<Deltas> for DeltasPacked {
//     fn into(self) -> Deltas {
//         let mut deltas = Deltas::default();
//         for (idx, voxel_id) in self.0 {
//             let lv = idx_to_lv(idx as usize);
//             deltas.0.insert(lv, voxel_id);
//         }

//         deltas
//     }
// }
