use glam::IVec3;
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

// Morton encoding logic
fn interleave_bits(input: u16) -> u64 {
    let mut x = input as u64;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    x
}

fn morton_encode(pos: IVec3) -> u64 {
    const OFFSET: i32 = 1 << 15;
    let x = interleave_bits((pos.x + OFFSET) as u16);
    let y = interleave_bits((pos.y + OFFSET) as u16) << 1;
    let z = interleave_bits((pos.z + OFFSET) as u16) << 2;
    x | y | z
}

#[derive(Default)]
pub struct MortonHasher {
    components: [i32; 3],
    count: usize,
}

impl Hasher for MortonHasher {
    fn finish(&self) -> u64 {
        debug_assert!(self.count == 3, "Incomplete IVec3 provided to hasher");
        let pos = IVec3::new(self.components[0], self.components[1], self.components[2]);

        morton_encode(pos)
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("This hasher is designed for structured data like IVec3");
    }

    fn write_i32(&mut self, i: i32) {
        if self.count < 3 {
            self.components[self.count] = i;
            self.count += 1;
        }
    }
}

pub type BuildMortonHasher = BuildHasherDefault<MortonHasher>;
pub type VoxelHashMap<T> = HashMap<IVec3, T, BuildMortonHasher>;

// Example
// =======================================================================
// fn main() {
//     let mut chunk_map: HashMap<IVec3, &str, BuildMortonHasher> =
//         HashMap::with_hasher(BuildMortonHasher::default());

//     let pos1 = IVec3::new(10, 20, 30);
//     let pos2 = IVec3::new(123, -456, 789);

//     chunk_map.insert(pos1, "My First Chunk");
//     chunk_map.insert(pos2, "Another Chunk");

//     if let Some(chunk_data) = chunk_map.get(&pos1) {
//         println!("Found data for position {}: {}", pos1, chunk_data);
//     }
// }
