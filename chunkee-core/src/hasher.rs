use glam::IVec3;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdentityHasher should only be used with integer keys");
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

pub type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

// 2. The Morton Encoder: It turns an IVec3 into a well-distributed u64.
// =======================================================================

// Helper function to spread the bits of a 16-bit integer out to 48 bits.
fn interleave_bits(input: u16) -> u64 {
    let mut x = input as u64;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    x
}

/// Encodes a 3D position into a 64-bit Z-order curve index.
/// Coordinates must be within the range [-32768, 32767].
pub fn morton_encode(pos: IVec3) -> u64 {
    const OFFSET: i32 = 1 << 15;
    const MIN_COORD: i32 = -OFFSET;
    const MAX_COORD: i32 = (1 << 16) - 1 - OFFSET;

    debug_assert!(
        (MIN_COORD..=MAX_COORD).contains(&pos.x)
            && (MIN_COORD..=MAX_COORD).contains(&pos.y)
            && (MIN_COORD..=MAX_COORD).contains(&pos.z),
        "Coordinate out of encodable range!"
    );

    let x = interleave_bits((pos.x + OFFSET) as u16);
    let y = interleave_bits((pos.y + OFFSET) as u16) << 1;
    let z = interleave_bits((pos.z + OFFSET) as u16) << 2;

    x | y | z
}
