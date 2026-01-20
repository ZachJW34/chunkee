use std::i32;

use glam::IVec3;

use crate::{
    block::BLOCK_FACES,
    coords::{CHUNK_SIZE, CHUNK_SIZE_F32},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub v: IVec3,
    pub lod: u8,
}

impl ChunkKey {
    pub const MAX: Self = Self {
        v: IVec3::MAX,
        lod: 0,
    };

    pub fn new(x: i32, y: i32, z: i32, lod: u8) -> Self {
        Self {
            v: IVec3::new(x, y, z),
            lod,
        }
    }

    #[inline(always)]
    pub fn to_wv(&self) -> IVec3 {
        let lod_scale = 1 << self.lod;
        let size = CHUNK_SIZE * lod_scale;

        IVec3::new(self.v.x * size, self.v.y * size, self.v.z * size)
    }

    #[inline(always)]
    pub fn lod_scale(&self) -> i32 {
        lod_scale(self.lod)
    }

    #[inline(always)]
    pub fn neighbors(&self) -> [ChunkKey; 6] {
        let v = IVec3::new(self.v.x, self.v.y, self.v.z);
        BLOCK_FACES.map(|face| {
            let n = face.into_normal() + v;
            ChunkKey::new(n.x, n.y, n.z, self.lod)
        })
    }

    #[inline(always)]
    pub fn parent(&self, max_lod: u8) -> Option<ChunkKey> {
        if self.lod == max_lod {
            return None;
        }
        Some(ChunkKey::new(
            self.v.x.div_euclid(2),
            self.v.y.div_euclid(2),
            self.v.z.div_euclid(2),
            self.lod + 1,
        ))
    }

    #[inline(always)]
    pub fn children(&self) -> Option<[ChunkKey; 8]> {
        if self.lod == 0 {
            return None;
        }

        let child_lod = self.lod - 1;
        let base_x = self.v.x * 2;
        let base_y = self.v.y * 2;
        let base_z = self.v.z * 2;

        Some([
            ChunkKey::new(base_x, base_y, base_z, child_lod),
            ChunkKey::new(base_x + 1, base_y, base_z, child_lod),
            ChunkKey::new(base_x, base_y + 1, base_z, child_lod),
            ChunkKey::new(base_x + 1, base_y + 1, base_z, child_lod),
            ChunkKey::new(base_x, base_y, base_z + 1, child_lod),
            ChunkKey::new(base_x + 1, base_y, base_z + 1, child_lod),
            ChunkKey::new(base_x, base_y + 1, base_z + 1, child_lod),
            ChunkKey::new(base_x + 1, base_y + 1, base_z + 1, child_lod),
        ])
    }
}
/// Configuration for the LOD system.
#[derive(Debug, Clone)]
pub struct LODConfig {
    /// Maximum LOD level (e.g., 2 means LOD0, LOD1, LOD2).
    pub max_lod: u8,
    /// Draw distance in "Chunk Counts" for each LOD level.
    /// IMPORTANT: These represent the "Half-Width" from the vertex center.
    /// They MUST be EVEN numbers to ensure symmetric holes.
    pub draw_distances: Vec<i32>,
    /// How much buffer before a shift occurs (0.5 = exact center, 0.75 = 75% to edge).
    pub hysteresis: f32,
}

impl LODConfig {
    pub fn validate(&self) {
        let max_lod_idx = self.max_lod as usize;

        // 1. Validate Array Length
        assert!(
            self.draw_distances.len() > max_lod_idx,
            "Config Error: `draw_distances` length ({}) is too small for max_lod ({}). Need {} entries.",
            self.draw_distances.len(),
            self.max_lod,
            max_lod_idx + 1
        );

        // 2. Validate Even Radii (Crucial for Even-Grid Symmetry)
        for (i, &dist) in self.draw_distances.iter().enumerate() {
            assert!(
                dist % 2 == 0,
                "Config Error: LOD{} radius ({}) is ODD. It must be EVEN to ensure symmetric holes in the vertex-centered grid.",
                i,
                dist
            );
        }

        // 3. Calculate Constants
        let stride = CHUNK_SIZE_F32 * 2u32.pow(self.max_lod as u32) as f32;
        let shift_threshold = stride * self.hysteresis;

        // 4. Constraint #1: The Hysteresis Trap
        // Note: In Vertex-Centered grid, the "Radius" is the distance to the edge.
        // E.g. Radius 4 covers [-4, 3]. The distance from center (0) to edge (-4) is 4 chunks.
        let lod0_world_radius = self.draw_distances[0] as f32 * CHUNK_SIZE_F32;

        assert!(
            lod0_world_radius > shift_threshold,
            "Config Error: LOD0 Radius ({}m) is smaller than Shift Threshold ({}m). \
             Player will see void before shift! Increase draw_distances[0].",
            lod0_world_radius,
            shift_threshold
        );

        // 5. Constraint #2: The Russian Doll Rule
        for i in 0..max_lod_idx {
            let current_scale = 2u32.pow(i as u32) as f32;
            let next_scale = 2u32.pow((i + 1) as u32) as f32;

            let current_radius_world =
                self.draw_distances[i] as f32 * CHUNK_SIZE_F32 * current_scale;
            let next_radius_world = self.draw_distances[i + 1] as f32 * CHUNK_SIZE_F32 * next_scale;

            assert!(
                next_radius_world >= current_radius_world,
                "Config Error: LOD{} World Radius ({}m) > LOD{} World Radius ({}m). \
                 Higher LODs must fully encompass lower LODs.",
                i,
                current_radius_world,
                i + 1,
                next_radius_world
            );
        }
    }

    pub fn uniform(radius: i32, max_lod: u8) -> Self {
        // Enforce even/power of two for perfect alignment
        let safe_radius = if radius % 2 != 0 { radius + 1 } else { radius };

        Self {
            max_lod,
            draw_distances: vec![safe_radius; (max_lod + 1) as usize],
            hysteresis: 0.75,
        }
    }

    pub fn num_chunks(&self) -> u32 {
        let mut total = 0;

        for lod in 0..=self.max_lod {
            let draw_distance = self.draw_distances[lod as usize];

            let width = 2 * draw_distance;
            let volume = width.pow(3);

            if lod == 0 {
                // LOD0 is solid
                total += volume;
            } else {
                // LOD1+ is a shell (Volume - Hole)
                let prev_draw_distance = self.draw_distances[(lod - 1) as usize];
                let hole_volume = prev_draw_distance.pow(3);

                total += volume - hole_volume;
            }
        }

        total as u32
    }
}

#[derive(Debug, Default)]
pub struct LodEvents {
    pub anchor_shifted: bool,
    pub new_anchor: IVec3,
    pub to_load: Vec<ChunkKey>,
    pub to_unload: Vec<ChunkKey>,
}

pub struct ClipMap {
    config: LODConfig,
    pub anchor: IVec3,
    step_size: f32,
    shift_threshold: f32,
    world_bounds: Bounds,
}

impl ClipMap {
    pub fn new(config: LODConfig) -> Self {
        let max_lod_scale = 2_i32.pow(config.max_lod as u32) as f32;
        let step_size = CHUNK_SIZE_F32 * max_lod_scale;
        let shift_threshold = step_size * config.hysteresis;

        Self {
            config,
            anchor: IVec3::MAX,
            step_size,
            shift_threshold,
            world_bounds: Bounds {
                min: IVec3::MAX,
                max: IVec3::MAX,
            },
        }
    }

    pub fn wv_to_chunk_key(&self, wv: IVec3) -> Option<ChunkKey> {
        // Guard against uninitialized state
        if self.anchor == IVec3::MAX || !self.world_bounds.contains(wv) {
            return None;
        }

        for lod in 0..=self.config.max_lod {
            let lod_scale = 1 << lod; // 1, 2, 4...
            let lod_chunk_size = CHUNK_SIZE * lod_scale;

            let chunk_x = wv.x.div_euclid(lod_chunk_size);
            let chunk_y = wv.y.div_euclid(lod_chunk_size);
            let chunk_z = wv.z.div_euclid(lod_chunk_size);
            let coord = IVec3::new(chunk_x, chunk_y, chunk_z);

            let bounds = self.get_chunk_bounds(lod, self.anchor);

            if bounds.contains(coord) {
                return Some(ChunkKey::new(chunk_x, chunk_y, chunk_z, lod));
            }
        }

        None
    }

    pub fn update(&mut self, player_pos: IVec3) -> LodEvents {
        let mut events = LodEvents::default();

        // ---------------------------------------------------------
        // 1. INITIALIZATION CASE
        // ---------------------------------------------------------
        if self.anchor == IVec3::MAX {
            self.anchor = self.snap_to_grid(player_pos);
            self.world_bounds = self.calculate_world_bounds(self.anchor);

            events.to_load = self.calculate_init(self.anchor);
            events.anchor_shifted = true;
            events.new_anchor = self.anchor;

            return events;
        }

        // ---------------------------------------------------------
        // 2. STANDARD UPDATE
        // ---------------------------------------------------------
        let mut shift_vector = IVec3::ZERO;
        shift_vector.x = self.calculate_axis_shift(self.anchor.x, player_pos.x);
        shift_vector.y = self.calculate_axis_shift(self.anchor.y, player_pos.y);
        shift_vector.z = self.calculate_axis_shift(self.anchor.z, player_pos.z);

        if shift_vector != IVec3::ZERO {
            let old_anchor = self.anchor;
            self.anchor = self.anchor + shift_vector;
            self.world_bounds = self.calculate_world_bounds(self.anchor);

            events.anchor_shifted = true;
            events.new_anchor = self.anchor;

            let (load, unload) = self.calculate_diff(old_anchor, self.anchor);
            events.to_load = load;
            events.to_unload = unload;
        }

        events
    }

    #[inline(always)]
    pub fn contains(&self, wv: IVec3) -> bool {
        self.world_bounds.contains(wv)
    }

    #[inline(always)]
    fn calculate_axis_shift(&self, anchor_val: i32, player_val: i32) -> i32 {
        let diff = (player_val - anchor_val) as f32;
        if diff.abs() <= self.shift_threshold {
            return 0;
        }

        let step = self.step_size as i32;
        let target_val = (player_val as i32 + step / 2).div_euclid(step) * step;

        target_val - anchor_val
    }

    #[inline(always)]
    fn snap_to_grid(&self, pos: IVec3) -> IVec3 {
        let step = self.step_size as i32;
        // Vertex Centering Logic:
        // We want to snap to the nearest Vertex.
        // A vertex is exactly at integer multiples of 'step'.
        // Standard rounding: floor(x / step + 0.5) * step
        let x = (pos.x + step / 2).div_euclid(step) * step;
        let y = (pos.y + step / 2).div_euclid(step) * step;
        let z = (pos.z + step / 2).div_euclid(step) * step;

        IVec3::new(x, y, z)
    }

    #[inline(always)]
    fn calculate_world_bounds(&self, anchor: IVec3) -> Bounds {
        let max_lod = self.config.max_lod;
        let radius_chunks = self.config.draw_distances[max_lod as usize];

        let scale = 1 << max_lod;
        let chunk_size_voxels = CHUNK_SIZE * scale;

        let world_radius = radius_chunks * chunk_size_voxels;

        Bounds {
            min: anchor - IVec3::splat(world_radius),
            // -1 because bounds are inclusive
            // (e.g. range [0, 100) becomes min 0, max 99)
            max: anchor + IVec3::splat(world_radius - 1),
        }
    }

    fn calculate_init(&self, anchor: IVec3) -> Vec<ChunkKey> {
        let mut to_load = Vec::new();

        for lod in 0..=self.config.max_lod {
            let bounds = self.get_chunk_bounds(lod, anchor);
            let hole = if lod > 0 {
                Some(self.get_hole_bounds(lod, anchor))
            } else {
                None
            };

            for z in bounds.min.z..=bounds.max.z {
                for y in bounds.min.y..=bounds.max.y {
                    for x in bounds.min.x..=bounds.max.x {
                        let coord = IVec3::new(x, y, z);

                        if let Some(h) = hole {
                            if h.contains(coord) {
                                continue;
                            }
                        }

                        to_load.push(ChunkKey::new(x, y, z, lod));
                    }
                }
            }
        }

        to_load
    }

    fn calculate_diff(
        &self,
        old_anchor: IVec3,
        new_anchor: IVec3,
    ) -> (Vec<ChunkKey>, Vec<ChunkKey>) {
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();

        for lod in 0..=self.config.max_lod {
            let old_box = self.get_chunk_bounds(lod, old_anchor);
            let new_box = self.get_chunk_bounds(lod, new_anchor);

            let old_hole = if lod > 0 {
                Some(self.get_hole_bounds(lod, old_anchor))
            } else {
                None
            };
            let new_hole = if lod > 0 {
                Some(self.get_hole_bounds(lod, new_anchor))
            } else {
                None
            };

            // 1. Find chunks to LOAD (In New, Not in Old)
            for z in new_box.min.z..=new_box.max.z {
                for y in new_box.min.y..=new_box.max.y {
                    for x in new_box.min.x..=new_box.max.x {
                        let coord = IVec3::new(x, y, z);

                        if let Some(hole) = new_hole {
                            if hole.contains(coord) {
                                continue;
                            }
                        }

                        let was_in_old_bounds = old_box.contains(coord);
                        let was_in_old_hole = old_hole.map_or(false, |h| h.contains(coord));

                        if !was_in_old_bounds || was_in_old_hole {
                            to_load.push(ChunkKey::new(x, y, z, lod));
                        }
                    }
                }
            }

            // 2. Find chunks to UNLOAD (In Old, Not in New)
            for z in old_box.min.z..=old_box.max.z {
                for y in old_box.min.y..=old_box.max.y {
                    for x in old_box.min.x..=old_box.max.x {
                        let coord = IVec3::new(x, y, z);

                        if let Some(hole) = old_hole {
                            if hole.contains(coord) {
                                continue;
                            }
                        }

                        let is_outside_new = !new_box.contains(coord);
                        let is_inside_new_hole = new_hole.map_or(false, |h| h.contains(coord));

                        if is_outside_new || is_inside_new_hole {
                            to_unload.push(ChunkKey::new(x, y, z, lod));
                        }
                    }
                }
            }
        }

        (to_load, to_unload)
    }

    /// Uses VERTEX-CENTERED Logic: [-Radius, Radius-1]
    #[inline(always)]
    fn get_chunk_bounds(&self, lod: u8, anchor_world: IVec3) -> Bounds {
        let lod_scale = 2_i32.pow(lod as u32);
        let lod_chunk_size = CHUNK_SIZE * lod_scale;
        let radius = self.config.draw_distances[lod as usize];

        let center_x = anchor_world.x.div_euclid(lod_chunk_size);
        let center_y = anchor_world.y.div_euclid(lod_chunk_size);
        let center_z = anchor_world.z.div_euclid(lod_chunk_size);

        Bounds {
            min: IVec3::new(center_x - radius, center_y - radius, center_z - radius),
            max: IVec3::new(
                center_x + radius - 1,
                center_y + radius - 1,
                center_z + radius - 1,
            ),
        }
    }

    #[inline(always)]
    fn get_hole_bounds(&self, current_lod: u8, anchor_world: IVec3) -> Bounds {
        let prev_lod = current_lod - 1;

        // SYMMETRY LOGIC:
        // In the even-grid system, the hole is exactly half the radius of the previous LOD.
        // Because previous LOD radius is in "Previous Units", we divide by 2 to get "Current Units".
        // Example: Prev Radius = 4 (prev-units). Hole = 2 (current-units).
        // This relies on Config::validate ensuring radii are Even numbers.
        let prev_radius = self.config.draw_distances[prev_lod as usize];
        let hole_radius = prev_radius / 2;

        let lod_scale = 2_i32.pow(current_lod as u32);
        let lod_chunk_size = CHUNK_SIZE * lod_scale;

        let center_x = anchor_world.x.div_euclid(lod_chunk_size);
        let center_y = anchor_world.y.div_euclid(lod_chunk_size);
        let center_z = anchor_world.z.div_euclid(lod_chunk_size);

        Bounds {
            min: IVec3::new(
                center_x - hole_radius,
                center_y - hole_radius,
                center_z - hole_radius,
            ),
            max: IVec3::new(
                center_x + hole_radius - 1,
                center_y + hole_radius - 1,
                center_z + hole_radius - 1,
            ),
        }
    }

    #[inline]
    pub fn is_chunk_valid(&self, key: ChunkKey) -> bool {
        let bounds = self.get_chunk_bounds(key.lod, self.anchor);

        if !bounds.contains(key.v) {
            return false;
        }

        if key.lod > 0 {
            let hole = self.get_hole_bounds(key.lod, self.anchor);

            if hole.contains(key.v) {
                return false;
            }
        }

        true
    }
}

#[inline(always)]
pub fn lod_scale(lod: u8) -> i32 {
    1 << lod
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    min: IVec3,
    max: IVec3,
}

impl Bounds {
    #[inline(always)]
    fn contains(&self, point: IVec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use std::collections::HashSet;

    #[test]
    fn config_validation() {
        // LOD0 too small
        let invalid_config = LODConfig {
            max_lod: 2,
            draw_distances: vec![2, 2, 2],
            hysteresis: 0.5,
        };
        let result = std::panic::catch_unwind(|| invalid_config.validate());
        assert!(result.is_err());
    }

    #[test]
    fn lods() {
        let r8 = 8_i32;
        let r8_volume: i32 = (r8 * 2).pow(3);
        let r8_shell_volume = r8_volume - r8.pow(3);

        for max_lod in 0..=3 {
            let lod_config = LODConfig::uniform(r8, max_lod);
            lod_config.validate();
            let mut clipmap = ClipMap::new(lod_config);
            let events = clipmap.update(IVec3::ZERO);
            assert!(events.to_unload.is_empty());
            assert_eq!(
                events.to_load.len() as i32,
                r8_volume + r8_shell_volume * (max_lod as i32)
            );
        }
    }

    #[test]
    fn initialization() {
        let r4: i32 = 4_i32;
        let r4_volume = (2 * r4).pow(3);
        let r4_shell_volume = r4_volume - r4.pow(3);
        let max_lod = 2;

        let lod_config = LODConfig::uniform(r4, max_lod);
        lod_config.validate();
        let mut clipmap = ClipMap::new(lod_config);
        let events = clipmap.update(IVec3::ZERO);
        assert!(events.to_unload.is_empty());
        assert_eq!(
            events.to_load.len() as i32,
            r4_volume + r4_shell_volume * (max_lod as i32)
        );

        let keys: HashSet<_> = events.to_load.iter().collect();
        assert_eq!(events.to_load.len(), keys.len());
        for x in -16..16 {
            for y in -16..16 {
                for z in -16..16 {
                    let key: ChunkKey;
                    if (x >= -4 && x < 4) && (y >= -4 && y < 4) && (z >= -4 && z < 4) {
                        key = ChunkKey::new(x, y, z, 0);
                    } else if (x >= -8 && x < 8) && (y >= -8 && y < 8) && (z >= -8 && z < 8) {
                        key = ChunkKey::new(x.div_euclid(2), y.div_euclid(2), z.div_euclid(2), 1);
                    } else {
                        key = ChunkKey::new(x.div_euclid(4), y.div_euclid(4), z.div_euclid(4), 2);
                    }

                    assert!(keys.contains(&key));

                    let wv = IVec3::new(x, y, z) * CHUNK_SIZE;
                    let calc_key = clipmap.wv_to_chunk_key(wv);
                    assert_eq!(calc_key, Some(key));
                }
            }
        }
    }

    #[test]
    fn no_shift() {
        let lod_config = LODConfig::uniform(4, 2);
        lod_config.validate();

        let max_no_shift: f32 = 32.0 * 2_f32.powi(2) * 0.75; // 32 voxels * 2^max_lod * hysteresis
        for x in (-1..=1).step_by(2) {
            for y in (-1..=1).step_by(2) {
                for z in (-1..=1).step_by(2) {
                    let pos = Vec3::new(x as f32, y as f32, z as f32) * max_no_shift;
                    let mut clipmap = ClipMap::new(lod_config.clone());
                    clipmap.update(IVec3::ZERO);
                    let events = clipmap.update(pos.as_ivec3());
                    assert!(!events.anchor_shifted);
                    assert!(events.to_load.is_empty());
                    assert!(events.to_unload.is_empty());
                }
            }
        }
    }

    #[test]
    fn shifts() {
        let lod_config = LODConfig::uniform(4, 2);
        lod_config.validate();

        let sr = || -4..4;
        let hr = || -2..2;

        // _s=shell, _h = hole, _u = unload, _l = load, _p = postive, _n = negative
        let l0_u_p = || -4..0;
        let l0_l_p = || 4..8;
        let l1_s_u_p = || -4..-2;
        let l1_h_u_p = || 2..4;
        let l1_s_l_p = || 4..6;
        let l1_h_l_p = || -2..0;
        let l2_s_u_p = || -4..-3;
        let l2_h_u_p = || 2..3;
        let l2_s_l_p = || 4..5;
        let l2_h_l_p = || -2..-1;

        let l0_u_n = || 0..4;
        let l0_l_n = || -8..-4;
        let l1_s_u_n = || 2..4;
        let l1_h_u_n = || -4..-2;
        let l1_s_l_n = || -6..-4;
        let l1_h_l_n = || 0..2;
        let l2_s_u_n = || 3..4;
        let l2_h_u_n = || -3..-2;
        let l2_s_l_n = || -5..-4;
        let l2_h_l_n = || 1..2;

        let permutations = [
            // Postive
            (
                IVec3::new(1, 0, 0),
                [
                    [l0_u_p(), sr(), sr()],
                    [l0_l_p(), sr(), sr()],
                    [l1_s_u_p(), sr(), sr()],
                    [l1_h_u_p(), hr(), hr()],
                    [l1_s_l_p(), sr(), sr()],
                    [l1_h_l_p(), hr(), hr()],
                    [l2_s_u_p(), sr(), sr()],
                    [l2_h_u_p(), hr(), hr()],
                    [l2_s_l_p(), sr(), sr()],
                    [l2_h_l_p(), hr(), hr()],
                ],
            ),
            (
                IVec3::new(0, 1, 0),
                [
                    [sr(), l0_u_p(), sr()],
                    [sr(), l0_l_p(), sr()],
                    [sr(), l1_s_u_p(), sr()],
                    [hr(), l1_h_u_p(), hr()],
                    [sr(), l1_s_l_p(), sr()],
                    [hr(), l1_h_l_p(), hr()],
                    [sr(), l2_s_u_p(), sr()],
                    [hr(), l2_h_u_p(), hr()],
                    [sr(), l2_s_l_p(), sr()],
                    [hr(), l2_h_l_p(), hr()],
                ],
            ),
            (
                IVec3::new(0, 0, 1),
                [
                    [sr(), sr(), l0_u_p()],
                    [sr(), sr(), l0_l_p()],
                    [sr(), sr(), l1_s_u_p()],
                    [hr(), hr(), l1_h_u_p()],
                    [sr(), sr(), l1_s_l_p()],
                    [hr(), hr(), l1_h_l_p()],
                    [sr(), sr(), l2_s_u_p()],
                    [hr(), hr(), l2_h_u_p()],
                    [sr(), sr(), l2_s_l_p()],
                    [hr(), hr(), l2_h_l_p()],
                ],
            ),
            // Negative
            (
                IVec3::new(-1, 0, 0),
                [
                    [l0_u_n(), sr(), sr()],
                    [l0_l_n(), sr(), sr()],
                    [l1_s_u_n(), sr(), sr()],
                    [l1_h_u_n(), hr(), hr()],
                    [l1_s_l_n(), sr(), sr()],
                    [l1_h_l_n(), hr(), hr()],
                    [l2_s_u_n(), sr(), sr()],
                    [l2_h_u_n(), hr(), hr()],
                    [l2_s_l_n(), sr(), sr()],
                    [l2_h_l_n(), hr(), hr()],
                ],
            ),
            (
                IVec3::new(0, -1, 0),
                [
                    [sr(), l0_u_n(), sr()],
                    [sr(), l0_l_n(), sr()],
                    [sr(), l1_s_u_n(), sr()],
                    [hr(), l1_h_u_n(), hr()],
                    [sr(), l1_s_l_n(), sr()],
                    [hr(), l1_h_l_n(), hr()],
                    [sr(), l2_s_u_n(), sr()],
                    [hr(), l2_h_u_n(), hr()],
                    [sr(), l2_s_l_n(), sr()],
                    [hr(), l2_h_l_n(), hr()],
                ],
            ),
            (
                IVec3::new(0, 0, -1),
                [
                    [sr(), sr(), l0_u_n()],
                    [sr(), sr(), l0_l_n()],
                    [sr(), sr(), l1_s_u_n()],
                    [hr(), hr(), l1_h_u_n()],
                    [sr(), sr(), l1_s_l_n()],
                    [hr(), hr(), l1_h_l_n()],
                    [sr(), sr(), l2_s_u_n()],
                    [hr(), hr(), l2_h_u_n()],
                    [sr(), sr(), l2_s_l_n()],
                    [hr(), hr(), l2_h_l_n()],
                ],
            ),
        ];

        for (shift_vector, perm) in permutations {
            let mut clipmap = ClipMap::new(lod_config.clone());
            clipmap.update(IVec3::ZERO);

            let max_no_shift = 32.0 * 2_f32.powi(2) * 0.75;
            let shift_dist = max_no_shift + 1.0;
            let pos = shift_vector.as_vec3() * shift_dist;
            let mut events = clipmap.update(pos.as_ivec3());
            assert!(events.anchor_shifted);
            assert!(!events.to_load.is_empty());
            assert!(!events.to_unload.is_empty());

            let [mut u_lod0s, mut u_lod1s, mut u_lod2s] =
                [HashSet::new(), HashSet::new(), HashSet::new()];

            for chunk_key in events.to_unload.drain(..) {
                match chunk_key.lod {
                    0 => u_lod0s.insert(chunk_key),
                    1 => u_lod1s.insert(chunk_key),
                    2 => u_lod2s.insert(chunk_key),
                    _ => unreachable!(),
                };
            }

            let [mut l_lod0s, mut l_lod1s, mut l_lod2s] =
                [HashSet::new(), HashSet::new(), HashSet::new()];

            for chunk_key in events.to_load.drain(..) {
                match chunk_key.lod {
                    0 => l_lod0s.insert(chunk_key),
                    1 => l_lod1s.insert(chunk_key),
                    2 => l_lod2s.insert(chunk_key),
                    _ => unreachable!(),
                };
            }

            let [
                [l0_s_u_xr, l0_s_u_yr, l0_s_u_zr],
                [l0_s_l_xr, l0_s_l_yr, l0_s_l_zr],
                [l1_s_u_xr, l1_s_u_yr, l1_s_u_zr],
                [l1_h_u_xr, l1_h_u_yr, l1_h_u_zr],
                [l1_s_l_xr, l1_s_l_yr, l1_s_l_zr],
                [l1_h_l_xr, l1_h_l_yr, l1_h_l_zr],
                [l2_s_u_xr, l2_s_u_yr, l2_s_u_zr],
                [l2_h_u_xr, l2_h_u_yr, l2_h_u_zr],
                [l2_s_l_xr, l2_s_l_yr, l2_s_l_zr],
                [l2_h_l_xr, l2_h_l_yr, l2_h_l_zr],
            ] = perm;

            // ---- LOD0 ----
            // Unload: [−4,0]×[−4,4]×[−4,4]
            let unloads: Vec<_> = ranges_3d(l0_s_u_xr, l0_s_u_yr, l0_s_u_zr).collect();
            assert_eq!(u_lod0s.len(), unloads.len());
            for (x, y, z) in unloads {
                let chunk_key = ChunkKey::new(x, y, z, 0);
                assert!(u_lod0s.contains(&chunk_key))
            }
            // Load: [4,8]×[−4,4]×[−4,4]
            let loads: Vec<_> = ranges_3d(l0_s_l_xr, l0_s_l_yr, l0_s_l_zr).collect();
            assert_eq!(l_lod0s.len(), loads.len());
            for (x, y, z) in loads {
                let chunk_key = ChunkKey::new(x, y, z, 0);
                assert!(l_lod0s.contains(&chunk_key))
            }

            // ---- LOD1 ----
            // Unload: [−4,-2]×[−4,4]×[−4,4] + [2,4]×[-2,2]×[-2,2]
            let shell_unloads: Vec<_> = ranges_3d(l1_s_u_xr, l1_s_u_yr, l1_s_u_zr).collect();
            let hole_unloads: Vec<_> = ranges_3d(l1_h_u_xr, l1_h_u_yr, l1_h_u_zr).collect();
            let unloads: Vec<_> = shell_unloads.into_iter().chain(hole_unloads).collect();
            assert_eq!(u_lod1s.len(), unloads.len());
            for (x, y, z) in unloads {
                let chunk_key = ChunkKey::new(x, y, z, 1);
                assert!(u_lod1s.contains(&chunk_key))
            }
            // Load: [4,6]×[−4,4]×[−4,4] + [-2,0]×[-2,2]×[-2,2]
            let shell_loads: Vec<_> = ranges_3d(l1_s_l_xr, l1_s_l_yr, l1_s_l_zr).collect();
            let hole_loads: Vec<_> = ranges_3d(l1_h_l_xr, l1_h_l_yr, l1_h_l_zr).collect();
            let loads: Vec<_> = shell_loads.into_iter().chain(hole_loads).collect();
            assert_eq!(l_lod1s.len(), loads.len());
            for (x, y, z) in loads {
                let chunk_key = ChunkKey::new(x, y, z, 1);
                assert!(l_lod1s.contains(&chunk_key))
            }

            // ---- LOD2 ----
            // Unload: [−4,-3]×[−4,4]×[−4,4] + [2,3]×[-2,2]×[-2,2]
            let shell_unloads: Vec<_> = ranges_3d(l2_s_u_xr, l2_s_u_yr, l2_s_u_zr).collect();
            let hole_unloads: Vec<_> = ranges_3d(l2_h_u_xr, l2_h_u_yr, l2_h_u_zr).collect();
            let unloads: Vec<_> = shell_unloads.into_iter().chain(hole_unloads).collect();
            assert_eq!(u_lod2s.len(), unloads.len());
            for (x, y, z) in unloads {
                let chunk_key = ChunkKey::new(x, y, z, 2);
                assert!(u_lod2s.contains(&chunk_key))
            }
            // Load: [4,5]×[−4,4]×[−4,4] + [-2,-1]×[-2,2]×[-2,2]
            let shell_loads: Vec<_> = ranges_3d(l2_s_l_xr, l2_s_l_yr, l2_s_l_zr).collect();
            let hole_loads: Vec<_> = ranges_3d(l2_h_l_xr, l2_h_l_yr, l2_h_l_zr).collect();
            let loads: Vec<_> = shell_loads.into_iter().chain(hole_loads).collect();
            assert_eq!(l_lod2s.len(), loads.len());
            for (x, y, z) in loads {
                let chunk_key = ChunkKey::new(x, y, z, 2);
                assert!(l_lod2s.contains(&chunk_key))
            }
        }
    }

    #[test]
    fn teleport() {
        let lod_config = LODConfig::uniform(4, 2);
        lod_config.validate();
        let mut clipmap = ClipMap::new(lod_config.clone());
        let events = clipmap.update(IVec3::ZERO);
        let max_load = events.to_load.len();

        let full_shift_dist = 32.0 * 2_f32.powi(2);
        let teleport_dist = full_shift_dist * 8.0;

        let events = clipmap.update(IVec3::new(teleport_dist as i32, 0, 0));
        assert!(events.anchor_shifted);
        assert_eq!(events.to_load.len(), max_load);
        assert_eq!(events.to_unload.len(), max_load);
    }

    fn ranges_3d(
        xr: std::ops::Range<i32>,
        yr: std::ops::Range<i32>,
        zr: std::ops::Range<i32>,
    ) -> impl Iterator<Item = (i32, i32, i32)> {
        xr.flat_map(move |x| {
            let zr = zr.clone();
            let yr = yr.clone();

            yr.flat_map(move |y| zr.clone().map(move |z| (x, y, z)))
        })
    }
}
