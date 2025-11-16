// https://github.com/bitshifter/glam-rs/discussions/564

use glam::{BVec3A, IVec3, Vec3A, Vec3Swizzles};

/// use glam::Vec3A;
/// const MAX_STEPS: usize = 128;
/// let ro = Vec3A::new(0.0, 1.8, 0.0); // origin
/// let rd = Vec3A::new(0.0, 0.0, 1.0); // direction
/// let mut dda = DDAState::from_pos_and_dir(ro, rd);
///
/// for _ in 0..MAX_STEPS {
/// 	// check if dda.next_voxelpos is 'inside' a voxel
/// 	// use the dda.hit_* functions here
/// 	if false { break; }
/// 	// no hit? step the ray
/// 	dda.step_mut();
/// }
#[derive(Debug, Clone, Copy)]
pub struct DDAState {
    /// The initial position of the ray.
    pub ray_origin: Vec3A,

    /// The direction the ray is going.
    pub ray_direction: Vec3A,

    /// The current largest component of/to the next boundary.
    pub max_boundary_mask: BVec3A,

    /// Per-component signum to the next voxel position.
    pub diff_voxelpos: IVec3, // = signum(dir)

    /// The next voxel position to be visited.
    pub next_voxelpos: IVec3,

    /// Per-component distance to next boundary plane.
    pub diff_boundary: Vec3A,

    /// The distances to the next voxel boundary on the X/Y/Z axes.
    pub next_boundary: Vec3A,
}

// This exists so DDAState can be directly used as Iterator.
impl Iterator for DDAState {
    type Item = DDAState;
    fn next(&mut self) -> Option<Self::Item> {
        self.step_mut();
        Some(*self)
    }
}

impl DDAState {
    #[inline(always)]
    pub fn from_pos_to_pos(ray_origin: glam::Vec3A, ray_target: glam::Vec3A) -> Self {
        let ray_direction = (ray_target - ray_origin).normalize_or_zero();
        Self::from_pos_and_dir(ray_origin, ray_direction)
    }

    #[inline(always)]
    pub fn from_pos_and_dir(ray_origin: glam::Vec3A, ray_direction: glam::Vec3A) -> Self {
        let ray_origin_grid = ray_origin.floor();
        let ray_origin_grid_i = ray_origin_grid.as_ivec3();

        let mut ray_direction = ray_direction;
        if ray_direction.x == 0.0 {
            ray_direction.x = 0.00001;
        }
        if ray_direction.y == 0.0 {
            ray_direction.y = 0.00001;
        }
        if ray_direction.z == 0.0 {
            ray_direction.z = 0.00001;
        }
        let ray_direction = ray_direction.normalize();

        let ray_sign = ray_direction.signum(); // Step
        let ray_dir_inv = Vec3A::ONE / ray_direction; // invDir

        // vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;
        let ray_dist = ray_dir_inv.abs();
        let next_dist =
            (ray_sign * (ray_origin_grid - ray_origin) + (ray_sign * 0.5) + 0.5) * ray_dist;

        Self {
            ray_origin,
            ray_direction,

            // Values that stay fixed.
            diff_boundary: ray_dist,
            diff_voxelpos: ray_sign.as_ivec3(),

            max_boundary_mask: BVec3A::FALSE,
            next_boundary: next_dist,
            next_voxelpos: ray_origin_grid_i,
        }
    }

    #[inline(always)]
    pub fn step_mut(&mut self) {
        // Find the longest component of `next_boundary` and store it.
        self.max_boundary_mask = self
            .next_boundary
            .xyz()
            .cmple(self.next_boundary.yzx().min(self.next_boundary.zxy()));
        self.next_voxelpos += self.diff_voxelpos * IVec3::from(self.max_boundary_mask);
        self.next_boundary += self.diff_boundary * Vec3A::from(self.max_boundary_mask);
    }

    #[inline(always)]
    pub fn step_new(&self) -> Self {
        let mut next = *self;
        next.step_mut();
        next
    }

    #[inline(always)]
    pub fn hit_distance(&self) -> f32 {
        ((self.next_boundary - self.diff_boundary) * Vec3A::from(self.max_boundary_mask))
            .element_sum()
    }

    /// Global position of the hit, but at a different distance.
    #[inline(always)]
    pub fn hit_position_at_distance(&self, ray_distance: f32) -> Vec3A {
        self.ray_origin + (self.ray_direction * ray_distance)
    }

    /// Global position of the hit on the next voxels boundary.
    #[inline(always)]
    pub fn hit_position(&self) -> Vec3A {
        self.hit_position_at_distance(self.hit_distance())
    }

    /// Global position and distance of/to the hit on the next voxels boundary.
    #[inline(always)]
    pub fn hit_position_and_distance(&self) -> glam::Vec4 {
        let ray_distance = self.hit_distance();
        (self.hit_position_at_distance(ray_distance), ray_distance).into()
    }

    /// Local position of the hit on the next voxels boundary.
    #[inline(always)]
    pub fn hit_boundary(&self) -> Vec3A {
        self.hit_position() - self.next_voxelpos.as_vec3a()
    }

    /// Normal of the hit on the next voxels boundary.
    #[inline(always)]
    pub fn hit_normal(&self) -> Vec3A {
        -self.diff_voxelpos.as_vec3a() * Vec3A::from(self.max_boundary_mask)
    }
}
