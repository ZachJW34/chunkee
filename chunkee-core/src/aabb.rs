use glam::IVec3;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AABB {
    pub min: IVec3,
    pub max: IVec3,
}

impl AABB {
    pub fn new(p1: IVec3, p2: IVec3) -> Self {
        Self {
            min: p1.min(p2),
            max: p1.max(p2),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min.cmpge(self.max).any()
    }
    /// Calculates the intersection of two AABBs.
    ///
    /// If there is no overlap, an empty/invalid AABB is returned.
    pub fn intersection(&self, other: &Self) -> Self {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        Self { min, max }
    }

    /// Calculates the parts of this AABB that do not overlap with another AABB.
    ///
    /// This is effectively `self - other`. The result is a list of up to
    /// 6 non-overlapping AABBs that constitute the remaining volume.
    pub fn subtract(&self, other: &Self) -> Vec<AABB> {
        // First, find the actual intersection, which must be contained within `self`.
        let intersection = self.intersection(other);
        if intersection.is_empty() {
            return vec![*self]; // If no overlap, the difference is the whole box.
        }

        let mut result = Vec::new();

        // We slice `self` into up to 6 pieces around the `intersection` box.

        // 1. Box "below" the intersection (min Y)
        let below = AABB::new(
            self.min,
            IVec3::new(self.max.x, intersection.min.y, self.max.z),
        );
        if !below.is_empty() {
            result.push(below);
        }

        // 2. Box "above" the intersection (max Y)
        let above = AABB::new(
            IVec3::new(self.min.x, intersection.max.y, self.min.z),
            self.max,
        );
        if !above.is_empty() {
            result.push(above);
        }

        // 3. Box to the "left" of the intersection (min X)
        // Note: We clamp the YZ dimensions to the intersection's slice to avoid overlap.
        let left = AABB::new(
            IVec3::new(self.min.x, intersection.min.y, self.min.z),
            IVec3::new(intersection.min.x, intersection.max.y, self.max.z),
        );
        if !left.is_empty() {
            result.push(left);
        }

        // 4. Box to the "right" of the intersection (max X)
        let right = AABB::new(
            IVec3::new(intersection.max.x, intersection.min.y, self.min.z),
            IVec3::new(self.max.x, intersection.max.y, self.max.z),
        );
        if !right.is_empty() {
            result.push(right);
        }

        // 5. Box "behind" the intersection (min Z)
        let back = AABB::new(
            IVec3::new(intersection.min.x, intersection.min.y, self.min.z),
            IVec3::new(intersection.max.x, intersection.max.y, intersection.min.z),
        );
        if !back.is_empty() {
            result.push(back);
        }

        // 6. Box "in front" of the intersection (max Z)
        let front = AABB::new(
            IVec3::new(intersection.min.x, intersection.min.y, intersection.max.z),
            IVec3::new(intersection.max.x, intersection.max.y, self.max.z),
        );
        if !front.is_empty() {
            result.push(front);
        }

        result
    }

    /// Calculates the volumetric difference between two AABBs.
    ///
    /// Returns a tuple containing two lists of AABBs:
    /// 1. `removed`: The volume that is in `self` but not in `other`.
    /// 2. `added`: The volume that is in `other` but not in `self`.
    pub fn difference(&self, other: &Self) -> (Vec<AABB>, Vec<AABB>) {
        let self_minus_other = self.subtract(other);
        let other_minus_self = other.subtract(self);
        (self_minus_other, other_minus_self)
    }

    pub fn iter(&self) -> AABBIter {
        let is_empty = self.is_empty();
        AABBIter {
            min: self.min,
            max: self.max,
            current: self.min,
            done: is_empty, // Start as done if the AABB is empty
        }
    }
}

pub struct AABBIter {
    min: IVec3,
    max: IVec3,
    current: IVec3,
    done: bool,
}

impl Iterator for AABBIter {
    type Item = IVec3;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let current_pos = self.current;

        self.current.x += 1;
        if self.current.x >= self.max.x {
            self.current.x = self.min.x;
            self.current.y += 1;
            if self.current.y >= self.max.y {
                self.current.y = self.min.y;
                self.current.z += 1;
                if self.current.z >= self.max.z {
                    self.done = true;
                }
            }
        }

        Some(current_pos)
    }
}
