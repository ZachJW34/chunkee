use std::fmt;

use block_mesh::Voxel as BlockMeshVoxel;
use glam::{IVec3, Mat3, Vec3};
use serde::{Deserialize, Serialize};

pub type BlockTypeId = u8;
pub type TextureId = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Rotation(u8);

impl Rotation {
    const MATRICES: [Mat3; 24] = [
        // Pointing +Y up
        Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::Z),
        Mat3::from_cols(Vec3::Z, Vec3::Y, Vec3::NEG_X),
        Mat3::from_cols(Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z),
        Mat3::from_cols(Vec3::NEG_Z, Vec3::Y, Vec3::X),
        // Pointing -Y up
        Mat3::from_cols(Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z),
        Mat3::from_cols(Vec3::NEG_Z, Vec3::NEG_Y, Vec3::NEG_X),
        Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Y, Vec3::Z),
        Mat3::from_cols(Vec3::Z, Vec3::NEG_Y, Vec3::X),
        // Pointing +Z up
        Mat3::from_cols(Vec3::X, Vec3::Z, Vec3::NEG_Y),
        Mat3::from_cols(Vec3::NEG_Y, Vec3::Z, Vec3::NEG_X),
        Mat3::from_cols(Vec3::NEG_X, Vec3::Z, Vec3::Y),
        Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::X),
        // Pointing -Z up
        Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y),
        Mat3::from_cols(Vec3::Y, Vec3::NEG_Z, Vec3::NEG_X),
        Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Z, Vec3::NEG_Y),
        Mat3::from_cols(Vec3::NEG_Y, Vec3::NEG_Z, Vec3::X),
        // Pointing +X up
        Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::Z),
        Mat3::from_cols(Vec3::Z, Vec3::X, Vec3::NEG_Y),
        Mat3::from_cols(Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
        Mat3::from_cols(Vec3::NEG_Z, Vec3::X, Vec3::Y),
        // Pointing -X up
        Mat3::from_cols(Vec3::Z, Vec3::NEG_X, Vec3::Y),
        Mat3::from_cols(Vec3::Y, Vec3::NEG_X, Vec3::NEG_Z),
        Mat3::from_cols(Vec3::NEG_Z, Vec3::NEG_X, Vec3::NEG_Y),
        Mat3::from_cols(Vec3::NEG_Y, Vec3::NEG_X, Vec3::Z),
    ];

    #[inline]
    pub fn matrix(self) -> Mat3 {
        Self::MATRICES[self.0 as usize]
    }

    // TODO: Allow players to place blocks with camera relative rotations
    #[inline]
    pub fn from_dirs(_world_forward: IVec3, _world_up: IVec3) -> Self {
        Rotation(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct VoxelId(u16);

mod layout {
    pub const TYPE_ID_BITS: u32 = 8;
    pub const ROTATION_BITS: u32 = 5;

    pub const TYPE_ID_SHIFT: u32 = 0;
    pub const ROTATION_SHIFT: u32 = TYPE_ID_SHIFT + TYPE_ID_BITS;

    pub const TYPE_ID_MASK: u16 = (1 << TYPE_ID_BITS) - 1;
    pub const ROTATION_MASK: u16 = (1 << ROTATION_BITS) - 1;
}

const _: () = {
    let total_bits = layout::TYPE_ID_BITS + layout::ROTATION_BITS;
    assert!(total_bits <= 16, "VoxelId fields exceed 16 bits");
};

impl VoxelId {
    pub const AIR: Self = Self(0);

    pub fn new(type_id: BlockTypeId, rotation: Rotation) -> Self {
        let type_id_part = type_id as u16;
        let rotation_part = rotation.0 as u16;
        Self((type_id_part << layout::TYPE_ID_SHIFT) | (rotation_part << layout::ROTATION_SHIFT))
    }

    #[inline(always)]
    pub fn type_id(self) -> BlockTypeId {
        ((self.0 >> layout::TYPE_ID_SHIFT) & layout::TYPE_ID_MASK) as BlockTypeId
    }

    #[inline(always)]
    pub fn with_type_id(mut self, type_id: BlockTypeId) -> Self {
        self.0 &= !(layout::TYPE_ID_MASK << layout::TYPE_ID_SHIFT);
        self.0 |= (type_id as u16) << layout::TYPE_ID_SHIFT;
        self
    }

    #[inline(always)]
    pub fn rotation(self) -> Rotation {
        Rotation(((self.0 >> layout::ROTATION_SHIFT) & layout::ROTATION_MASK) as u8)
    }

    #[inline(always)]
    pub fn with_rotation(mut self, rotation: Rotation) -> Self {
        self.0 &= !(layout::ROTATION_MASK << layout::ROTATION_SHIFT);
        self.0 |= (rotation.0 as u16) << layout::ROTATION_SHIFT;
        self
    }
}

impl fmt::Debug for VoxelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "VoxelId(AIR)");
        }
        f.debug_struct("VoxelId")
            .field("type", &self.type_id())
            .field("rot", &self.rotation())
            .finish()
    }
}

// number is used to represent neighbor_mask bit locations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockFace {
    Bottom = 0, // -Y
    Top = 1,    // +Y
    Front = 2,  // -Z
    Back = 3,   // +Z
    Left = 4,   // -X
    Right = 5,  // +X
}

impl BlockFace {
    pub fn into_normal(&self) -> IVec3 {
        match self {
            BlockFace::Bottom => IVec3::NEG_Y,
            BlockFace::Top => IVec3::Y,
            BlockFace::Front => IVec3::NEG_Z,
            BlockFace::Back => IVec3::Z,
            BlockFace::Left => IVec3::NEG_X,
            BlockFace::Right => IVec3::X,
        }
    }

    pub fn from_normal(normal: IVec3) -> Self {
        match normal {
            IVec3::NEG_Y => BlockFace::Bottom,
            IVec3::Y => BlockFace::Top,
            IVec3::NEG_Z => BlockFace::Front,
            IVec3::Z => BlockFace::Back,
            IVec3::NEG_X => BlockFace::Left,
            IVec3::X => BlockFace::Right,
            _ => panic!("Invalid normal"),
        }
    }

    pub fn opposite(&self) -> Self {
        match self {
            BlockFace::Bottom => BlockFace::Top,
            BlockFace::Top => BlockFace::Bottom,
            BlockFace::Front => BlockFace::Back,
            BlockFace::Back => BlockFace::Front,
            BlockFace::Left => BlockFace::Right,
            BlockFace::Right => BlockFace::Left,
        }
    }
}

pub fn neighbors_mask_to_faces(mask: u8) -> [Option<BlockFace>; 6] {
    std::array::from_fn(|i| {
        if (mask >> i) & 1 != 0 {
            BlockFace::try_from(i as u8).ok()
        } else {
            None
        }
    })
}

pub type NeighborsMask = u8;

impl TryFrom<NeighborsMask> for BlockFace {
    type Error = &'static str;

    fn try_from(value: NeighborsMask) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BlockFace::Bottom),
            1 => Ok(BlockFace::Top),
            2 => Ok(BlockFace::Front),
            3 => Ok(BlockFace::Back),
            4 => Ok(BlockFace::Left),
            5 => Ok(BlockFace::Right),
            _ => Err("Invalid value: must be between 0 and 5 for BlockFace"),
        }
    }
}

// Any fn that operates on BLOCK_FACES and returns [Option<BlockFace>; 6]
// should guarentee that Some(face) respects the idx below
// DON"T CHANGE ORDER UNLESS YOU REFACTOR MAPPINGS ABOVE
pub const BLOCK_FACES: [BlockFace; 6] = [
    BlockFace::Bottom,
    BlockFace::Top,
    BlockFace::Front,
    BlockFace::Back,
    BlockFace::Left,
    BlockFace::Right,
];

#[derive(Debug, Clone, Copy)]
pub enum TextureMapping {
    All(TextureId),
    PerFace([TextureId; 6]),
    None,
}

pub trait Block: Default {
    fn name(&self) -> &'static str;
    fn texture_mapping(&self) -> TextureMapping;
    fn is_solid(&self) -> bool {
        true
    }
    fn texture_id(&self, world_face: BlockFace, rotation: Rotation) -> TextureId {
        match self.texture_mapping() {
            TextureMapping::All(id) => id,
            TextureMapping::PerFace(ids) => {
                let world_normal = world_face.into_normal();
                let inverse_rotation_matrix = rotation.matrix().transpose(); // For orthonormal matrices, inverse is transpose.
                let local_normal = inverse_rotation_matrix * world_normal.as_vec3();
                let local_face = BlockFace::from_normal(local_normal.round().as_ivec3());
                ids[local_face as usize]
            }
            TextureMapping::None => 0,
        }
    }
}

pub trait Where<T> {}
impl<T, U> Where<U> for T {}

pub trait ChunkeeVoxel:
    Block + Copy + Default + From<BlockTypeId> + Into<BlockTypeId> + BlockMeshVoxel
{
}

// #[macro_export]
// macro_rules! define_voxels {
//     (
//         pub enum $name:ident {
//             $( ($id:expr, $type:ident) ),*
//             ,
//         }
//     ) => {
//         #[derive(Debug, Clone, Copy)]
//         pub enum $name {
//             $( $type($type) ),*
//         }

//         impl $crate::block::Block for $name {
//             fn name(&self) -> &'static str {
//                 match self {
//                     $( $name::$type(inner) => inner.name() ),*
//                 }
//             }
//             fn texture_mapping(&self, rotation: $crate::block::Rotation) -> $crate::block::TextureMapping {
//                 match self {
//                     $( $name::$type(inner) => inner.texture_mapping(rotation) ),*
//                 }
//             }
//             fn is_solid(&self) -> bool {
//                 match self {
//                     $( $name::$type(inner) => inner.is_solid() ),*
//                 }
//             }
//             fn light_emitted(&self) -> $crate::block::LightLevel {
//                 match self {
//                     $( $name::$type(inner) => inner.light_emitted() ),*
//                 }
//             }
//         }
//         impl Default for $name {
//             fn default() -> Self {
//                 <Self as $crate::block::Voxel>::from_type_id(0)
//             }
//         }
//         impl $crate::block::Voxel for $name {
//             fn from_type_id(type_id: $crate::block::BlockTypeId) -> Self {
//                 match type_id {
//                     $( $id => $name::$type(<$type>::default()), )*
//                     _ => Self::default()
//                 }
//             }
//             fn to_type_id(&self) -> $crate::block::BlockTypeId {
//                 match self {
//                     $( $name::$type(_) => $id, )*
//                 }
//             }
//         }
//     }
// }
