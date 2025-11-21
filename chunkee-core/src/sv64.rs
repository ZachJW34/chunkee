use std::{
    collections::hash_map::Entry,
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock, Weak},
};

use fxhash::{FxHashMap, FxHasher64};
use glam::IVec3;

use crate::{block::VoxelId, chunk::Chunk, coords::LocalVector};

pub const REGION_32_8: [IVec3; 64] = [
    IVec3::new(0, 0, 0),
    IVec3::new(8, 0, 0),
    IVec3::new(16, 0, 0),
    IVec3::new(24, 0, 0),
    IVec3::new(0, 8, 0),
    IVec3::new(8, 8, 0),
    IVec3::new(16, 8, 0),
    IVec3::new(24, 8, 0),
    IVec3::new(0, 16, 0),
    IVec3::new(8, 16, 0),
    IVec3::new(16, 16, 0),
    IVec3::new(24, 16, 0),
    IVec3::new(0, 24, 0),
    IVec3::new(8, 24, 0),
    IVec3::new(16, 24, 0),
    IVec3::new(24, 24, 0),
    IVec3::new(0, 0, 8),
    IVec3::new(8, 0, 8),
    IVec3::new(16, 0, 8),
    IVec3::new(24, 0, 8),
    IVec3::new(0, 8, 8),
    IVec3::new(8, 8, 8),
    IVec3::new(16, 8, 8),
    IVec3::new(24, 8, 8),
    IVec3::new(0, 16, 8),
    IVec3::new(8, 16, 8),
    IVec3::new(16, 16, 8),
    IVec3::new(24, 16, 8),
    IVec3::new(0, 24, 8),
    IVec3::new(8, 24, 8),
    IVec3::new(16, 24, 8),
    IVec3::new(24, 24, 8),
    IVec3::new(0, 0, 16),
    IVec3::new(8, 0, 16),
    IVec3::new(16, 0, 16),
    IVec3::new(24, 0, 16),
    IVec3::new(0, 8, 16),
    IVec3::new(8, 8, 16),
    IVec3::new(16, 8, 16),
    IVec3::new(24, 8, 16),
    IVec3::new(0, 16, 16),
    IVec3::new(8, 16, 16),
    IVec3::new(16, 16, 16),
    IVec3::new(24, 16, 16),
    IVec3::new(0, 24, 16),
    IVec3::new(8, 24, 16),
    IVec3::new(16, 24, 16),
    IVec3::new(24, 24, 16),
    IVec3::new(0, 0, 24),
    IVec3::new(8, 0, 24),
    IVec3::new(16, 0, 24),
    IVec3::new(24, 0, 24),
    IVec3::new(0, 8, 24),
    IVec3::new(8, 8, 24),
    IVec3::new(16, 8, 24),
    IVec3::new(24, 8, 24),
    IVec3::new(0, 16, 24),
    IVec3::new(8, 16, 24),
    IVec3::new(16, 16, 24),
    IVec3::new(24, 16, 24),
    IVec3::new(0, 24, 24),
    IVec3::new(8, 24, 24),
    IVec3::new(16, 24, 24),
    IVec3::new(24, 24, 24),
];

pub const REGION_8_2: [IVec3; 64] = [
    IVec3::new(0, 0, 0),
    IVec3::new(2, 0, 0),
    IVec3::new(4, 0, 0),
    IVec3::new(6, 0, 0),
    IVec3::new(0, 2, 0),
    IVec3::new(2, 2, 0),
    IVec3::new(4, 2, 0),
    IVec3::new(6, 2, 0),
    IVec3::new(0, 4, 0),
    IVec3::new(2, 4, 0),
    IVec3::new(4, 4, 0),
    IVec3::new(6, 4, 0),
    IVec3::new(0, 6, 0),
    IVec3::new(2, 6, 0),
    IVec3::new(4, 6, 0),
    IVec3::new(6, 6, 0),
    IVec3::new(0, 0, 2),
    IVec3::new(2, 0, 2),
    IVec3::new(4, 0, 2),
    IVec3::new(6, 0, 2),
    IVec3::new(0, 2, 2),
    IVec3::new(2, 2, 2),
    IVec3::new(4, 2, 2),
    IVec3::new(6, 2, 2),
    IVec3::new(0, 4, 2),
    IVec3::new(2, 4, 2),
    IVec3::new(4, 4, 2),
    IVec3::new(6, 4, 2),
    IVec3::new(0, 6, 2),
    IVec3::new(2, 6, 2),
    IVec3::new(4, 6, 2),
    IVec3::new(6, 6, 2),
    IVec3::new(0, 0, 4),
    IVec3::new(2, 0, 4),
    IVec3::new(4, 0, 4),
    IVec3::new(6, 0, 4),
    IVec3::new(0, 2, 4),
    IVec3::new(2, 2, 4),
    IVec3::new(4, 2, 4),
    IVec3::new(6, 2, 4),
    IVec3::new(0, 4, 4),
    IVec3::new(2, 4, 4),
    IVec3::new(4, 4, 4),
    IVec3::new(6, 4, 4),
    IVec3::new(0, 6, 4),
    IVec3::new(2, 6, 4),
    IVec3::new(4, 6, 4),
    IVec3::new(6, 6, 4),
    IVec3::new(0, 0, 6),
    IVec3::new(2, 0, 6),
    IVec3::new(4, 0, 6),
    IVec3::new(6, 0, 6),
    IVec3::new(0, 2, 6),
    IVec3::new(2, 2, 6),
    IVec3::new(4, 2, 6),
    IVec3::new(6, 2, 6),
    IVec3::new(0, 4, 6),
    IVec3::new(2, 4, 6),
    IVec3::new(4, 4, 6),
    IVec3::new(6, 4, 6),
    IVec3::new(0, 6, 6),
    IVec3::new(2, 6, 6),
    IVec3::new(4, 6, 6),
    IVec3::new(6, 6, 6),
];

pub const REGION_2_1: [IVec3; 8] = [
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 1),
];

#[derive(Debug)]
pub struct HashedNode {
    pub hash: u64,
    pub node: NodeKind,
}

impl HashedNode {
    fn new_uniform(voxel: VoxelId) -> Self {
        let mut hasher = FxHasher64::default();
        0u8.hash(&mut hasher); // tag for Uniform
        voxel.hash(&mut hasher);
        let hash = hasher.finish();

        Self {
            hash,
            node: NodeKind::Uniform(UniformNode { voxel }),
        }
    }

    fn new_branch(mask: u64, children: Box<[Arc<HashedNode>]>) -> Self {
        let mut hasher = FxHasher64::default();
        1u8.hash(&mut hasher); // tag for branch

        mask.hash(&mut hasher);
        for (idx, child) in children.iter().enumerate() {
            idx.hash(&mut hasher);
            child.hash.hash(&mut hasher);
        }

        let hash = hasher.finish();

        Self {
            hash,
            node: NodeKind::Branch(BranchNode { mask, children }),
        }
    }

    fn new_leaf(voxels: [VoxelId; 8]) -> Self {
        let mut hasher = FxHasher64::default();
        2u8.hash(&mut hasher); // tag for leaf

        for (idx, voxel) in voxels.iter().enumerate() {
            idx.hash(&mut hasher);
            voxel.hash(&mut hasher);
        }

        let hash = hasher.finish();

        Self {
            hash,
            node: NodeKind::Leaf(LeafNode { voxels }),
        }
    }
}

#[derive(Debug)]
pub struct BranchNode {
    pub mask: u64,
    pub children: Box<[Arc<HashedNode>]>,
}

#[derive(Debug)]
pub struct LeafNode {
    pub voxels: [VoxelId; 8],
}

#[derive(Debug)]
pub struct UniformNode {
    pub voxel: VoxelId,
}

#[derive(Debug)]
pub enum NodeKind {
    Branch(BranchNode),
    Leaf(LeafNode),
    Uniform(UniformNode),
}

const L0_NO_LEAF: &str = "S64T L0 node for 32^3 chunk cannot be a leaf";
const L1_NO_LEAF: &str = "S64T L1 node for 32^3 chunk cannot be a leaf";
const L2_NO_BRANCH: &str = "S64T L2 node for 32^3 chunk cannot be a branch";

static SHARED_AIR_UNIFORM: LazyLock<Arc<HashedNode>> =
    LazyLock::new(|| Arc::new(HashedNode::new_uniform(VoxelId::AIR)));

// TODO: Memory pool the nodes
#[derive(Debug, Clone)]
pub struct SV64Tree {
    pub root: Arc<HashedNode>,
}

impl Default for SV64Tree {
    fn default() -> Self {
        Self {
            root: SHARED_AIR_UNIFORM.clone(),
        }
    }
}

impl SV64Tree {
    pub fn from_chunk(&chunk: &Chunk) -> Self {
        let mut l0_mask = 0;
        let mut l0_children: Vec<Arc<HashedNode>> = Vec::with_capacity(64);
        let l0_voxel_sample = chunk.get_voxel(IVec3::ZERO);
        let mut l0_uniform = true;

        for (l1_idx, l1_v) in REGION_32_8.into_iter().enumerate() {
            let mut l1_mask = 0;
            let mut l1_children: Vec<Arc<HashedNode>> = Vec::with_capacity(64);
            let l1_voxel_sample = chunk.get_voxel(l1_v);
            let mut l1_uniform = true;

            for (l2_idx, l2_offset) in REGION_8_2.into_iter().enumerate() {
                let l2_v = l1_v + l2_offset;
                let l2_voxel_sample = chunk.get_voxel(l2_v);
                let mut l2_uniform: bool = true;
                let voxels: [VoxelId; 8] = std::array::from_fn(|i| {
                    let lv: IVec3 = l2_v + REGION_2_1[i];
                    let voxel = chunk.get_voxel(lv);
                    l2_uniform &= l2_voxel_sample == voxel;
                    chunk.get_voxel(lv)
                });

                l1_uniform &= l2_uniform && l1_voxel_sample == l2_voxel_sample;

                if l2_uniform && l2_voxel_sample == VoxelId::AIR {
                    continue;
                }

                let l2 = if l2_uniform {
                    HashedNode::new_uniform(l2_voxel_sample)
                } else {
                    HashedNode::new_leaf(voxels)
                };

                l1_children.push(Arc::new(l2));
                l1_mask |= 1u64 << (l2_idx as u64);
            }

            l0_uniform &= l1_uniform && l0_voxel_sample == l1_voxel_sample;

            if l1_uniform && l1_voxel_sample == VoxelId::AIR {
                continue;
            }

            let l1 = if l1_uniform {
                Arc::new(HashedNode::new_uniform(l1_voxel_sample))
            } else {
                let children = l1_children.into_boxed_slice();
                let new_branch = HashedNode::new_branch(l1_mask, children);
                Arc::new(new_branch)
            };

            l0_children.push(l1);
            l0_mask |= 1u64 << (l1_idx as u64);
        }

        let l0 = if l0_uniform {
            Arc::new(HashedNode::new_uniform(l0_voxel_sample))
        } else {
            let children = l0_children.into_boxed_slice();
            let new_branch = HashedNode::new_branch(l0_mask, children);
            Arc::new(new_branch)
        };

        Self { root: l0 }
    }

    pub fn to_chunk(&self) -> Chunk {
        let mut chunk = Chunk::new();

        match &self.root.node {
            NodeKind::Leaf { .. } => unreachable!("{L0_NO_LEAF}"),
            NodeKind::Uniform(uniform) => {
                if uniform.voxel != VoxelId::AIR {
                    chunk.fill(uniform.voxel);
                }
            }
            NodeKind::Branch(l0) => {
                let mut child_idx = 0;
                for (l1_idx, l1_offset) in REGION_32_8.iter().enumerate() {
                    if (l0.mask >> l1_idx) & 1 == 1 {
                        match &l0.children[child_idx].node {
                            NodeKind::Leaf { .. } => unreachable!("{L1_NO_LEAF}"),
                            NodeKind::Uniform(uniform) => {
                                for z in 0..8 {
                                    for y in 0..8 {
                                        for x in 0..8 {
                                            let lv = *l1_offset + IVec3::new(x, y, z);
                                            chunk.set_voxel(lv, uniform.voxel);
                                        }
                                    }
                                }
                            }
                            NodeKind::Branch(l1) => {
                                let mut child_idx = 0;
                                for (l2_idx, l2_offset) in REGION_8_2.iter().enumerate() {
                                    if (l1.mask >> l2_idx) & 1 == 1 {
                                        match &l1.children[child_idx].node {
                                            NodeKind::Branch(_) => unreachable!("{L2_NO_BRANCH}"),
                                            NodeKind::Uniform(uniform) => {
                                                for l3_offset in REGION_2_1 {
                                                    let lv = l1_offset + l2_offset + l3_offset;
                                                    chunk.set_voxel(lv, uniform.voxel);
                                                }
                                            }
                                            NodeKind::Leaf(leaf) => {
                                                for (l3_idx, l3_offset) in
                                                    REGION_2_1.iter().enumerate()
                                                {
                                                    let lv = l1_offset + l2_offset + l3_offset;
                                                    chunk.set_voxel(lv, leaf.voxels[l3_idx]);
                                                }
                                            }
                                        }
                                        child_idx += 1;
                                    }
                                }
                            }
                        };

                        child_idx += 1;
                    }
                }
            }
        }

        chunk
    }

    pub fn set_voxels(&mut self, voxels: &[(LocalVector, VoxelId)]) {
        if voxels.is_empty() {
            return;
        }

        let mut nodes: FxHashMap<u64, FxHashMap<u64, Vec<(u64, VoxelId)>>> =
            FxHashMap::with_hasher(Default::default());

        for (lv, voxel) in voxels {
            let l1_idx = l1_idx_from_lv(*lv);
            let l2_idx = l2_idx_from_lv(*lv);
            let l3_idx = l3_idx_from_lv(*lv);

            let l1 = nodes.entry(l1_idx).or_insert_with(|| Default::default());
            let l2 = l1.entry(l2_idx).or_insert_with(|| Default::default());
            l2.push((l3_idx, *voxel));
        }

        let empty = NodeKind::Uniform(UniformNode {
            voxel: VoxelId::AIR,
        });

        let mut l0_children = FxHashMap::with_hasher(Default::default());

        match &self.root.node {
            NodeKind::Uniform(uniform) if uniform.voxel != VoxelId::AIR => {
                for i in 0..64 {
                    l0_children.insert(i, Arc::new(HashedNode::new_uniform(uniform.voxel)));
                }
            }
            NodeKind::Branch(l0_branch) => {
                let mut child_idx = 0;
                for i in 0..64 {
                    if (l0_branch.mask >> i) & 1 == 1 {
                        l0_children.insert(i, l0_branch.children[child_idx].clone());
                        child_idx += 1;
                    }
                }
            }
            _ => {}
        }

        for (l1_idx, l1) in nodes {
            let old_l1 = match &self.root.node {
                NodeKind::Leaf { .. } => unreachable!("{L0_NO_LEAF}"),
                NodeKind::Branch(branch) => {
                    if (branch.mask >> l1_idx) & 1 == 0 {
                        &empty
                    } else {
                        let child_idx = idx_from_mask(l1_idx, branch.mask);
                        &branch.children[child_idx].node
                    }
                }
                NodeKind::Uniform(_) => &self.root.node,
            };

            let mut l1_children = FxHashMap::with_hasher(Default::default());

            if let NodeKind::Branch(branch) = old_l1 {
                let mut child_idx = 0;
                for i in 0..64 {
                    if (branch.mask >> i) & 1 == 1 {
                        l1_children.insert(i, branch.children[child_idx].clone());
                        child_idx += 1;
                    }
                }
            };

            match old_l1 {
                NodeKind::Uniform(uniform) if uniform.voxel != VoxelId::AIR => {
                    for i in 0..64 {
                        l1_children.insert(i, Arc::new(HashedNode::new_uniform(uniform.voxel)));
                    }
                }
                NodeKind::Branch(l1_branch) => {
                    let mut child_idx = 0;
                    for i in 0..64 {
                        if (l1_branch.mask >> i) & 1 == 1 {
                            l1_children.insert(i, l1_branch.children[child_idx].clone());
                            child_idx += 1;
                        }
                    }
                }
                _ => {}
            }

            for (l2_idx, l2) in l1 {
                let old_l2 = match &old_l1 {
                    NodeKind::Leaf { .. } => unreachable!("{L1_NO_LEAF}"),
                    NodeKind::Branch(branch) => {
                        if (branch.mask >> l2_idx) & 1 == 0 {
                            &empty
                        } else {
                            let child_idx = idx_from_mask(l2_idx, branch.mask);
                            &branch.children[child_idx].node
                        }
                    }
                    NodeKind::Uniform { .. } => old_l1,
                };

                let mut voxels = match &old_l2 {
                    NodeKind::Branch { .. } => unreachable!("{L2_NO_BRANCH}"),
                    NodeKind::Leaf(leaf) => leaf.voxels,
                    NodeKind::Uniform(uniform) => [uniform.voxel; 8],
                };

                for (v_idx, voxel) in l2 {
                    voxels[v_idx as usize] = voxel;
                }

                let voxel_sample = &voxels[0];
                let uniform = voxels.iter().all(|v| v == voxel_sample);

                if uniform && *voxel_sample == VoxelId::AIR {
                    l1_children.remove(&l2_idx);
                    continue;
                }

                let l2 = if uniform {
                    HashedNode::new_uniform(*voxel_sample)
                } else {
                    HashedNode::new_leaf(voxels)
                };

                l1_children.insert(l2_idx, Arc::new(l2));
            }

            if l1_children.is_empty() {
                l0_children.remove(&l1_idx);
                continue;
            }

            let mut l1_children: Vec<_> = l1_children.into_iter().collect();
            let l1_voxel_sample = match &l1_children[0].1.node {
                NodeKind::Branch { .. } => unreachable!("{L2_NO_BRANCH}"),
                NodeKind::Leaf(leaf) => leaf.voxels[0],
                NodeKind::Uniform(uniform) => uniform.voxel,
            };
            let l1_uniform = l1_children.len() == 64 && l1_children.iter().all(|e| matches!(&e.1.node, NodeKind::Uniform(uniform) if uniform.voxel == l1_voxel_sample));

            let l1 = if l1_uniform {
                let l1 = HashedNode::new_uniform(l1_voxel_sample);
                Arc::new(l1)
            } else {
                l1_children.sort_by_key(|e| e.0);
                let mut mask = 0;
                let mut children = Vec::with_capacity(l1_children.len());
                for (l2_idx, l2) in l1_children.drain(..) {
                    mask |= 1u64 << (l2_idx as u64);
                    children.push(l2);
                }
                let l1 = HashedNode::new_branch(mask, children.into_boxed_slice());
                Arc::new(l1)
            };

            l0_children.insert(l1_idx, l1);
        }

        if l0_children.is_empty() {
            self.root = Arc::new(HashedNode::new_uniform(VoxelId::AIR));
        }

        let mut l0_children: Vec<_> = l0_children.into_iter().collect();
        let l0_voxel_sample = voxels[0].1;
        let l0_uniform = l0_children.len() == 64 && l0_children.iter().all(
            |e| matches!(&e.1.node, NodeKind::Uniform(uniform) if uniform.voxel == l0_voxel_sample),
        );

        let l0 = if l0_uniform {
            let l0 = HashedNode::new_uniform(l0_voxel_sample);
            Arc::new(l0)
        } else {
            l0_children.sort_by_key(|e| e.0);
            let mut mask = 0;
            let mut children = Vec::with_capacity(l0_children.len());
            for (l2_idx, l2) in l0_children.drain(..) {
                mask |= 1u64 << (l2_idx as u64);
                children.push(l2);
            }
            let l0 = HashedNode::new_branch(mask, children.into_boxed_slice());
            Arc::new(l0)
        };

        self.root = l0;
    }

    pub fn get_voxel(&self, lv: IVec3) -> VoxelId {
        let l0_idx = l1_idx_from_lv(lv);

        let current_node = match &self.root.node {
            NodeKind::Leaf { .. } => unreachable!("{L0_NO_LEAF}"),
            NodeKind::Uniform(uniform) => return uniform.voxel,
            NodeKind::Branch(branch) => {
                if (branch.mask >> l0_idx) & 1 == 0 {
                    return VoxelId::AIR;
                }
                let child_idx = idx_from_mask(l0_idx, branch.mask);
                &branch.children[child_idx]
            }
        };

        let l1_idx = l2_idx_from_lv(lv);

        let current_node = match &current_node.node {
            NodeKind::Leaf { .. } => unreachable!("{L1_NO_LEAF}"),
            NodeKind::Uniform(uniform) => return uniform.voxel,
            NodeKind::Branch(branch) => {
                if (branch.mask >> l1_idx) & 1 == 0 {
                    return VoxelId::AIR;
                }
                let child_idx = idx_from_mask(l1_idx, branch.mask);
                &branch.children[child_idx]
            }
        };

        let l2_idx = l3_idx_from_lv(lv);

        match &current_node.node {
            NodeKind::Branch { .. } => unreachable!("{L2_NO_BRANCH}"),
            NodeKind::Uniform(uniform) => return uniform.voxel,
            NodeKind::Leaf(leaf) => leaf.voxels[l2_idx as usize],
        }
    }
}

pub struct Interner {
    pub map: FxHashMap<u64, Weak<HashedNode>>,
    refresh_count: u32,
    count: u32,
}

impl Interner {
    pub fn new(refresh_count: u32) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(64 * 64, Default::default()),
            refresh_count,
            count: 0,
        }
    }

    pub fn intern_tree(&mut self, tree: SV64Tree) -> SV64Tree {
        self.count += 1;
        self.maybe_prune();

        let l0 = &tree.root;
        let new_l0 = match &l0.node {
            NodeKind::Leaf(_) => unreachable!("{L0_NO_LEAF}"),
            NodeKind::Uniform(_) => self.insert(tree.root),
            NodeKind::Branch(l0_branch) => {
                let new_l0_children: Vec<_> = l0_branch
                    .children
                    .iter()
                    .map(|l1| match &l1.node {
                        NodeKind::Leaf(_) => unreachable!("{L1_NO_LEAF}"),
                        NodeKind::Uniform(_) => self.insert(l1.clone()),
                        NodeKind::Branch(l1_branch) => {
                            let new_l1_children: Vec<_> = l1_branch
                                .children
                                .iter()
                                .map(|l2| self.insert(l2.clone()))
                                .collect();
                            let new_l1_node = HashedNode {
                                hash: l1.hash,
                                node: NodeKind::Branch(BranchNode {
                                    mask: l1_branch.mask,
                                    children: new_l1_children.into_boxed_slice(),
                                }),
                            };

                            self.insert(Arc::new(new_l1_node))
                        }
                    })
                    .collect();

                let new_l0_node = HashedNode {
                    hash: l0.hash,
                    node: NodeKind::Branch(BranchNode {
                        mask: l0_branch.mask,
                        children: new_l0_children.into_boxed_slice(),
                    }),
                };
                self.insert(Arc::new(new_l0_node))
            }
        };

        SV64Tree { root: new_l0 }
    }

    fn insert(&mut self, node: Arc<HashedNode>) -> Arc<HashedNode> {
        match self.map.entry(node.hash) {
            Entry::Occupied(mut occ) => {
                if let Some(existing) = occ.get().upgrade() {
                    existing
                } else {
                    occ.insert(Arc::downgrade(&node));
                    node
                }
            }
            Entry::Vacant(vac) => {
                vac.insert(Arc::downgrade(&node));
                node
            }
        }
    }

    fn maybe_prune(&mut self) {
        if self.count < self.refresh_count {
            return;
        }

        self.map.retain(|_k, v| v.upgrade().is_some());

        self.count = 0;
    }
}

#[inline(always)]
fn idx_from_mask(idx: u64, mask: u64) -> usize {
    (mask & ((1_u64 << idx) - 1)).count_ones() as usize
}

#[inline(always)]
fn l1_idx_from_lv(lv: IVec3) -> u64 {
    let l0_x = (lv.x >> 3) as u64;
    let l0_y = (lv.y >> 3) as u64;
    let l0_z = (lv.z >> 3) as u64;

    l0_x + l0_y * 4 + l0_z * 16
}

#[inline(always)]
fn l2_idx_from_lv(lv: IVec3) -> u64 {
    let l1_x = ((lv.x & 7) >> 1) as u64;
    let l1_y = ((lv.y & 7) >> 1) as u64;
    let l1_z = ((lv.z & 7) >> 1) as u64;

    l1_x + l1_y * 4 + l1_z * 16
}

#[inline(always)]
fn l3_idx_from_lv(lv: IVec3) -> u64 {
    let l2_x = (lv.x & 1) as u64;
    let l2_y = (lv.y & 1) as u64;
    let l2_z = (lv.z & 1) as u64;

    l2_x + l2_y * 2 + l2_z * 4
}

#[cfg(test)]
mod tests {
    use block_mesh::VoxelVisibility;

    use crate::{
        block::{Block, BlockTypeId, ChunkeeVoxel, TextureMapping, VoxelCollision},
        chunk::CHUNK_VOLUME_32,
    };

    use super::*;

    #[derive(Clone, Copy)]
    pub enum TestVoxels {
        Air = 0,
        Grass = 1,
    }

    impl From<TestVoxels> for BlockTypeId {
        fn from(voxel: TestVoxels) -> Self {
            voxel as BlockTypeId
        }
    }

    impl From<BlockTypeId> for TestVoxels {
        fn from(id: BlockTypeId) -> Self {
            match id {
                0 => TestVoxels::Air,
                1 => TestVoxels::Grass,
                _ => TestVoxels::Air,
            }
        }
    }

    impl Default for TestVoxels {
        fn default() -> Self {
            TestVoxels::Air
        }
    }

    impl Block for TestVoxels {
        fn name(&self) -> &'static str {
            match self {
                TestVoxels::Air => "Air",
                TestVoxels::Grass => "Grass",
            }
        }

        fn texture_mapping(&self) -> TextureMapping {
            match self {
                TestVoxels::Air => unreachable!(),
                TestVoxels::Grass => TextureMapping::All(0),
            }
        }

        fn collision(&self) -> VoxelCollision {
            match self {
                TestVoxels::Air => VoxelCollision::None,
                TestVoxels::Grass => VoxelCollision::Solid,
            }
        }

        fn visibilty(&self) -> VoxelVisibility {
            match self {
                TestVoxels::Air => VoxelVisibility::Empty,
                TestVoxels::Grass => VoxelVisibility::Opaque,
            }
        }
    }

    impl ChunkeeVoxel for TestVoxels {}

    #[test]
    fn from_chunk_to_chunk() {
        let mut in_chunk = Chunk::new();
        let voxel = VoxelId::new(TestVoxels::Grass.into());
        let positions: Vec<IVec3> = vec![
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 31),
            IVec3::new(31, 0, 0),
            IVec3::new(31, 0, 31),
            IVec3::new(0, 31, 0),
            IVec3::new(31, 31, 0),
            IVec3::new(0, 31, 31),
            IVec3::new(31, 31, 31),
        ];
        for lv in &positions {
            in_chunk.set_voxel(*lv, voxel);
        }

        let sv64 = SV64Tree::from_chunk(&in_chunk);
        for lv in &positions {
            assert_eq!(sv64.get_voxel(*lv), voxel);
        }

        let out_chunk = sv64.to_chunk();
        for idx in 0..CHUNK_VOLUME_32 {
            assert_eq!(&in_chunk.voxels[idx], &out_chunk.voxels[idx])
        }
    }

    #[test]
    fn l0_uniformity() {
        let grass_voxel = VoxelId::new(TestVoxels::Grass.into());

        let chunk = Chunk::new();

        let sv64 = SV64Tree::from_chunk(&chunk);
        match &sv64.root.node {
            NodeKind::Uniform(uniform) => {
                assert_eq!(uniform.voxel, VoxelId::AIR)
            }
            _ => unreachable!(),
        }

        let chunk = Chunk::with_voxel(grass_voxel);
        let sv64 = SV64Tree::from_chunk(&chunk);
        match &sv64.root.node {
            NodeKind::Uniform(uniform) => {
                assert_eq!(uniform.voxel, grass_voxel)
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn l1_uniformity() {
        let grass_voxel = VoxelId::new(TestVoxels::Grass.into());

        let mut chunk = Chunk::new();
        // Skip every other l1
        for l1_offset in REGION_32_8.iter().step_by(2) {
            for l2_offset in REGION_8_2 {
                for v_offset in REGION_2_1 {
                    let lv = l1_offset + l2_offset + v_offset;
                    chunk.set_voxel(lv, grass_voxel);
                }
            }
        }

        let sv64 = SV64Tree::from_chunk(&chunk);
        match &sv64.root.node {
            NodeKind::Branch(branch) => {
                // every other child is full due to step_by(2) above (...0101 = x5)
                assert_eq!(branch.mask, 0x5555555555555555);
                assert_eq!(branch.children.len(), 32);

                for child in &branch.children {
                    match &child.node {
                        NodeKind::Uniform(uniform) => {
                            assert_eq!(uniform.voxel, grass_voxel)
                        }
                        _ => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn l2_uniformity() {
        let grass_voxel = VoxelId::new(TestVoxels::Grass.into());

        let mut chunk = Chunk::new();
        for l1_offset in REGION_32_8 {
            // Skip every other l2
            for l2_offset in REGION_8_2.iter().step_by(2) {
                for v_offset in REGION_2_1 {
                    let lv = l1_offset + l2_offset + v_offset;
                    chunk.set_voxel(lv, grass_voxel);
                }
            }
        }

        let sv64 = SV64Tree::from_chunk(&chunk);
        match &sv64.root.node {
            NodeKind::Branch(branch) => {
                // every child is full
                assert_eq!(branch.mask, u64::MAX);
                assert_eq!(branch.children.len(), 64);

                for child in &branch.children {
                    match &child.node {
                        NodeKind::Branch(branch) => {
                            // every other child is full due to step_by(2) above (...0101 = x5)
                            assert_eq!(branch.mask, 0x5555555555555555);
                            assert_eq!(branch.children.len(), 32);

                            for child in &branch.children {
                                match &child.node {
                                    NodeKind::Uniform(uniform) => {
                                        assert_eq!(uniform.voxel, grass_voxel)
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn set_voxel() {
        let grass_voxel = VoxelId::new(TestVoxels::Grass.into());

        // let mut sv64 = SV64Tree::from_chunk(&Chunk::with_voxel(grass_voxel));
        // let lv = IVec3::new(31, 31, 31);
        // sv64.set_voxels(&[(lv, VoxelId::AIR)]);
        // assert_eq!(sv64.get_voxel(lv), VoxelId::AIR);
        // match &sv64.root.node {
        //     NodeKind::Branch(l0_branch) => {
        //         assert_eq!(l0_branch.mask, u64::MAX);
        //     }
        //     _ => unreachable!(),
        // }

        // // Testing that an entire l2 region is replaced with air
        // let mut sv64 = SV64Tree::from_chunk(&Chunk::with_voxel(grass_voxel));
        // let mut edits = Vec::new();
        // let l2_start = IVec3::ZERO;
        // for l2_v in REGION_2_1 {
        //     let lv = l2_start + l2_v;
        //     edits.push((lv, VoxelId::AIR));
        // }
        // sv64.set_voxels(&edits);
        // for (lv, v) in &edits {
        //     assert_eq!(sv64.get_voxel(*lv), *v);
        // }

        // Testing that an entire l1 region is replaced with air
        let mut sv64 = SV64Tree::from_chunk(&Chunk::with_voxel(grass_voxel));
        let mut edits = Vec::new();
        let l1_start = IVec3::ZERO;
        for l1_v in REGION_8_2 {
            for l2_v in REGION_2_1 {
                let lv = l1_start + l1_v + l2_v;
                edits.push((lv, VoxelId::AIR));
            }
        }
        sv64.set_voxels(&edits);
        for (lv, v) in &edits {
            assert_eq!(sv64.get_voxel(*lv), *v);
        }

        // let chunk = Chunk::new();
        // let mut sv64 = SV64Tree::from_chunk(&chunk);
        // for (l1_idx, l1_offset) in REGION_32_8.iter().enumerate() {
        //     for (l2_idx, l2_offset) in REGION_8_2.iter().enumerate() {
        //         for l3_offset in REGION_2_1 {
        //             let lv = l1_offset + l2_offset + l3_offset;
        //             assert_eq!(sv64.get_voxel(lv), VoxelId::AIR);

        //             sv64.set_voxels(&[(lv, grass_voxel)]);
        //             assert_eq!(sv64.get_voxel(lv), grass_voxel);
        //         }

        //         match &sv64.root.node {
        //             NodeKind::Branch(l0) => match &l0.children[l1_idx].node {
        //                 NodeKind::Branch(l1) => {
        //                     assert!(l2_idx < 63);
        //                     match &l1.children[l2_idx].node {
        //                         NodeKind::Uniform(uniform) => {
        //                             assert_eq!(uniform.voxel, grass_voxel)
        //                         }
        //                         _ => unreachable!(),
        //                     }
        //                 }
        //                 NodeKind::Uniform(uniform) => {
        //                     assert!(l2_idx == 63);
        //                     assert_eq!(uniform.voxel, grass_voxel)
        //                 }
        //                 _ => unreachable!(),
        //             },
        //             NodeKind::Uniform(uniform) => {
        //                 assert!(l1_idx == 63);
        //                 assert_eq!(uniform.voxel, grass_voxel);
        //             }
        //             _ => unreachable!(),
        //         }
        //     }
        // }
    }

    #[test]
    fn interning() {
        let grass_voxel = VoxelId::new(TestVoxels::Grass.into());

        let mut chunk = Chunk::new();
        chunk.set_voxel(IVec3::ZERO, grass_voxel);
        let mut interner = Interner::new(10000);
        let sv64t_1 = interner.intern_tree(SV64Tree::from_chunk(&chunk));
        let sv64t_2 = interner.intern_tree(SV64Tree::from_chunk(&chunk));

        assert!(Arc::ptr_eq(&sv64t_1.root, &sv64t_2.root));
        match (&sv64t_1.root.node, &sv64t_2.root.node) {
            (NodeKind::Branch(l0_1), NodeKind::Branch(l0_2)) => {
                let (l1_1, l1_2) = (&l0_1.children[0], &l0_2.children[0]);
                assert!(Arc::ptr_eq(l1_1, l1_2));

                match (&l1_1.node, &l1_2.node) {
                    (NodeKind::Branch(l1_1), NodeKind::Branch(l1_2)) => {
                        let (l2_1, l2_2) = (&l1_1.children[0], &l1_2.children[0]);
                        assert!(Arc::ptr_eq(l2_1, l2_2));
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}
