use std::marker::PhantomData;

use block_mesh::{
    GreedyQuadsBuffer, MergeVoxel, RIGHT_HANDED_Y_UP_CONFIG, Voxel, VoxelVisibility, greedy_quads,
    ndshape::ConstShape3u32,
};
use glam::IVec3;

use crate::{
    block::{BLOCK_FACES, BlockFace, ChunkeeVoxel, VoxelId},
    chunk::Chunk,
    coords::{CHUNK_SIZE, CHUNK_VOLUME, ChunkVector, cv_to_wv},
};

#[derive(Debug)]
pub struct ChunkMeshData {
    pub indices: Vec<u32>,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub layers: Vec<f32>,
    pub tangents: Vec<f32>,
}

impl Default for ChunkMeshData {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            layers: Vec::new(),
            tangents: Vec::new(),
        }
    }
}

impl ChunkMeshData {
    fn idx(&self, face: usize, vert: usize) -> usize {
        self.indices[face * 3 + vert] as usize
    }
}

impl mikktspace::Geometry for ChunkMeshData {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.idx(face, vert)]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.idx(face, vert)]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.idx(face, vert)]
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bi_tangent: [f32; 3],
        _f_mag_s: f32,
        _f_mag_t: f32,
        bi_tangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let global_vert_idx = self.idx(face, vert);

        let tangents_idx = global_vert_idx * 4;
        let w = if bi_tangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        self.tangents[tangents_idx] = tangent[0];
        self.tangents[tangents_idx + 1] = tangent[1];
        self.tangents[tangents_idx + 2] = tangent[2];
        self.tangents[tangents_idx + 3] = w;
    }
}

pub const PADDED_CHUNK_SIZE: i32 = CHUNK_SIZE + 2;
pub const PADDED_CHUNK_VOLUME: usize =
    (PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE) as usize;

#[derive(Debug, Clone)]
pub struct PaddedChunk {
    pub voxels: [VoxelId; PADDED_CHUNK_VOLUME],
}

impl Default for PaddedChunk {
    fn default() -> Self {
        Self {
            voxels: [VoxelId::AIR; PADDED_CHUNK_VOLUME],
        }
    }
}

impl PaddedChunk {
    #[inline]
    fn pos_to_idx(pos: IVec3) -> usize {
        (pos.x + pos.y * PADDED_CHUNK_SIZE + pos.z * PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE) as usize
    }

    pub fn set_voxel(&mut self, pos: IVec3, voxel: VoxelId) {
        self.voxels[Self::pos_to_idx(pos)] = voxel;
    }
}

pub const PADDED_CHUNK_SIZE_U32: u32 = (CHUNK_SIZE + 2) as u32;
pub type Chunk34ShapeU =
    ConstShape3u32<PADDED_CHUNK_SIZE_U32, PADDED_CHUNK_SIZE_U32, PADDED_CHUNK_SIZE_U32>;

#[derive(Clone, Copy)]
#[repr(transparent)]
struct MesherVoxel<T>(VoxelId, PhantomData<T>);

impl<V: ChunkeeVoxel> Voxel for MesherVoxel<V> {
    fn get_visibility(&self) -> VoxelVisibility {
        let type_id = self.0.type_id();
        let voxel_enum = V::from(type_id);
        voxel_enum.get_visibility()
    }
}

impl<V: ChunkeeVoxel> MergeVoxel for MesherVoxel<V> {
    type MergeValue = VoxelId; // Merge based on the entire VoxelId

    fn merge_value(&self) -> Self::MergeValue {
        self.0 // Return the full VoxelId (type + rotation)
    }
}

pub fn mesh_chunk<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: &Box<Chunk>,
    neighbors: &Box<[Option<Chunk>; 6]>,
) -> ChunkMeshData {
    let padded_chunk = build_padded_chunk(chunk, neighbors);
    let mesher_voxels: &[MesherVoxel<V>] = unsafe {
        std::slice::from_raw_parts(
            padded_chunk.voxels.as_ptr() as *const MesherVoxel<V>,
            padded_chunk.voxels.len(),
        )
    };
    let mut buffer = GreedyQuadsBuffer::new((CHUNK_VOLUME * 6) as usize);

    greedy_quads(
        &mesher_voxels,
        &Chunk34ShapeU {},
        [0; 3],
        [(CHUNK_SIZE + 1) as u32; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );
    build_mesh_from_quads::<V>(&buffer, &padded_chunk, cv)
}

fn build_padded_chunk(chunk: &Box<Chunk>, neighbors: &Box<[Option<Chunk>; 6]>) -> PaddedChunk {
    let mut padded_chunk = PaddedChunk::default();

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let local_pos = IVec3::new(x, y, z);
                let padded_pos = local_pos + 1; // Offset into the center
                let voxel = chunk.get_voxel(local_pos);
                padded_chunk.set_voxel(padded_pos, voxel);
            }
        }
    }

    for face in BLOCK_FACES {
        if let Some(neighbor_chunk) = &neighbors[face as usize] {
            match face {
                BlockFace::Left => {
                    for y in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(CHUNK_SIZE - 1, y, z);
                            let padded_pos = IVec3::new(0, y + 1, z + 1);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
                BlockFace::Right => {
                    for y in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(0, y, z);
                            let padded_pos = IVec3::new(PADDED_CHUNK_SIZE - 1, y + 1, z + 1);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
                BlockFace::Bottom => {
                    for x in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(x, CHUNK_SIZE - 1, z);
                            let padded_pos = IVec3::new(x + 1, 0, z + 1);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
                BlockFace::Top => {
                    for x in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(x, 0, z);
                            let padded_pos = IVec3::new(x + 1, PADDED_CHUNK_SIZE - 1, z + 1);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
                BlockFace::Front => {
                    for x in 0..CHUNK_SIZE {
                        for y in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(x, y, CHUNK_SIZE - 1);
                            let padded_pos = IVec3::new(x + 1, y + 1, 0);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
                BlockFace::Back => {
                    for x in 0..CHUNK_SIZE {
                        for y in 0..CHUNK_SIZE {
                            let neighbor_pos = IVec3::new(x, y, 0);
                            let padded_pos = IVec3::new(x + 1, y + 1, PADDED_CHUNK_SIZE - 1);
                            padded_chunk
                                .set_voxel(padded_pos, neighbor_chunk.get_voxel(neighbor_pos));
                        }
                    }
                }
            }
        }
    }

    padded_chunk
}

fn build_mesh_from_quads<V: ChunkeeVoxel>(
    buffer: &GreedyQuadsBuffer,
    padded_chunk: &PaddedChunk,
    cv: ChunkVector,
) -> ChunkMeshData {
    let chunk_offset = cv_to_wv(cv).to_array().map(|s| s as f32);
    let num_quads = buffer.quads.num_quads();
    let num_indices = num_quads * 6;
    let num_vertices = num_quads * 4;

    let mut mesh = ChunkMeshData {
        indices: Vec::with_capacity(num_indices),
        positions: Vec::with_capacity(num_vertices),
        normals: Vec::with_capacity(num_vertices),
        uvs: Vec::with_capacity(num_vertices),
        layers: Vec::with_capacity(num_vertices),
        tangents: vec![0.0; num_vertices * 4],
    };

    for (face, quads) in RIGHT_HANDED_Y_UP_CONFIG
        .faces
        .iter()
        .zip(buffer.quads.groups.iter())
    {
        for quad in quads.iter() {
            let start_vertex = mesh.positions.len() as u32;
            let mut new_indices = face.quad_mesh_indices(start_vertex);

            new_indices.swap(1, 2);
            new_indices.swap(4, 5);
            mesh.indices.extend_from_slice(&new_indices);

            let mut new_positions = face.quad_mesh_positions(quad, 1.0);
            for v in &mut new_positions {
                v[0] = v[0] - 1.0 + chunk_offset[0];
                v[1] = v[1] - 1.0 + chunk_offset[1];
                v[2] = v[2] - 1.0 + chunk_offset[2];
            }
            mesh.positions.extend_from_slice(&new_positions);
            mesh.normals.extend_from_slice(&face.quad_mesh_normals());
            mesh.uvs.extend_from_slice(&face.tex_coords(
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                true,
                quad,
            ));

            let voxel_pos = IVec3::from_array(quad.minimum.map(|s| s as i32));
            let voxel_id = padded_chunk.voxels[PaddedChunk::pos_to_idx(voxel_pos)];
            let voxel_enum = V::from(voxel_id.type_id());
            let normal = IVec3::from_array(face.signed_normal().to_array());
            let block_face = BlockFace::from_normal(normal);
            let rotation = voxel_id.rotation();
            let layer_id = voxel_enum.texture_id(block_face, rotation) as f32;
            mesh.layers
                .extend_from_slice(&[layer_id, layer_id, layer_id, layer_id]);
        }
    }

    mikktspace::generate_tangents(&mut mesh);

    mesh
}
