use std::marker::PhantomData;

use block_mesh::{
    GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, RIGHT_HANDED_Y_UP_CONFIG, UnorientedQuad,
    Voxel, VoxelVisibility, greedy_quads,
    ndshape::{ConstShape, ConstShape3u32},
};
use glam::{IVec3, Vec3};

use crate::{
    block::{BLOCK_FACES, BlockFace, ChunkeeVoxel, Rotation, VoxelCollision, VoxelId},
    chunk::{CHUNK_SIDE_32, Chunk},
    coords::{ChunkVector, cv_to_wv},
};

type Shape34 = ConstShape3u32<34, 34, 34>;

#[derive(Debug)]
pub struct ChunkMeshGroup {
    pub opaque: ChunkMeshData,
    pub translucent: ChunkMeshData,
}

impl Default for ChunkMeshGroup {
    fn default() -> Self {
        Self {
            opaque: ChunkMeshData::default(),
            translucent: ChunkMeshData::default(),
        }
    }
}

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

#[derive(Clone, Copy)]
#[repr(transparent)]
struct MesherVoxel<T>(VoxelId, PhantomData<T>);

impl<V: ChunkeeVoxel> Voxel for MesherVoxel<V> {
    fn get_visibility(&self) -> VoxelVisibility {
        self.0.to_voxel::<V>().visibilty()
    }
}

// TODO: Dont merge water, or allow options for not merging certain faces for use in shaders (like water)
impl<V: ChunkeeVoxel> MergeVoxel for MesherVoxel<V> {
    type MergeValue = VoxelId;

    fn merge_value(&self) -> Self::MergeValue {
        self.0 // Return the full VoxelId (type + rotation)
    }
}

pub fn mesh_chunk<V: ChunkeeVoxel>(
    cv: ChunkVector,
    chunk: Box<Chunk>,
    neighbors: Box<[Chunk; 6]>,
) -> ChunkMeshGroup {
    let padded_side = CHUNK_SIDE_32 + 2;
    let padded_volume = padded_side * padded_side * padded_side;
    let mut padded_voxels = vec![VoxelId::AIR; padded_volume as usize];

    build_padded_buffer::<V>(*chunk, &neighbors, &mut padded_voxels);

    let mesher_voxels: &[MesherVoxel<V>] = unsafe {
        std::slice::from_raw_parts(
            padded_voxels.as_ptr() as *const MesherVoxel<V>,
            padded_voxels.len(),
        )
    };

    run_greedy_mesher(cv, mesher_voxels)
}

fn run_greedy_mesher<V: ChunkeeVoxel>(
    cv: ChunkVector,
    padded_voxels: &[MesherVoxel<V>],
) -> ChunkMeshGroup {
    run_greedy_mesher_generic::<V, Shape34>(cv, padded_voxels, &Shape34 {})
}

fn run_greedy_mesher_generic<V: ChunkeeVoxel, S: ConstShape<3, Coord = u32>>(
    cv: ChunkVector,
    padded_voxels: &[MesherVoxel<V>],
    shape: &S,
) -> ChunkMeshGroup {
    let mut buffer = GreedyQuadsBuffer::new(padded_voxels.len());
    let padded_side = S::ARRAY[0];
    let side = padded_side - 2;

    greedy_quads(
        padded_voxels,
        shape,
        [0; 3],
        [side + 1; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );

    build_mesh_from_quads::<V, S>(&buffer, padded_voxels, cv)
}

fn build_padded_buffer<V: ChunkeeVoxel>(
    chunk: Chunk,
    neighbors: &[Chunk; 6],
    padded_voxels: &mut [VoxelId],
) {
    let padded_side = CHUNK_SIDE_32 + 2;
    let pos_to_idx =
        |p: IVec3| (p.x + p.y * padded_side + p.z * padded_side * padded_side) as usize;

    for z in 0..CHUNK_SIDE_32 {
        for y in 0..CHUNK_SIDE_32 {
            for x in 0..CHUNK_SIDE_32 {
                let local_pos = IVec3::new(x, y, z);
                let padded_pos = local_pos + 1;
                padded_voxels[pos_to_idx(padded_pos)] = chunk.get_voxel(local_pos);
            }
        }
    }

    for face in BLOCK_FACES {
        let neighbor_chunk = neighbors[face as usize];

        for i in 0..CHUNK_SIDE_32 {
            for j in 0..CHUNK_SIDE_32 {
                let (n_x, n_y, n_z, p_x, p_y, p_z) = match face {
                    BlockFace::Right => (0, j, i, CHUNK_SIDE_32 + 1, j + 1, i + 1),
                    BlockFace::Left => (CHUNK_SIDE_32 - 1, j, i, 0, j + 1, i + 1),
                    BlockFace::Top => (i, 0, j, i + 1, CHUNK_SIDE_32 + 1, j + 1),
                    BlockFace::Bottom => (i, CHUNK_SIDE_32 - 1, j, i + 1, 0, j + 1),
                    BlockFace::Back => (i, j, 0, i + 1, j + 1, CHUNK_SIDE_32 + 1),
                    BlockFace::Front => (i, j, CHUNK_SIDE_32 - 1, i + 1, j + 1, 0),
                };

                let neighbor_pos = IVec3::new(n_x, n_y, n_z);
                let padded_pos = IVec3::new(p_x, p_y, p_z);

                let voxel = neighbor_chunk.get_voxel(neighbor_pos);
                padded_voxels[pos_to_idx(padded_pos)] = voxel;
            }
        }
    }
}

fn build_mesh_from_quads<V: ChunkeeVoxel, S: ConstShape<3, Coord = u32>>(
    buffer: &GreedyQuadsBuffer,
    padded_voxels: &[MesherVoxel<V>],
    cv: ChunkVector,
) -> ChunkMeshGroup {
    let chunk_offset = cv_to_wv(cv).to_array().map(|s| s as f32);
    let num_quads = buffer.quads.num_quads();
    let num_indices = num_quads * 6;
    let num_vertices = num_quads * 4;

    let mut opaque = ChunkMeshData {
        indices: Vec::with_capacity(num_indices),
        positions: Vec::with_capacity(num_vertices),
        normals: Vec::with_capacity(num_vertices),
        uvs: Vec::with_capacity(num_vertices),
        layers: Vec::with_capacity(num_vertices),
        tangents: vec![],
    };

    let mut translucent = ChunkMeshData {
        indices: Vec::with_capacity(num_indices),
        positions: Vec::with_capacity(num_vertices),
        normals: Vec::with_capacity(num_vertices),
        uvs: Vec::with_capacity(num_vertices),
        layers: Vec::with_capacity(num_vertices),
        tangents: vec![],
    };

    for (face, quads) in RIGHT_HANDED_Y_UP_CONFIG
        .faces
        .iter()
        .zip(buffer.quads.groups.iter())
    {
        for quad in quads.iter() {
            let quad_minimum = quad.minimum;
            let voxel_idx = S::linearize(quad_minimum) as usize;
            let voxel_id = padded_voxels[voxel_idx];
            let voxel = V::from(voxel_id.0.type_id());
            let rotation = voxel_id.0.rotation();
            let mesh = if voxel.visibilty() == VoxelVisibility::Opaque {
                &mut opaque
            } else {
                &mut translucent
            };

            process_quad(mesh, quad, face, chunk_offset, voxel, rotation);
        }
    }

    optimize_mesh(&mut opaque);
    generate_tangents(&mut opaque);
    optimize_mesh(&mut translucent);
    generate_tangents(&mut translucent);

    ChunkMeshGroup {
        opaque,
        translucent,
    }
}

fn optimize_mesh(mesh_data: &mut ChunkMeshData) {
    if mesh_data.indices.is_empty() {
        return;
    }

    let num_vertices = mesh_data.positions.len();

    meshopt::optimize_vertex_cache_in_place(&mut mesh_data.indices, num_vertices);

    let positions_adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&mesh_data.positions),
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .unwrap();

    meshopt::optimize_overdraw_in_place(&mut mesh_data.indices, &positions_adapter, 1.05);

    let remap_table = meshopt::optimize_vertex_fetch_remap(&mesh_data.indices, num_vertices);

    mesh_data.positions =
        meshopt::remap_vertex_buffer(&mesh_data.positions, num_vertices, &remap_table);
    mesh_data.normals =
        meshopt::remap_vertex_buffer(&mesh_data.normals, num_vertices, &remap_table);
    mesh_data.uvs = meshopt::remap_vertex_buffer(&mesh_data.uvs, num_vertices, &remap_table);
    mesh_data.layers = meshopt::remap_vertex_buffer(&mesh_data.layers, num_vertices, &remap_table);

    mesh_data.indices =
        meshopt::remap_index_buffer(Some(&mesh_data.indices), num_vertices, &remap_table);
}

fn process_quad<V: ChunkeeVoxel>(
    mesh_data: &mut ChunkMeshData,
    quad: &UnorientedQuad,
    face: &OrientedBlockFace,
    chunk_offset: [f32; 3],
    voxel: V,
    rotation: Rotation,
) {
    let start_vertex = mesh_data.positions.len() as u32;
    let mut new_indices = face.quad_mesh_indices(start_vertex);
    new_indices.swap(1, 2);
    new_indices.swap(4, 5);

    // positions
    let mut new_positions = face.quad_mesh_positions(quad, 1.0);

    let chunk_offset_vec = Vec3::from(chunk_offset);
    for v in &mut new_positions {
        let scaled_local_pos = Vec3::from(*v) - Vec3::ONE;
        *v = (scaled_local_pos + chunk_offset_vec).to_array();
    }

    // uvs
    let new_uvs = face.tex_coords(RIGHT_HANDED_Y_UP_CONFIG.u_flip_face, true, quad);

    // layers
    let normal = IVec3::from_array(face.signed_normal().to_array());
    let block_face = BlockFace::from_normal(normal);
    let layer_id = voxel.texture_id(block_face, rotation) as f32;
    let new_layers = [layer_id; 4];

    mesh_data.indices.extend_from_slice(&new_indices);
    mesh_data.positions.extend_from_slice(&new_positions);
    mesh_data
        .normals
        .extend_from_slice(&face.quad_mesh_normals());
    mesh_data.uvs.extend_from_slice(&new_uvs);
    mesh_data.layers.extend_from_slice(&new_layers);
}

fn generate_tangents(mesh_data: &mut ChunkMeshData) {
    if mesh_data.positions.len() > 0 {
        mesh_data.tangents = vec![0.0; mesh_data.positions.len() * 4];
        mikktspace::generate_tangents(mesh_data);
    }
}

//////////////////// Physics ////////////////////////////////
#[derive(Clone, Copy)]
#[repr(transparent)]
struct PhysicsMesherVoxel<T>(VoxelId, PhantomData<T>);

impl<V: ChunkeeVoxel> Voxel for PhysicsMesherVoxel<V> {
    fn get_visibility(&self) -> VoxelVisibility {
        match self.0.to_voxel::<V>().collision() {
            VoxelCollision::None => VoxelVisibility::Empty,
            VoxelCollision::Solid => VoxelVisibility::Opaque,
        }
    }
}

impl<V: ChunkeeVoxel> MergeVoxel for PhysicsMesherVoxel<V> {
    type MergeValue = bool;

    fn merge_value(&self) -> Self::MergeValue {
        true
    }
}

pub fn mesh_physics_chunk<V: ChunkeeVoxel>(cv: ChunkVector, chunk: Box<Chunk>) -> Vec<Vec3> {
    let padded_side = CHUNK_SIDE_32 + 2;
    let padded_volume = padded_side * padded_side * padded_side;
    let mut padded_voxels = vec![VoxelId::AIR; padded_volume as usize];
    let pos_to_idx =
        |p: IVec3| (p.x + p.y * padded_side + p.z * padded_side * padded_side) as usize;

    for z in 0..CHUNK_SIDE_32 {
        for y in 0..CHUNK_SIDE_32 {
            for x in 0..CHUNK_SIDE_32 {
                let local_pos = IVec3::new(x, y, z);
                let padded_pos = local_pos + 1;
                padded_voxels[pos_to_idx(padded_pos)] = chunk.get_voxel(local_pos);
            }
        }
    }

    let mesher_voxels: &[PhysicsMesherVoxel<V>] = unsafe {
        std::slice::from_raw_parts(
            padded_voxels.as_ptr() as *const PhysicsMesherVoxel<V>,
            padded_voxels.len(),
        )
    };

    let mut buffer = GreedyQuadsBuffer::new(padded_voxels.len());
    greedy_quads(
        mesher_voxels,
        &Shape34 {},
        [0; 3],
        [33; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );

    let chunk_offset = cv_to_wv(cv).as_vec3();
    let mut triangles = Vec::with_capacity(buffer.quads.num_quads() * 2 * 3);

    for (face, quads) in RIGHT_HANDED_Y_UP_CONFIG
        .faces
        .iter()
        .zip(buffer.quads.groups.into_iter())
    {
        for quad in quads.into_iter() {
            let positions = face.quad_mesh_positions(&quad.into(), 1.0);

            let indices = face.quad_mesh_indices(0);

            let p = [
                Vec3::from_array(positions[0]),
                Vec3::from_array(positions[1]),
                Vec3::from_array(positions[2]),
                Vec3::from_array(positions[3]),
            ];

            // 4. Use the canonical indices to build the triangle soup correctly.
            for &i in indices.iter() {
                let local_pos = p[i as usize];
                let world_pos = (local_pos - Vec3::ONE) + chunk_offset;
                triangles.push(world_pos);
            }
        }
    }

    triangles
}
