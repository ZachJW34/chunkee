mod conversions;
mod generator;
mod voxels;

use std::collections::HashMap;

use chunkee_core::{
    coords::{ChunkVector, WorldVector},
    glam::Vec3,
    meshing::ChunkMeshData,
    streaming::should_unload,
    world::{ChunkeeWorld, ChunkeeWorldConfig, VoxelRaycast},
};
use godot::{
    classes::{
        ArrayMesh, Input, MeshInstance3D, ShaderMaterial, StandardMaterial3D, SurfaceTool,
        base_material_3d::ShadingMode,
        mesh::{ArrayFormat, PrimitiveType},
    },
    global::Key,
    obj::NewAlloc,
    prelude::*,
};

use crate::{conversions::*, generator::WorldGenerator, voxels::MyVoxels};

struct ChunkeeGodotExtension;

#[gdextension]
unsafe impl ExtensionLibrary for ChunkeeGodotExtension {}

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct ChunkeeWorldNode {
    base: Base<Node3D>,
    voxel_world: ChunkeeWorld<MyVoxels>,
    rendered_chunks: HashMap<ChunkVector, Gd<Node3D>>,
    voxel_hit: Option<(WorldVector, MyVoxels)>,
    outline_node: Option<Gd<MeshInstance3D>>,

    // #[export]
    // pub texture_atlas: Option<Gd<CompressedTexture2DArray>>,
    #[export]
    pub opaque_material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub translucent_material: Option<Gd<ShaderMaterial>>,
}

#[godot_api]
impl INode3D for ChunkeeWorldNode {
    fn init(base: Base<Node3D>) -> Self {
        println!("Initializing ChunkeeWorldNode");
        let config = ChunkeeWorldConfig {
            radius_xz: 10,
            radius_y: 5,
            generator: Box::new(WorldGenerator::new()),
        };
        let voxel_world: ChunkeeWorld<MyVoxels> = ChunkeeWorld::new(config);

        Self {
            base,
            voxel_world,
            opaque_material: None,
            translucent_material: None,
            rendered_chunks: HashMap::new(),
            voxel_hit: None,
            outline_node: None,
        }
    }

    fn ready(&mut self) {
        self.voxel_world.enable_pipeline();
        let mut outline = create_voxel_outline();
        outline.set_visible(false);
        self.base_mut().add_child(&outline);
        self.outline_node = Some(outline);

        // Debug
        // self.voxel_world.update(camera_pos);
        // std::thread::sleep(std::time::Duration::from_millis(100));
        // self.voxel_world
        //     .set_voxel_at(MyVoxels::Grass, IVec3::new(0, 0, 0));
        // self.voxel_world
        //     .set_voxel_at(MyVoxels::Grass, IVec3::new(-1, 0, 0));
        // self.voxel_world
        //     .set_voxel_at(MyVoxels::Grass, IVec3::new(0, 0, -1));
        // self.voxel_world
        //     .set_voxel_at(MyVoxels::Grass, IVec3::new(0, -1, 0));
    }

    fn process(&mut self, _delta: f64) {
        if let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) {
            let camera_pos = Vec3::from_array(camera.get_global_position().to_array());
            self.voxel_world.update(camera.to_camera_data());
            self.render(
                camera_pos,
                self.voxel_world.radius_xz,
                self.voxel_world.radius_y,
            );

            let input = Input::singleton();

            if input.is_key_pressed(Key::SPACE)
                && let Some(hit) = self.voxel_hit.as_ref()
            {
                self.voxel_world.set_voxel_at(MyVoxels::Air, hit.0);
            }

            let forward_direction = -camera.get_global_transform().basis.col_c();
            match self
                .voxel_world
                .raycast_hit(camera_pos, forward_direction.as_vec3(), 5)
            {
                VoxelRaycast::Hit(hit) => {
                    self.voxel_hit = Some(hit);
                }
                VoxelRaycast::Miss => self.voxel_hit = None,
                VoxelRaycast::None => {}
            }
        } else {
            println!("Cannot update without camera")
        }
    }
}

const ARRAY_VERTEX: usize = 0;
const ARRAY_NORMAL: usize = 1;
const ARRAY_TANGENT: usize = 2;
// const ARRAY_COLOR: usize = 3;
const ARRAY_TEX_UV: usize = 4;
// const ARRAY_TEX_UV2: usize = 5;
const ARRAY_CUSTOM0: usize = 6;
// const ARRAY_CUSTOM1: usize = 7;
// const ARRAY_CUSTOM2: usize = 8;
// const ARRAY_CUSTOM3: usize = 9;
// const ARRAY_BONES: usize = 10;
// const ARRAY_WEIGHTS: usize = 11;
const ARRAY_INDEX: usize = 12;
const ARRAY_MAX: usize = 13;

#[godot_api]
impl ChunkeeWorldNode {
    fn render(&mut self, camera_pos: Vec3, radius_xz: u32, radius_y: u32) {
        let array_format = ArrayFormat::VERTEX
            | ArrayFormat::NORMAL
            | ArrayFormat::TANGENT
            | ArrayFormat::TEX_UV
            | ArrayFormat::INDEX
            | ArrayFormat::CUSTOM0
            | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

        self.rendered_chunks.retain(|cv, node| {
            if should_unload(*cv, camera_pos, radius_xz, radius_y) {
                node.queue_free();
                false
            } else {
                true
            }
        });

        if let (Some(opaque_material), Some(translucent_material)) = (
            &self.opaque_material.clone(),
            &self.translucent_material.clone(),
        ) {
            while let Some((cv, mesh_group)) = self.voxel_world.mesh_queue.pop() {
                if let Some(mut node) = self.rendered_chunks.remove(&cv) {
                    node.queue_free();
                }
                let mut parent = Node3D::new_alloc();

                if mesh_group.opaque.positions.len() > 0 {
                    let mesh_instance =
                        build_mesh_instance(mesh_group.opaque, opaque_material, array_format);
                    parent.add_child(&mesh_instance);
                }

                if mesh_group.translucent.positions.len() > 0 {
                    let mesh_instance = build_mesh_instance(
                        mesh_group.translucent,
                        translucent_material,
                        array_format,
                    );
                    parent.add_child(&mesh_instance);
                }
                self.base_mut().add_child(&parent);
                parent.set_owner(&*self.base_mut());

                self.rendered_chunks.insert(cv, parent);
            }

            if let Some(outline_node) = self.outline_node.as_mut() {
                // Check if a voxel is currently being hit
                if let Some((pos, _)) = self.voxel_hit {
                    // A voxel is hit: move the outline and make it visible
                    let new_pos = Vector3::new(pos.x as f32, pos.y as f32, pos.z as f32);
                    outline_node.set_position(new_pos);
                    outline_node.set_visible(true);
                } else {
                    // No voxel is hit: just hide the outline
                    outline_node.set_visible(false);
                }
            }
        }
    }
}

fn build_mesh_instance(
    mesh_data: ChunkMeshData,
    material: &Gd<ShaderMaterial>,
    array_format: ArrayFormat,
) -> Gd<MeshInstance3D> {
    let indices = PackedInt32Array::from_iter(mesh_data.indices.iter().map(|i| *i as i32));
    let positions =
        PackedVector3Array::from_iter(mesh_data.positions.iter().map(|p| Vector3::from_array(*p)));
    let normals =
        PackedVector3Array::from_iter(mesh_data.normals.iter().map(|n| Vector3::from_array(*n)));
    let tangents = PackedFloat32Array::from(mesh_data.tangents);
    let uvs =
        PackedVector2Array::from_iter(mesh_data.uvs.iter().map(|uv| Vector2::from_array(*uv)));
    let layers = PackedFloat32Array::from(mesh_data.layers);

    let mut array = VariantArray::new();
    array.resize(ARRAY_MAX, &Variant::nil());

    array.set(ARRAY_VERTEX, &positions.to_variant());
    array.set(ARRAY_NORMAL, &normals.to_variant());
    array.set(ARRAY_TANGENT, &tangents.to_variant());
    array.set(ARRAY_TEX_UV, &uvs.to_variant());
    array.set(ARRAY_INDEX, &indices.to_variant());
    array.set(ARRAY_CUSTOM0, &layers.to_variant());

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays_ex(PrimitiveType::TRIANGLES, &array)
        .flags(array_format)
        .done();

    mesh.surface_set_material(0, material);

    let mut mesh_instance = MeshInstance3D::new_alloc();
    mesh_instance.set_mesh(&mesh);
    mesh_instance
}

fn create_voxel_outline() -> Gd<MeshInstance3D> {
    let mut st = SurfaceTool::new_gd();

    st.begin(PrimitiveType::LINES);

    // Create a 1x1x1 cube. Add a small offset to avoid Z-fighting with the actual voxel mesh.
    let size = 1.001;
    let offset = -0.0005; // Helps center the outline on the voxel block

    // Define the 8 vertices of the cube
    let v = [
        Vector3::new(offset, offset, offset), // 0: bottom-left-front
        Vector3::new(size, offset, offset),   // 1: bottom-right-front
        Vector3::new(size, size, offset),     // 2: top-right-front
        Vector3::new(offset, size, offset),   // 3: top-left-front
        Vector3::new(offset, offset, size),   // 4: bottom-left-back
        Vector3::new(size, offset, size),     // 5: bottom-right-back
        Vector3::new(size, size, size),       // 6: top-right-back
        Vector3::new(offset, size, size),     // 7: top-left-back
    ];

    // Draw the 12 edges of the cube by adding pairs of vertices

    // Bottom face
    st.add_vertex(v[0]);
    st.add_vertex(v[1]);
    st.add_vertex(v[1]);
    st.add_vertex(v[5]);
    st.add_vertex(v[5]);
    st.add_vertex(v[4]);
    st.add_vertex(v[4]);
    st.add_vertex(v[0]);

    // Top face
    st.add_vertex(v[3]);
    st.add_vertex(v[2]);
    st.add_vertex(v[2]);
    st.add_vertex(v[6]);
    st.add_vertex(v[6]);
    st.add_vertex(v[7]);
    st.add_vertex(v[7]);
    st.add_vertex(v[3]);

    // Vertical connecting edges
    st.add_vertex(v[0]);
    st.add_vertex(v[3]);
    st.add_vertex(v[1]);
    st.add_vertex(v[2]);
    st.add_vertex(v[4]);
    st.add_vertex(v[7]);
    st.add_vertex(v[5]);
    st.add_vertex(v[6]);

    // Commit the surface to create the mesh.
    // .unwrap() is fine here since we know the operations are valid.
    let mut mesh = st.commit().unwrap();

    // Create a simple, unlit material for the outline
    let mut material = StandardMaterial3D::new_gd();
    material.set_shading_mode(ShadingMode::UNSHADED);
    material.set_albedo(Color::from_rgb(1.0, 1.0, 0.0)); // Bright Yellow

    // Apply the material to the mesh surface
    mesh.surface_set_material(0, &material);

    let mut mesh_instance = MeshInstance3D::new_alloc();
    mesh_instance.set_mesh(&mesh);
    mesh_instance
}
