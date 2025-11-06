mod conversions;
mod generator;
mod voxels;

use crate::{conversions::*, generator::WorldGenerator, voxels::MyVoxels};
use chunkee_core::{
    coords::ChunkVector,
    glam::{IVec3, Vec3},
    hasher::VoxelHashMap,
    manager::{ChunkeeConfig, ChunkeeManager, Update, VoxelRaycast},
    meshing::{ChunkMeshData, ChunkMeshGroup, PhysicsMesh},
    streaming::ChunkRadius,
};
use godot::{
    classes::{
        CollisionShape3D, ConcavePolygonShape3D, IStaticBody3D, Input, MeshInstance3D,
        RenderingServer, ShaderMaterial, StandardMaterial3D, StaticBody3D, SurfaceTool,
        base_material_3d::ShadingMode,
        mesh::PrimitiveType as MeshPrimitiveType,
        rendering_server::{ArrayFormat, PrimitiveType as RenderingServerPrimitiveType},
    },
    obj::NewAlloc,
    prelude::*,
};

struct ChunkeeGodotExtension;

#[gdextension]
unsafe impl ExtensionLibrary for ChunkeeGodotExtension {}

#[derive(GodotClass)]
#[class(base=StaticBody3D)]
pub struct ChunkeeWorldNode {
    base: Base<StaticBody3D>,
    chunkee_manager: ChunkeeManager<MyVoxels>,
    rendered_chunks: VoxelHashMap<Vec<(Rid, Rid)>>,
    physics_chunks: VoxelHashMap<Gd<CollisionShape3D>>,
    physics_debug_meshes: VoxelHashMap<Gd<MeshInstance3D>>,
    world_scenario: Rid,
    voxel_raycast: VoxelRaycast<MyVoxels>,
    outline_node: Option<Gd<MeshInstance3D>>,
    pub show_physics_debug_mesh: bool,
    #[export]
    pub opaque_material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub translucent_material: Option<Gd<ShaderMaterial>>,
    // #[export]
    pub voxel_size: f32,
}

#[godot_api]
impl IStaticBody3D for ChunkeeWorldNode {
    fn init(base: Base<StaticBody3D>) -> Self {
        println!("Initializing ChunkeeWorldNode");
        let voxel_size = 1.0;
        let config = ChunkeeConfig {
            radius: ChunkRadius(25),
            generator: Box::new(WorldGenerator::new()),
            voxel_size,
            thread_count: 4,
        };
        let chunkee_manager: ChunkeeManager<MyVoxels> = ChunkeeManager::new(config);

        Self {
            base,
            chunkee_manager,
            world_scenario: Rid::Invalid,
            rendered_chunks: Default::default(),
            physics_chunks: Default::default(),
            physics_debug_meshes: Default::default(),
            opaque_material: None,
            translucent_material: None,
            voxel_raycast: VoxelRaycast::None,
            outline_node: None,
            show_physics_debug_mesh: false,
            voxel_size,
        }
    }

    fn ready(&mut self) {
        #[cfg(feature = "profile")]
        {
            use tracing_subscriber::Registry;
            use tracing_subscriber::prelude::*;
            use tracing_tracy::TracyLayer;

            let subscriber = Registry::default().with(TracyLayer::default());
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set global default subscriber");
        }

        let world = self
            .base()
            .get_world_3d()
            .expect("ChunkeeWorldNode must be placed in a 3D world.");
        self.world_scenario = world.get_scenario();
        godot_print!("World Scenario RID: {:?}", self.world_scenario);

        let mut outline = create_voxel_outline(self.voxel_size);
        outline.set_visible(false);
        self.base_mut().add_child(&outline);
        self.outline_node = Some(outline);

        self.chunkee_manager.start();
    }

    fn process(&mut self, _delta: f64) {
        #[cfg(feature = "profile")]
        let _frame_span = tracing::info_span!("frame").entered();

        let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) else {
            println!("No camera: noop");
            return;
        };
        let camera_data = camera.to_camera_data();

        if Input::singleton().is_action_just_pressed("toggle_debug_physics_mesh") {
            self.show_physics_debug_mesh = !self.show_physics_debug_mesh;
            for (_, mesh) in self.physics_debug_meshes.iter_mut() {
                mesh.set_visible(self.show_physics_debug_mesh)
            }
        }

        let mut edits = Vec::new();

        if Input::singleton().is_action_pressed("break_block")
            && let VoxelRaycast::Hit(hit) = &self.voxel_raycast
        {
            let radius = 10;
            let radius_sq = radius * radius;
            let mut sphere_removals = vec![];

            for x in -radius..=radius {
                for y in -radius..=radius {
                    for z in -radius..=radius {
                        let offset = IVec3::new(x, y, z);
                        if offset.length_squared() <= radius_sq {
                            let wv = hit.0 + offset;
                            sphere_removals.push((wv, MyVoxels::Air));
                        }
                    }
                }
            }
            edits.extend_from_slice(&sphere_removals);
        }

        if Input::singleton().is_action_just_pressed("add_block")
            && let VoxelRaycast::Hit(hit) = &self.voxel_raycast
        {
            let radius = 4;
            let radius_sq = radius * radius;
            let mut sphere_additions = vec![];

            for x in -radius..=radius {
                for y in -radius..=radius {
                    for z in -radius..=radius {
                        let offset = IVec3::new(x, y, z);
                        if offset.length_squared() <= radius_sq {
                            let wv = hit.0 + offset;
                            sphere_additions.push((wv, MyVoxels::Stone));
                        }
                    }
                }
            }
            edits.extend_from_slice(&sphere_additions);
        }

        let physics_entities = vec![camera_data.pos];
        self.chunkee_manager
            .update(camera_data, physics_entities, &edits);
        self.voxel_raycast = self
            .chunkee_manager
            .raycast(camera_data.pos, camera_data.forward, 20);
        self.render();
    }

    fn exit_tree(&mut self) {
        godot_print!("Exiting tree, cleaning up all rendering server RIDs.");
        let mut rs = RenderingServer::singleton();

        for (_, rids_to_free) in self.rendered_chunks.drain() {
            for (instance_rid, mesh_rid) in rids_to_free {
                if instance_rid.is_valid() {
                    rs.instance_set_scenario(instance_rid, Rid::Invalid);
                    rs.free_rid(instance_rid);
                }
                if mesh_rid.is_valid() {
                    rs.free_rid(mesh_rid);
                }
            }
        }
        godot_print!("Rendering server cleanup complete.");
    }
}

const ARRAY_VERTEX: usize = 0;
const ARRAY_NORMAL: usize = 1;
const ARRAY_TANGENT: usize = 2;
const ARRAY_TEX_UV: usize = 4;
const ARRAY_CUSTOM0: usize = 6;
const ARRAY_INDEX: usize = 12;
const ARRAY_MAX: usize = 13;

#[godot_api]
impl ChunkeeWorldNode {
    #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
    fn render(&mut self) {
        let (Some(opaque_material), Some(translucent_material)) = (
            self.opaque_material.clone(),
            self.translucent_material.clone(),
        ) else {
            return;
        };

        let mut rs = RenderingServer::singleton();

        while let Some(update) = self.chunkee_manager.updates.pop() {
            match update {
                Update::Mesh(cv, chunk_mesh_group) => self.render_mesh_group(
                    &mut rs,
                    cv,
                    chunk_mesh_group,
                    &opaque_material,
                    &translucent_material,
                ),
                Update::MeshUnload(cv) => self.remove_mesh_group(cv, &mut rs),
                Update::Physics(cv, pmesh) => self.render_physics_mesh(cv, pmesh),
                Update::PhysicsUnload(cv) => self.remove_physics_mesh(cv),
            }
        }

        self.render_raycast_outline();
    }

    fn render_mesh_group(
        &mut self,
        rs: &mut RenderingServer,
        cv: ChunkVector,
        mesh_group: Box<ChunkMeshGroup>,
        opaque_material: &Gd<ShaderMaterial>,
        translucent_material: &Gd<ShaderMaterial>,
    ) {
        self.remove_mesh_group(cv, rs);

        let mut new_rids = Vec::new();
        if !mesh_group.opaque.positions.is_empty() {
            let (instance_rid, mesh_rid) =
                create_render_instance(self.world_scenario, rs, mesh_group.opaque, opaque_material);
            new_rids.push((instance_rid, mesh_rid));
        }

        if !mesh_group.translucent.positions.is_empty() {
            let (instance_rid, mesh_rid) = create_render_instance(
                self.world_scenario,
                rs,
                mesh_group.translucent,
                translucent_material,
            );
            new_rids.push((instance_rid, mesh_rid));
        }

        if !new_rids.is_empty() {
            self.rendered_chunks.insert(cv, new_rids);
        }
    }

    fn remove_mesh_group(&mut self, cv: ChunkVector, rs: &mut RenderingServer) {
        if let Some(old_rids) = self.rendered_chunks.remove(&cv) {
            for (instance_rid, mesh_rid) in old_rids {
                rs.free_rid(instance_rid);
                rs.free_rid(mesh_rid);
            }
        }
    }

    fn render_physics_mesh(&mut self, cv: ChunkVector, pmesh: PhysicsMesh) {
        self.remove_physics_mesh(cv);

        if pmesh.is_empty() {
            return;
        }

        let collision_shape_node = create_physics_mesh(&pmesh);
        self.base_mut().add_child(&collision_shape_node);
        if let Some(mut old_col) = self.physics_chunks.insert(cv, collision_shape_node) {
            old_col.queue_free();
        }

        let mut debug_mesh_instance = create_physics_debug_mesh(&pmesh);
        debug_mesh_instance.set_visible(self.show_physics_debug_mesh);
        self.base_mut().add_child(&debug_mesh_instance);
        if let Some(mut old_debug_mesh) = self.physics_debug_meshes.insert(cv, debug_mesh_instance)
        {
            old_debug_mesh.queue_free();
        }
    }

    fn remove_physics_mesh(&mut self, cv: ChunkVector) {
        if let Some(mut old_debug_mesh) = self.physics_debug_meshes.remove(&cv) {
            old_debug_mesh.queue_free();
        }

        if let Some(mut old_col) = self.physics_chunks.remove(&cv) {
            old_col.queue_free();
        }
    }

    fn render_raycast_outline(&mut self) {
        if let Some(outline_node) = self.outline_node.as_mut() {
            if let VoxelRaycast::Hit((wv, _)) = self.voxel_raycast {
                let scaled_wv =
                    Vector3::new(wv.x as f32, wv.y as f32, wv.z as f32) * self.voxel_size;
                outline_node.set_position(scaled_wv.to_godot());
                outline_node.set_visible(true);
            } else {
                outline_node.set_visible(false);
            }
        }
    }
}

// #[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
// fn create_render_instance(
//     scenario: Rid,
//     rs: &mut RenderingServer,
//     mesh_data: ChunkMeshData,
//     material: &Gd<ShaderMaterial>,
// ) -> (Rid, Rid) {
//     let array_format = ArrayFormat::VERTEX
//         | ArrayFormat::NORMAL
//         | ArrayFormat::TANGENT
//         | ArrayFormat::TEX_UV
//         | ArrayFormat::INDEX
//         | ArrayFormat::CUSTOM0
//         | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

//     let indices = PackedInt32Array::from(
//         mesh_data
//             .indices
//             .iter()
//             .map(|i| *i as i32)
//             .collect::<Vec<i32>>(),
//     );
//     let positions = PackedVector3Array::from(unsafe {
//         std::slice::from_raw_parts(
//             mesh_data.positions.as_ptr() as *const Vector3,
//             mesh_data.positions.len(),
//         )
//     });
//     let normals = PackedVector3Array::from(unsafe {
//         std::slice::from_raw_parts(
//             mesh_data.normals.as_ptr() as *const Vector3,
//             mesh_data.normals.len(),
//         )
//     });
//     let uvs = PackedVector2Array::from(unsafe {
//         std::slice::from_raw_parts(
//             mesh_data.uvs.as_ptr() as *const Vector2,
//             mesh_data.uvs.len(),
//         )
//     });
//     let tangents = PackedFloat32Array::from(bytemuck::cast_slice(&mesh_data.tangents));
//     let layers = PackedFloat32Array::from(bytemuck::cast_slice(&mesh_data.layers));

//     let mut arrays = VariantArray::new();
//     arrays.resize(ARRAY_MAX, &Variant::nil());

//     arrays.set(ARRAY_VERTEX, &positions.to_variant());
//     arrays.set(ARRAY_NORMAL, &normals.to_variant());
//     arrays.set(ARRAY_TANGENT, &tangents.to_variant());
//     arrays.set(ARRAY_TEX_UV, &uvs.to_variant());
//     arrays.set(ARRAY_INDEX, &indices.to_variant());
//     arrays.set(ARRAY_CUSTOM0, &layers.to_variant());

//     // 2. Create mesh resource on the server
//     let mesh_rid = rs.mesh_create();
//     rs.mesh_add_surface_from_arrays_ex(mesh_rid, RenderingServerPrimitiveType::TRIANGLES, &arrays)
//         .compress_format(array_format)
//         .done();
//     rs.mesh_surface_set_material(mesh_rid, 0, material.get_rid());

//     // 3. Create an instance to render the mesh
//     let instance_rid = rs.instance_create();
//     rs.instance_set_base(instance_rid, mesh_rid);
//     rs.instance_set_scenario(instance_rid, scenario);

//     (instance_rid, mesh_rid)
// }

#[cfg_attr(feature = "profile", tracing::instrument(skip_all))]
fn create_render_instance(
    scenario: Rid,
    rs: &mut RenderingServer,
    mesh_data: ChunkMeshData,
    material: &Gd<ShaderMaterial>,
) -> (Rid, Rid) {
    let array_format = ArrayFormat::VERTEX
        | ArrayFormat::NORMAL
        | ArrayFormat::TANGENT
        | ArrayFormat::TEX_UV
        | ArrayFormat::INDEX
        | ArrayFormat::CUSTOM0
        | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

    let indices = PackedInt32Array::from_iter(mesh_data.indices.iter().map(|i| *i as i32));
    let positions =
        PackedVector3Array::from_iter(mesh_data.positions.iter().map(|p| Vector3::from_array(*p)));
    let normals =
        PackedVector3Array::from_iter(mesh_data.normals.iter().map(|n| Vector3::from_array(*n)));
    let tangents = PackedFloat32Array::from(mesh_data.tangents.as_slice());
    let uvs =
        PackedVector2Array::from_iter(mesh_data.uvs.iter().map(|uv| Vector2::from_array(*uv)));
    let layers = PackedFloat32Array::from(mesh_data.layers.as_slice());

    let mut arrays = VariantArray::new();
    arrays.resize(ARRAY_MAX, &Variant::nil());

    arrays.set(ARRAY_VERTEX, &positions.to_variant());
    arrays.set(ARRAY_NORMAL, &normals.to_variant());
    arrays.set(ARRAY_TANGENT, &tangents.to_variant());
    arrays.set(ARRAY_TEX_UV, &uvs.to_variant());
    arrays.set(ARRAY_INDEX, &indices.to_variant());
    arrays.set(ARRAY_CUSTOM0, &layers.to_variant());

    // 2. Create mesh resource on the server
    let mesh_rid = rs.mesh_create();
    rs.mesh_add_surface_from_arrays_ex(mesh_rid, RenderingServerPrimitiveType::TRIANGLES, &arrays)
        .compress_format(array_format)
        .done();
    rs.mesh_surface_set_material(mesh_rid, 0, material.get_rid());

    // 3. Create an instance to render the mesh
    let instance_rid = rs.instance_create();
    rs.instance_set_base(instance_rid, mesh_rid);
    rs.instance_set_scenario(instance_rid, scenario);

    (instance_rid, mesh_rid)
}

fn create_voxel_outline(voxel_size: f32) -> Gd<MeshInstance3D> {
    let mut st = SurfaceTool::new_gd();

    st.begin(MeshPrimitiveType::LINES);

    // Create a 1x1x1 cube. Add a small offset to avoid Z-fighting with the actual voxel mesh.
    let size = (1.0 * voxel_size) + (0.001 * voxel_size);
    let offset = -0.0005 * voxel_size; // Helps center the outline on the voxel block

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

    let mut mesh = st.commit().unwrap();

    let mut material = StandardMaterial3D::new_gd();
    material.set_shading_mode(ShadingMode::UNSHADED);
    material.set_albedo(Color::from_rgb(1.0, 1.0, 0.0)); // Bright Yellow

    mesh.surface_set_material(0, &material);

    let mut mesh_instance = MeshInstance3D::new_alloc();
    mesh_instance.set_mesh(&mesh);
    mesh_instance
}

fn create_physics_mesh(pmesh: &Vec<Vec3>) -> Gd<CollisionShape3D> {
    let faces = PackedVector3Array::from_iter(
        pmesh
            .into_iter()
            .map(|tri| Vector3::from_array(tri.to_array())),
    );

    let mut shape_resource = ConcavePolygonShape3D::new_gd();
    shape_resource.set_faces(&faces);
    shape_resource.set_backface_collision_enabled(true);
    let mut shape_node = CollisionShape3D::new_alloc();
    shape_node.set_shape(&shape_resource);

    shape_node
}

fn create_physics_debug_mesh(pmesh: &Vec<Vec3>) -> Gd<MeshInstance3D> {
    let mut st = SurfaceTool::new_gd();

    st.begin(MeshPrimitiveType::LINES);

    for triangle in pmesh.chunks_exact(3) {
        let p0 = Vector3::from_array(triangle[0].to_array());
        let p1 = Vector3::from_array(triangle[1].to_array());
        let p2 = Vector3::from_array(triangle[2].to_array());

        st.add_vertex(p0);
        st.add_vertex(p1);

        st.add_vertex(p1);
        st.add_vertex(p2);

        st.add_vertex(p2);
        st.add_vertex(p0);
    }

    let mut mesh = st.commit().unwrap();

    let mut material = StandardMaterial3D::new_gd();
    material.set_shading_mode(ShadingMode::UNSHADED);
    material.set_albedo(Color::from_rgb(0.0, 1.0, 0.0)); //Bright Green

    mesh.surface_set_material(0, &material);

    let mut mesh_instance = MeshInstance3D::new_alloc();
    mesh_instance.set_mesh(&mesh);
    mesh_instance
}
