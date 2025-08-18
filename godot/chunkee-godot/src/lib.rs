mod conversions;
mod generator;
mod voxels;

use std::time::{Duration, Instant};

use chunkee_core::{
    define_metrics,
    glam::{IVec3, Vec3},
    hasher::VoxelHashMap,
    meshing::ChunkMeshData,
    metrics::Metrics,
    world::{ChunkeeWorld, ChunkeeWorldConfig, VoxelRaycast},
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

use crate::{conversions::*, generator::WorldGenerator, voxels::MyVoxels};

struct ChunkeeGodotExtension;

#[gdextension]
unsafe impl ExtensionLibrary for ChunkeeGodotExtension {}

#[derive(GodotClass)]
#[class(base=StaticBody3D)]
pub struct ChunkeeWorldNode {
    base: Base<StaticBody3D>,
    voxel_world: ChunkeeWorld<MyVoxels>,
    rendered_chunks: VoxelHashMap<Vec<(Rid, Rid)>>,
    physics_chunks: VoxelHashMap<Gd<CollisionShape3D>>,
    physics_debug_meshes: VoxelHashMap<Gd<MeshInstance3D>>,
    world_scenario: Rid,
    voxel_raycast: VoxelRaycast<MyVoxels>,
    outline_node: Option<Gd<MeshInstance3D>>,
    metrics: Metrics<ChunkeeWorldNodeMetrics>,
    pub show_physics_debug_mesh: bool,
    #[export]
    pub opaque_material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub translucent_material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub voxel_size: f32,
}

#[godot_api]
impl IStaticBody3D for ChunkeeWorldNode {
    fn init(base: Base<StaticBody3D>) -> Self {
        env_logger::init();
        println!("Initializing ChunkeeWorldNode");
        let voxel_size = 0.5;
        let config = ChunkeeWorldConfig {
            radius: 12,
            generator: Box::new(WorldGenerator::new()),
            voxel_size,
        };
        let voxel_world: ChunkeeWorld<MyVoxels> = ChunkeeWorld::new(config);

        Self {
            base,
            voxel_world,
            world_scenario: Rid::Invalid,
            rendered_chunks: Default::default(),
            physics_chunks: Default::default(),
            physics_debug_meshes: Default::default(),
            opaque_material: None,
            translucent_material: None,
            voxel_raycast: VoxelRaycast::None,
            outline_node: None,
            show_physics_debug_mesh: false,
            metrics: Metrics::new(Duration::from_secs(2)),
            voxel_size,
        }
    }

    fn ready(&mut self) {
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

        self.voxel_world.enable_pipeline();
    }

    fn process(&mut self, _delta: f64) {
        if let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) {
            let process_time = Instant::now();
            let camera_pos = Vec3::from_array(camera.get_global_position().to_array());
            self.voxel_world.update(camera.to_camera_data());
            let input = Input::singleton();

            if input.is_action_just_pressed("toggle_debug_physics_mesh") {
                self.show_physics_debug_mesh = !self.show_physics_debug_mesh;
                for (_, mesh) in self.physics_debug_meshes.iter_mut() {
                    mesh.set_visible(self.show_physics_debug_mesh)
                }
            }

            let forward_direction = -camera.get_global_transform().basis.col_c();
            let raycast_time = Instant::now();
            self.voxel_raycast =
                self.voxel_world
                    .try_raycast(camera_pos, forward_direction.as_vec3(), 100);
            self.metrics
                .get_mut(ChunkeeWorldNodeMetrics::Raycast)
                .record(raycast_time.elapsed());

            if input.is_action_pressed("break_block")
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
                self.voxel_world.set_voxels_at(&sphere_removals);
            }

            self.render();

            self.metrics
                .get_mut(ChunkeeWorldNodeMetrics::Process)
                .record(process_time.elapsed());
            self.metrics.batch_print();
        } else {
            println!("Cannot update without camera")
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        if let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) {
            let camera_pos = Vec3::from_array(camera.get_global_position().to_array());

            let mut entities = Vec::new();
            entities.push(chunkee_core::pipeline::PhysicsEntity {
                id: camera.instance_id().to_i64(),
                pos: camera_pos,
            });

            self.voxel_world.update_physics_entities(entities);
            self.process_physics_meshes();
        }
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

define_metrics! {
    enum ChunkeeWorldNodeMetrics {
        Process => "ChunkeeWorldNode::process",
        Render => "ChunkeeWorldNode::render",
        Raycast => "ChunkeeWorld::try_raycast"
    }
}

#[godot_api]
impl ChunkeeWorldNode {
    fn render(&mut self) {
        let mesh_render_time = Instant::now();
        let mut rs = RenderingServer::singleton();

        let drain_limit = 100;

        self.rendered_chunks.retain(|cv, rids| {
            if !self.voxel_world.chunk_in_range(*cv) {
                for (instance_rid, mesh_rid) in rids {
                    rs.free_rid(*instance_rid);
                    rs.free_rid(*mesh_rid);
                }

                return false;
            }

            true
        });

        if let (Some(opaque_material), Some(translucent_material)) =
            (&self.opaque_material, &self.translucent_material)
        {
            let mesh_queue_len = self.voxel_world.results.mesh_load.len();

            for _ in 0..drain_limit {
                let Some((cv, mesh_group)) = self.voxel_world.results.mesh_load.pop() else {
                    break;
                };

                if let Some(old_rids) = self.rendered_chunks.remove(&cv) {
                    for (instance_rid, mesh_rid) in old_rids {
                        rs.free_rid(instance_rid);
                        rs.free_rid(mesh_rid);
                    }
                }

                let mut new_rids = Vec::new();
                if !mesh_group.opaque.positions.is_empty() {
                    let (instance_rid, mesh_rid) = create_render_instance(
                        self.world_scenario,
                        mesh_group.opaque,
                        opaque_material,
                    );
                    new_rids.push((instance_rid, mesh_rid));
                }

                if !mesh_group.translucent.positions.is_empty() {
                    let (instance_rid, mesh_rid) = create_render_instance(
                        self.world_scenario,
                        mesh_group.translucent,
                        translucent_material,
                    );
                    new_rids.push((instance_rid, mesh_rid));
                }

                if !new_rids.is_empty() {
                    self.rendered_chunks.insert(cv, new_rids);
                }
            }

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

            if mesh_queue_len > 0 {
                self.metrics
                    .get_mut(ChunkeeWorldNodeMetrics::Render)
                    .record(mesh_render_time.elapsed());
            }
        }
    }

    fn process_physics_meshes(&mut self) {
        if let Some((cv, triangles)) = self.voxel_world.results.physics_load.pop() {
            if triangles.is_empty() {
                if let Some(mut old_debug_mesh) = self.physics_debug_meshes.remove(&cv) {
                    old_debug_mesh.queue_free();
                }

                if let Some(mut old_col) = self.physics_chunks.remove(&cv) {
                    old_col.queue_free();
                }
            } else {
                // --- Debug Visualization Mesh ---
                let mut debug_mesh_instance = create_physics_debug_mesh(triangles.clone());
                debug_mesh_instance.set_visible(self.show_physics_debug_mesh);
                self.base_mut().add_child(&debug_mesh_instance);
                if let Some(mut old_debug_mesh) =
                    self.physics_debug_meshes.insert(cv, debug_mesh_instance)
                {
                    old_debug_mesh.queue_free();
                }
                // ---------------------------------------

                let collision_shape_node = create_physics_mesh(triangles);
                self.base_mut().add_child(&collision_shape_node);
                if let Some(mut old_col) = self.physics_chunks.insert(cv, collision_shape_node) {
                    old_col.queue_free();
                }
            }
        }

        if let Some(cv) = self.voxel_world.results.physics_unload.pop() {
            if let Some(mut shape_to_remove) = self.physics_chunks.remove(&cv) {
                shape_to_remove.queue_free();
            }

            if let Some(mut debug_mesh_to_remove) = self.physics_debug_meshes.remove(&cv) {
                debug_mesh_to_remove.queue_free();
            }
        }
    }
}

fn create_render_instance(
    scenario: Rid,
    mesh_data: ChunkMeshData,
    material: &Gd<ShaderMaterial>,
) -> (Rid, Rid) {
    let mut rs = RenderingServer::singleton();

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

fn create_physics_mesh(triangles: Vec<Vec3>) -> Gd<CollisionShape3D> {
    let faces = PackedVector3Array::from_iter(
        triangles
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

fn create_physics_debug_mesh(triangles: Vec<Vec3>) -> Gd<MeshInstance3D> {
    let mut st = SurfaceTool::new_gd();

    st.begin(MeshPrimitiveType::LINES);

    for triangle in triangles.chunks_exact(3) {
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
