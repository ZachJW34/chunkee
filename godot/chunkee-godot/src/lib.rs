mod conversions;
mod generator;
mod voxels;

use std::time::{Duration, Instant};

use chunkee_core::{
    coords::WorldVector,
    define_metrics,
    glam::{IVec3, Vec3},
    hasher::VoxelHashMap,
    meshing::ChunkMeshData,
    metrics::Metrics,
    world::{ChunkeeWorld, ChunkeeWorldConfig, VoxelRaycast},
};
use godot::{
    classes::{
        ArrayMesh, CollisionShape3D, ConcavePolygonShape3D, IStaticBody3D, Input, MeshInstance3D,
        ShaderMaterial, StandardMaterial3D, StaticBody3D, SurfaceTool,
        base_material_3d::ShadingMode,
        mesh::{ArrayFormat, PrimitiveType},
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
    rendered_chunks: VoxelHashMap<Gd<Node3D>>,
    physics_chunks: VoxelHashMap<Gd<CollisionShape3D>>,
    physics_debug_meshes: VoxelHashMap<Gd<MeshInstance3D>>,
    voxel_hit: Option<(WorldVector, MyVoxels)>,
    outline_node: Option<Gd<MeshInstance3D>>,
    metrics: Metrics<RenderMetrics>,

    pub show_physics_debug_mesh: bool,
    #[export]
    pub opaque_material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub translucent_material: Option<Gd<ShaderMaterial>>,
}

#[godot_api]
impl IStaticBody3D for ChunkeeWorldNode {
    fn init(base: Base<StaticBody3D>) -> Self {
        env_logger::init();
        println!("Initializing ChunkeeWorldNode");
        let config = ChunkeeWorldConfig {
            radius: 8,
            generator: Box::new(WorldGenerator::new()),
        };
        let voxel_world: ChunkeeWorld<MyVoxels> = ChunkeeWorld::new(config);

        Self {
            base,
            voxel_world,
            opaque_material: None,
            translucent_material: None,
            rendered_chunks: Default::default(),
            physics_chunks: Default::default(),
            physics_debug_meshes: Default::default(),
            voxel_hit: None,
            outline_node: None,
            show_physics_debug_mesh: false,
            metrics: Metrics::new(Duration::from_secs(2)),
        }
    }

    fn ready(&mut self) {
        self.voxel_world.enable_pipeline();
        let mut outline = create_voxel_outline();
        outline.set_visible(false);
        self.base_mut().add_child(&outline);
        self.outline_node = Some(outline);
    }

    fn process(&mut self, _delta: f64) {
        if let Some(camera) = self.base().get_viewport().and_then(|vp| vp.get_camera_3d()) {
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
            match self
                .voxel_world
                .raycast_hit(camera_pos, forward_direction.as_vec3(), 20)
            {
                VoxelRaycast::Hit(hit) => {
                    self.voxel_hit = Some(hit);
                }
                VoxelRaycast::Miss => self.voxel_hit = None,
                VoxelRaycast::None => {}
            }

            if input.is_action_pressed("break_block")
                && let Some(hit) = self.voxel_hit.as_ref()
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

define_metrics! {
  enum RenderMetrics {
      ChunkMeshRender => "Render::ChunkMesh",
  }
}

#[godot_api]
impl ChunkeeWorldNode {
    fn render(&mut self) {
        // let render_time = Instant::now();
        let array_format = ArrayFormat::VERTEX
            | ArrayFormat::NORMAL
            | ArrayFormat::TANGENT
            | ArrayFormat::TEX_UV
            | ArrayFormat::INDEX
            | ArrayFormat::CUSTOM0
            | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

        while let Some(cv) = self.voxel_world.results.mesh_unload.pop() {
            if let Some(mut node) = self.rendered_chunks.remove(&cv) {
                node.queue_free();
            }
        }

        if let (Some(opaque_material), Some(translucent_material)) = (
            &self.opaque_material.clone(),
            &self.translucent_material.clone(),
        ) {
            let mesh_render_time = Instant::now();
            let mesh_queue_len = self.voxel_world.results.mesh_load.len();
            let drain_limit = 100;

            for _ in 0..drain_limit {
                if self.voxel_world.results.mesh_load.is_empty() {
                    break;
                }

                let (cv, mesh_group) = self.voxel_world.results.mesh_load.pop().unwrap();
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
                // parent.set_visible(false);
                self.base_mut().add_child(&parent);
                parent.set_owner(&*self.base_mut());

                if let Some(mut old_node) = self.rendered_chunks.insert(cv, parent) {
                    old_node.queue_free();
                }
            }

            if mesh_queue_len > 0 {
                self.metrics
                    .get_mut(RenderMetrics::ChunkMeshRender)
                    .record(mesh_render_time.elapsed());
            }

            if let Some(outline_node) = self.outline_node.as_mut() {
                if let Some((pos, _)) = self.voxel_hit {
                    let new_pos = Vector3::new(pos.x as f32, pos.y as f32, pos.z as f32);
                    outline_node.set_position(new_pos);
                    outline_node.set_visible(true);
                } else {
                    outline_node.set_visible(false);
                }
            }
        }

        // println!("Total Render time: {:?}", render_time.elapsed());
    }

    fn process_physics_meshes(&mut self) {
        let physics_load_limit = 10;
        for _ in 0..physics_load_limit {
            if self.voxel_world.results.physics_load.is_empty() {
                break;
            }

            let (cv, triangles) = self.voxel_world.results.physics_load.pop().unwrap();

            if triangles.is_empty() {
                continue;
            }

            // --- Debug Visualization Mesh ---
            let mut debug_mesh_instance = build_physics_debug_mesh(triangles.clone());
            debug_mesh_instance.set_visible(self.show_physics_debug_mesh);
            self.base_mut().add_child(&debug_mesh_instance);
            if let Some(mut old_debug_mesh) =
                self.physics_debug_meshes.insert(cv, debug_mesh_instance)
            {
                old_debug_mesh.queue_free();
            }
            // ---------------------------------------

            let collision_shape_node = build_physics_mesh(triangles);
            self.base_mut().add_child(&collision_shape_node);
            if let Some(mut old_col) = self.physics_chunks.insert(cv, collision_shape_node) {
                old_col.queue_free();
            }
        }

        while let Some(cv) = self.voxel_world.results.physics_unload.pop() {
            if let Some(mut shape_to_remove) = self.physics_chunks.remove(&cv) {
                shape_to_remove.queue_free();
            }

            if let Some(mut debug_mesh_to_remove) = self.physics_debug_meshes.remove(&cv) {
                debug_mesh_to_remove.queue_free();
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

fn build_physics_mesh(triangles: Vec<Vec3>) -> Gd<CollisionShape3D> {
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

fn build_physics_debug_mesh(triangles: Vec<Vec3>) -> Gd<MeshInstance3D> {
    let mut st = SurfaceTool::new_gd();

    st.begin(PrimitiveType::LINES);

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
