mod conversions;
mod generator;
mod voxels;

use std::collections::HashMap;

use chunkee_core::{
    coords::ChunkVector,
    glam::Vec3,
    streaming::is_chunk_in_range,
    world::{ChunkeeWorld, ChunkeeWorldConfig},
};
use godot::{
    classes::{
        ArrayMesh, MeshInstance3D, ShaderMaterial,
        mesh::{ArrayFormat, PrimitiveType},
    },
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
    rendered_chunks: HashMap<ChunkVector, Gd<MeshInstance3D>>,

    // #[export]
    // pub texture_atlas: Option<Gd<CompressedTexture2DArray>>,
    #[export]
    pub material: Option<Gd<ShaderMaterial>>,
}

#[godot_api]
impl INode3D for ChunkeeWorldNode {
    fn init(base: Base<Node3D>) -> Self {
        println!("Initializing ChunkeeWorldNode");
        let config = ChunkeeWorldConfig {
            radius: 20,
            generator: Box::new(WorldGenerator::new()),
        };
        let voxel_world: ChunkeeWorld<MyVoxels> = ChunkeeWorld::new(config);

        Self {
            base,
            voxel_world,
            material: None,
            rendered_chunks: HashMap::new(),
        }
    }

    fn ready(&mut self) {
        self.voxel_world.enable_pipeline();

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
            self.render(self.voxel_world.radius, camera_pos);
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
    fn render(&mut self, radius: u32, camera_pos: Vec3) {
        let array_format = ArrayFormat::VERTEX
            | ArrayFormat::NORMAL
            | ArrayFormat::TANGENT
            | ArrayFormat::TEX_UV
            | ArrayFormat::INDEX
            | ArrayFormat::CUSTOM0
            | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

        // while let Some(cv_to_unload) = self.voxel_world.unload_queue.pop() {
        //     if let Some(mut mesh) = self.rendered_chunks.remove(&cv_to_unload) {
        //         let camera_pos = Vec3::from_array(
        //             self.camera_3d
        //                 .as_ref()
        //                 .unwrap()
        //                 .get_global_position()
        //                 .to_array(),
        //         );

        //         let radius = 4 as f32;
        //         let cvw = cv_to_wv(cv_to_unload).as_vec3();
        //         let is_in_range = cvw.distance_squared(camera_pos) <= radius * radius;
        //         println!(
        //             "Freeing cv: {cv_to_unload:?} (camera = {camera_pos} cvw={cvw} is_in_range={is_in_range})",
        //         );
        //         mesh.queue_free();
        //     }
        // }

        self.rendered_chunks.retain(|cv, node| {
            if is_chunk_in_range(*cv, camera_pos, radius) {
                true
            } else {
                node.queue_free();
                false
            }
        });

        if let Some(material) = &self.material.clone() {
            while let Some((cv, mesh_data)) = self.voxel_world.mesh_queue.pop() {
                if let Some(mut old_mesh) = self.rendered_chunks.remove(&cv) {
                    old_mesh.queue_free();
                }

                if mesh_data.positions.len() <= 1 {
                    continue;
                }

                let indices =
                    PackedInt32Array::from_iter(mesh_data.indices.iter().map(|i| *i as i32));
                let positions = PackedVector3Array::from_iter(
                    mesh_data.positions.iter().map(|p| Vector3::from_array(*p)),
                );
                let normals = PackedVector3Array::from_iter(
                    mesh_data.normals.iter().map(|n| Vector3::from_array(*n)),
                );
                let tangents = PackedFloat32Array::from(mesh_data.tangents);
                let uvs = PackedVector2Array::from_iter(
                    mesh_data.uvs.iter().map(|uv| Vector2::from_array(*uv)),
                );
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

                self.base_mut().add_child(&mesh_instance);
                mesh_instance.set_owner(&*self.base_mut());

                self.rendered_chunks.insert(cv, mesh_instance.upcast());
            }
        }
    }
}
