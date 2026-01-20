use chunkee_core::{
    glam::Vec3,
    streaming::{CameraData, Frustum, Plane},
};
use godot::{builtin::Vector3, classes::Camera3D, obj::Gd};

pub trait ToGlamVec3 {
    fn as_vec3(&self) -> Vec3;
}

impl ToGlamVec3 for Vector3 {
    fn as_vec3(&self) -> Vec3 {
        Vec3::from_array(self.to_array())
    }
}

pub trait ToCameraData {
    fn to_camera_data(&self) -> CameraData;
}

impl ToCameraData for Gd<Camera3D> {
    fn to_camera_data(&self) -> CameraData {
        let pos = self.get_global_position().as_vec3();
        let planes_arr = self.get_frustum();
        let planes = std::array::from_fn(|idx| {
            let plane = planes_arr.at(idx).normalized();

            Plane {
                // Godot’s get_frustum returns planes with outward normals,
                // Chunkee is expecting inward so we flip
                normal: -plane.normal.as_vec3(),
                d: plane.d,
            }
        });

        let frustum = Frustum { planes };
        let forward = -self.get_global_transform().basis.col_c().as_vec3();

        CameraData {
            pos,
            frustum,
            forward,
        }
    }
}
