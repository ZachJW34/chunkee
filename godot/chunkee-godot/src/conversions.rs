use chunkee_core::{
    glam::Vec3,
    streaming::{CameraData, Frustum, Plane},
};
use godot::{builtin::Vector3, classes::Camera3D, obj::Gd};

trait ToGlamVec3 {
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
        let planes = self.get_frustum();
        let frustum = Frustum {
            planes: std::array::from_fn(|idx| Plane {
                normal: planes.at(idx).normal.as_vec3(),
                d: planes.at(idx).d,
            }),
        };

        CameraData { pos, frustum }
    }
}
