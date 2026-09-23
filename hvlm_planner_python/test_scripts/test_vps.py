import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import spark_dsg._dsg_bindings as spark_dsg
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for viewpoint selection")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the point clouds and viewpoints",
    )
    parser.add_argument(
        "--visualize_score", action="store_true", help="Whether to visualize the score"
    )
    parser.add_argument(
        "--radius_margin",
        type=float,
        default=0.2,
        help="Margin to add to the computed radius for surrounding point selection",
    )
    return parser.parse_args()


def get_object_pts(
    attrs: spark_dsg.ObjectNodeAttributes, all_points: np.ndarray
) -> np.ndarray:
    """Get the point cloud corresponding to an object node."""
    return all_points[attrs.mesh_connections]


def compute_viewing_rad_from_bb(
    bbox: spark_dsg.BoundingBox, fx: float, fy: float, height: float, width: float
) -> float:
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))
    tan_half_fov_x = np.tan(fov_x / 2)
    tan_half_fov_y = np.tan(fov_y / 2)

    corners = bbox.corners()
    max_xy_r = 0.0
    max_z_r = 0.0

    for corner in corners:
        diff = corner - bbox.world_P_center
        xy_norm = np.linalg.norm(diff[:2])
        max_xy_r = max(max_xy_r, xy_norm)
        max_z_r = max(max_z_r, abs(diff[2]))

    return max(max_xy_r / tan_half_fov_x, max_z_r / tan_half_fov_y)


def get_surrounding_pts(
    object_attrs: spark_dsg.ObjectNodeAttributes, all_points: np.ndarray, radius: float
) -> np.ndarray:
    """Get the point cloud corresponding to an object node."""
    tree = KDTree(all_points)
    obj_center = object_attrs.position
    idxs = tree.query_ball_point(obj_center, radius)
    obj_pt_indices = object_attrs.mesh_connections
    # Remove object points from surrounding points
    idxs = [idx for idx in idxs if idx not in obj_pt_indices]
    return all_points[idxs]


def vis_obj_and_surrounding_pts(
    object_pts: np.ndarray,
    surrounding_pts: np.ndarray,
    mesh: o3d.geometry.TriangleMesh = None,
):
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pts)
    obj_pcd.paint_uniform_color([1, 0, 0])  # Red for object points

    surrounding_pcd = o3d.geometry.PointCloud()
    surrounding_pcd.points = o3d.utility.Vector3dVector(surrounding_pts)
    surrounding_pcd.paint_uniform_color([0, 1, 0])  # Green for surrounding points

    # Create coordinate frame at viewpoint
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=np.zeros(3)
    )

    if mesh is not None:
        o3d.visualization.draw_geometries([obj_pcd, surrounding_pcd, mesh, frame])
    else:
        o3d.visualization.draw_geometries([obj_pcd, surrounding_pcd])


def get_rpy(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw angles."""
    r = R.from_matrix(rotation_matrix)
    return r.as_euler("xyz", degrees=False)


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)

    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Circle2DSelectionConfig:
    num_angle_samples: int = 5
    top_n: int = 1
    min_angular_separation_degrees: int = 15
    radius_margin: float = 0.2
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    width: int = 640
    height: int = 480
    visualize_score: bool = False


@dataclass
class ViewPoint:
    position: np.ndarray
    orientation: np.ndarray
    score: float


class Circle2DSelection:
    def __init__(self, config: Circle2DSelectionConfig):
        self.config = config

    def compute_best_viewpoints(
        self,
        object_pts: np.ndarray,
        surrounding_pts: np.ndarray,
        radius: float,
        centroid: np.ndarray,
        height: float,
    ) -> list[ViewPoint]:
        candidates: list[ViewPoint] = []
        margined_radius = radius * (1 + self.config.radius_margin)
        for i in range(self.config.num_angle_samples):
            theta = (2 * np.pi / self.config.num_angle_samples) * i
            candidates.append(
                self.evaluate_angle(
                    theta,
                    centroid,
                    object_pts,
                    surrounding_pts,
                    margined_radius,
                    height,
                )
            )

        # Sort candidates by score: descending
        candidates.sort(key=lambda vp: vp.score, reverse=True)

        selected_viewpoints: list[ViewPoint] = []
        for candidate in candidates:
            valid = True
            for selected_vp in selected_viewpoints:
                cand_decomp = get_rpy(candidate.orientation)
                selected_decomp = get_rpy(selected_vp.orientation)
                d = abs(cand_decomp[2] - selected_decomp[2])  # Compare yaw angles
                if d > np.pi:
                    d = 2 * np.pi - d  # Account for angle wrap-around
                if d < np.radians(self.config.min_angular_separation_degrees):
                    valid = False
                    break

            if valid:
                selected_viewpoints.append(candidate)
                if len(selected_viewpoints) >= self.config.top_n:
                    break

        return selected_viewpoints

    def evaluate_angle(
        self,
        theta: float,
        centroid: np.ndarray,
        object_pts: np.ndarray,
        surrounding_pts: np.ndarray,
        radius: float,
        height: float,
    ) -> ViewPoint:
        # 1. Position on circle (world frame)
        cam_pos = centroid + radius * np.array([np.cos(theta), np.sin(theta), 0])
        cam_pos[2] = height

        # 2. Compute yaw that faces object horizontally
        dir_xy = centroid[:2] - cam_pos[:2]
        yaw = np.arctan2(dir_xy[1], dir_xy[0])

        # 3. World rotation (camera-to-world)
        R_cw = yaw_to_rotation_matrix(yaw)

        # 4. World-to-camera
        R_wc = R_cw.T

        # 5. Score using yaw-only extrinsics
        score = self.score_view(R_wc, cam_pos, object_pts, surrounding_pts)

        return ViewPoint(
            position=cam_pos,
            orientation=R_cw,  # world rotation
            score=score,
        )

    def score_view(self, R_wc, cam_pos, object_pts, surrounding_pts):
        H = self.config.height
        W = self.config.width

        fx = self.config.fx
        fy = self.config.fy
        cx = self.config.cx
        cy = self.config.cy

        # --------------------------------------------------
        # 1️⃣ Build scene depth buffer (surroundings only)
        # --------------------------------------------------
        depth_scene = np.full((H, W), np.inf)
        # Mirror depth scene
        depth_scene = np.flipud(depth_scene)

        pts = surrounding_pts - cam_pos
        cam_pts = (R_wc @ pts.T).T

        valid = cam_pts[:, 0] > 0
        cam_pts = cam_pts[valid]

        if cam_pts.shape[0] > 0:
            u = fx * cam_pts[:, 1] / cam_pts[:, 0] + cx
            v = fy * cam_pts[:, 2] / cam_pts[:, 0] + cy

            valid_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

            u = u[valid_img].astype(int)
            v = v[valid_img].astype(int)
            depth = cam_pts[valid_img, 0]

            for i in range(len(u)):
                if depth[i] < depth_scene[v[i], u[i]]:
                    depth_scene[v[i], u[i]] = depth[i]

        # --------------------------------------------------
        # 2️⃣ Rasterize object into its own depth buffer
        # --------------------------------------------------
        depth_object = np.full((H, W), np.inf)

        pts = object_pts - cam_pos
        cam_obj = (R_wc @ pts.T).T

        valid = cam_obj[:, 0] > 0
        cam_obj = cam_obj[valid]

        if cam_obj.shape[0] == 0:
            return 0.0

        u = fx * cam_obj[:, 1] / cam_obj[:, 0] + cx
        v = fy * cam_obj[:, 2] / cam_obj[:, 0] + cy

        valid_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = u[valid_img].astype(int)
        v = v[valid_img].astype(int)
        depth = cam_obj[valid_img, 0]

        for i in range(len(u)):
            if depth[i] < depth_object[v[i], u[i]]:
                depth_object[v[i], u[i]] = depth[i]

        # --------------------------------------------------
        # 3️⃣ Visibility test per pixel
        # --------------------------------------------------
        object_pixels = depth_object < np.inf
        visible_pixels = depth_object < depth_scene

        visible_count = np.sum(object_pixels & visible_pixels)
        total_object_pixels = np.sum(object_pixels)

        if total_object_pixels == 0:
            return 0.0

        # --------------------------------------------------
        # 4️⃣ Return visibility ratio (scale invariant)
        # --------------------------------------------------
        visibility_ratio = visible_count / object_pts.shape[0]

        if self.config.visualize_score:
            print(f"Visibility ratio: {visibility_ratio:.4f}")
            print(
                f"Visible pixels: {visible_count}, Total object pixels: {total_object_pixels}"
            )
            print(f"Unobscured pixels: {np.sum(visible_pixels)}")
            depth_scene_img = depth_scene.copy()
            depth_scene_img[depth_scene_img == np.inf] = 0
            depth_object_img = depth_object.copy()
            depth_object_img[depth_object_img == np.inf] = 0
            fig, axs = plt.subplots(2, 1, figsize=(8, 12))
            axs[0].imshow(depth_scene_img, cmap="gray")
            axs[0].set_title("Scene Depth Buffer")
            axs[1].imshow(depth_object_img, cmap="gray")
            axs[1].set_title("Object Depth Buffer")
            plt.show()

        return float(visibility_ratio)


def visualize_viewpoints(
    viewpoints: list[ViewPoint],
    object_pts: np.ndarray,
    mesh: o3d.geometry.TriangleMesh = None,
):
    # Draw viewpoints as small coordinate frames
    geometries = []

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pts)
    obj_pcd.paint_uniform_color([1, 0, 0])  # Red for object points
    geometries.append(obj_pcd)

    for vp in viewpoints:
        # Create coordinate frame at viewpoint
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=vp.position
        )
        frame.rotate(vp.orientation, center=vp.position)
        geometries.append(frame)

    if mesh is not None:
        geometries.append(mesh)

    o3d.visualization.draw_geometries(geometries)


# Parse command line arguments
def main(args):
    path = "/developer/ros2_hydra_ws/os_dsg.sparkdsg"
    dsg = spark_dsg.DynamicSceneGraph.load(path)
    obj_id = 5692549928996306944
    agent_node_pose = np.array([0.0, 0.0, 0.7])
    config = Circle2DSelectionConfig(
        radius_margin=args.radius_margin, visualize_score=args.visualize_score
    )
    vp_selector = Circle2DSelection(config)

    all_points = dsg.mesh.get_vertices().T
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_points[:, :3])
    mesh.triangles = o3d.utility.Vector3iVector(dsg.mesh.get_faces().T)
    mesh.vertex_colors = o3d.utility.Vector3dVector(all_points[:, 3:])
    all_points = all_points[:, :3]

    obj_node_attrs = dsg.get_node(obj_id).attributes
    object_cloud = get_object_pts(obj_node_attrs, all_points)

    opt_radius = compute_viewing_rad_from_bb(
        obj_node_attrs.bounding_box, config.fx, config.fy, config.height, config.width
    )

    surrounding_cloud = get_surrounding_pts(
        obj_node_attrs, all_points, opt_radius * (1 + config.radius_margin)
    )

    # if args.visualize:
    #     vis_obj_and_surrounding_pts(object_cloud, surrounding_cloud, mesh)

    viewpoints = vp_selector.compute_best_viewpoints(
        object_cloud,
        surrounding_cloud,
        opt_radius,
        obj_node_attrs.position,
        agent_node_pose[2],
    )

    if args.visualize:
        visualize_viewpoints(viewpoints, object_cloud, mesh)


if __name__ == "__main__":
    args = parse_args()
    main(args)
