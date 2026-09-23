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
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_score", action="store_true")
    parser.add_argument("--radius_margin", type=float, default=0.2)
    return parser.parse_args()


def get_object_pts(
    attrs: spark_dsg.ObjectNodeAttributes, all_points: np.ndarray
) -> np.ndarray:
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
    tree = KDTree(all_points)
    obj_center = object_attrs.position

    idxs = tree.query_ball_point(obj_center, radius)

    obj_pt_indices = object_attrs.mesh_connections
    idxs = [idx for idx in idxs if idx not in obj_pt_indices]

    return all_points[idxs]


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)

    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def get_rpy(rotation_matrix: np.ndarray) -> np.ndarray:
    r = R.from_matrix(rotation_matrix)
    return r.as_euler("xyz", degrees=False)


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


def visualize_viewpoints(
    viewpoints: list[ViewPoint], object_pts: np.ndarray, mesh: o3d.geometry.TriangleMesh
):
    geometries = []

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pts)
    obj_pcd.paint_uniform_color([1, 0, 0])

    geometries.append(obj_pcd)

    for vp in viewpoints:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=vp.position
        )

        frame.rotate(vp.orientation, center=vp.position)

        geometries.append(frame)

    geometries.append(mesh)

    o3d.visualization.draw_geometries(geometries)


class Circle2DSelection:
    def __init__(self, config: Circle2DSelectionConfig):
        self.config = config
        self.mesh = None
        self.renderer = None

    def init_renderer(self):
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.config.width, self.config.height
        )

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"

        self.renderer.scene.add_geometry("mesh", self.mesh, mat)

    def compute_best_viewpoints(
        self,
        object_pts: np.ndarray,
        surrounding_pts: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
        radius: float,
        centroid: np.ndarray,
        object_mesh_connections: list[int],
        height: float,
    ) -> list[ViewPoint]:
        # Remove object points from mesh
        mesh.remove_vertices_by_index(object_mesh_connections)
        self.mesh = mesh
        self.init_renderer()

        candidates: list[ViewPoint] = []

        margined_radius = radius * (1 + self.config.radius_margin)

        for i in range(self.config.num_angle_samples):
            theta = (2 * np.pi / self.config.num_angle_samples) * i

            vp = self.evaluate_angle(
                theta, centroid, object_pts, surrounding_pts, margined_radius, height, i
            )

            candidates.append(vp)

        candidates.sort(key=lambda vp: vp.score, reverse=True)

        selected_viewpoints: list[ViewPoint] = []

        for candidate in candidates:
            valid = True

            for selected_vp in selected_viewpoints:
                cand_decomp = get_rpy(candidate.orientation)
                selected_decomp = get_rpy(selected_vp.orientation)

                d = abs(cand_decomp[2] - selected_decomp[2])

                if d > np.pi:
                    d = 2 * np.pi - d

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
        angle_index: int,
    ) -> ViewPoint:
        cam_pos = centroid + radius * np.array([np.cos(theta), np.sin(theta), 0])

        cam_pos[2] = height

        dir_xy = centroid[:2] - cam_pos[:2]

        yaw = np.arctan2(dir_xy[1], dir_xy[0])

        R_cw = yaw_to_rotation_matrix(yaw)

        R_wc = R_cw.T

        score = self.score_view(R_wc, cam_pos, object_pts, angle_index)

        return ViewPoint(position=cam_pos, orientation=R_cw, score=score)

    def score_view(self, R_wc, cam_pos, object_pts, angle_index) -> float:
        H = self.config.height
        W = self.config.width

        fx = self.config.fx
        fy = self.config.fy
        cx = self.config.cx
        cy = self.config.cy

        A = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])

        R_o3d = A @ R_wc
        t_o3d = -R_o3d @ cam_pos

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_o3d
        extrinsic[:3, 3] = t_o3d

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        self.renderer.setup_camera(intrinsic, extrinsic)

        depth_scene = np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True)
        )

        depth_scene[depth_scene == 0] = np.inf

        cam_pts = (
            extrinsic @ np.hstack((object_pts, np.ones((object_pts.shape[0], 1)))).T
        )
        cam_pts = cam_pts.T[:, :3]

        valid = cam_pts[:, 2] > 0
        cam_pts = cam_pts[valid]

        if cam_pts.shape[0] == 0:
            return 0.0

        u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
        v = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy
        depth = cam_pts[:, 2]

        valid_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = u[valid_img].astype(int)
        v = v[valid_img].astype(int)
        depth = depth[valid_img]

        if depth.shape[0] == 0:
            return 0.0

        visible = depth < depth_scene[v, u]

        visibility_ratio = np.sum(visible) / object_pts.shape[0]

        if self.config.visualize_score:
            print(f"Visibility ratio: {visibility_ratio:.4f}")
            print(f"Visible pixels: {visible.sum()}")
            fig, ax = plt.subplots(2, 2, figsize=(30, 30))
            # Add cam_pts to the depth scene visualization
            print(depth_scene.max(), depth_scene.min())
            depth_scene_vis = np.ones((H, W, 3), dtype=np.uint8) * 255
            depth_scene_vis[(depth_scene > 0) & (depth_scene < np.inf)] = np.zeros(
                3, dtype=np.uint8
            )
            depth_scene_vis[v, u] = np.array(
                [0, 255, 0], dtype=np.uint8
            )  # Mark projected object points in green
            # Make marked points bigger and more visible
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    uu = np.clip(u + du, 0, W - 1)
                    vv = np.clip(v + dv, 0, H - 1)
                    depth_scene_vis[vv, uu] = np.array([0, 255, 0], dtype=np.uint8)
            ax[0, 0].imshow(depth_scene_vis)
            ax[0, 0].set_title("Scene Mesh with Projected Object Points", fontsize=25)
            ax[0, 0].axis("off")
            ax[0, 1].imshow(depth_scene, cmap="gray")
            ax[0, 1].set_title("Depth Scene", fontsize=25)
            ax[0, 1].axis("off")
            img_scene = np.asarray(self.renderer.render_to_image())
            ax[1, 0].imshow(img_scene)
            ax[1, 0].set_title("Rendered Image", fontsize=25)
            ax[1, 0].axis("off")

            # Remove last subplot
            ax[1, 1].axis("off")
            fig.savefig(f"/developer/ros2_hydra_ws/scene_depth_{angle_index}.png")
            plt.close(fig)

            # visualize_viewpoints(
            #     [ViewPoint(position=cam_pos, orientation=R_wc.T, score=visibility_ratio)],
            #     object_pts,
            #     self.mesh
            # )

        return float(visibility_ratio)


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

    viewpoints = vp_selector.compute_best_viewpoints(
        object_cloud,
        surrounding_cloud,
        mesh,
        opt_radius,
        obj_node_attrs.position,
        obj_node_attrs.mesh_connections,
        agent_node_pose[2],
    )

    if args.visualize:
        visualize_viewpoints(viewpoints, object_cloud, mesh)


if __name__ == "__main__":
    args = parse_args()
    main(args)
