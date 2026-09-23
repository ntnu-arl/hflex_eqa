import argparse
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import spark_dsg._dsg_bindings as spark_dsg


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for viewpoint selection")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_score", action="store_true")
    parser.add_argument("--num_angle_samples", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--radius_margin", type=float, default=0.2)
    parser.add_argument("--disable_coarse_to_fine", action="store_true")
    parser.add_argument("--coarse_to_fine_min_samples", type=int, default=9)
    parser.add_argument("--coarse_stride", type=int, default=0)
    parser.add_argument("--coarse_seed_count", type=int, default=0)
    parser.add_argument("--coarse_refine_window", type=int, default=0)
    return parser.parse_args()


def get_object_pts(
    attrs: spark_dsg.ObjectNodeAttributes, all_points: np.ndarray
) -> np.ndarray:
    return all_points[attrs.mesh_connections]


def compute_viewing_rad_from_bb(
    bbox: spark_dsg.BoundingBox,
    fx: float,
    fy: float,
    image_height: float,
    image_width: float,
) -> float:
    # tan(arctan(x)) == x; avoids extra trig work.
    tan_half_fov_x = image_width / (2 * fx)
    tan_half_fov_y = image_height / (2 * fy)

    corners = bbox.corners()
    corners_np = np.asarray(corners)
    center = np.asarray(bbox.world_P_center)

    diffs = corners_np - center
    max_xy_r = np.linalg.norm(diffs[:, :2], axis=1).max(initial=0.0)
    max_z_r = np.abs(diffs[:, 2]).max(initial=0.0)

    return max(max_xy_r / tan_half_fov_x, max_z_r / tan_half_fov_y)


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
    coarse_to_fine_enabled: bool = True
    coarse_to_fine_min_samples: int = 9
    coarse_stride: int = 0
    coarse_seed_count: int = 0
    coarse_refine_window: int = 0

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
        self._A = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.config.width,
            self.config.height,
            self.config.fx,
            self.config.fy,
            self.config.cx,
            self.config.cy,
        )
        self._extrinsic = np.eye(4, dtype=float)

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

        max_radius = radius * (1 + max(0.0, self.config.radius_margin))
        if max_radius <= radius:
            radius_candidates = np.array([radius], dtype=float)
        else:
            radius_candidates = np.linspace(radius, max_radius, num=5, dtype=float)
        num_samples = max(1, self.config.num_angle_samples)
        theta_scale = (2.0 * np.pi) / num_samples
        angle_ids = np.arange(num_samples, dtype=int)
        cos_vals = np.cos(theta_scale * angle_ids)
        sin_vals = np.sin(theta_scale * angle_ids)
        object_pts_t = object_pts.T
        object_count = object_pts.shape[0]
        if object_count == 0:
            return []

        candidates_by_id: dict[int, ViewPoint] = {}

        def eval_angle_id(angle_id: int) -> ViewPoint:
            wrapped_id = int(angle_id % num_samples)
            candidate = candidates_by_id.get(wrapped_id)
            if candidate is None:
                candidate = self.evaluate_angle(
                    wrapped_id,
                    centroid,
                    object_pts_t,
                    radius_candidates,
                    height,
                    cos_vals[wrapped_id],
                    sin_vals[wrapped_id],
                )
                candidates_by_id[wrapped_id] = candidate
            return candidate

        use_coarse_to_fine = self.config.coarse_to_fine_enabled and num_samples >= max(
            1, self.config.coarse_to_fine_min_samples
        )

        if not use_coarse_to_fine:
            for angle_id in range(num_samples):
                eval_angle_id(angle_id)
        else:
            coarse_stride = self.config.coarse_stride
            if coarse_stride <= 0:
                coarse_stride = max(2, int(np.sqrt(num_samples)))
            coarse_stride = max(1, min(coarse_stride, num_samples))

            coarse_ids = list(range(0, num_samples, coarse_stride))
            for angle_id in coarse_ids:
                eval_angle_id(angle_id)

            coarse_ids.sort(key=lambda idx: candidates_by_id[idx].score, reverse=True)

            seed_count = self.config.coarse_seed_count
            if seed_count <= 0:
                seed_count = max(self.config.top_n * 2, 2)
            seed_count = min(len(coarse_ids), seed_count)

            window = self.config.coarse_refine_window
            if window <= 0:
                window = max(1, coarse_stride // 2)

            for seed_id in coarse_ids[:seed_count]:
                for delta in range(-window, window + 1):
                    eval_angle_id(seed_id + delta)

        candidates = list(candidates_by_id.values())

        candidates.sort(key=lambda vp: vp.score, reverse=True)

        selected_viewpoints: list[ViewPoint] = []
        selected_yaws: list[float] = []
        min_sep = np.radians(self.config.min_angular_separation_degrees)

        for candidate in candidates:
            valid = True
            cand_yaw = np.arctan2(
                candidate.orientation[1, 0], candidate.orientation[0, 0]
            )

            for selected_yaw in selected_yaws:
                d = abs(cand_yaw - selected_yaw)

                if d > np.pi:
                    d = 2 * np.pi - d

                if d < min_sep:
                    valid = False
                    break

            if valid:
                selected_viewpoints.append(candidate)
                selected_yaws.append(cand_yaw)

                if len(selected_viewpoints) >= self.config.top_n:
                    break

        return selected_viewpoints

    def evaluate_angle(
        self,
        angle_index: int,
        centroid: np.ndarray,
        object_pts_t: np.ndarray,
        radius_candidates: np.ndarray,
        height: float,
        cos_theta: float,
        sin_theta: float,
    ) -> ViewPoint:
        yaw = np.arctan2(-sin_theta, -cos_theta)

        R_cw = yaw_to_rotation_matrix(yaw)

        R_wc = R_cw.T

        scores: list[float] = []
        cam_positions: list[np.ndarray] = []
        for radius in radius_candidates:
            cam_pos = centroid + radius * np.array([cos_theta, sin_theta, 0.0])
            cam_pos[2] = height

            score = self.score_view(R_wc, cam_pos, object_pts_t, angle_index, radius)
            scores.append(score)
            cam_positions.append(cam_pos)

        if not scores:
            return ViewPoint(position=centroid.copy(), orientation=R_cw, score=0.0)

        score_arr = np.asarray(scores, dtype=float)
        top_k = min(2, score_arr.shape[0])
        top_ids = np.argsort(score_arr)[-top_k:]
        robust_score = float(score_arr[top_ids].mean())
        best_idx = int(np.argmax(score_arr))

        return ViewPoint(
            position=cam_positions[best_idx], orientation=R_cw, score=robust_score
        )

    def score_view(self, R_wc, cam_pos, object_pts_t, angle_index, radius) -> float:
        H = self.config.height
        W = self.config.width

        fx = self.config.fx
        fy = self.config.fy
        cx = self.config.cx
        cy = self.config.cy

        R_o3d = self._A @ R_wc
        t_o3d = -R_o3d @ cam_pos

        extrinsic = self._extrinsic
        extrinsic[:3, :3] = R_o3d
        extrinsic[:3, 3] = t_o3d
        extrinsic[3, :3] = 0.0
        extrinsic[3, 3] = 1.0

        self.renderer.setup_camera(self._intrinsic, extrinsic)

        depth_scene = np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True)
        )

        depth_scene[depth_scene == 0] = np.inf

        cam_pts = (R_o3d @ object_pts_t).T + t_o3d

        in_front = cam_pts[:, 2] > 0
        cam_pts = cam_pts[in_front]
        num_in_front = int(np.count_nonzero(in_front))

        if num_in_front == 0:
            return 0.0

        u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
        v = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy
        depth = cam_pts[:, 2]

        valid_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = np.rint(u[valid_img]).astype(int)
        v = np.rint(v[valid_img]).astype(int)
        depth = depth[valid_img]
        num_in_image = depth.shape[0]

        if num_in_image == 0:
            return 0.0

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        # Take the local minimum depth in a 3x3 neighborhood to reduce rasterization sensitivity.
        neighborhood_depths = []
        for du in (-1, 0, 1):
            uu = np.clip(u + du, 0, W - 1)
            for dv in (-1, 0, 1):
                vv = np.clip(v + dv, 0, H - 1)
                neighborhood_depths.append(depth_scene[vv, uu])
        depth_scene_local = np.minimum.reduce(neighborhood_depths)

        depth_epsilon = 0.02
        visible = depth <= (depth_scene_local + depth_epsilon)

        visible_count = int(np.count_nonzero(visible))
        occlusion_ratio = visible_count / num_in_front
        framing_ratio = num_in_image / num_in_front
        visibility_ratio = (0.7 * occlusion_ratio) + (0.3 * framing_ratio)

        if self.config.visualize_score:
            import matplotlib.pyplot as plt

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
            fig.savefig(
                f"/developer/ros2_hydra_ws/scene_depth_{angle_index}_{radius}.png"
            )
            plt.close(fig)

            visualize_viewpoints(
                [
                    ViewPoint(
                        position=cam_pos, orientation=R_wc.T, score=visibility_ratio
                    )
                ],
                object_pts_t.T,
                self.mesh,
            )

        return float(visibility_ratio)


def main(args):
    path = "/developer/ros2_hydra_ws/os_dsg.sparkdsg"

    dsg = spark_dsg.DynamicSceneGraph.load(path)

    obj_id = 5692549928996306944

    agent_node_pose = np.array([0.0, 0.0, 0.7])

    config = Circle2DSelectionConfig(
        num_angle_samples=args.num_angle_samples,
        top_n=args.top_n,
        radius_margin=args.radius_margin,
        visualize_score=args.visualize_score,
        coarse_to_fine_enabled=not args.disable_coarse_to_fine,
        coarse_to_fine_min_samples=args.coarse_to_fine_min_samples,
        coarse_stride=args.coarse_stride,
        coarse_seed_count=args.coarse_seed_count,
        coarse_refine_window=args.coarse_refine_window,
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

    viewpoints = vp_selector.compute_best_viewpoints(
        object_cloud,
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
