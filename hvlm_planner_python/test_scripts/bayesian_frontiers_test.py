import argparse
import copy
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

USE_SIGLIP = False
ALPHA = 0.5
BETA = 0.5
NORMALIZE_SIMILARITIES = False

POOLING_MODE = "average"  # Matches AllCosineSimilarity::PoolingMode::AVERAGE.
WEIGHTS_MODE = "uniform"
EXPONENTIAL_DECAY_RATE = 0.5

RELEVANT_OBJECTS = ["blanket", "bed", "pillow", "sheets", "nightstand"]
GRAPH_PATH = "eqa_planner_output_debug_dsg.sparkdsg"
DEBUG_JSON_PATH = "eqa_planner_output_debug.json"
MESH_PATH = "mesh.sparkdsg"
OUTPUT_DIR = "bayesian_frontiers_debug"
TOP_K_DEBUG = 15
MAX_MESH_FACES = 60000
EPS = 1.0e-12


def load_clip_module():
    try:
        import clip
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the OpenAI CLIP package. Activate the environment that "
            "provides `clip`, or install it before running this script."
        ) from exc
    return clip


def load_spark_dsg_module():
    try:
        import spark_dsg._dsg_bindings as spark_dsg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import spark_dsg. Source the ROS workspace or activate the "
            "environment that provides the Spark DSG Python bindings."
        ) from exc
    return spark_dsg


def load_open3d_module():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import open3d. Install Open3D or activate the environment that "
            "provides it before running this script."
        ) from exc
    return o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score frontier nodes with object and room CLIP probabilities."
    )
    parser.add_argument("--graph", default=GRAPH_PATH)
    parser.add_argument("--debug-json", default=DEBUG_JSON_PATH)
    parser.add_argument("--mesh", default=MESH_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--objects", nargs="+", default=RELEVANT_OBJECTS)
    parser.add_argument("--use-siglip", action="store_true", default=USE_SIGLIP)
    parser.add_argument(
        "--normalize-similarities",
        action="store_true",
        default=NORMALIZE_SIMILARITIES,
        help="Min-max normalize each logit vector before softmax.",
    )
    parser.add_argument(
        "--pooling-mode", choices=["average", "max"], default=POOLING_MODE
    )
    parser.add_argument(
        "--weights-mode",
        choices=["uniform", "linear_decay", "exponential_decay"],
        default=WEIGHTS_MODE,
    )
    parser.add_argument(
        "--exponential-decay-rate", type=float, default=EXPONENTIAL_DECAY_RATE
    )
    parser.add_argument("--top-k-debug", type=int, default=TOP_K_DEBUG)
    parser.add_argument("--max-mesh-faces", type=int, default=MAX_MESH_FACES)
    parser.add_argument(
        "--frontier-radius",
        type=float,
        default=None,
        help="Radius for Open3D frontier spheres. Defaults to a mesh-relative size.",
    )
    parser.add_argument(
        "--feature-point-radius",
        type=float,
        default=None,
        help="Radius for Open3D feature-point spheres. "
        "Defaults to half frontier radius.",
    )
    parser.add_argument("--show", action="store_true", help="Show Open3D windows.")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path.cwd().parent / path,
        Path(__file__).resolve().parents[4] / path,
        Path(__file__).resolve().parents[5] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def l2_normalize(features):
    return features / torch.clamp(features.norm(dim=-1, keepdim=True), min=EPS)


def stable_softmax(logits, dim=0):
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)


def minmax_normalize(logits, dim=0):
    min_values = logits.min(dim=dim, keepdim=True).values
    max_values = logits.max(dim=dim, keepdim=True).values
    ranges = max_values - min_values
    return torch.where(
        ranges > EPS, (logits - min_values) / ranges, torch.zeros_like(logits)
    )


def compute_weights(num_features, mode, exponential_decay_rate):
    if num_features == 0:
        return torch.empty(0, dtype=torch.float32)

    if mode == "uniform":
        return torch.full((num_features,), 1.0 / float(num_features))

    if mode == "linear_decay":
        n = float(num_features)
        denom = n * (n + 1.0) * 0.5
        return torch.tensor(
            [(n - float(i)) / denom for i in range(num_features)], dtype=torch.float32
        )

    if mode == "exponential_decay":
        weights = torch.tensor(
            [exponential_decay_rate**i for i in range(num_features)],
            dtype=torch.float32,
        )
        return weights / weights.sum()

    raise ValueError(f"Unknown weights mode: {mode}")


def all_cosine_similarity_logits(
    frontier_feature_sets,
    reference_features,
    pooling_mode="average",
    weights_mode="uniform",
    exponential_decay_rate=0.5,
):
    """Match AllCosineSimilarity::compute for one logit per frontier."""
    if len(frontier_feature_sets) == 0 or reference_features.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    reference_features = l2_normalize(reference_features.cpu())
    weights = compute_weights(
        reference_features.shape[0], weights_mode, exponential_decay_rate
    ).to(reference_features)

    logits = []
    for frontier_features in frontier_feature_sets:
        if frontier_features.numel() == 0:
            logits.append(torch.tensor(-1.0, dtype=torch.float32))
            continue

        frontier_features = l2_normalize(frontier_features.cpu())
        similarities = frontier_features @ reference_features.T

        if pooling_mode == "average":
            weighted_per_frontier_feature = similarities @ weights
            logits.append(weighted_per_frontier_feature.mean())
        elif pooling_mode == "max":
            logits.append(similarities.max())
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

    return torch.stack(logits)


def feature_object_logits(
    feature_sets,
    point_sets,
    frontier_poses,
    reference_features,
    pooling_mode="average",
    weights_mode="uniform",
    exponential_decay_rate=0.5,
    normalize_similarities=False,
):
    reference_features = l2_normalize(reference_features.cpu())
    weights = compute_weights(
        reference_features.shape[0], weights_mode, exponential_decay_rate
    ).to(reference_features)

    feature_points = []
    feature_frontier_indices = []
    feature_indices = []
    logits = []

    for frontier_idx, (features, points) in enumerate(zip(feature_sets, point_sets)):
        if points is None or features.numel() == 0:
            continue

        usable_count = min(features.shape[0], len(points))
        if usable_count == 0:
            continue

        features = l2_normalize(features[:usable_count].cpu())
        similarities = features @ reference_features.T

        if pooling_mode == "average":
            feature_logits = similarities @ weights
        elif pooling_mode == "max":
            feature_logits = similarities.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

        z = frontier_poses[frontier_idx, 2]
        points_3d = np.column_stack(
            [
                points[:usable_count, 0],
                points[:usable_count, 1],
                np.full(usable_count, z, dtype=np.float32),
            ]
        )

        feature_points.append(points_3d)
        feature_frontier_indices.extend([frontier_idx] * usable_count)
        feature_indices.extend(range(usable_count))
        logits.append(feature_logits)

    if not logits:
        empty = np.empty((0, 3), dtype=np.float32)
        return {
            "points": empty,
            "frontier_indices": np.empty(0, dtype=np.int32),
            "feature_indices": np.empty(0, dtype=np.int32),
            "logits": torch.empty(0, dtype=torch.float32),
            "probabilities": torch.empty(0, dtype=torch.float32),
        }

    logits = torch.cat(logits)
    softmax_logits = (
        minmax_normalize(logits, dim=0) if normalize_similarities else logits
    )
    probabilities = stable_softmax(softmax_logits, dim=0)

    return {
        "points": np.vstack(feature_points),
        "frontier_indices": np.asarray(feature_frontier_indices, dtype=np.int32),
        "feature_indices": np.asarray(feature_indices, dtype=np.int32),
        "logits": logits,
        "probabilities": probabilities,
    }


def encode_clip_text(clip_module, model, texts, device):
    tokens = clip_module.tokenize(texts).to(device)
    with torch.no_grad():
        return model.encode_text(tokens).float().cpu()


def siglip_text_features(model, inputs):
    features = model.get_text_features(**inputs)
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def load_siglip_model(device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    model = (
        AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").to(device).eval()
    )
    return processor, model


def encode_siglip_texts(processor, model, texts, device):
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        return siglip_text_features(model, inputs).float().cpu()


def feature_matrix_from_attr(value):
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return torch.from_numpy(array).float()


def point_matrix_from_attr(value):
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] == 2 and array.shape[1] != 2:
        array = array.T
    if array.shape[1] < 2:
        return None
    return array[:, :2]


def load_frontiers(spark_dsg, graph):
    frontier_layer = graph.get_layer(spark_dsg.DsgLayers.FRONTIERS)
    frontier_ids = []
    frontier_poses = []
    semantic_features = []
    all_feature_sets = []
    all_feature_point_sets = []
    all_feature_points = []

    for node in frontier_layer.nodes:
        attrs = node.attributes
        semantic_feature = feature_matrix_from_attr(attrs.semantic_feature)
        feature_set = feature_matrix_from_attr(getattr(attrs, "features", None))
        feature_points = point_matrix_from_attr(getattr(attrs, "feature_points", None))

        if semantic_feature is None:
            continue
        if feature_set is None:
            feature_set = semantic_feature
        feature_set_for_points = feature_set
        if feature_points is not None and len(feature_points) != feature_set.shape[0]:
            usable_count = min(len(feature_points), feature_set.shape[0])
            print(
                f"Warning: frontier {node.id} has {feature_set.shape[0]} features "
                f"but {len(feature_points)} feature_points; using {usable_count} pairs."
            )
            feature_set_for_points = feature_set[:usable_count]
            feature_points = feature_points[:usable_count]

        frontier_ids.append(str(node.id))
        frontier_poses.append(np.asarray(attrs.position, dtype=np.float32))
        semantic_features.append(semantic_feature.squeeze(0))
        all_feature_sets.append(feature_set)
        all_feature_point_sets.append(feature_set_for_points)
        all_feature_points.append(feature_points)

    if not frontier_ids:
        raise RuntimeError("No frontier nodes with valid semantic features found.")

    return (
        frontier_ids,
        np.vstack(frontier_poses),
        torch.stack(semantic_features),
        all_feature_sets,
        all_feature_point_sets,
        all_feature_points,
    )


def compute_scores(
    frontier_feature_sets,
    frontier_semantic_features,
    object_embeddings,
    room_embeddings_for_frontiers,
    room_embeddings_for_question,
    question_embedding,
    alpha,
    beta,
    pooling_mode,
    weights_mode,
    exponential_decay_rate,
    normalize_similarities,
):
    if frontier_semantic_features.shape[1] != room_embeddings_for_frontiers.shape[1]:
        raise ValueError(
            "Frontier semantic feature dimension does not match CLIP room embedding "
            f"dimension: {frontier_semantic_features.shape[1]} != "
            f"{room_embeddings_for_frontiers.shape[1]}."
        )
    if room_embeddings_for_question.shape[1] != question_embedding.shape[1]:
        raise ValueError(
            "Question embedding dimension does not match room embedding dimension: "
            f"{question_embedding.shape[1]} != {room_embeddings_for_question.shape[1]}."
        )

    object_logits = all_cosine_similarity_logits(
        frontier_feature_sets,
        object_embeddings,
        pooling_mode=pooling_mode,
        weights_mode=weights_mode,
        exponential_decay_rate=exponential_decay_rate,
    )
    object_softmax_logits = (
        minmax_normalize(object_logits, dim=0)
        if normalize_similarities
        else object_logits
    )
    object_probs = stable_softmax(object_softmax_logits, dim=0)

    room_question_logits = l2_normalize(room_embeddings_for_question) @ l2_normalize(
        question_embedding
    ).squeeze(0)
    room_question_softmax_logits = (
        minmax_normalize(room_question_logits, dim=0)
        if normalize_similarities
        else room_question_logits
    )
    room_question_probs = stable_softmax(room_question_softmax_logits, dim=0)

    room_frontier_logits = (
        l2_normalize(room_embeddings_for_frontiers)
        @ l2_normalize(frontier_semantic_features).T
    )
    room_frontier_softmax_logits = (
        minmax_normalize(room_frontier_logits, dim=0)
        if normalize_similarities
        else room_frontier_logits
    )
    room_frontier_probs = stable_softmax(room_frontier_softmax_logits, dim=0)

    room_products = room_frontier_probs * room_question_probs[:, None]
    room_scores = room_products.sum(dim=0)
    total_scores = alpha * object_probs + beta * room_scores

    return {
        "object_logits": object_logits,
        "object_probs": object_probs,
        "room_question_logits": room_question_logits,
        "room_question_probs": room_question_probs,
        "room_frontier_logits": room_frontier_logits,
        "room_frontier_probs": room_frontier_probs,
        "room_products": room_products,
        "room_scores": room_scores,
        "total_scores": total_scores,
    }


def load_mesh(spark_dsg, mesh_path, graph):
    mesh_path = resolve_path(mesh_path)
    if mesh_path.exists():
        mesh_graph = spark_dsg.DynamicSceneGraph.load(str(mesh_path))
        if mesh_graph.has_mesh():
            return mesh_graph.mesh

    if graph.has_mesh():
        return graph.mesh
    return None


def mesh_arrays(mesh):
    if mesh is None:
        return None, None, None

    vertices = mesh.get_vertices()
    points = vertices[:3, :].T
    colors = vertices[3:, :].T
    faces = mesh.get_faces().T
    return points, colors, faces


def blue_to_red_colors(values):
    values = np.asarray(values, dtype=np.float64)
    value_range = values.max() - values.min()
    if value_range < EPS:
        normalized = np.full_like(values, 0.5)
    else:
        normalized = (values - values.min()) / value_range

    blue = np.array([0.0, 0.15, 1.0])
    red = np.array([1.0, 0.05, 0.0])
    return (1.0 - normalized[:, None]) * blue + normalized[:, None] * red


def jet_colors(values):
    values = np.asarray(values, dtype=np.float64)
    value_range = values.max() - values.min()
    if value_range < EPS:
        normalized = np.full_like(values, 0.5)
    else:
        normalized = (values - values.min()) / value_range

    colormap = plt.get_cmap("jet")
    return colormap(normalized)[:, :3]  # Return RGB values only


def make_open3d_mesh(o3d, mesh, max_faces):
    points, colors, faces = mesh_arrays(mesh)
    if points is None or faces is None or len(faces) == 0:
        return None, None

    if len(faces) > max_faces:
        sampled_indices = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = faces[sampled_indices]

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(points.astype(np.float64))
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(colors, 0.0, 1.0).astype(np.float64)
    )
    open3d_mesh.compute_vertex_normals()
    return open3d_mesh, points


def estimate_frontier_radius(mesh_points, frontier_poses, requested_radius):
    if requested_radius is not None:
        return requested_radius

    if mesh_points is None:
        points = frontier_poses
    else:
        points = np.vstack([mesh_points, frontier_poses])

    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return max(0.12, diagonal * 0.006)


def make_frontier_sphere(o3d, position, color, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    sphere.translate(position.astype(np.float64))
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def make_top_frontier_marker(o3d, position, radius):
    try:
        marker = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=radius * 1.8,
            tube_radius=radius * 0.12,
            radial_resolution=28,
            tubular_resolution=8,
        )
        marker.translate(position.astype(np.float64))
    except AttributeError:
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius * 1.35, resolution=12
        )
        marker.translate(position.astype(np.float64))
    marker.paint_uniform_color([1.0, 0.9, 0.0])
    marker.compute_vertex_normals()
    return marker


def write_open3d_score_scenes(
    output_dir,
    mesh,
    frontier_ids,
    frontier_poses,
    score_sets,
    top_k,
    max_mesh_faces,
    frontier_radius,
    show,
):
    o3d = load_open3d_module()
    base_mesh, mesh_points = make_open3d_mesh(o3d, mesh, max_mesh_faces)
    radius = estimate_frontier_radius(mesh_points, frontier_poses, frontier_radius)
    scene_paths = []

    for filename, title, values in score_sets:
        colors = jet_colors(values)
        best_indices = set(np.argsort(-values)[:top_k])
        geometries = []
        if base_mesh is not None:
            geometries.append(copy.deepcopy(base_mesh))

        for idx, (position, color) in enumerate(zip(frontier_poses, colors)):
            geometries.append(make_frontier_sphere(o3d, position, color, radius))
            if idx in best_indices:
                geometries.append(make_top_frontier_marker(o3d, position, radius))

        combined = o3d.geometry.TriangleMesh()
        for geometry in geometries:
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                combined += geometry

        scene_path = output_dir / f"{filename}.ply"
        o3d.io.write_triangle_mesh(str(scene_path), combined, write_ascii=False)
        scene_paths.append(scene_path)

        if show:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"{title}: blue low, red high, yellow top {top_k}",
            )

    legend_path = output_dir / "open3d_score_scene_legend.txt"
    with legend_path.open("w") as f:
        f.write("Open3D frontier score scenes\n")
        f.write("Blue frontier spheres are low values; red spheres are high values.\n")
        f.write(
            f"Yellow torus markers indicate the top {top_k} frontiers in each scene.\n"
        )
        f.write("Scene files:\n")
        for scene_path in scene_paths:
            f.write(f"  {scene_path.name}\n")

    scene_paths.append(legend_path)
    return scene_paths


def write_open3d_feature_point_scenes(
    output_dir,
    mesh,
    frontier_poses,
    feature_scores,
    score_sets,
    max_mesh_faces,
    frontier_radius,
    feature_point_radius,
    show,
):
    if len(feature_scores["points"]) == 0:
        return []

    o3d = load_open3d_module()
    base_mesh, mesh_points = make_open3d_mesh(o3d, mesh, max_mesh_faces)
    center_radius = estimate_frontier_radius(
        mesh_points, frontier_poses, frontier_radius
    )
    point_radius = (
        feature_point_radius
        if feature_point_radius is not None
        else max(center_radius * 0.45, 0.05)
    )
    scene_paths = []

    for filename, title, values in score_sets:
        colors = jet_colors(values)
        geometries = []
        if base_mesh is not None:
            geometries.append(copy.deepcopy(base_mesh))

        for position in frontier_poses:
            geometries.append(
                make_frontier_sphere(o3d, position, [0.92, 0.92, 0.92], center_radius)
            )

        for position, color in zip(feature_scores["points"], colors):
            geometries.append(make_frontier_sphere(o3d, position, color, point_radius))

        combined = o3d.geometry.TriangleMesh()
        for geometry in geometries:
            if isinstance(geometry, o3d.geometry.TriangleMesh):
                combined += geometry

        scene_path = output_dir / f"{filename}.ply"
        o3d.io.write_triangle_mesh(str(scene_path), combined, write_ascii=False)
        scene_paths.append(scene_path)

        if show:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"{title}: feature points blue low, red high",
            )

    legend_path = output_dir / "open3d_feature_point_scene_legend.txt"
    with legend_path.open("w") as f:
        f.write("Open3D frontier feature-point scenes\n")
        f.write("Colored small spheres are individual frontier feature_points.\n")
        f.write(
            "Blue feature-point spheres are low values; red spheres are high values.\n"
        )
        f.write("Light gray larger spheres are frontier centroids for context.\n")
        f.write(
            "Each 2D feature_point uses the z height of its owning frontier node.\n"
        )
        f.write("Scene files:\n")
        for scene_path in scene_paths:
            f.write(f"  {scene_path.name}\n")

    scene_paths.append(legend_path)
    return scene_paths


def write_room_matrix_csv(output_dir, filename, frontier_ids, room_labels, matrix):
    path = output_dir / filename
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frontier_id", *room_labels])
        for frontier_idx, frontier_id in enumerate(frontier_ids):
            row_values = [
                float(matrix[room_idx, frontier_idx])
                for room_idx in range(len(room_labels))
            ]
            writer.writerow([frontier_id, *row_values])
    return path


def write_feature_point_csv(output_dir, frontier_ids, feature_scores):
    path = output_dir / "frontier_feature_points_debug.csv"
    logits = feature_scores["logits"].detach().cpu().numpy()
    probabilities = feature_scores["probabilities"].detach().cpu().numpy()

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frontier_id",
                "frontier_index",
                "feature_index",
                "x",
                "y",
                "z",
                "object_similarity_logit",
                "p_feature_point_given_objects",
            ],
        )
        writer.writeheader()
        for idx, point in enumerate(feature_scores["points"]):
            frontier_index = int(feature_scores["frontier_indices"][idx])
            writer.writerow(
                {
                    "frontier_id": frontier_ids[frontier_index],
                    "frontier_index": frontier_index,
                    "feature_index": int(feature_scores["feature_indices"][idx]),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "object_similarity_logit": float(logits[idx]),
                    "p_feature_point_given_objects": float(probabilities[idx]),
                }
            )
    return path


def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


def write_debug_outputs(
    output_dir,
    frontier_ids,
    frontier_poses,
    room_labels,
    scores,
    object_logits,
    object_probs,
    room_scores,
    room_question_probs,
    room_frontier_probs,
    room_products,
    alpha,
    beta,
):
    scores_np = scores.detach().cpu().numpy()
    object_logits_np = object_logits.detach().cpu().numpy()
    object_probs_np = object_probs.detach().cpu().numpy()
    room_scores_np = room_scores.detach().cpu().numpy()
    room_question_probs_np = room_question_probs.detach().cpu().numpy()
    room_frontier_probs_np = room_frontier_probs.detach().cpu().numpy()
    room_products_np = room_products.detach().cpu().numpy()

    rows = []
    for i, frontier_id in enumerate(frontier_ids):
        top_room_idx = int(np.argmax(room_frontier_probs_np[:, i]))
        weighted_top_room_idx = int(np.argmax(room_products_np[:, i]))
        room_terms = []
        for j, room_label in enumerate(room_labels):
            room_terms.append(
                {
                    "room": room_label,
                    "p_room_given_frontier": float(room_frontier_probs_np[j, i]),
                    "p_room_given_question": float(room_question_probs_np[j]),
                    "product": float(room_products_np[j, i]),
                }
            )

        rows.append(
            {
                "frontier_id": frontier_id,
                "position": frontier_poses[i].tolist(),
                "object_logit": float(object_logits_np[i]),
                "p_frontier_given_objects": float(object_probs_np[i]),
                "alpha_object_term": float(alpha * object_probs_np[i]),
                "room_score": float(room_scores_np[i]),
                "beta_room_term": float(beta * room_scores_np[i]),
                "total_score": float(scores_np[i]),
                "top_room_by_frontier": room_labels[top_room_idx],
                "top_room_by_product": room_labels[weighted_top_room_idx],
                "room_terms": room_terms,
            }
        )

    rows.sort(key=lambda row: row["total_score"], reverse=True)

    with (output_dir / "frontier_scores_debug.json").open("w") as f:
        json.dump(
            {
                "alpha": alpha,
                "beta": beta,
                "room_question_probabilities": {
                    room_labels[i]: float(room_question_probs_np[i])
                    for i in range(len(room_labels))
                },
                "frontiers": rows,
            },
            f,
            indent=2,
        )

    csv_path = output_dir / "frontier_scores_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "frontier_id",
                "total_score",
                "p_frontier_given_objects",
                "room_score",
                "alpha_object_term",
                "beta_room_term",
                "object_logit",
                "top_room_by_frontier",
                "top_room_by_product",
                "x",
                "y",
                "z",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "frontier_id": row["frontier_id"],
                    "total_score": row["total_score"],
                    "p_frontier_given_objects": row["p_frontier_given_objects"],
                    "room_score": row["room_score"],
                    "alpha_object_term": row["alpha_object_term"],
                    "beta_room_term": row["beta_room_term"],
                    "object_logit": row["object_logit"],
                    "top_room_by_frontier": row["top_room_by_frontier"],
                    "top_room_by_product": row["top_room_by_product"],
                    "x": row["position"][0],
                    "y": row["position"][1],
                    "z": row["position"][2],
                }
            )

    return rows, csv_path


def print_summary(rows, room_labels, room_question_probs, top_k, objects):
    print("\nP(room | question):")
    room_question = list(zip(room_labels, tensor_to_list(room_question_probs)))
    for room, probability in sorted(
        room_question, key=lambda item: item[1], reverse=True
    ):
        print(f"  {room}: {probability:.4f}")

    print(f"\nTop {min(top_k, len(rows))} frontiers:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"  {rank:02d}. {row['frontier_id']} "
            f"score={row['total_score']:.4f} "
            f"P(f|objects)={row['p_frontier_given_objects']:.4f} "
            f"room_score={row['room_score']:.4f} "
            f"top_room={row['top_room_by_product']}"
        )
    print(f"Objects considered: {', '.join(objects)}")


def main():
    args = parse_args()
    if abs(args.alpha + args.beta - 1.0) > 1.0e-6:
        raise ValueError(f"alpha + beta must equal 1. Got {args.alpha + args.beta}.")
    if args.alpha < 0.0 or args.beta < 0.0:
        raise ValueError("alpha and beta must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = resolve_path(args.graph)
    debug_json_path = resolve_path(args.debug_json)

    spark_dsg = load_spark_dsg_module()
    graph = spark_dsg.DynamicSceneGraph.load(str(graph_path))
    with debug_json_path.open("r") as f:
        debug_output = json.load(f)

    room_labels = debug_output["floorplan_rooms"]
    question = debug_output["question"]
    mesh = load_mesh(spark_dsg, args.mesh, graph)

    (
        frontier_ids,
        frontier_poses,
        frontier_semantic_features,
        frontier_feature_sets,
        frontier_feature_point_sets,
        frontier_feature_points,
    ) = load_frontiers(spark_dsg, graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_module = load_clip_module()
    model, _ = clip_module.load(
        "ViT-B/32", device=device, download_root=os.environ.get("CLIP_CACHE_DIR")
    )
    model.eval()

    object_embeddings = encode_clip_text(clip_module, model, args.objects, device)
    room_embeddings_clip = encode_clip_text(clip_module, model, room_labels, device)
    feature_point_scores = feature_object_logits(
        frontier_feature_point_sets,
        frontier_feature_points,
        frontier_poses,
        object_embeddings,
        pooling_mode=args.pooling_mode,
        weights_mode=args.weights_mode,
        exponential_decay_rate=args.exponential_decay_rate,
        normalize_similarities=args.normalize_similarities,
    )

    if args.use_siglip:
        siglip_processor, siglip_model = load_siglip_model(device)
        room_embeddings_for_question = encode_siglip_texts(
            siglip_processor, siglip_model, room_labels, device
        )
        question_embedding = encode_siglip_texts(
            siglip_processor, siglip_model, [question], device
        )
    else:
        room_embeddings_for_question = room_embeddings_clip
        question_embedding = encode_clip_text(clip_module, model, [question], device)

    scores = compute_scores(
        frontier_feature_sets=frontier_feature_sets,
        frontier_semantic_features=frontier_semantic_features,
        object_embeddings=object_embeddings,
        room_embeddings_for_frontiers=room_embeddings_clip,
        room_embeddings_for_question=room_embeddings_for_question,
        question_embedding=question_embedding,
        alpha=args.alpha,
        beta=args.beta,
        pooling_mode=args.pooling_mode,
        weights_mode=args.weights_mode,
        exponential_decay_rate=args.exponential_decay_rate,
        normalize_similarities=args.normalize_similarities,
    )

    rows, csv_path = write_debug_outputs(
        output_dir=output_dir,
        frontier_ids=frontier_ids,
        frontier_poses=frontier_poses,
        room_labels=room_labels,
        scores=scores["total_scores"],
        object_logits=scores["object_logits"],
        object_probs=scores["object_probs"],
        room_scores=scores["room_scores"],
        room_question_probs=scores["room_question_probs"],
        room_frontier_probs=scores["room_frontier_probs"],
        room_products=scores["room_products"],
        alpha=args.alpha,
        beta=args.beta,
    )

    total_scores_np = scores["total_scores"].detach().cpu().numpy()
    object_probs_np = scores["object_probs"].detach().cpu().numpy()
    room_scores_np = scores["room_scores"].detach().cpu().numpy()
    room_frontier_probs_np = scores["room_frontier_probs"].detach().cpu().numpy()
    room_products_np = scores["room_products"].detach().cpu().numpy()

    scene_paths = write_open3d_score_scenes(
        output_dir=output_dir,
        mesh=mesh,
        frontier_ids=frontier_ids,
        frontier_poses=frontier_poses,
        score_sets=[
            ("open3d_total_score_frontiers", "total score", total_scores_np),
            ("open3d_object_probability_frontiers", "P(f | objects)", object_probs_np),
            ("open3d_room_score_frontiers", "room score", room_scores_np),
        ],
        top_k=min(args.top_k_debug, len(frontier_ids)),
        max_mesh_faces=args.max_mesh_faces,
        frontier_radius=args.frontier_radius,
        show=args.show,
    )
    feature_point_scene_paths = write_open3d_feature_point_scenes(
        output_dir=output_dir,
        mesh=mesh,
        frontier_poses=frontier_poses,
        feature_scores=feature_point_scores,
        score_sets=[
            (
                "open3d_feature_point_object_similarity",
                "feature-point object similarity",
                feature_point_scores["logits"].detach().cpu().numpy(),
            ),
            (
                "open3d_feature_point_object_probability",
                "P(feature point | objects)",
                feature_point_scores["probabilities"].detach().cpu().numpy(),
            ),
        ],
        max_mesh_faces=args.max_mesh_faces,
        frontier_radius=args.frontier_radius,
        feature_point_radius=args.feature_point_radius,
        show=args.show,
    )
    room_prob_csv = write_room_matrix_csv(
        output_dir,
        "frontier_room_probabilities.csv",
        frontier_ids,
        room_labels,
        room_frontier_probs_np,
    )
    room_product_csv = write_room_matrix_csv(
        output_dir,
        "frontier_room_products.csv",
        frontier_ids,
        room_labels,
        room_products_np,
    )
    feature_point_csv = write_feature_point_csv(
        output_dir, frontier_ids, feature_point_scores
    )

    print_summary(
        rows=rows,
        room_labels=room_labels,
        room_question_probs=scores["room_question_probs"],
        top_k=min(args.top_k_debug, len(frontier_ids)),
        objects=args.objects,
    )
    print("\nWrote debug artifacts:")
    for scene_path in scene_paths:
        print(f"  {scene_path}")
    for scene_path in feature_point_scene_paths:
        print(f"  {scene_path}")
    print(f"  {csv_path}")
    print(f"  {room_prob_csv}")
    print(f"  {room_product_csv}")
    print(f"  {feature_point_csv}")
    print(f"  {output_dir / 'frontier_scores_debug.json'}")


if __name__ == "__main__":
    main()
