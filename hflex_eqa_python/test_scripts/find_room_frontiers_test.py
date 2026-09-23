#!/usr/bin/env python3
"""Debug find-room frontier scoring from saved DSG and floorplan output."""

import argparse
import collections
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from bayesian_frontiers_test import (
    EPS,
    all_cosine_similarity_logits,
    encode_clip_text,
    feature_matrix_from_attr,
    load_clip_module,
    load_mesh,
    load_spark_dsg_module,
    resolve_path,
    write_open3d_score_scenes,
    write_room_matrix_csv,
)

GRAPH_PATH = "eqa_planner_output_debug_dsg.sparkdsg"
DEBUG_JSON_PATH = "eqa_planner_output_debug.json"
MESH_PATH = "mesh.sparkdsg"
OUTPUT_DIR = "find_room_frontiers_debug"
TOP_K_DEBUG = 15
MAX_MESH_FACES = 60000

ALPHA = 0.25
BETA = 0.75
GAMMA = 0.0
POOLING_MODE = "max"
WEIGHTS_MODE = "uniform"
EXPONENTIAL_DECAY_RATE = 0.5

TRANSITION_PROMPT_TEMPLATES = [
    "doorway to a {room}",
    "entrance to a {room}",
    "hallway leading to a {room}",
    "opening into a {room}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score frontier nodes with the find_room semantic search cost."
    )
    parser.add_argument("--graph", default=GRAPH_PATH)
    parser.add_argument("--debug-json", default=DEBUG_JSON_PATH)
    parser.add_argument("--mesh", default=MESH_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--current-room", default=None)
    parser.add_argument("--target-room", default=None)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
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
    parser.add_argument("--show", action="store_true", help="Show Open3D windows.")
    return parser.parse_args()


def clean_room_name(room):
    """Normalize a floorplan room node name to its semantic room type."""
    room = re.sub(r"^tie:\s*", "", room).strip()
    room = re.sub(r"[_-]\d+$", "", room).strip()
    room = re.sub(r"\s+\d+$", "", room).strip()
    room = room.replace("_", " ")
    room = re.sub(r"\s+", " ", room).strip()
    return room


def room_name_parts(room):
    """Split a possibly tied room label into normalized semantic room types."""
    room = re.sub(r"^tie:\s*", "", room).strip()
    return [clean_room_name(part) for part in room.split("&")]


def unique_room_names(room_list):
    seen = set()
    result = []
    for room in room_list:
        for cleaned in room_name_parts(room):
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
    return result


def softmax(scores):
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    exp_sum = sum(exp_scores)
    if exp_sum <= 0.0:
        return [1.0 / float(len(scores))] * len(scores)
    return [score / exp_sum for score in exp_scores]


def floorplan_instance_adjacency(nodes, edges):
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        if len(edge) != 2:
            continue
        a, b = edge
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def floorplan_distance_to_target_instances(start_node, target_label, adjacency):
    target_label = clean_room_name(target_label)
    if target_label in room_name_parts(start_node):
        return 0
    if start_node not in adjacency:
        return None

    visited = {start_node}
    node_queue = collections.deque([(start_node, 0)])
    while node_queue:
        node, distance = node_queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor in visited:
                continue
            if target_label in room_name_parts(neighbor):
                return distance + 1
            visited.add(neighbor)
            node_queue.append((neighbor, distance + 1))
    return None


def compute_floorplan_label_scores(
    nodes, edges, room_labels, current_room, target_room
):
    """Compute softmax(-distance_to_target) over valid next-room labels only."""
    room_labels = room_labels or unique_room_names(nodes)
    if not room_labels:
        return [], []
    if current_room is None or target_room is None:
        return room_labels, [0.0] * len(room_labels)

    current_room_labels = set(room_name_parts(current_room))
    target_room = clean_room_name(target_room)
    adjacency = floorplan_instance_adjacency(nodes, edges)
    current_nodes = [
        node
        for node in nodes
        if any(label in current_room_labels for label in room_name_parts(node))
    ]
    if not current_nodes:
        return room_labels, [0.0] * len(room_labels)

    candidate_distance_by_label = {}
    for current_node in current_nodes:
        for neighbor in adjacency.get(current_node, set()):
            neighbor_distance = floorplan_distance_to_target_instances(
                neighbor, target_room, adjacency
            )
            if neighbor_distance is None:
                continue
            for neighbor_label in room_name_parts(neighbor):
                if neighbor_label not in room_labels:
                    continue
                candidate_distance_by_label[neighbor_label] = min(
                    candidate_distance_by_label.get(neighbor_label, neighbor_distance),
                    neighbor_distance,
                )

    if not candidate_distance_by_label:
        return room_labels, [0.0] * len(room_labels)

    candidate_labels = list(candidate_distance_by_label.keys())
    candidate_weights = softmax(
        [-float(candidate_distance_by_label[label]) for label in candidate_labels]
    )
    score_by_label = {label: 0.0 for label in room_labels}
    for label, weight in zip(candidate_labels, candidate_weights, strict=True):
        score_by_label[label] = weight
    return room_labels, [score_by_label[label] for label in room_labels]


def floorplan_nodes_with_label(nodes, room_label):
    """Find all floorplan instance nodes that can represent a semantic room label."""
    labels = set(room_name_parts(room_label))
    return {
        node
        for node in nodes
        if any(label in labels for label in room_name_parts(node))
    }


def save_floorplan_figure(output_dir, nodes, edges, current_room, target_room):
    """Save a floorplan graph figure with possible current and target nodes marked."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import matplotlib/networkx. Install them or activate the "
            "environment that provides them before saving the floorplan figure."
        ) from exc

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge for edge in edges if len(edge) == 2)

    current_nodes = floorplan_nodes_with_label(nodes, current_room)
    target_nodes = floorplan_nodes_with_label(nodes, target_room)

    node_colors = []
    node_edge_colors = []
    node_sizes = []
    for node in graph.nodes:
        is_current = node in current_nodes
        is_target = node in target_nodes
        if is_current and is_target:
            node_colors.append("#8e44ad")
            node_edge_colors.append("#4a235a")
            node_sizes.append(2100)
        elif is_current:
            node_colors.append("#27ae60")
            node_edge_colors.append("#145a32")
            node_sizes.append(1900)
        elif is_target:
            node_colors.append("#e74c3c")
            node_edge_colors.append("#922b21")
            node_sizes.append(1900)
        else:
            node_colors.append("#d5d8dc")
            node_edge_colors.append("#566573")
            node_sizes.append(1500)

    width = max(9.0, min(18.0, 1.15 * max(1, len(nodes))))
    height = max(6.0, min(14.0, 0.85 * max(1, len(nodes))))
    plt.figure(figsize=(width, height))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_edges(graph, pos, edge_color="#85929e", width=1.8)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=1.8,
        node_size=node_sizes,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#27ae60",
            markeredgecolor="#145a32",
            markersize=12,
            label=f"possible current: {current_room}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markeredgecolor="#922b21",
            markersize=12,
            label=f"target: {target_room}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#8e44ad",
            markeredgecolor="#4a235a",
            markersize=12,
            label="current and target",
        ),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.title(
        f"Find-room floorplan context: current={current_room}, target={target_room}"
    )
    plt.axis("off")
    plt.tight_layout()

    figure_path = output_dir / "find_room_floorplan_context.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()
    return figure_path


def floorplan_node_scores(nodes, room_labels, floorplan_label_weights):
    """Assign each floorplan instance node the best score of its semantic labels."""
    weight_by_label = {
        label: float(weight)
        for label, weight in zip(room_labels, floorplan_label_weights, strict=True)
    }
    scores = {}
    for node in nodes:
        scores[node] = max(
            (weight_by_label.get(label, 0.0) for label in room_name_parts(node)),
            default=0.0,
        )
    return scores


def save_final_floorplan_score_figure(
    output_dir,
    nodes,
    edges,
    current_room,
    target_room,
    room_labels,
    floorplan_label_weights,
):
    """Save the floorplan graph with current/target nodes and final node scores."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import matplotlib/networkx. Install them or activate the "
            "environment that provides them before saving the floorplan figure."
        ) from exc

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge for edge in edges if len(edge) == 2)

    current_nodes = floorplan_nodes_with_label(nodes, current_room)
    target_nodes = floorplan_nodes_with_label(nodes, target_room)
    node_score_by_name = floorplan_node_scores(
        nodes, room_labels, floorplan_label_weights
    )
    node_scores = np.asarray(
        [node_score_by_name.get(node, 0.0) for node in graph.nodes], dtype=np.float64
    )
    max_node_score = float(node_scores.max()) if node_scores.size else 0.0

    width = max(10.0, min(20.0, 1.25 * max(1, len(nodes))))
    height = max(7.0, min(15.0, 0.95 * max(1, len(nodes))))
    fig, ax = plt.subplots(figsize=(width, height))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#85929e", width=1.8)
    nodes_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_scores,
        cmap="viridis",
        vmin=0.0,
        vmax=max(max_node_score, EPS),
        edgecolors="#34495e",
        linewidths=1.5,
        node_size=2100,
    )

    def draw_highlight(node_set, edge_color, node_size, linewidth):
        if not node_set:
            return
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=[node for node in graph.nodes if node in node_set],
            node_color="none",
            edgecolors=edge_color,
            linewidths=linewidth,
            node_size=node_size,
        )

    both_nodes = current_nodes & target_nodes
    draw_highlight(current_nodes - both_nodes, "#27ae60", 2650, 4.0)
    draw_highlight(target_nodes - both_nodes, "#e74c3c", 2650, 4.0)
    draw_highlight(both_nodes, "#8e44ad", 2850, 5.0)

    labels = {}
    for node in graph.nodes:
        markers = []
        if node in current_nodes:
            markers.append("C")
        if node in target_nodes:
            markers.append("T")
        marker_text = f" [{'|'.join(markers)}]" if markers else ""
        labels[node] = f"{node}{marker_text}\nscore={node_score_by_name[node]:.3f}"

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "boxstyle": "round,pad=0.18",
        },
    )
    colorbar = fig.colorbar(nodes_artist, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label("floorplan next-label score")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#27ae60",
            markeredgewidth=3,
            markersize=14,
            label=f"possible current: {current_room}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#e74c3c",
            markeredgewidth=3,
            markersize=14,
            label=f"target: {target_room}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#8e44ad",
            markeredgewidth=3,
            markersize=14,
            label="current and target",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(
        "Find-room final floorplan scores "
        f"(current={current_room}, target={target_room})"
    )
    ax.axis("off")
    fig.tight_layout()

    figure_path = output_dir / "find_room_floorplan_final_scores.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def count_attr_points(value):
    if value is None:
        return 0
    array = np.asarray(value)
    if array.size == 0:
        return 0
    if array.ndim == 1:
        return 1
    if array.shape[0] in (2, 3) and array.shape[1] not in (2, 3):
        return int(array.shape[1])
    return int(array.shape[0])


def load_frontiers_with_free_space(spark_dsg, graph):
    frontier_layer = graph.get_layer(spark_dsg.DsgLayers.FRONTIERS)
    frontier_ids = []
    frontier_poses = []
    semantic_features = []
    all_feature_sets = []
    free_space_counts = []

    for node in frontier_layer.nodes:
        attrs = node.attributes
        semantic_feature = feature_matrix_from_attr(attrs.semantic_feature)
        feature_set = feature_matrix_from_attr(getattr(attrs, "features", None))

        if semantic_feature is None:
            continue
        if feature_set is None:
            feature_set = semantic_feature

        frontier_ids.append(str(node.id))
        frontier_poses.append(np.asarray(attrs.position, dtype=np.float32))
        semantic_features.append(semantic_feature.squeeze(0))
        all_feature_sets.append(feature_set)
        free_space_counts.append(
            count_attr_points(getattr(attrs, "frontier_points", None))
        )

    if not frontier_ids:
        raise RuntimeError("No frontier nodes with valid semantic features found.")

    return (
        frontier_ids,
        np.vstack(frontier_poses),
        torch.stack(semantic_features),
        all_feature_sets,
        np.asarray(free_space_counts, dtype=np.float32),
    )


def room_label_from_node(room_node):
    for attr_name in ("label", "name"):
        room_label = getattr(room_node.attributes, attr_name, None)
        if room_label:
            return str(room_label)
    return None


def infer_current_room_from_graph(spark_dsg, graph):
    if not graph.has_layer(spark_dsg.DsgLayers.AGENTS):
        return None

    agent_layer_id = graph.get_layer_key(spark_dsg.DsgLayers.AGENTS).layer
    agent_partition = None
    for layerkey in graph.layer_keys:
        if layerkey.partition != 0:
            agent_partition = layerkey.partition
            break
    if agent_partition is None:
        return None

    agent_layer = graph.get_layer(agent_layer_id, agent_partition)
    if agent_layer.num_nodes() == 0:
        return None

    current_agent_node = list(agent_layer.nodes)[-1]
    if not current_agent_node.has_parent():
        return None
    nav_node = graph.get_node(current_agent_node.get_parent())
    if not nav_node.has_parent():
        return None
    room_node = graph.get_node(nav_node.get_parent())
    return room_label_from_node(room_node)


def choose_current_room(args, debug_output, spark_dsg, graph):
    if args.current_room:
        return clean_room_name(args.current_room)

    for key in ("current_room", "current_room_label", "room_label"):
        value = debug_output.get(key)
        if value:
            return clean_room_name(value)

    current_state = debug_output.get("current_agent_state")
    if isinstance(current_state, str):
        match = re.search(r"room_label\s+([^,\n]+)", current_state)
        if match and match.group(1).strip() != "null":
            return clean_room_name(match.group(1))

    inferred = infer_current_room_from_graph(spark_dsg, graph)
    if inferred:
        return clean_room_name(inferred)
    return None


def mean_prompt_embeddings(clip_module, model, room_labels, device):
    all_embeddings = []
    for room_label in room_labels:
        prompts = [
            template.format(room=room_label) for template in TRANSITION_PROMPT_TEMPLATES
        ]
        prompt_embeddings = encode_clip_text(clip_module, model, prompts, device)
        all_embeddings.append(prompt_embeddings.mean(dim=0))
    return torch.stack(all_embeddings)


def compute_find_room_scores(
    frontier_feature_sets,
    target_transition_embeddings,
    floorplan_room_embeddings,
    floorplan_label_weights,
    free_space_counts,
    alpha,
    beta,
    gamma,
    pooling_mode,
    weights_mode,
    exponential_decay_rate,
):
    if len(frontier_feature_sets) == 0:
        raise ValueError("No frontier feature sets were provided.")
    if floorplan_room_embeddings.shape[0] != len(floorplan_label_weights):
        raise ValueError(
            "Room embedding count does not match floorplan label weight count: "
            f"{floorplan_room_embeddings.shape[0]} != {len(floorplan_label_weights)}."
        )

    transition_scores = all_cosine_similarity_logits(
        frontier_feature_sets,
        target_transition_embeddings,
        pooling_mode=pooling_mode,
        weights_mode=weights_mode,
        exponential_decay_rate=exponential_decay_rate,
    )

    room_similarity_rows = []
    for room_embedding in floorplan_room_embeddings:
        room_similarity_rows.append(
            all_cosine_similarity_logits(
                frontier_feature_sets,
                room_embedding.unsqueeze(0),
                pooling_mode=pooling_mode,
                weights_mode=weights_mode,
                exponential_decay_rate=exponential_decay_rate,
            )
        )
    room_similarity_matrix = torch.stack(room_similarity_rows)
    label_weights = torch.tensor(floorplan_label_weights, dtype=torch.float32)
    progress_scores = (label_weights[:, None] * room_similarity_matrix).sum(dim=0)

    max_free_space = float(free_space_counts.max()) if len(free_space_counts) else 0.0
    if max_free_space > EPS:
        novelty_scores = torch.from_numpy(free_space_counts / max_free_space).float()
    else:
        novelty_scores = torch.zeros_like(transition_scores)

    total_scores = (
        alpha * transition_scores + beta * progress_scores + gamma * novelty_scores
    )

    return {
        "transition_scores": transition_scores,
        "room_similarity_matrix": room_similarity_matrix,
        "progress_scores": progress_scores,
        "novelty_scores": novelty_scores,
        "total_scores": total_scores,
    }


def write_debug_outputs(
    output_dir,
    frontier_ids,
    frontier_poses,
    room_labels,
    floorplan_label_weights,
    free_space_counts,
    scores,
    current_room,
    target_room,
    alpha,
    beta,
    gamma,
):
    total_np = scores["total_scores"].detach().cpu().numpy()
    transition_np = scores["transition_scores"].detach().cpu().numpy()
    progress_np = scores["progress_scores"].detach().cpu().numpy()
    novelty_np = scores["novelty_scores"].detach().cpu().numpy()
    room_similarity_np = scores["room_similarity_matrix"].detach().cpu().numpy()

    rows = []
    for i, frontier_id in enumerate(frontier_ids):
        top_room_idx = int(np.argmax(room_similarity_np[:, i]))
        weighted_terms = room_similarity_np[:, i] * np.asarray(floorplan_label_weights)
        top_weighted_room_idx = int(np.argmax(weighted_terms))
        room_terms = []
        for j, room_label in enumerate(room_labels):
            room_terms.append(
                {
                    "room": room_label,
                    "floorplan_weight": float(floorplan_label_weights[j]),
                    "transition_similarity": float(room_similarity_np[j, i]),
                    "weighted_similarity": float(weighted_terms[j]),
                }
            )

        rows.append(
            {
                "frontier_id": frontier_id,
                "position": frontier_poses[i].tolist(),
                "total_score": float(total_np[i]),
                "target_transition_score": float(transition_np[i]),
                "progress_score": float(progress_np[i]),
                "novelty_score": float(novelty_np[i]),
                "alpha_transition_term": float(alpha * transition_np[i]),
                "beta_progress_term": float(beta * progress_np[i]),
                "gamma_novelty_term": float(gamma * novelty_np[i]),
                "free_space_points": int(free_space_counts[i]),
                "top_room_by_similarity": room_labels[top_room_idx],
                "top_room_by_weighted_similarity": room_labels[top_weighted_room_idx],
                "room_terms": room_terms,
            }
        )

    rows.sort(key=lambda row: row["total_score"], reverse=True)

    with (output_dir / "find_room_frontier_scores_debug.json").open("w") as f:
        json.dump(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "current_room": current_room,
                "target_room": target_room,
                "floorplan_label_weights": {
                    room_labels[i]: float(floorplan_label_weights[i])
                    for i in range(len(room_labels))
                },
                "frontiers": rows,
            },
            f,
            indent=2,
        )

    csv_path = output_dir / "find_room_frontier_scores_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "frontier_id",
                "total_score",
                "target_transition_score",
                "progress_score",
                "novelty_score",
                "alpha_transition_term",
                "beta_progress_term",
                "gamma_novelty_term",
                "free_space_points",
                "top_room_by_similarity",
                "top_room_by_weighted_similarity",
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
                    "target_transition_score": row["target_transition_score"],
                    "progress_score": row["progress_score"],
                    "novelty_score": row["novelty_score"],
                    "alpha_transition_term": row["alpha_transition_term"],
                    "beta_progress_term": row["beta_progress_term"],
                    "gamma_novelty_term": row["gamma_novelty_term"],
                    "free_space_points": row["free_space_points"],
                    "top_room_by_similarity": row["top_room_by_similarity"],
                    "top_room_by_weighted_similarity": row[
                        "top_room_by_weighted_similarity"
                    ],
                    "x": row["position"][0],
                    "y": row["position"][1],
                    "z": row["position"][2],
                }
            )

    return rows, csv_path


def write_label_weights_csv(output_dir, room_labels, floorplan_label_weights):
    path = output_dir / "floorplan_label_weights.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["room_label", "weight"])
        for room_label, weight in zip(
            room_labels, floorplan_label_weights, strict=True
        ):
            writer.writerow([room_label, float(weight)])
    return path


def print_summary(
    rows, room_labels, floorplan_label_weights, current_room, target_room, top_k
):
    print(f"\nCurrent room: {current_room}")
    print(f"Target room: {target_room}")
    print("\nFloorplan next-label weights:")
    for room, weight in sorted(
        zip(room_labels, floorplan_label_weights, strict=True),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {room}: {weight:.4f}")

    print(f"\nTop {min(top_k, len(rows))} frontiers:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"  {rank:02d}. {row['frontier_id']} "
            f"score={row['total_score']:.4f} "
            f"target={row['target_transition_score']:.4f} "
            f"progress={row['progress_score']:.4f} "
            f"novelty={row['novelty_score']:.4f} "
            f"top_room={row['top_room_by_weighted_similarity']}"
        )


def main():
    args = parse_args()
    if args.alpha < 0.0 or args.beta < 0.0 or args.gamma < 0.0:
        raise ValueError("alpha, beta, and gamma must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = resolve_path(args.graph)
    debug_json_path = resolve_path(args.debug_json)

    spark_dsg = load_spark_dsg_module()
    graph = spark_dsg.DynamicSceneGraph.load(str(graph_path))
    with debug_json_path.open("r") as f:
        debug_output = json.load(f)

    floorplan_nodes = debug_output["floorplan_nodes"]
    floorplan_edges = debug_output["floorplan_edges"]
    room_labels = debug_output.get("floorplan_rooms") or unique_room_names(
        floorplan_nodes
    )
    target_room_value = args.target_room or debug_output.get("target_room")
    if not target_room_value:
        raise RuntimeError(
            "Could not find target room in the debug JSON. "
            "Pass --target-room explicitly."
        )
    target_room = clean_room_name(target_room_value)
    current_room = choose_current_room(args, debug_output, spark_dsg, graph)
    if current_room is None:
        raise RuntimeError(
            "Could not infer current room from the debug DSG/JSON. "
            "Pass --current-room explicitly."
        )

    floorplan_figure_path = save_floorplan_figure(
        output_dir, floorplan_nodes, floorplan_edges, current_room, target_room
    )

    room_labels, floorplan_label_weights = compute_floorplan_label_scores(
        floorplan_nodes, floorplan_edges, room_labels, current_room, target_room
    )
    if sum(floorplan_label_weights) <= EPS:
        print(
            "Warning: floorplan label weights are all zero. "
            "The progress term will contribute zero."
        )

    mesh = load_mesh(spark_dsg, args.mesh, graph)
    (
        frontier_ids,
        frontier_poses,
        frontier_semantic_features,
        frontier_feature_sets,
        free_space_counts,
    ) = load_frontiers_with_free_space(spark_dsg, graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_module = load_clip_module()
    model, _ = clip_module.load(
        "ViT-B/32", device=device, download_root=os.environ.get("CLIP_CACHE_DIR")
    )
    model.eval()

    target_prompts = [
        template.format(room=target_room) for template in TRANSITION_PROMPT_TEMPLATES
    ]
    target_transition_embeddings = encode_clip_text(
        clip_module, model, target_prompts, device
    )
    floorplan_room_embeddings = mean_prompt_embeddings(
        clip_module, model, room_labels, device
    )

    if frontier_semantic_features.shape[1] != target_transition_embeddings.shape[1]:
        raise ValueError(
            "Frontier semantic feature dimension does not match CLIP prompt "
            f"dimension: {frontier_semantic_features.shape[1]} != "
            f"{target_transition_embeddings.shape[1]}."
        )

    scores = compute_find_room_scores(
        frontier_feature_sets=frontier_feature_sets,
        target_transition_embeddings=target_transition_embeddings,
        floorplan_room_embeddings=floorplan_room_embeddings,
        floorplan_label_weights=floorplan_label_weights,
        free_space_counts=free_space_counts,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        pooling_mode=args.pooling_mode,
        weights_mode=args.weights_mode,
        exponential_decay_rate=args.exponential_decay_rate,
    )

    rows, csv_path = write_debug_outputs(
        output_dir=output_dir,
        frontier_ids=frontier_ids,
        frontier_poses=frontier_poses,
        room_labels=room_labels,
        floorplan_label_weights=floorplan_label_weights,
        free_space_counts=free_space_counts,
        scores=scores,
        current_room=current_room,
        target_room=target_room,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    total_scores_np = scores["total_scores"].detach().cpu().numpy()
    transition_scores_np = scores["transition_scores"].detach().cpu().numpy()
    progress_scores_np = scores["progress_scores"].detach().cpu().numpy()
    novelty_scores_np = scores["novelty_scores"].detach().cpu().numpy()
    room_similarity_np = scores["room_similarity_matrix"].detach().cpu().numpy()

    scene_paths = write_open3d_score_scenes(
        output_dir=output_dir,
        mesh=mesh,
        frontier_ids=frontier_ids,
        frontier_poses=frontier_poses,
        score_sets=[
            (
                "open3d_find_room_total_score_frontiers",
                "find-room total score",
                total_scores_np,
            ),
            (
                "open3d_find_room_target_transition_frontiers",
                "target transition score",
                transition_scores_np,
            ),
            (
                "open3d_find_room_progress_frontiers",
                "floorplan progress score",
                progress_scores_np,
            ),
            ("open3d_find_room_novelty_frontiers", "novelty score", novelty_scores_np),
        ],
        top_k=min(args.top_k_debug, len(frontier_ids)),
        max_mesh_faces=args.max_mesh_faces,
        frontier_radius=args.frontier_radius,
        show=args.show,
    )
    room_similarity_csv = write_room_matrix_csv(
        output_dir,
        "find_room_frontier_room_transition_similarities.csv",
        frontier_ids,
        room_labels,
        room_similarity_np,
    )
    label_weights_csv = write_label_weights_csv(
        output_dir, room_labels, floorplan_label_weights
    )
    final_floorplan_figure_path = save_final_floorplan_score_figure(
        output_dir,
        floorplan_nodes,
        floorplan_edges,
        current_room,
        target_room,
        room_labels,
        floorplan_label_weights,
    )

    print_summary(
        rows=rows,
        room_labels=room_labels,
        floorplan_label_weights=floorplan_label_weights,
        current_room=current_room,
        target_room=target_room,
        top_k=min(args.top_k_debug, len(frontier_ids)),
    )
    print("\nTarget transition prompts:")
    for prompt in target_prompts:
        print(f"  {prompt}")
    print("\nWrote debug artifacts:")
    print(f"  {floorplan_figure_path}")
    print(f"  {final_floorplan_figure_path}")
    for scene_path in scene_paths:
        print(f"  {scene_path}")
    print(f"  {csv_path}")
    print(f"  {room_similarity_csv}")
    print(f"  {label_weights_csv}")
    print(f"  {output_dir / 'find_room_frontier_scores_debug.json'}")


if __name__ == "__main__":
    main()
