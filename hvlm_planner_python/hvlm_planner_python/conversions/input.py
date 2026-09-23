# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""DSG parser."""

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
import spark_dsg._dsg_bindings as spark_dsg

from hvlm_planner_python.conversions.image_sampling import BaseImageSampler
from hvlm_planner_python.misc import LoggerWarn


@dataclass
class DsgVLMFormat:
    """Data class for the DSG format used for the VLM input."""

    dsg: str
    images: list[np.ndarray]
    images_ordering: list[str]
    agent_layer_id: int
    agent_layer_partition: int
    node_id_to_index: dict[str, int]


@dataclass
class RoomImage:
    """Image and camera metadata captured at an agent node."""

    image: np.ndarray
    image_feature: np.ndarray
    position: np.ndarray
    yaw: float | None
    node_id: str


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """Draw readable text onto an RGB image."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (220, 220, 220),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (45, 45, 45),
        thickness,
        cv2.LINE_AA,
    )


def _stitch_room_images(images: list[np.ndarray], room_id: str) -> np.ndarray | None:
    """Stitch a room's selected images into a compact grid."""
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    stitched = np.zeros(
        (rows * max_height, cols * max_width, images[0].shape[2]), dtype=images[0].dtype
    )

    for idx, image in enumerate(images):
        resized = cv2.resize(image, (max_width, max_height))
        label = f"Image {idx + 1} from {room_id}"
        _draw_text(resized, label, (12, 28))
        row = idx // cols
        col = idx % cols
        stitched[
            row * max_height : (row + 1) * max_height,
            col * max_width : (col + 1) * max_width,
        ] = resized

    return stitched


def _quaternion_to_yaw(rotation) -> float | None:
    """Extract world-frame yaw from a spark_dsg Quaternion-like object."""
    if rotation is None:
        return None

    try:
        w = float(rotation.w)
        x = float(rotation.x)
        y = float(rotation.y)
        z = float(rotation.z)
    except AttributeError:
        return None

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _agent_position(attrs) -> np.ndarray:
    """Return the agent translation as a 3D vector."""
    position = np.asarray(attrs.position, dtype=np.float32).reshape(-1)
    if position.size >= 3:
        return position[:3].copy()

    padded = np.zeros(3, dtype=np.float32)
    padded[: position.size] = position
    return padded


def _selected_image_indices(
    images: list[np.ndarray], selected_images: list[np.ndarray]
) -> list[int]:
    """Map sampled image objects back to their source indices."""
    used_indices = set()
    selected_indices = []
    for selected_image in selected_images:
        for idx, image in enumerate(images):
            if idx in used_indices:
                continue
            if selected_image is image:
                selected_indices.append(idx)
                used_indices.add(idx)
                break
        else:
            LoggerWarn.warning(
                "Sampled image did not match an input image object; "
                "falling back to the first available room image"
            )
            for idx in range(len(images)):
                if idx not in used_indices:
                    selected_indices.append(idx)
                    used_indices.add(idx)
                    break

    return selected_indices


def _room_image_sort_key(room_image: RoomImage) -> tuple[float, float]:
    """Sort by yaw when available, otherwise by position."""
    if room_image.yaw is not None:
        return (0.0, room_image.yaw)

    return (1.0, math.atan2(room_image.position[1], room_image.position[0]))


def _draw_pose_map(
    map_image: np.ndarray,
    room_images: list[RoomImage],
    ordered_room_images: list[RoomImage],
) -> None:
    """Draw a top-down pose map into the provided panel."""
    map_image[:, :] = (245, 245, 245)
    height, width = map_image.shape[:2]
    margin = max(28, min(width, height) // 12)

    positions = np.asarray([room_image.position[:2] for room_image in room_images])
    min_xy = positions.min(axis=0)
    max_xy = positions.max(axis=0)
    span = max_xy - min_xy
    span[span < 1e-3] = 1.0

    def to_pixel(position: np.ndarray) -> tuple[int, int]:
        normalized = (position[:2] - min_xy) / span
        x_px = margin + normalized[0] * (width - 2 * margin)
        y_px = height - margin - normalized[1] * (height - 2 * margin)
        return int(round(x_px)), int(round(y_px))

    cv2.rectangle(map_image, (0, 0), (width - 1, height - 1), (180, 180, 180), 2)
    _draw_text(map_image, "Top-down agent poses", (12, 28), scale=0.6, thickness=2)

    if len(room_images) > 1:
        path_points = np.asarray(
            [to_pixel(room_image.position) for room_image in room_images],
            dtype=np.int32,
        )
        cv2.polylines(map_image, [path_points], False, (160, 160, 160), 2, cv2.LINE_AA)

    ordered_indices = {
        id(room_image): idx + 1 for idx, room_image in enumerate(ordered_room_images)
    }
    for room_image in room_images:
        x_px, y_px = to_pixel(room_image.position)
        color = (40, 115, 225) if id(room_image) in ordered_indices else (100, 100, 100)
        cv2.circle(map_image, (x_px, y_px), 8, color, -1, cv2.LINE_AA)
        cv2.circle(map_image, (x_px, y_px), 10, (35, 35, 35), 2, cv2.LINE_AA)

        if room_image.yaw is not None:
            arrow_len = max(24, min(width, height) // 10)
            end_point = (
                int(round(x_px + arrow_len * math.cos(room_image.yaw))),
                int(round(y_px - arrow_len * math.sin(room_image.yaw))),
            )
            cv2.arrowedLine(
                map_image,
                (x_px, y_px),
                end_point,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

        if id(room_image) in ordered_indices:
            _draw_text(
                map_image,
                str(ordered_indices[id(room_image)]),
                (x_px + 12, y_px - 10),
                scale=0.55,
                thickness=2,
            )


def _stitch_room_images_by_pose(
    selected_room_images: list[RoomImage],
    all_room_images: list[RoomImage],
    room_id: str,
) -> np.ndarray | None:
    """Stitch room images as a pose-aware contact sheet."""
    if not selected_room_images:
        return None

    if len(selected_room_images) == 1:
        return selected_room_images[0].image

    ordered_room_images = sorted(selected_room_images, key=_room_image_sort_key)
    selected_images = [room_image.image for room_image in ordered_room_images]
    max_height = max(image.shape[0] for image in selected_images)
    max_width = max(image.shape[1] for image in selected_images)
    cols = math.ceil(math.sqrt(len(selected_images)))
    rows = math.ceil(len(selected_images) / cols)

    stitched = np.zeros(
        (rows * max_height, (cols + 1) * max_width, selected_images[0].shape[2]),
        dtype=selected_images[0].dtype,
    )

    map_panel = stitched[:, :max_width]
    _draw_pose_map(map_panel, all_room_images, ordered_room_images)

    for idx, room_image in enumerate(ordered_room_images):
        resized = cv2.resize(room_image.image, (max_width, max_height))
        label = f"View {idx + 1} from {room_id}"
        _draw_text(resized, label, (12, 28))
        if room_image.yaw is not None:
            yaw_degrees = math.degrees(room_image.yaw)
            pose_label = (
                f"x={room_image.position[0]:.1f}, y={room_image.position[1]:.1f}, "
                f"yaw={yaw_degrees:.0f} deg"
            )
        else:
            pose_label = (
                f"x={room_image.position[0]:.1f}, y={room_image.position[1]:.1f}"
            )
        _draw_text(resized, pose_label, (12, 56), scale=0.55, thickness=2)

        row = idx // cols
        col = idx % cols
        x_start = (col + 1) * max_width
        stitched[
            row * max_height : (row + 1) * max_height,
            x_start : x_start + max_width,
        ] = resized

    return stitched


def _resolve_room_image_stitching(
    stitch_room_images: bool, room_image_stitching: str
) -> str:
    """Resolve legacy boolean stitching and the newer mode string."""
    mode = str(room_image_stitching).lower()
    valid_modes = {"none", "grid", "pose"}
    if mode not in valid_modes:
        LoggerWarn.warning(
            f"Unknown room_image_stitching mode '{room_image_stitching}', "
            "falling back to 'none'"
        )
        return "none"

    if mode != "none":
        return mode

    return "grid" if stitch_room_images else "none"


def to_vlm_format(
    graph: spark_dsg.DynamicSceneGraph,
    image_sampler: BaseImageSampler,
    stitch_room_images: bool = False,
    room_image_stitching: str = "none",
) -> DsgVLMFormat:
    """Convert the input to a format suitable for the VLM.
    :param hlp_input: Input for the High-Level Planner.
    :param image_sampler: BaseImageSampler object to use
                          for sampling images for the VLM.
    :param stitch_room_images: Legacy flag to stitch images
                               from the same room into
                               a single grid image.
    :param room_image_stitching: Room image stitching mode: "none", "grid", or "pose".
    :return: The DSG prompt string, VLM images, image ordering,
             and graph metadata.
    """
    dsg_json = {}
    room_images = {}
    room_image_features = {}
    room_image_records = {}
    images = []
    images_ordering = []
    node_id_to_index = {}
    layer_names = graph.layer_names
    stitching_mode = _resolve_room_image_stitching(
        stitch_room_images, room_image_stitching
    )

    # Get labelspace to map label ids to names
    labelspace = graph.get_labelspace(layer_names[spark_dsg.DsgLayers.OBJECTS].layer)

    # Get room nodes
    room_layer = graph.get_layer(spark_dsg.DsgLayers.ROOMS)
    for node in room_layer.nodes:
        attrs = node.attributes
        dsg_json[node.id.str()] = {
            "position": attrs.position.tolist(),
            "label": attrs.label,
            "objects": [],
            "frontiers": [],
        }
        node_id_to_index[node.id.str()] = node.id.value
        room_images[node.id.str()] = []
        room_image_features[node.id.str()] = []
        room_image_records[node.id.str()] = []

    single_room = False
    if room_layer.num_nodes() == 0:
        dsg_json["single_room"] = {
            "objects": [],
            "frontiers": [],
        }
        single_room = True
        room_images["single_room"] = []
        room_image_features["single_room"] = []
        room_image_records["single_room"] = []

    # Get object nodes
    object_layer = graph.get_layer(spark_dsg.DsgLayers.OBJECTS)
    for node in object_layer.nodes:
        attrs = node.attributes
        obj_data = {
            "position": attrs.position.tolist(),
            "label": labelspace.get_category(attrs.semantic_label),
            "id": node.id.str(),
        }
        node_id_to_index[node.id.str()] = node.id.value
        if single_room:
            dsg_json["single_room"]["objects"].append(obj_data)
        else:
            if not node.has_parent():
                LoggerWarn.warning(f"Object node {node.id} has no parent nav, skipping")
                continue
            nav_layer_parent_id = node.get_parent()
            nav_node = graph.get_node(nav_layer_parent_id)
            if not nav_node.has_parent():
                LoggerWarn.warning(
                    f"Navigation node {nav_layer_parent_id} has no parent room, "
                    f"skipping object node {node.id}"
                )
                continue
            room_parent_node = graph.get_node(nav_node.get_parent())
            dsg_json[room_parent_node.id.str()]["objects"].append(obj_data)

    # Get frontier nodes
    frontier_layer = graph.get_layer(spark_dsg.DsgLayers.FRONTIERS)
    for node in frontier_layer.nodes:
        attrs = node.attributes
        frontier_data = {
            "position": attrs.position.tolist(),
            "id": node.id.str(),
            "nearby_objects": [
                spark_dsg.NodeSymbol(object_id).str()
                for object_id in attrs.connected_objects
            ],
        }
        node_id_to_index[node.id.str()] = node.id.value
        if single_room:
            dsg_json["single_room"]["frontiers"].append(frontier_data)
        else:
            if attrs.connected_nav is None:
                LoggerWarn.warning(
                    f"Frontier node {node.id} has no connected navigation "
                    f"node, skipping"
                )
                continue
            if not graph.has_node(attrs.connected_nav):
                LoggerWarn.warning(
                    f"Connected navigation node {attrs.connected_nav} for "
                    f"frontier node {node.id} "
                    f"does not exist in the graph, skipping"
                )
                continue
            nav_node = graph.get_node(attrs.connected_nav)
            if not nav_node.has_parent():
                LoggerWarn.warning(
                    f"Navigation node {nav_node.id} connected to frontier "
                    f"node {node.id} has no parent room, skipping"
                )
                continue
            room_parent_node = graph.get_node(nav_node.get_parent())
            dsg_json[room_parent_node.id.str()]["frontiers"].append(frontier_data)

    # Get agent nodes
    agent_layer_id = graph.get_layer_key(spark_dsg.DsgLayers.AGENTS).layer
    agent_partition = None
    for layerkey in graph.layer_keys:
        if layerkey.partition != 0:
            agent_partition = layerkey.partition
            break
    if agent_partition is not None:
        agent_layer = graph.get_layer(agent_layer_id, agent_partition)
        for node in agent_layer.nodes:
            attrs = node.attributes
            agent_image = attrs.image
            if agent_image.shape[0] > 0:
                agent_image_feature = attrs.image_feature
                room_image = RoomImage(
                    image=agent_image,
                    image_feature=agent_image_feature,
                    position=_agent_position(attrs),
                    yaw=_quaternion_to_yaw(getattr(attrs, "world_R_body", None)),
                    node_id=node.id.str(),
                )
                if not single_room:
                    if node.has_parent():
                        nav_node = graph.get_node(node.get_parent())
                        if nav_node.has_parent():
                            room_node = graph.get_node(nav_node.get_parent())
                            room_images[room_node.id.str()].append(agent_image)
                            room_image_features[room_node.id.str()].append(
                                agent_image_feature
                            )
                            room_image_records[room_node.id.str()].append(room_image)
                else:
                    room_images["single_room"].append(agent_image)
                    room_image_features["single_room"].append(agent_image_feature)
                    room_image_records["single_room"].append(room_image)

    for room_id, img_list in room_images.items():
        features_list = room_image_features[room_id]
        selected_images = image_sampler.sample_images(img_list, features_list)
        if not selected_images:
            continue

        if stitching_mode != "none":
            if stitching_mode == "pose":
                selected_indices = _selected_image_indices(img_list, selected_images)
                selected_room_images = [
                    room_image_records[room_id][idx] for idx in selected_indices
                ]
                stitched_image = _stitch_room_images_by_pose(
                    selected_room_images, room_image_records[room_id], room_id
                )
            else:
                stitched_image = _stitch_room_images(selected_images, room_id)
            if stitched_image is None:
                continue
            images.append(stitched_image)
            images_ordering.append(
                f"{stitching_mode}-stitched relevant views from room {room_id} "
                f"containing {len(selected_images)} images"
            )
            continue

        images.extend(selected_images)
        images_ordering.extend(
            [
                f"relevant view {idx + 1}/{len(selected_images)} from room {room_id}"
                for idx in range(len(selected_images))
            ]
        )

    return DsgVLMFormat(
        dsg=json.dumps(dsg_json),
        images=images,
        images_ordering=images_ordering,
        agent_layer_id=agent_layer_id,
        agent_layer_partition=agent_partition,
        node_id_to_index=node_id_to_index,
    )
