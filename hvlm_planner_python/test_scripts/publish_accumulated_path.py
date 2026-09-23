#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
"""Publish one nav_msgs/msg/Path from an accumulated path JSON file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_TOPIC = "/hvlm_planner/low_level/planned_path"

rclpy = None
Time = None
PoseStamped = None
NavPath = None
Node = None
DurabilityPolicy = None
QoSProfile = None
ReliabilityPolicy = None


def load_ros_modules() -> None:
    global rclpy
    global Time
    global PoseStamped
    global NavPath
    global Node
    global DurabilityPolicy
    global QoSProfile
    global ReliabilityPolicy

    import rclpy as rclpy_module
    from builtin_interfaces.msg import Time as ros_time
    from geometry_msgs.msg import PoseStamped as ros_pose_stamped
    from nav_msgs.msg import Path as ros_nav_path
    from rclpy.node import Node as ros_node
    from rclpy.qos import DurabilityPolicy as ros_durability_policy
    from rclpy.qos import QoSProfile as ros_qos_profile
    from rclpy.qos import ReliabilityPolicy as ros_reliability_policy

    rclpy = rclpy_module
    Time = ros_time
    PoseStamped = ros_pose_stamped
    NavPath = ros_nav_path
    Node = ros_node
    DurabilityPolicy = ros_durability_policy
    QoSProfile = ros_qos_profile
    ReliabilityPolicy = ros_reliability_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSON file produced by extract_accumulated_path_from_bag.py "
            "and publish its poses as one nav_msgs/msg/Path message."
        )
    )
    parser.add_argument(
        "path_file",
        type=Path,
        help="Accumulated path JSON file to publish.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help=(
            "Topic to publish on. Defaults to the saved source_topic, or "
            f"{DEFAULT_TOPIC} if the file has no source_topic."
        ),
    )
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Optional frame_id override for the published Path and poses.",
    )
    parser.add_argument(
        "--stamp-now",
        action="store_true",
        help="Stamp the Path and every pose with the current ROS time before publishing.",
    )
    parser.add_argument(
        "--publish-delay",
        type=float,
        default=0.25,
        help="Seconds to wait before publishing once. Defaults to 0.25.",
    )
    parser.add_argument(
        "--keep-alive-seconds",
        type=float,
        default=5.0,
        help=(
            "Seconds to keep the transient-local publisher alive after publishing. "
            "Defaults to 5.0."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Accumulated path file does not exist: {path}")
    with path.open(encoding="utf-8") as path_file:
        data = json.load(path_file)
    if not isinstance(data, dict):
        raise ValueError(f"Accumulated path file must contain a JSON object: {path}")
    if not isinstance(data.get("poses", []), list):
        raise ValueError(f"Accumulated path file field 'poses' must be a list: {path}")
    return data


def dict_to_stamp(data: dict | None) -> Time:
    stamp = Time()
    if not isinstance(data, dict):
        return stamp
    stamp.sec = int(data.get("sec", 0))
    stamp.nanosec = int(data.get("nanosec", 0))
    return stamp


def fill_pose_stamped(entry: dict, fallback_frame_id: str) -> PoseStamped:
    pose_stamped = PoseStamped()
    header = entry.get("header", {})
    pose_stamped.header.stamp = dict_to_stamp(header.get("stamp"))
    pose_stamped.header.frame_id = header.get("frame_id") or fallback_frame_id

    pose = entry.get("pose", {})
    position = pose.get("position", {})
    orientation = pose.get("orientation", {})

    pose_stamped.pose.position.x = float(position.get("x", 0.0))
    pose_stamped.pose.position.y = float(position.get("y", 0.0))
    pose_stamped.pose.position.z = float(position.get("z", 0.0))
    pose_stamped.pose.orientation.x = float(orientation.get("x", 0.0))
    pose_stamped.pose.orientation.y = float(orientation.get("y", 0.0))
    pose_stamped.pose.orientation.z = float(orientation.get("z", 0.0))
    pose_stamped.pose.orientation.w = float(orientation.get("w", 1.0))
    return pose_stamped


def accumulated_data_to_path_msg(data: dict, frame_id_override: str | None) -> NavPath:
    path_msg = NavPath()
    path_header = data.get("path_header", {})
    saved_frame_id = path_header.get("frame_id", "")
    frame_id = frame_id_override if frame_id_override is not None else saved_frame_id

    path_msg.header.stamp = dict_to_stamp(path_header.get("stamp"))
    path_msg.header.frame_id = frame_id
    path_msg.poses = [
        fill_pose_stamped(entry, frame_id)
        for entry in data.get("poses", [])
        if isinstance(entry, dict)
    ]

    if frame_id_override is not None:
        for pose_stamped in path_msg.poses:
            pose_stamped.header.frame_id = frame_id_override

    return path_msg


def make_path_publisher(topic: str, path_msg: NavPath) -> Node:
    class SinglePathPublisher(Node):
        def __init__(self, topic: str, path_msg: NavPath) -> None:
            super().__init__("single_accumulated_path_publisher")
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._publisher = self.create_publisher(NavPath, topic, qos)
            self._topic = topic
            self._path_msg = path_msg

        def publish_once(self) -> None:
            self._publisher.publish(self._path_msg)
            self.get_logger().info(
                "Published one nav_msgs/msg/Path with "
                f"{len(self._path_msg.poses)} poses on {self._topic}"
            )

    return SinglePathPublisher(topic, path_msg)


def main() -> None:
    args = parse_args()
    data = load_json(args.path_file)
    topic = args.topic or data.get("source_topic") or DEFAULT_TOPIC

    load_ros_modules()
    rclpy.init()
    node = None
    try:
        path_msg = accumulated_data_to_path_msg(data, args.frame_id)
        node = make_path_publisher(topic, path_msg)

        if args.stamp_now:
            now = node.get_clock().now().to_msg()
            path_msg.header.stamp = now
            for pose_stamped in path_msg.poses:
                pose_stamped.header.stamp = now

        delay_end = time.monotonic() + max(args.publish_delay, 0.0)
        while rclpy.ok() and time.monotonic() < delay_end:
            rclpy.spin_once(node, timeout_sec=0.05)

        node.publish_once()

        keep_alive_end = time.monotonic() + max(args.keep_alive_seconds, 0.0)
        while rclpy.ok() and time.monotonic() < keep_alive_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
