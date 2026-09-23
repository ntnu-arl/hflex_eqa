#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
"""Extract one accumulated nav_msgs/msg/Path from rosbag2 Path messages."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TOPIC = "/hvlm_planner/low_level/planned_path"
DEFAULT_OUTPUT = "accumulated_path.json"
DEFAULT_STORAGE_ID = "mcap"
PATH_TYPE = "nav_msgs/msg/Path"
FORMAT = "hvlm_planner_python.accumulated_nav_path.v1"
DECOMPRESSION_CHUNK_SIZE = 1024 * 1024

NavPath = None
deserialize_message = None
rosbag2_py = None
yaml = None
zstandard = None


def load_rosbag_modules() -> None:
    global NavPath
    global deserialize_message
    global rosbag2_py
    global yaml
    global zstandard

    import rosbag2_py as rosbag2_module
    import yaml as yaml_module
    import zstandard as zstandard_module
    from nav_msgs.msg import Path as ros_nav_path
    from rclpy.serialization import deserialize_message as ros_deserialize_message

    NavPath = ros_nav_path
    deserialize_message = ros_deserialize_message
    rosbag2_py = rosbag2_module
    yaml = yaml_module
    zstandard = zstandard_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read nav_msgs/msg/Path messages from rosbag2 and save all poses as "
            "one accumulated JSON path."
        )
    )
    parser.add_argument(
        "bag_path",
        type=Path,
        help=(
            "Path to a rosbag2 directory, a directory containing rosbag2 directories, "
            "or an .mcap/.mcap.zstd file inside a rosbag2 directory."
        ),
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Path topic to read. Defaults to {DEFAULT_TOPIC}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--storage-id",
        default=DEFAULT_STORAGE_ID,
        help=f"rosbag2 storage plugin id. Defaults to {DEFAULT_STORAGE_ID}.",
    )
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Optional frame_id override for the accumulated path.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help=(
            "Minimum distance in meters between appended consecutive poses. "
            "Defaults to 0.0, which only removes exact consecutive duplicates."
        ),
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Append every read pose, including consecutive duplicates.",
    )
    return parser.parse_args()


def resolve_bag_dirs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {path}")

    if path.is_file():
        if path.name == "metadata.yaml":
            return [path.parent]
        if path.suffixes[-2:] == [".mcap", ".zstd"] or path.suffix == ".mcap":
            return [path.parent]
        raise ValueError(
            f"Expected a rosbag2 metadata.yaml or .mcap/.mcap.zstd file: {path}"
        )

    if (path / "metadata.yaml").exists():
        return [path]

    bag_dirs = sorted({metadata.parent for metadata in path.rglob("metadata.yaml")})
    if not bag_dirs:
        raise ValueError(
            f"Could not find any rosbag2 bags under {path}. "
            "Expected directories containing metadata.yaml."
        )
    return bag_dirs


def open_reader(bag_uri: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_uri),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="",
        output_serialization_format="",
    )
    reader.open(storage_options, converter_options)
    return reader


def bag_metadata(bag_uri: Path) -> dict:
    metadata_path = bag_uri / "metadata.yaml"
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return yaml.safe_load(metadata_file)


def bag_file_paths(metadata: dict) -> list[Path]:
    bag_info = metadata.get("rosbag2_bagfile_information", {})
    return [Path(path) for path in bag_info.get("relative_file_paths", [])]


def has_zstd_bag_files(bag_uri: Path) -> bool:
    metadata = bag_metadata(bag_uri)
    return any(path.suffix == ".zstd" for path in bag_file_paths(metadata))


def decompress_zstd_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zstandard.ZstdDecompressor()
    with input_path.open("rb") as compressed_file:
        with output_path.open("wb") as decompressed_file:
            decompressor.copy_stream(
                compressed_file,
                decompressed_file,
                read_size=DECOMPRESSION_CHUNK_SIZE,
                write_size=DECOMPRESSION_CHUNK_SIZE,
            )


def write_uncompressed_metadata(metadata: dict, output_metadata_path: Path) -> None:
    metadata = dict(metadata)
    bag_info = dict(metadata["rosbag2_bagfile_information"])
    bag_info["relative_file_paths"] = [
        str(Path(path).with_suffix(""))
        if Path(path).suffix == ".zstd"
        else str(Path(path))
        for path in bag_info.get("relative_file_paths", [])
    ]
    bag_info["compression_format"] = ""
    bag_info["compression_mode"] = ""
    metadata["rosbag2_bagfile_information"] = bag_info

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        yaml.safe_dump(metadata, metadata_file, sort_keys=False)


@contextmanager
def prepared_bag_dir(bag_uri: Path):
    if not has_zstd_bag_files(bag_uri):
        yield bag_uri
        return

    metadata = bag_metadata(bag_uri)
    with tempfile.TemporaryDirectory(
        prefix=f"{bag_uri.name}_uncompressed_"
    ) as temp_dir:
        temp_bag_uri = Path(temp_dir) / bag_uri.name
        temp_bag_uri.mkdir(parents=True)

        for relative_path in bag_file_paths(metadata):
            source_path = bag_uri / relative_path
            target_relative_path = (
                relative_path.with_suffix("")
                if relative_path.suffix == ".zstd"
                else relative_path
            )
            target_path = temp_bag_uri / target_relative_path

            if relative_path.suffix == ".zstd":
                print(f"Decompressing {source_path.name}...", file=sys.stderr)
                decompress_zstd_file(source_path, target_path)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

        write_uncompressed_metadata(metadata, temp_bag_uri / "metadata.yaml")
        yield temp_bag_uri


def topic_type_lookup(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def stamp_to_dict(stamp) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def header_to_dict(header, frame_id_override: str | None = None) -> dict:
    return {
        "stamp": stamp_to_dict(header.stamp),
        "frame_id": frame_id_override
        if frame_id_override is not None
        else header.frame_id,
    }


def pose_to_dict(pose) -> dict:
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def position_distance(first: dict[str, float], second: dict[str, float]) -> float:
    dx = first["x"] - second["x"]
    dy = first["y"] - second["y"]
    dz = first["z"] - second["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def should_append_pose(
    poses: list[dict],
    pose_entry: dict,
    keep_duplicates: bool,
    min_distance: float,
) -> bool:
    if keep_duplicates or not poses:
        return True

    previous_position = poses[-1]["pose"]["position"]
    current_position = pose_entry["pose"]["position"]
    return position_distance(previous_position, current_position) > min_distance


def extract_bag_path(
    bag_uri: Path,
    topic: str,
    storage_id: str,
    poses: list[dict],
    frame_id_override: str | None,
    keep_duplicates: bool,
    min_distance: float,
) -> tuple[int, int, int, dict | None]:
    with prepared_bag_dir(bag_uri) as readable_bag_uri:
        reader = open_reader(readable_bag_uri, storage_id)
        topics = topic_type_lookup(reader)

        if topic not in topics:
            print(f"Skipping {bag_uri}: topic {topic} not found", file=sys.stderr)
            return 0, 0, 0, None

        if topics[topic] != PATH_TYPE:
            raise TypeError(
                f"Topic {topic} in {bag_uri} has type {topics[topic]}, "
                f"expected {PATH_TYPE}."
            )

        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

        message_count = 0
        received_pose_count = 0
        skipped_pose_count = 0
        path_header = None

        while reader.has_next():
            read_topic, serialized_msg, _timestamp = reader.read_next()
            if read_topic != topic:
                continue

            path_msg = deserialize_message(serialized_msg, NavPath)
            message_count += 1
            received_pose_count += len(path_msg.poses)
            path_header = header_to_dict(path_msg.header, frame_id_override)

            for pose_stamped in path_msg.poses:
                pose_entry = {
                    "header": header_to_dict(pose_stamped.header, frame_id_override),
                    "pose": pose_to_dict(pose_stamped.pose),
                }
                if not should_append_pose(
                    poses, pose_entry, keep_duplicates, min_distance
                ):
                    skipped_pose_count += 1
                    continue
                poses.append(pose_entry)

        return message_count, received_pose_count, skipped_pose_count, path_header


def write_output(
    output_path: Path,
    source_bag_paths: list[Path],
    source_topic: str,
    path_header: dict | None,
    message_count: int,
    received_pose_count: int,
    skipped_pose_count: int,
    poses: list[dict],
    frame_id_override: str | None,
) -> None:
    frame_id = frame_id_override
    if frame_id is None and path_header is not None:
        frame_id = path_header.get("frame_id", "")
    if frame_id is None:
        frame_id = ""

    path_header = path_header or {
        "stamp": {"sec": 0, "nanosec": 0},
        "frame_id": frame_id,
    }
    path_header["frame_id"] = frame_id

    data = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_topic": source_topic,
        "source_bag_paths": [str(path) for path in source_bag_paths],
        "path_header": path_header,
        "message_count": message_count,
        "received_pose_count": received_pose_count,
        "skipped_pose_count": skipped_pose_count,
        "pose_count": len(poses),
        "poses": poses,
    }

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    load_rosbag_modules()
    bag_dirs = resolve_bag_dirs(args.bag_path)

    poses: list[dict] = []
    path_header = None
    total_messages = 0
    total_received_poses = 0
    total_skipped_poses = 0

    for bag_dir in bag_dirs:
        message_count, received_pose_count, skipped_pose_count, bag_path_header = (
            extract_bag_path(
                bag_uri=bag_dir,
                topic=args.topic,
                storage_id=args.storage_id,
                poses=poses,
                frame_id_override=args.frame_id,
                keep_duplicates=args.keep_duplicates,
                min_distance=args.min_distance,
            )
        )
        total_messages += message_count
        total_received_poses += received_pose_count
        total_skipped_poses += skipped_pose_count
        if bag_path_header is not None:
            path_header = bag_path_header

        print(
            f"{bag_dir}: {message_count} path messages, "
            f"{received_pose_count} poses read"
        )

    if total_messages == 0:
        raise RuntimeError(f"No {PATH_TYPE} messages found on topic {args.topic}")

    write_output(
        output_path=args.output,
        source_bag_paths=bag_dirs,
        source_topic=args.topic,
        path_header=path_header,
        message_count=total_messages,
        received_pose_count=total_received_poses,
        skipped_pose_count=total_skipped_poses,
        poses=poses,
        frame_id_override=args.frame_id,
    )

    print(
        f"Wrote {len(poses)} accumulated poses from {total_messages} path messages "
        f"to {args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
