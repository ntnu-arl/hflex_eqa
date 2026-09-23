#!/usr/bin/env python3

import argparse
import math
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import rosbag2_py
import yaml
import zstandard
from nav_msgs.msg import Path as NavPath
from rclpy.serialization import deserialize_message

DEFAULT_TOPIC = "/hflex_eqa/low_level/planned_path"
DEFAULT_STORAGE_ID = "mcap"
PATH_TYPE = "nav_msgs/msg/Path"
DECOMPRESSION_CHUNK_SIZE = 1024 * 1024


@dataclass
class BagPathStats:
    bag_uri: Path
    num_messages: int = 0
    num_poses: int = 0
    path_length_m: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the accumulated trajectory length from nav_msgs/msg/Path "
            "messages stored in rosbag2 MCAP bags."
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
        "--storage-id",
        default=DEFAULT_STORAGE_ID,
        help=f"rosbag2 storage plugin id. Defaults to {DEFAULT_STORAGE_ID}.",
    )
    parser.add_argument(
        "--planar",
        action="store_true",
        help="Compute x/y distance only, ignoring z.",
    )
    parser.add_argument(
        "--no-per-bag",
        action="store_true",
        help="Only print the final total.",
    )
    return parser.parse_args()


def resolve_bag_dirs(path):
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


def open_reader(bag_uri, storage_id):
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


def bag_metadata(bag_uri):
    metadata_path = bag_uri / "metadata.yaml"
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return yaml.safe_load(metadata_file)


def bag_file_paths(metadata):
    bag_info = metadata.get("rosbag2_bagfile_information", {})
    return [Path(path) for path in bag_info.get("relative_file_paths", [])]


def has_zstd_bag_files(bag_uri):
    metadata = bag_metadata(bag_uri)
    return any(path.suffix == ".zstd" for path in bag_file_paths(metadata))


def decompress_zstd_file(input_path, output_path):
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


def write_uncompressed_metadata(metadata, output_metadata_path):
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
def prepared_bag_dir(bag_uri):
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


def topic_type_lookup(reader):
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def path_length(path_msg, planar=False):
    poses = path_msg.poses
    if len(poses) < 2:
        return 0.0

    length = 0.0
    previous = poses[0].pose.position
    for pose_stamped in poses[1:]:
        current = pose_stamped.pose.position
        dx = current.x - previous.x
        dy = current.y - previous.y
        dz = 0.0 if planar else current.z - previous.z
        length += math.sqrt(dx * dx + dy * dy + dz * dz)
        previous = current
    return length


def compute_bag_path_stats(bag_uri, topic, storage_id, planar=False):
    with prepared_bag_dir(bag_uri) as readable_bag_uri:
        return compute_readable_bag_path_stats(
            bag_uri=bag_uri,
            readable_bag_uri=readable_bag_uri,
            topic=topic,
            storage_id=storage_id,
            planar=planar,
        )


def compute_readable_bag_path_stats(
    bag_uri, readable_bag_uri, topic, storage_id, planar=False
):
    reader = open_reader(readable_bag_uri, storage_id)
    topics = topic_type_lookup(reader)

    if topic not in topics:
        return BagPathStats(bag_uri=bag_uri)

    if topics[topic] != PATH_TYPE:
        raise TypeError(
            f"Topic {topic} in {bag_uri} has type {topics[topic]}, expected {PATH_TYPE}."
        )

    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    stats = BagPathStats(bag_uri=bag_uri)
    while reader.has_next():
        read_topic, serialized_msg, _timestamp = reader.read_next()
        if read_topic != topic:
            continue

        path_msg = deserialize_message(serialized_msg, NavPath)
        stats.num_messages += 1
        stats.num_poses += len(path_msg.poses)
        stats.path_length_m += path_length(path_msg, planar=planar)

    return stats


def print_stats(stats, root_path):
    try:
        bag_name = stats.bag_uri.relative_to(root_path)
    except ValueError:
        bag_name = stats.bag_uri
    if str(bag_name) == ".":
        bag_name = stats.bag_uri.name

    print(
        f"{bag_name}: "
        f"{stats.path_length_m:.3f} m "
        f"({stats.num_messages} path messages, {stats.num_poses} poses)"
    )


def main():
    args = parse_args()
    bag_dirs = resolve_bag_dirs(args.bag_path)

    all_stats = [
        compute_bag_path_stats(
            bag_uri=bag_dir,
            topic=args.topic,
            storage_id=args.storage_id,
            planar=args.planar,
        )
        for bag_dir in bag_dirs
    ]

    root_path = args.bag_path.expanduser().resolve()
    if root_path.is_file():
        root_path = root_path.parent

    if not args.no_per_bag:
        for stats in all_stats:
            print_stats(stats, root_path)

    total_messages = sum(stats.num_messages for stats in all_stats)
    total_poses = sum(stats.num_poses for stats in all_stats)
    total_length_m = sum(stats.path_length_m for stats in all_stats)
    print(
        f"Total trajectory length: {total_length_m:.3f} m "
        f"({total_messages} path messages, {total_poses} poses, {len(all_stats)} bags)"
    )


if __name__ == "__main__":
    main()
