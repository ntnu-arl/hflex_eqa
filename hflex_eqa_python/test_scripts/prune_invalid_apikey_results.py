#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
"""Prune invalid result subdirectories from a parent folder."""

import argparse
import json
import shutil
from pathlib import Path

VALID_FINAL_STATES = {"STUCK", "finished", "FINISHED"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete invalid result subdirectories based on log.json contents."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Parent folder whose immediate subdirectories will be checked.",
    )
    return parser.parse_args()


def _remove_directory(directory: Path, reason: str) -> None:
    shutil.rmtree(directory)
    print(f"Deleted {directory} ({reason})")


def _should_delete_from_log(log_path: Path) -> bool:
    with log_path.open("r", encoding="utf-8") as handle:
        log_data = json.load(handle)

    final_state = log_data.get("final_state")
    num_iters = log_data.get("num_iters")
    return final_state not in VALID_FINAL_STATES and num_iters == 50


def main() -> None:
    args = _parse_args()
    root_folder = args.folder.expanduser().resolve()

    if not root_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")
    if not root_folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_folder}")

    deleted_count = 0
    kept_count = 0

    for subdir in sorted(root_folder.iterdir()):
        if not subdir.is_dir():
            continue

        log_path = subdir / "log.json"
        if not log_path.exists():
            _remove_directory(subdir, "missing log.json")
            deleted_count += 1
            continue

        if _should_delete_from_log(log_path):
            _remove_directory(
                subdir,
                "invalid final_state with num_iters == 50",
            )
            deleted_count += 1
            continue

        kept_count += 1

    print(f"Finished. Kept {kept_count} subdirectories and deleted {deleted_count}.")


if __name__ == "__main__":
    main()
