/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, NTNU Autonomous Robots Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#pragma once
#include <spark_dsg/scene_graph_types.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "hvlm_planner/common/types.h"
#include "hvlm_planner/low_level/semantic_search.h"

namespace hvlm_planner {

struct LowLevelOutput {
  using Ptr = std::shared_ptr<LowLevelOutput>;

  LowLevelOutput() = default;
  virtual ~LowLevelOutput() = default;

  //! Timestamp
  uint64_t timestamp_ns;
  //! Goal node
  spark_dsg::NodeId goal;
  //! Goal pose
  Pose goal_pose;
  //! Start pose
  Pose start_pose;
  // Nav graph nodes to follow
  std::vector<spark_dsg::NodeId> nav_graph_nodes;
  // Poses of nodes to follow
  std::vector<Pose> nav_graph_poses;
  // Waypoints
  std::vector<Pose> waypoints;
  // Success flag
  bool success = false;
  // Whether the goal is found
  bool valid_goal = false;
  // Original selected frontier, if the executable goal was projected to traversability.
  bool has_selected_frontier_pose = false;
  Pose selected_frontier_pose;

  void fromSearchOutput(const SearchOutput& search_output) {
    goal = search_output.goal;
    goal_pose = search_output.goal_pose;
    selected_frontier_pose = search_output.selected_frontier_pose;
    has_selected_frontier_pose = search_output.has_selected_frontier_pose;
    start_pose = search_output.start_pose;
    nav_graph_nodes = search_output.nav_nodes;
    nav_graph_poses = search_output.nav_poses;
    success = search_output.success;
    valid_goal = search_output.valid_goal;
  }
};

}  // namespace hvlm_planner
