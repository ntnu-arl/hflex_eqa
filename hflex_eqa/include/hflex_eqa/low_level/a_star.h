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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "hflex_eqa/common/input.h"  // adjust include path as needed

namespace hflex_eqa {
using namespace input;

using Pose = Eigen::Isometry3d;

class AStarPlanner {
 public:
  using Ptr = std::unique_ptr<AStarPlanner>;
  using Path = std::vector<Pose>;

  struct Config {
    bool use_diagonal = true;
    bool use_robot_clearance = false;
    double robot_clearance_radius_m = 0.0;
    double start_robot_clearance_radius_m = -1.0;
    double goal_robot_clearance_radius_m = -1.0;
    bool use_initial_robot_pose_clearance = false;
    double initial_robot_pose_clearance_radius_m = 0.0;
  } const config;

  enum class ClearanceSegment {
    DEFAULT,
    START_TO_TRAVERSABILITY,
    TRAVERSABILITY_TO_GOAL,
  };

  explicit AStarPlanner(const Config& config);

  bool plan(const OccupancyGrid::Ptr& occupancy_grid,
            const Pose& start,
            const Pose& goal,
            Path& path,
            ClearanceSegment clearance_segment = ClearanceSegment::DEFAULT);

 private:
  struct Node {
    GridIndex index;
    double g = 0.0;
    double f = 0.0;

    bool operator>(const Node& other) const { return f > other.f; }
  };

  double heuristic(const GridIndex& a, const GridIndex& b) const;

  std::vector<GridIndex> getNeighbors(const GridIndex& idx) const;

  double clearanceRadius(ClearanceSegment clearance_segment) const;

  bool isTraversable(
      const OccupancyGrid& grid,
      const GridIndex& idx,
      double clearance_radius_m,
      const std::optional<GridIndex>& initial_clearance_center = std::nullopt) const;

  bool isInsideInitialClearance(
      const OccupancyGrid& grid,
      const GridIndex& idx,
      const std::optional<GridIndex>& initial_clearance_center) const;

  void reconstructPath(
      const std::unordered_map<GridIndex, GridIndex, GridIndexHash>& came_from,
      const GridIndex& current,
      const OccupancyGrid& grid,
      const double& z_value,
      Path& path) const;
  const int offsets8[8][2] = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int offsets4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
};

void declare_config(AStarPlanner::Config& config);

}  // namespace hflex_eqa
