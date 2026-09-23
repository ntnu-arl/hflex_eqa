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
#include "hvlm_planner/low_level/a_star.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>

namespace hvlm_planner {

void declare_config(AStarPlanner::Config& config) {
  using namespace config;
  name("AStarPlannerConfig");
  field(config.use_diagonal, "use_diagonal");
  field(config.use_robot_clearance, "use_robot_clearance");
  field(config.robot_clearance_radius_m, "robot_clearance_radius_m", "m");
  field(config.start_robot_clearance_radius_m, "start_robot_clearance_radius_m", "m");
  field(config.goal_robot_clearance_radius_m, "goal_robot_clearance_radius_m", "m");
  field(config.use_initial_robot_pose_clearance, "use_initial_robot_pose_clearance");
  field(config.initial_robot_pose_clearance_radius_m,
        "initial_robot_pose_clearance_radius_m",
        "m");
  checkCondition(config.robot_clearance_radius_m >= 0.0,
                 "robot_clearance_radius_m must be non-negative.");
  checkCondition(config.start_robot_clearance_radius_m >= 0.0 ||
                     config.start_robot_clearance_radius_m == -1.0,
                 "start_robot_clearance_radius_m must be non-negative.");
  checkCondition(config.goal_robot_clearance_radius_m >= 0.0 ||
                     config.goal_robot_clearance_radius_m == -1.0,
                 "goal_robot_clearance_radius_m must be non-negative.");
  checkCondition(config.initial_robot_pose_clearance_radius_m >= 0.0,
                 "initial_robot_pose_clearance_radius_m must be non-negative.");
}

AStarPlanner::AStarPlanner(const Config& config) : config(config::checkValid(config)) {}

bool AStarPlanner::plan(const input::OccupancyGrid::Ptr& occupancy_grid,
                        const Pose& start,
                        const Pose& goal,
                        Path& path,
                        ClearanceSegment clearance_segment) {
  path.clear();
  if (!occupancy_grid || occupancy_grid->empty()) {
    LOG(ERROR) << "AStarPlanner: Occupancy grid is null or empty.";
    return false;
  }

  const auto& grid = *occupancy_grid;

  const auto start_idx = grid.worldToGrid(start.translation().head<2>().cast<float>());
  const auto goal_idx = grid.worldToGrid(goal.translation().head<2>().cast<float>());
  const double clearance_radius_m = clearanceRadius(clearance_segment);
  const std::optional<GridIndex> initial_clearance_center =
      clearance_segment == ClearanceSegment::START_TO_TRAVERSABILITY &&
              config.use_initial_robot_pose_clearance &&
              config.initial_robot_pose_clearance_radius_m > 0.0
          ? std::make_optional(start_idx)
          : std::nullopt;
  VLOG(2) << "AStarPlanner: using robot clearance radius " << clearance_radius_m
          << " m.";
  if (initial_clearance_center) {
    VLOG(2) << "AStarPlanner: using initial robot pose clearance radius "
            << config.initial_robot_pose_clearance_radius_m << " m.";
  }

  if (!isTraversable(grid, start_idx, clearance_radius_m, initial_clearance_center) ||
      !isTraversable(grid, goal_idx, clearance_radius_m, initial_clearance_center)) {
    LOG(ERROR) << "AStarPlanner: Start or goal is not traversable.";
    return false;
  }

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;

  std::unordered_map<GridIndex, double, GridIndexHash> g_score;
  std::unordered_map<GridIndex, GridIndex, GridIndexHash> came_from;

  g_score[start_idx] = 0.0;

  open_set.push(Node{start_idx, 0.0, heuristic(start_idx, goal_idx)});

  while (!open_set.empty()) {
    Node current = open_set.top();
    open_set.pop();

    if (current.index == goal_idx) {
      reconstructPath(came_from, goal_idx, grid, start.translation().z(), path);
      path.insert(path.end(), goal);  // Ensure the final pose is exactly the goal
      LOG(INFO) << "AStarPlanner: Path found with " << path.size() << " waypoints.";
      return true;
    }

    for (const auto& neighbor : getNeighbors(current.index)) {
      if (!isTraversable(
              grid, neighbor, clearance_radius_m, initial_clearance_center)) {
        continue;
      }

      double move_cost =
          heuristic(current.index, neighbor);  // exact cost (1 or sqrt(2))

      double tentative_g = g_score[current.index] + move_cost;

      auto it = g_score.find(neighbor);
      if (it == g_score.end() || tentative_g < it->second) {
        came_from[neighbor] = current.index;
        g_score[neighbor] = tentative_g;

        double f = tentative_g + heuristic(neighbor, goal_idx);

        open_set.push(Node{neighbor, tentative_g, f});
      }
    }
  }

  LOG(ERROR) << "AStarPlanner: Failed to find a path from start to goal.";
  return false;
}

double AStarPlanner::heuristic(const GridIndex& a, const GridIndex& b) const {
  const double dx = static_cast<double>(a.second - b.second);
  const double dy = static_cast<double>(a.first - b.first);
  return std::sqrt(dx * dx + dy * dy);
}

std::vector<GridIndex> AStarPlanner::getNeighbors(const GridIndex& idx) const {
  std::vector<GridIndex> neighbors;
  size_t num_neighbors = config.use_diagonal ? 8 : 4;
  neighbors.reserve(num_neighbors);

  const auto& offsets = config.use_diagonal ? offsets8 : offsets4;
  for (size_t i = 0; i < num_neighbors; ++i) {
    neighbors.emplace_back(idx.first + offsets[i][0], idx.second + offsets[i][1]);
  }

  return neighbors;
}

double AStarPlanner::clearanceRadius(ClearanceSegment clearance_segment) const {
  switch (clearance_segment) {
    case ClearanceSegment::START_TO_TRAVERSABILITY:
      if (config.start_robot_clearance_radius_m >= 0.0) {
        return config.start_robot_clearance_radius_m;
      }
      break;
    case ClearanceSegment::TRAVERSABILITY_TO_GOAL:
      if (config.goal_robot_clearance_radius_m >= 0.0) {
        return config.goal_robot_clearance_radius_m;
      }
      break;
    case ClearanceSegment::DEFAULT:
      break;
  }

  return config.robot_clearance_radius_m;
}

bool AStarPlanner::isTraversable(
    const OccupancyGrid& grid,
    const GridIndex& idx,
    double clearance_radius_m,
    const std::optional<GridIndex>& initial_clearance_center) const {
  const bool is_initial_pose =
      initial_clearance_center && idx == initial_clearance_center.value();
  if (!grid.isFree(idx) && !is_initial_pose) {
    return false;
  }

  if (!config.use_robot_clearance || clearance_radius_m <= 0.0) {
    return true;
  }

  if (grid.resolution <= 0.0f) {
    LOG(ERROR) << "AStarPlanner: Occupancy grid resolution must be positive.";
    return false;
  }

  const int radius_cells =
      static_cast<int>(std::ceil(clearance_radius_m / grid.resolution));
  const double radius_cells_sq = std::pow(clearance_radius_m / grid.resolution, 2.0);

  for (int dr = -radius_cells; dr <= radius_cells; ++dr) {
    for (int dc = -radius_cells; dc <= radius_cells; ++dc) {
      if (static_cast<double>(dr * dr + dc * dc) > radius_cells_sq) {
        continue;
      }

      const GridIndex neighbor{idx.first + dr, idx.second + dc};
      if (!grid.isFree(neighbor) &&
          !isInsideInitialClearance(grid, neighbor, initial_clearance_center)) {
        return false;
      }
    }
  }

  return true;
}

bool AStarPlanner::isInsideInitialClearance(
    const OccupancyGrid& grid,
    const GridIndex& idx,
    const std::optional<GridIndex>& initial_clearance_center) const {
  if (!initial_clearance_center ||
      config.initial_robot_pose_clearance_radius_m <= 0.0) {
    return false;
  }
  if (grid.resolution <= 0.0f) {
    return false;
  }

  const double radius_cells =
      config.initial_robot_pose_clearance_radius_m / grid.resolution;
  const double dr = static_cast<double>(idx.first - initial_clearance_center->first);
  const double dc = static_cast<double>(idx.second - initial_clearance_center->second);
  return dr * dr + dc * dc <= radius_cells * radius_cells;
}

void AStarPlanner::reconstructPath(
    const std::unordered_map<GridIndex, GridIndex, GridIndexHash>& came_from,
    const GridIndex& goal,
    const OccupancyGrid& grid,
    const double& z_value,
    Path& path) const {
  path.clear();

  std::vector<GridIndex> index_path;
  index_path.reserve(came_from.size() + 1);
  GridIndex current = goal;
  index_path.push_back(current);

  while (true) {
    auto it = came_from.find(current);
    if (it == came_from.end()) break;

    current = it->second;
    index_path.push_back(current);
  }

  std::reverse(index_path.begin(), index_path.end());
  if (index_path.size() < 2) {
    LOG(WARNING) << "AStarPlanner: Path has less than 2 waypoints.";
    return;
  }

  path.reserve(index_path.size());
  for (size_t i = 1; i < index_path.size() - 1; ++i) {
    Point2f world = grid.gridToWorld(index_path[i]);
    Pose pose = Pose::Identity();
    pose.translation().x() = world.x();
    pose.translation().y() = world.y();
    pose.translation().z() = z_value;

    Point2f next_world = grid.gridToWorld(index_path[i + 1]);
    Eigen::Vector2d direction = (next_world - world).cast<double>();
    if (direction.norm() > 1e-6) {
      direction.normalize();
      double yaw = std::atan2(direction.y(), direction.x());
      pose.linear() =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    path.push_back(pose);
  }
}

}  // namespace hvlm_planner
