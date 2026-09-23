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
#include "hvlm_planner/low_level/planner.h"

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

void declare_config(Planner::Config& config) {
  using namespace config;
  name("PlannerConfig");
  field(config.interpolate_yaw_waypoints, "interpolate_yaw_waypoints");
  field(config.append_goal_yaw_rotation, "append_goal_yaw_rotation");
  field(config.add_goal, "add_goal");
  field(config.delta_distance, "delta_distance", "m");
}

Planner::Planner(const Config& config) : config(config) {}

std::vector<Pose> Planner::interpolateWaypoints(const std::vector<Pose>& input,
                                                size_t min_points) const {
  if (input.size() >= min_points) {
    return input;
  }

  std::vector<Pose> result = input;

  // Interpolate until we have enough points
  size_t idx = 0;
  while (result.size() < min_points) {
    const Pose& a = result[idx];
    const Pose& b = result[idx + 1];

    Pose mid = Pose::Identity();
    mid.translate(0.5 * (a.translation() + b.translation()));

    result.insert(result.begin() + idx + 1, mid);
  }

  return result;
}

std::vector<Pose> Planner::getWaypoints(const SearchOutput& search_output) const {
  std::vector<Pose> waypoints = search_output.nav_poses;
  // waypoints.insert(
  //     waypoints.end(), search_output.nav_poses.begin(),
  //     search_output.nav_poses.end());
  // if (!search_output.start_pose.translation().isApprox(
  //         search_output.nav_poses.front().translation())) {
  //   waypoints.insert(waypoints.begin(), search_output.start_pose);
  // }
  // if (config.add_goal && !search_output.goal_pose.translation().isApprox(
  //                            search_output.nav_poses.back().translation())) {
  //   waypoints.push_back(search_output.goal_pose);
  // }
  return waypoints;
}

void Planner::finalizePath(std::vector<Pose>& path,
                           const SearchOutput& search_output,
                           const std::vector<Pose>& extra_poses_start,
                           const std::vector<Pose>& extra_poses_end) const {
  std::vector<Eigen::Quaterniond> quats;

  if (extra_poses_start.size() > 0) {
    math_utils::slerpIsometries<double>(
        search_output.start_pose, path.front(), extra_poses_start.size(), quats);
    for (size_t i = 0; i < extra_poses_start.size(); ++i) {
      Pose pose = extra_poses_start[i];
      pose.linear() = quats[i].toRotationMatrix();
      path.insert(path.begin(), pose);
    }
  }
  if (extra_poses_end.size() > 0) {
    quats.clear();
    math_utils::slerpIsometries<double>(
        path.back(), search_output.goal_pose, extra_poses_end.size(), quats);
    for (size_t i = 0; i < extra_poses_end.size(); ++i) {
      Pose pose = extra_poses_end[i];
      pose.linear() = quats[i].toRotationMatrix();
      path.push_back(pose);
    }
  }
  if (!search_output.start_pose.translation().isApprox(path.front().translation())) {
    path.insert(path.begin(), search_output.start_pose);
  }
  if (config.add_goal &&
      !search_output.goal_pose.translation().isApprox(path.back().translation())) {
    path.push_back(search_output.goal_pose);
  }

  if (config.append_goal_yaw_rotation && !path.empty()) {
    static constexpr size_t kNumYawRotationSegments = 36;
    const Pose goal_pose = path.back();
    for (size_t i = 1; i < kNumYawRotationSegments; ++i) {
      const double yaw_offset = 2.0 * M_PI * static_cast<double>(i) /
                                static_cast<double>(kNumYawRotationSegments);
      Pose pose = goal_pose;
      pose.linear() =
          goal_pose.linear() * math_utils::yawToRotationMatrix<double>(yaw_offset);
      path.push_back(pose);
    }
    path.push_back(goal_pose);
  }
}

}  // namespace hvlm_planner
