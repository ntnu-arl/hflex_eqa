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
#include "hvlm_planner/low_level/linear_interpolator_planner.h"

#include <config_utilities/config.h>
#include <glog/logging.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

void declare_config(LinearInterpolatorPlanner::Config& config) {
  using namespace config;
  base<Planner::Config>(config);
  name("LinearInterpolatorPlanner::Config");
}

LinearInterpolatorPlanner::LinearInterpolatorPlanner(const Config& config)
    : Planner(config), config(config::checkValid(config)) {}

bool LinearInterpolatorPlanner::plan(const SearchOutput& search_output,
                                     const input::OccupancyGrid::Ptr&,
                                     std::vector<Pose>& path) const {
  std::vector<Pose> all_waypoints = getWaypoints(search_output);

  generateLinearPathFromWaypoints(all_waypoints, path);
  finalizePath(path,
               search_output,
               search_output.extra_poses_start,
               search_output.extra_poses_end);
  return true;
}

void LinearInterpolatorPlanner::generateLinearPathFromWaypoints(
    const std::vector<Pose>& waypoints, std::vector<Pose>& path) const {
  path.clear();

  path.push_back(waypoints.front());  // Ensure starting point is included
  for (size_t i = 1; i < waypoints.size(); ++i) {
    size_t num_points_segment = static_cast<size_t>(
        std::ceil((waypoints[i].translation() - waypoints[i - 1].translation()).norm() /
                  config.delta_distance));
    if (num_points_segment < 2) {
      path.push_back(waypoints[i]);
      continue;
    }
    std::vector<Eigen::Quaterniond> quats;
    if (config.interpolate_yaw_waypoints) {
      math_utils::slerpIsometries<double>(
          waypoints[i - 1], waypoints[i], num_points_segment, quats);
    }
    for (size_t j = 1; j <= num_points_segment; ++j) {
      double t = static_cast<double>(j) / static_cast<double>(num_points_segment);
      Translation p =
          (1.0 - t) * waypoints[i - 1].translation() + t * waypoints[i].translation();

      Pose pose = Pose::Identity();
      pose.translate(p);
      if (config.interpolate_yaw_waypoints) {
        pose.linear() = quats[j - 1].toRotationMatrix();
      }
      path.push_back(pose);
    }
  }

  if (!config.interpolate_yaw_waypoints) {
    std::vector<Eigen::Quaterniond> quats;
    math_utils::slerpIsometries<double>(
        waypoints.front(), waypoints.back(), path.size(), quats);
    for (size_t i = 0; i < path.size(); ++i) {
      path[i].linear() = quats[i].toRotationMatrix();
    }
  }
}

}  // namespace hvlm_planner
