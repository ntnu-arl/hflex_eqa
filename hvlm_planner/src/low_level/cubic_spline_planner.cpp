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
#include "hvlm_planner/low_level/cubic_spline_planner.h"

#include <config_utilities/config.h>
#include <glog/logging.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {
void declare_config(CubicSplinePlanner::Config& config) {
  using namespace config;
  base<Planner::Config>(config);
  name("CubicSplinePlannerConfig");
  field(config.num_points_integration, "num_points_integration");
}

CubicSplinePlanner::CubicSplinePlanner(const Config& config)
    : Planner(config), config(config::checkValid(config)) {
  spline_ = std::make_unique<Spline3d>();
}

bool CubicSplinePlanner::plan(const SearchOutput& search_output,
                              const input::OccupancyGrid::Ptr&,
                              std::vector<Pose>& path) const {
  std::vector<Pose> all_waypoints = getWaypoints(search_output);

  if (all_waypoints.size() < Degree + 1) {
    LOG(WARNING) << "Not enough waypoints for cubic spline planning. "
                    "Interpolating waypoints before fitting.";
    all_waypoints = interpolateWaypoints(all_waypoints, Degree + 1);
  }

  for (size_t i = 0; i < all_waypoints.size(); ++i) {
    LOG(INFO) << "Waypoint " << i << ": " << all_waypoints[i].translation().transpose();
  }

  generateSplineFromWaypoints(all_waypoints);

  const double spline_length = computeSplineLength();
  VLOG(1) << "Computed spline length: " << spline_length;
  if (spline_length < 1e-6) {
    LOG(WARNING) << "Spline length is too small. Returning straight line path.";
    path = all_waypoints;
    finalizePath(path,
                 search_output,
                 search_output.extra_poses_start,
                 search_output.extra_poses_end);
    return true;
  }

  // Interpolate orientations using SLERP
  size_t path_points =
      static_cast<size_t>(std::ceil(spline_length / config.delta_distance));
  VLOG(2) << "Number of path points: " << path_points;
  std::vector<Eigen::Quaterniond> quats;
  if (!config.interpolate_yaw_waypoints) {
    math_utils::slerpIsometries<double>(
        all_waypoints.front(), all_waypoints.back(), path_points, quats);
  }

  for (double t = 0.0; t <= 1.0; t += config.delta_distance / spline_length) {
    if (t > 1.0) {
      t = 1.0;  // Ensure we include the last point
    }
    Translation point = (*spline_)(t);
    Pose pose = Pose::Identity();
    pose.translate(point);

    if (config.interpolate_yaw_waypoints) {
      if (t == 0.0) {
        pose.linear() = search_output.start_pose.linear();
      } else if (t == 1.0) {
        pose.linear() = search_output.goal_pose.linear();
      } else {
        const auto point_difference = point - path[path.size() - 1].translation();
        double target_yaw = std::atan2(point_difference.y(), point_difference.x());
        pose.linear() = math_utils::yawToRotationMatrix<double>(target_yaw);
      }
    } else {
      pose.linear() = quats[path.size()].toRotationMatrix();
    }
    path.push_back(pose);
  }

  finalizePath(path,
               search_output,
               search_output.extra_poses_start,
               search_output.extra_poses_end);
  return true;
}

void CubicSplinePlanner::generateSplineFromWaypoints(
    const std::vector<Pose>& waypoints) const {
  Eigen::Matrix<double, Dimension, Eigen::Dynamic> controlPoints(Dimension,
                                                                 waypoints.size());
  for (size_t i = 0; i < waypoints.size(); ++i) {
    controlPoints.col(i) = waypoints[i].translation();
  }
  *spline_ = Eigen::SplineFitting<Spline3d>::Interpolate(controlPoints, Degree);
}

double CubicSplinePlanner::computeSplineLength() const {
  if (!spline_) {
    return -1.0;
  }
  double length = 0.0;
  Translation prev_point = (*spline_)(0.0);
  for (size_t i = 1; i <= config.num_points_integration; ++i) {
    double t =
        static_cast<double>(i) / static_cast<double>(config.num_points_integration);
    Translation point = (*spline_)(t);
    length += (point - prev_point).norm();
    prev_point = point;
  }
  return length;
}

}  // namespace hvlm_planner
