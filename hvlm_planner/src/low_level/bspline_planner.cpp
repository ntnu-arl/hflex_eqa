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
#include "hvlm_planner/low_level/bspline_planner.h"

#include <config_utilities/config.h>
#include <glog/logging.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

void declare_config(BSplinePlanner::Config& config) {
  using namespace config;
  base<Planner::Config>(config);
  name("BSplinePlannerConfig");
  field(config.num_points_integration, "num_points_integration");
}

BSplinePlanner::BSplinePlanner(const Config& config)
    : Planner(config), config(config::checkValid(config)) {
  spline_ = std::make_unique<splines::BSpline>();
}

bool BSplinePlanner::plan(const SearchOutput& search_output,
                          const input::OccupancyGrid::Ptr&,
                          std::vector<Pose>& path) const {
  std::vector<Pose> all_waypoints = getWaypoints(search_output);
  if (all_waypoints.size() < 2) {
    LOG(WARNING) << "Not enough waypoints for spline planning. Returning linear path.";
    path = all_waypoints;
    finalizePath(path,
                 search_output,
                 search_output.extra_poses_start,
                 search_output.extra_poses_end);
    return true;
  }

  if (all_waypoints.size() < Degree + 1) {
    LOG(WARNING) << "Not enough waypoints for bspline spline planning. "
                    "Interpolating waypoints before fitting.";
    all_waypoints = interpolateWaypoints(all_waypoints, Degree + 1);
  }

  if (!generateSplineFromWaypoints(all_waypoints)) {
    // Set path equal to waypoints if spline generation fails to ensure we return a
    // valid path
    LOG(WARNING)
        << "Spline generation failed. Returning linear path through waypoints.";
    path = all_waypoints;
    finalizePath(path,
                 search_output,
                 search_output.extra_poses_start,
                 search_output.extra_poses_end);
    return true;
  }

  size_t num_points_segment = spline_->nodeCount() / (all_waypoints.size() - 1);
  std::vector<Eigen::Quaterniond> quats;
  if (config.interpolate_yaw_waypoints) {
    for (size_t i = 0; i < all_waypoints.size() - 1; ++i) {
      std::vector<Eigen::Quaterniond> segment_quats;
      math_utils::slerpIsometries<double>(
          all_waypoints[i], all_waypoints[i + 1], num_points_segment, segment_quats);
      quats.insert(quats.end(), segment_quats.begin(), segment_quats.end());
    }
    for (size_t j = quats.size(); j < spline_->nodeCount(); ++j) {
      quats.push_back(Eigen::Quaterniond(all_waypoints.back().linear()));
    }
  } else {
    math_utils::slerpIsometries<double>(
        all_waypoints.front(), all_waypoints.back(), spline_->nodeCount(), quats);
  }
  size_t last_added_idx = 0;
  for (size_t i = 0; i < spline_->nodeCount(); ++i) {
    double length = spline_->lengthBetweenNodes(i, last_added_idx);
    if (length < config.delta_distance) {
      continue;  // Skip points that are too close to the last added point
    }
    const splines::Vector node = spline_->node(i);
    Pose pose = Pose::Identity();
    pose.translate(Translation(node.x, node.y, node.z));
    pose.linear() = quats[i].toRotationMatrix();
    path.push_back(pose);
    last_added_idx = i;
  }

  // Linearly interpolate last point to last waypoint to ensure exact match
  const double distance_to_last =
      (all_waypoints.back().translation() - path.back().translation()).norm();
  if (distance_to_last > 1e-6) {
    size_t num_extra_points =
        static_cast<size_t>(std::ceil(distance_to_last / config.delta_distance));
    for (size_t i = 1; i <= num_extra_points; ++i) {
      double t = static_cast<double>(i) / static_cast<double>(num_extra_points);
      Translation p = (1.0 - t) * path.back().translation() +
                      t * all_waypoints.back().translation();
      Pose pose = Pose::Identity();
      pose.translate(p);
      pose.linear() = all_waypoints.back().linear();
      path.push_back(pose);
    }
  }

  finalizePath(path,
               search_output,
               search_output.extra_poses_start,
               search_output.extra_poses_end);
  return true;
}

bool BSplinePlanner::generateSplineFromWaypoints(
    const std::vector<Pose>& waypoints) const {
  spline_->clear();
  // Sum up the total distance of the path through the waypoints to determine if spline
  // can be fitted
  double total_distance = 0.0;
  for (size_t i = 1; i < waypoints.size(); ++i) {
    total_distance +=
        (waypoints[i].translation() - waypoints[i - 1].translation()).norm();
  }
  if (total_distance < config.delta_distance) {
    LOG(WARNING) << "Total distance between waypoints is too small for spline fitting.";
    return false;
  }
  spline_->setSteps(config.num_points_integration);
  for (const auto& waypoint : waypoints) {
    spline_->addWayPoint(splines::Vector(waypoint.translation().x(),
                                         waypoint.translation().y(),
                                         waypoint.translation().z()));
  }
  spline_->finalize();
  return true;
}

}  // namespace hvlm_planner
