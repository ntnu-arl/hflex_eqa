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
#include <config_utilities/config_utilities.h>
#include <config_utilities/virtual_config.h>

#include "hvlm_planner/common/types.h"
#include "hvlm_planner/low_level/semantic_search.h"

namespace hvlm_planner {

class Planner {
 public:
  using Ptr = std::unique_ptr<Planner>;

  struct Config {
    bool interpolate_yaw_waypoints = true;
    bool append_goal_yaw_rotation = false;
    bool add_goal = false;
    double delta_distance = 0.1;
  } const config;

  explicit Planner(const Config& config);

  virtual ~Planner() = default;

  virtual bool plan(const SearchOutput& search_output,
                    const input::OccupancyGrid::Ptr& occupancy_grid,
                    std::vector<Pose>& path) const = 0;

 protected:
  std::vector<Pose> interpolateWaypoints(const std::vector<Pose>& input,
                                         size_t min_points) const;

  std::vector<Pose> getWaypoints(const SearchOutput& search_output) const;

  void finalizePath(std::vector<Pose>& path,
                    const SearchOutput& search_output,
                    const std::vector<Pose>& extra_poses_start,
                    const std::vector<Pose>& extra_poses_end) const;
};

void declare_config(Planner::Config& config);

}  // namespace hvlm_planner
