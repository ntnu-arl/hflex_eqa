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
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>

#include <Eigen/Dense>
#include <memory>
#include <unsupported/Eigen/Splines>
#include <vector>

#include "hflex_eqa/common/types.h"
#include "hflex_eqa/low_level/planner.h"

namespace hflex_eqa {

class LinearInterpolatorPlanner : public Planner {
 public:
  struct Config : Planner::Config {
  } const config;

  explicit LinearInterpolatorPlanner(const Config& config);

  bool plan(const SearchOutput& search_output,
            const input::OccupancyGrid::Ptr& occupancy_grid,
            std::vector<Pose>& path) const override;

 protected:
  void generateLinearPathFromWaypoints(const std::vector<Pose>& waypoints,
                                       std::vector<Pose>& path) const;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<Planner, LinearInterpolatorPlanner, Config>(
          "LinearInterpolatorPlanner");
};

}  // namespace hflex_eqa
