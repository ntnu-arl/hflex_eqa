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

#include <memory>
#include <vector>

#include "hflex_eqa/low_level/planner.h"
#include "hflex_eqa/splines/bspline.h"

namespace hflex_eqa {

class BSplinePlanner : public Planner {
 public:
  static constexpr int Degree = 3;
  struct Config : Planner::Config {
    size_t num_points_integration = 300;
  } const config;

  explicit BSplinePlanner(const Config& config);

  bool plan(const SearchOutput& search_output,
            const input::OccupancyGrid::Ptr& occupancy_grid,
            std::vector<Pose>& path) const override;

 protected:
  bool generateSplineFromWaypoints(const std::vector<Pose>& waypoints) const;
  mutable std::unique_ptr<splines::BSpline> spline_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<Planner, BSplinePlanner, Config>("BSplinePlanner");
};

}  // namespace hflex_eqa
