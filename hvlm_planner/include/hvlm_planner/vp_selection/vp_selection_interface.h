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

#include <spark_dsg/mesh.h>

#include <Eigen/Dense>
#include <list>
#include <memory>
#include <vector>

#include "hvlm_planner/common/input.h"
#include "hvlm_planner/common/types.h"

namespace hvlm_planner {
namespace vp_selection {

struct Viewpoint {
  Pose position;
  float score;
};

struct CameraConfig {
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;
};

class VPSelectionInterface {
 public:
  struct Config {
    CameraConfig camera_config;
  } const config;
  using Ptr = std::unique_ptr<VPSelectionInterface>;
  explicit VPSelectionInterface(const Config& config);
  virtual ~VPSelectionInterface() = default;

  virtual std::vector<Viewpoint> computeBestViewpoints(
      const std::vector<Eigen::Vector3f>& object_cloud,
      const spark_dsg::Mesh::Ptr& mesh,
      const input::OccupancyGrid::Ptr& occupancy_grid,
      const float& radius,
      const Eigen::Vector3f& centroid,
      const std::list<size_t>& object_mesh_indices,
      const float& height = 0.0f) = 0;
};

void declare_config(CameraConfig& config);
void declare_config(VPSelectionInterface::Config& config);

}  // namespace vp_selection
}  // namespace hvlm_planner
