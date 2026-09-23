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

#include <open3d/Open3D.h>
#include <open3d/t/geometry/RaycastingScene.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "hvlm_planner/common/types.h"
#include "hvlm_planner/utils/offscreen_renderer.h"
#include "hvlm_planner/vp_selection/vp_selection_interface.h"

namespace hvlm_planner {
namespace vp_selection {

class Circle2DSelection : public VPSelectionInterface {
 public:
  struct Config : VPSelectionInterface::Config {
    float radius_margin = 0.1f;
    size_t num_radius_samples = 5;

    size_t num_angle_samples = 5;
    size_t top_n = 1;

    float min_angular_separation_degrees = 15;

    bool coarse_to_fine_enabled = true;
    size_t coarse_to_fine_min_samples = 9;
    size_t coarse_stride = 0;
    size_t coarse_seed_count = 0;
    size_t coarse_refine_window = 0;

    float visibility_occlusion_weight = 0.7f;
    float visibility_framing_weight = 0.3f;

    bool visualize_score = false;
  } const config;

  explicit Circle2DSelection(const Config& config);

  std::vector<Viewpoint> computeBestViewpoints(
      const std::vector<Eigen::Vector3f>& object_cloud,
      const spark_dsg::Mesh::Ptr& mesh,
      const input::OccupancyGrid::Ptr& occupancy_grid,
      const float& radius,
      const Eigen::Vector3f& centroid,
      const std::list<size_t>& object_mesh_indices,
      const float& height = 0.0f) override;

 private:
  void initializeScene(const spark_dsg::Mesh::Ptr& mesh,
                       const std::list<size_t>& object_mesh_indices);
  Viewpoint evaluateAngle(const Eigen::Vector3f& centroid,
                          const std::vector<Eigen::Vector3f>& object_pts,
                          const input::OccupancyGrid::Ptr& occupancy_grid,
                          const std::vector<float>& radius_candidates,
                          const float& height,
                          const float& cos_theta,
                          const float& sin_theta,
                          const size_t angle_index) const;

  float scoreView(const Pose& cam_pose_w,
                  const std::vector<Eigen::Vector3f>& object_cloud,
                  const input::OccupancyGrid::Ptr& occupancy_grid,
                  const size_t angle_index,
                  const float radius) const;

  std::unique_ptr<open3d::geometry::TriangleMesh> mesh_;
  std::unique_ptr<open3d::visualization::rendering::OffscreenRenderer> renderer_;
  Eigen::Matrix3d A_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
};

void declare_config(Circle2DSelection::Config& config);

}  // namespace vp_selection
}  // namespace hvlm_planner
