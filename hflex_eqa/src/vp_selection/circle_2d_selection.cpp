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
#include "hflex_eqa/vp_selection/circle_2d_selection.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>
#include <open3d/t/geometry/RaycastingScene.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "hflex_eqa/utils/dsg.h"
#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {
namespace vp_selection {

namespace {
static const auto registration =
    config::RegistrationWithConfig<VPSelectionInterface,
                                   Circle2DSelection,
                                   Circle2DSelection::Config>("circle_2d");
}  // namespace

void declare_config(Circle2DSelection::Config& config) {
  using namespace config;
  base<VPSelectionInterface::Config>(config);
  name("Circle2DSelectionConfig");
  field(config.radius_margin, "radius_margin");
  field(config.num_radius_samples, "num_radius_samples");
  field(config.num_angle_samples, "num_angle_samples");
  field(config.top_n, "top_n");
  field(config.min_angular_separation_degrees,
        "min_angular_separation_degrees",
        "degrees");
  field(config.coarse_to_fine_enabled, "coarse_to_fine_enabled");
  field(config.coarse_to_fine_min_samples, "coarse_to_fine_min_samples");
  field(config.coarse_stride, "coarse_stride");
  field(config.coarse_seed_count, "coarse_seed_count");
  field(config.coarse_refine_window, "coarse_refine_window");
  field(config.visibility_occlusion_weight, "visibility_occlusion_weight");
  field(config.visibility_framing_weight, "visibility_framing_weight");
  field(config.visualize_score, "visualize_score");
}

Circle2DSelection::Circle2DSelection(const Config& config)
    : VPSelectionInterface(config), config(config::checkValid(config)) {
  A_ << 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0;
  K_ << config.camera_config.fx, 0.0, config.camera_config.cx, 0.0,
      config.camera_config.fy, config.camera_config.cy, 0.0, 0.0, 1.0;
}

void Circle2DSelection::initializeScene(const spark_dsg::Mesh::Ptr& mesh,
                                        const std::list<size_t>& object_mesh_indices) {
  mesh_ = dsg_utils::convertToOpen3DMesh(mesh);
  mesh_->RemoveVerticesByIndex(
      std::vector<size_t>(object_mesh_indices.begin(), object_mesh_indices.end()));

  mesh_->RemoveUnreferencedVertices();
  mesh_->RemoveDegenerateTriangles();
  mesh_->RemoveDuplicatedTriangles();
  mesh_->RemoveDuplicatedVertices();
  mesh_->ComputeTriangleNormals();
  mesh_->ComputeVertexNormals();

  if (!renderer_) {
    renderer_ = std::make_unique<open3d::visualization::rendering::OffscreenRenderer>(
        config.camera_config.width, config.camera_config.height);
  }

  auto* scene = renderer_->GetScene();

  // Remove previously-added geometry before re-adding.
  scene->RemoveGeometry("mesh");

  open3d::visualization::rendering::MaterialRecord mat;
  mat.shader = "defaultLit";

  scene->AddGeometry("mesh", mesh_.get(), mat);
}

std::vector<Viewpoint> Circle2DSelection::computeBestViewpoints(
    const std::vector<Eigen::Vector3f>& object_cloud,
    const spark_dsg::Mesh::Ptr& mesh,
    const input::OccupancyGrid::Ptr& occupancy_grid,
    const float& radius,
    const Eigen::Vector3f& centroid,
    const std::list<size_t>& object_mesh_indices,
    const float& height) {
  // Initialize the raycasting scene with the mesh and remove object mesh indices
  initializeScene(mesh, object_mesh_indices);

  // Sample viewpoints on a circle around the object
  const float max_radius = radius * (1.0f + std::max(0.0f, config.radius_margin));
  std::vector<float> radius_candidates;
  if (max_radius <= radius) {
    radius_candidates.push_back(radius);
  } else {
    radius_candidates.reserve(config.num_radius_samples);
    for (size_t i = 0; i < config.num_radius_samples; ++i) {
      const float t =
          static_cast<float>(i) / static_cast<float>(config.num_radius_samples - 1);
      radius_candidates.push_back(radius + t * (max_radius - radius));
    }
  }
  size_t num_samples = std::max(static_cast<size_t>(1), config.num_angle_samples);
  float theta_scale = 2.0f * M_PI / num_samples;

  std::vector<float> cos_vals(num_samples);
  std::vector<float> sin_vals(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    cos_vals[i] = std::cos(theta_scale * static_cast<float>(i));
    sin_vals[i] = std::sin(theta_scale * static_cast<float>(i));
  }

  if (object_cloud.empty()) {
    return {};
  }

  // Evaluate angle and save candidates wrapper
  std::unordered_map<size_t, Viewpoint> candidates_by_id;
  auto eval_angle_id = [&](size_t angle_id) -> Viewpoint {
    const size_t wrapped_id = ((angle_id % num_samples) + num_samples) % num_samples;
    const auto it = candidates_by_id.find(wrapped_id);
    if (it != candidates_by_id.end()) {
      return it->second;
    }
    const auto candidate = evaluateAngle(centroid,
                                         object_cloud,
                                         occupancy_grid,
                                         radius_candidates,
                                         height,
                                         cos_vals[wrapped_id],
                                         sin_vals[wrapped_id],
                                         wrapped_id);
    candidates_by_id[wrapped_id] = candidate;
    return candidate;
  };

  const bool use_coarse_to_fine =
      config.coarse_to_fine_enabled &&
      num_samples >=
          std::max(static_cast<size_t>(1), config.coarse_to_fine_min_samples);

  if (!use_coarse_to_fine) {
    for (size_t angle_id = 0; angle_id < num_samples; ++angle_id) {
      eval_angle_id(angle_id);
    }
  } else {
    size_t coarse_stride = config.coarse_stride;
    if (coarse_stride <= 0) {
      coarse_stride =
          std::max(static_cast<size_t>(2), static_cast<size_t>(std::sqrt(num_samples)));
    }
    coarse_stride =
        std::max(static_cast<size_t>(1), std::min(coarse_stride, num_samples));

    std::vector<size_t> coarse_ids;
    for (size_t id = 0; id < num_samples; id += coarse_stride) {
      coarse_ids.push_back(id);
      eval_angle_id(id);
    }

    std::sort(coarse_ids.begin(), coarse_ids.end(), [&](size_t a, size_t b) {
      return candidates_by_id[a].score > candidates_by_id[b].score;
    });

    size_t seed_count = config.coarse_seed_count;
    if (seed_count <= 0) {
      seed_count = std::max(config.top_n * 2, static_cast<size_t>(2));
    }
    seed_count = std::min(seed_count, coarse_ids.size());

    size_t window = config.coarse_refine_window;
    if (window <= 0) {
      window = std::max(static_cast<size_t>(1), coarse_stride / 2);
    }

    for (size_t i = 0; i < seed_count; ++i) {
      const size_t seed_id = coarse_ids[i];
      for (int delta = -window; delta <= static_cast<int>(window); ++delta) {
        eval_angle_id(seed_id + delta);
      }
    }
  }

  std::vector<Viewpoint> candidates;
  candidates.reserve(candidates_by_id.size());
  for (const auto& kv : candidates_by_id) {
    candidates.push_back(kv.second);
  }
  std::sort(candidates.begin(),
            candidates.end(),
            [](const Viewpoint& a, const Viewpoint& b) { return a.score > b.score; });

  std::vector<Viewpoint> selected_viewpoints;
  std::vector<float> selected_yaws;
  const float min_sep = config.min_angular_separation_degrees * M_PI / 180.0;

  for (const auto& candidate : candidates) {
    bool valid = true;
    const float cand_yaw =
        std::atan2(static_cast<float>(candidate.position.linear()(1, 0)),
                   static_cast<float>(candidate.position.linear()(0, 0)));
    for (const float selected_yaw : selected_yaws) {
      float d = std::abs(cand_yaw - selected_yaw);
      if (d > M_PI) {
        d = 2.0 * M_PI - d;
      }
      if (d < min_sep) {
        valid = false;
        break;
      }
    }

    if (valid) {
      selected_viewpoints.push_back(candidate);
      selected_yaws.push_back(cand_yaw);
      if (selected_viewpoints.size() >= config.top_n) {
        break;
      }
    }
  }

  return selected_viewpoints;
}

Viewpoint Circle2DSelection::evaluateAngle(
    const Eigen::Vector3f& centroid,
    const std::vector<Eigen::Vector3f>& object_pts,
    const input::OccupancyGrid::Ptr& occupancy_grid,
    const std::vector<float>& radius_candidates,
    const float& height,
    const float& cos_theta,
    const float& sin_theta,
    const size_t angle_index) const {
  float yaw = std::atan2(-sin_theta, -cos_theta);

  Eigen::Matrix3f R_cw = math_utils::yawToRotationMatrix(yaw);
  Eigen::Matrix3f R_wc = R_cw.transpose();

  std::vector<float> scores;
  std::vector<Eigen::Vector3d> cam_positions;
  scores.reserve(radius_candidates.size());
  cam_positions.reserve(radius_candidates.size());

  for (const float radius : radius_candidates) {
    Pose cam_pose = Pose::Identity();
    cam_pose.translation() =
        centroid.cast<double>() +
        static_cast<double>(radius) * Eigen::Vector3d(cos_theta, sin_theta, 0.0);
    cam_pose.translation().z() = static_cast<double>(height);
    cam_pose.linear() = R_wc.cast<double>();

    scores.push_back(
        scoreView(cam_pose, object_pts, occupancy_grid, angle_index, radius));
    cam_positions.push_back(cam_pose.translation());
  }

  if (scores.empty()) {
    Viewpoint empty_vp;
    empty_vp.position = Pose::Identity();
    empty_vp.position.translation() = centroid.cast<double>();
    empty_vp.position.translation().z() = static_cast<double>(height);
    empty_vp.position.linear() = R_cw.cast<double>();
    empty_vp.score = 0.0f;
    return empty_vp;
  }

  const auto best_it = std::max_element(scores.begin(), scores.end());
  const size_t best_idx = std::distance(scores.begin(), best_it);

  std::vector<float> sorted_scores = scores;
  std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<float>());
  const size_t top_k = std::min(static_cast<size_t>(2), sorted_scores.size());
  float robust_score = 0.0f;
  for (size_t i = 0; i < top_k; ++i) {
    robust_score += sorted_scores[i];
  }
  robust_score /= static_cast<float>(top_k);

  Viewpoint vp;
  vp.position.translation() = cam_positions[best_idx];
  vp.position.linear() = R_cw.cast<double>();
  vp.score = robust_score;
  return vp;
}

float Circle2DSelection::scoreView(const Pose& cam_pose_w,
                                   const std::vector<Eigen::Vector3f>& object_cloud,
                                   const input::OccupancyGrid::Ptr& occupancy_grid,
                                   const size_t angle_index,
                                   const float radius) const {
  // Check that point is in free space in occupancy grid, otherwise return 0 score
  if (occupancy_grid) {
    if (!occupancy_grid->isFree(cam_pose_w.translation().head<2>().cast<float>())) {
      return 0.0f;
    }
  }
  const size_t H = config.camera_config.height;
  const size_t W = config.camera_config.width;

  const Eigen::MatrixXd R_o3d = A_ * cam_pose_w.linear();
  const Eigen::Vector3d t_o3d = -R_o3d * cam_pose_w.translation();

  Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
  extrinsic.block<3, 3>(0, 0) = R_o3d;
  extrinsic.block<3, 1>(0, 3) = t_o3d;
  extrinsic.block<1, 3>(3, 0).setZero();
  extrinsic(3, 3) = 1.0;

  renderer_->SetupCamera(K_, extrinsic);

  const auto depth_image = renderer_->RenderToDepthImage(true);
  if (!depth_image) {
    return 0.0;
  }

  int num_in_front = 0;
  int num_in_image = 0;
  int visible_count = 0;
  std::vector<int> us;
  std::vector<int> vs;

  for (const auto& pt_w : object_cloud) {
    const Eigen::Vector3f pt_c = R_o3d.cast<float>() * pt_w + t_o3d.cast<float>();
    if (pt_c.z() <= 0.0) {
      continue;
    }
    ++num_in_front;

    const float u_f =
        config.camera_config.fx * pt_c.x() / pt_c.z() + config.camera_config.cx;
    const float v_f =
        config.camera_config.fy * pt_c.y() / pt_c.z() + config.camera_config.cy;
    if (u_f < 0.0 || u_f >= static_cast<float>(W) || v_f < 0.0 ||
        v_f >= static_cast<float>(H)) {
      continue;
    }
    ++num_in_image;

    const int u =
        std::clamp(static_cast<int>(std::lround(u_f)), 0, static_cast<int>(W) - 1);
    const int v =
        std::clamp(static_cast<int>(std::lround(v_f)), 0, static_cast<int>(H) - 1);

    float depth_scene_local = std::numeric_limits<float>::infinity();
    for (int du = -1; du <= 1; ++du) {
      for (int dv = -1; dv <= 1; ++dv) {
        const int uu = std::clamp(u + du, 0, static_cast<int>(W) - 1);
        const int vv = std::clamp(v + dv, 0, static_cast<int>(H) - 1);
        float depth_scene = *depth_image->PointerAt<float>(uu, vv);
        if (depth_scene == 0.0f) {
          depth_scene = std::numeric_limits<float>::infinity();
        }
        depth_scene_local = std::min(depth_scene_local, depth_scene);
      }
    }

    constexpr float kDepthEpsilon = 0.02f;
    if (pt_c.z() <= depth_scene_local + kDepthEpsilon) {
      ++visible_count;
    }

    if (config.visualize_score) {
      us.push_back(u);
      vs.push_back(v);
    }
  }

  if (config.visualize_score) {
    const float occlusion_ratio =
        num_in_front > 0
            ? static_cast<float>(visible_count) / static_cast<float>(num_in_front)
            : 0.0f;
    const float framing_ratio = num_in_front > 0 ? static_cast<float>(num_in_image) /
                                                       static_cast<float>(num_in_front)
                                                 : 0.0f;
    const float weights_sum = std::max(
        1e-6f, config.visibility_occlusion_weight + config.visibility_framing_weight);
    const float visibility_ratio =
        (config.visibility_occlusion_weight * occlusion_ratio +
         config.visibility_framing_weight * framing_ratio) /
        weights_sum;

    LOG(INFO) << "Angle " << angle_index << " visibility ratio: " << visibility_ratio
              << " visible pixels: " << visible_count << std::endl;

    cv::Mat depth_vis(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t y = 0; y < H; ++y) {
      for (size_t x = 0; x < W; ++x) {
        float d = *depth_image->PointerAt<float>(x, y);
        if (d > 0.0f && std::isfinite(d)) {
          depth_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        }
      }
    }

    for (size_t i = 0; i < us.size(); ++i) {
      for (int du = -2; du <= 2; ++du) {
        for (int dv = -2; dv <= 2; ++dv) {
          const int uu = std::clamp(us[i] + du, 0, static_cast<int>(W) - 1);
          const int vv = std::clamp(vs[i] + dv, 0, static_cast<int>(H) - 1);
          depth_vis.at<cv::Vec3b>(vv, uu) = cv::Vec3b(0, 255, 0);
        }
      }
    }

    const std::string out = "/developer/ros2_hydra_ws/scene_depth_" +
                            std::to_string(angle_index) + "_" + std::to_string(radius) +
                            ".png";
    cv::imwrite(out, depth_vis);
  }

  if (num_in_front <= 0) {
    return 0.0f;
  }

  const float occlusion_ratio =
      static_cast<float>(visible_count) / static_cast<float>(num_in_front);
  const float framing_ratio =
      static_cast<float>(num_in_image) / static_cast<float>(num_in_front);
  const float weights_sum = std::max(
      1e-6f, config.visibility_occlusion_weight + config.visibility_framing_weight);

  return (config.visibility_occlusion_weight * occlusion_ratio +
          config.visibility_framing_weight * framing_ratio) /
         weights_sum;
}

}  // namespace vp_selection
}  // namespace hflex_eqa
