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

#include <spark_dsg/bounding_box.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <type_traits>
#include <vector>

namespace hvlm_planner {
namespace math_utils {

template <typename DerivedA, typename DerivedB>
auto cosineSimilarity(const Eigen::MatrixBase<DerivedA>& a,
                      const Eigen::MatrixBase<DerivedB>& b) {
  static_assert(std::is_arithmetic_v<typename DerivedA::Scalar>,
                "Eigen vector must have an arithmetic scalar type");

  using Scalar =
      std::common_type_t<typename DerivedA::Scalar, typename DerivedB::Scalar>;

  // Safety check
  assert(a.size() == b.size());

  Scalar dot = a.template cast<Scalar>().dot(b.template cast<Scalar>());
  Scalar norm_a = a.template cast<Scalar>().norm();
  Scalar norm_b = b.template cast<Scalar>().norm();

  if (norm_a == Scalar(0) || norm_b == Scalar(0)) return Scalar(0);

  return dot / (norm_a * norm_b + Scalar(1e-6));
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> yawToRotationMatrix(Scalar yaw) {
  Scalar c = std::cos(yaw);
  Scalar s = std::sin(yaw);

  Eigen::Matrix<Scalar, 3, 3> R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;

  return R;
}

template <typename T>
void slerpIsometries(const Eigen::Transform<T, 3, Eigen::Isometry>& start,
                     const Eigen::Transform<T, 3, Eigen::Isometry>& end,
                     const size_t& num_points,
                     std::vector<Eigen::Quaternion<T>>& out_quats) {
  out_quats.clear();
  if (num_points == 0) {
    return;
  }
  if (num_points == 1) {
    out_quats.push_back(Eigen::Quaternion<T>(end.rotation()));
    return;
  }

  Eigen::Quaternion<T> q0(start.rotation());
  Eigen::Quaternion<T> q1(end.rotation());

  // Ensure shortest path
  if (q0.dot(q1) < 0) q1.coeffs() = -q1.coeffs();

  for (size_t i = 0; i < num_points; ++i) {
    T t = static_cast<T>(i) / static_cast<T>(num_points - 1);
    Eigen::Quaternion<T> q_interp = q0.slerp(t, q1);
    out_quats.push_back(q_interp);
  }
}

template <typename Scalar>
struct PoseDifference {
  Scalar translation;
  Eigen::Matrix<Scalar, 3, 1> angles;
};

template <typename Scalar>
PoseDifference<Scalar> poseDifference(
    const Eigen::Transform<Scalar, 3, Eigen::Isometry>& from,
    const Eigen::Transform<Scalar, 3, Eigen::Isometry>& to,
    bool in_degrees = false) {
  static_assert(std::is_floating_point_v<Scalar>,
                "poseDifference requires a floating-point Scalar");

  PoseDifference<Scalar> result;
  result.translation = (to.translation() - from.translation()).norm();

  Eigen::Matrix<Scalar, 3, 3> R_rel = from.rotation().transpose() * to.rotation();

  // Extract roll, pitch, yaw
  Eigen::Matrix<Scalar, 3, 1> rpy = R_rel.eulerAngles(0, 1, 2);

  // Normalize each angle to [-pi, pi]
  for (int i = 0; i < 3; ++i) {
    rpy(i) = std::atan2(std::sin(rpy(i)), std::cos(rpy(i)));
  }

  // Convert to degrees if requested
  if (in_degrees) {
    rpy *= Scalar(180) / Scalar(M_PI);
  }

  result.angles = rpy;
  return result;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> getRPY(
    const Eigen::Transform<Scalar, 3, Eigen::Isometry>& T, bool degrees = false) {
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  Vector3 rpy = T.rotation().eulerAngles(2, 1, 0).reverse();

  if (degrees) {
    rpy *= Scalar(180.0) / Scalar(M_PI);
  }
  return rpy;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> yawToUnitVector(Scalar yaw, bool degrees = false) {
  if (degrees) {
    constexpr Scalar deg2rad = Scalar(M_PI) / Scalar(180.0);
    yaw *= deg2rad;
  }

  return Eigen::Matrix<Scalar, 2, 1>(std::cos(yaw), std::sin(yaw));
}

inline float computeViewingRadiusFromBoundingBox(const spark_dsg::BoundingBox& bbox,
                                                 float fx,
                                                 float fy,
                                                 int image_width,
                                                 int image_height) {
  if (!bbox.isValid()) {
    return -1.0f;
  }

  const float fov_x = 2.0f * std::atan(image_width / (2.0 * (fx)));
  const float fov_y = 2.0f * std::atan(image_height / (2.0 * (fy)));
  const float tan_half_fov_x = std::tan(fov_x * 0.5);
  const float tan_half_fov_y = std::tan(fov_y * 0.5);

  const auto corners = bbox.corners();
  float max_xy_radius = 0.0;
  float max_z_radius = 0.0;

  for (const auto& corner : corners) {
    Eigen::Vector3f diff = corner - bbox.world_P_center;

    // Horizontal radial extent (XY plane)
    float xy_norm = std::sqrt(diff.x() * diff.x() + diff.y() * diff.y());
    max_xy_radius = std::max(max_xy_radius, xy_norm);

    // Vertical extent
    max_z_radius = std::max(max_z_radius, std::abs(diff.z()));
  }

  const float r_x = max_xy_radius / tan_half_fov_x;
  const float r_y = max_z_radius / tan_half_fov_y;
  return std::max(r_x, r_y);
}

}  // namespace math_utils
}  // namespace hvlm_planner
