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

#include <Eigen/Dense>
#include <unordered_set>

namespace hvlm_planner {

using Label = uint32_t;
using FeatureVector = Eigen::VectorXf;
using Pose = Eigen::Isometry3d;
using Translation = Eigen::Vector3d;
using Rotation = Eigen::Quaterniond;

struct TranslationHash {
  std::size_t operator()(const Pose& p) const noexcept;
};

struct TranslationEqual {
  bool operator()(const Pose& a, const Pose& b) const noexcept;
};

// enum class Mode { AUTONOMOUS, TRIGGERED };
using FrontierBlackList = std::unordered_set<Pose, TranslationHash, TranslationEqual>;

namespace input {

using Point2f = Eigen::Vector2f;
using GridIndex = std::pair<int, int>;

enum class CellState : int8_t {
  Unknown = -1,
  ObservedFree = 0,
  ObservedOccupied = 100,
  OutOfBounds = 50
};

struct GridIndexHash {
  size_t operator()(const GridIndex& p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};
}  // namespace input

}  // namespace hvlm_planner
