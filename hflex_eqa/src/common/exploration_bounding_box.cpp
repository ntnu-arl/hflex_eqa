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
#include "hflex_eqa/common/exploration_bounding_box.h"

#include <config_utilities/config.h>

#include <cstddef>
#include <stdexcept>

namespace hflex_eqa {

void declare_config(ExplorationBoundingBoxConfig& config) {
  using namespace config;
  name("ExplorationBoundingBoxConfig");
  field(config.enabled, "enabled");
  field(config.min, "min");
  field(config.max, "max");
}

ExplorationBoundingBox& ExplorationBoundingBox::instance() {
  static ExplorationBoundingBox instance;
  return instance;
}

void ExplorationBoundingBox::setConfig(const ExplorationBoundingBoxConfig& config) {
  if (config.enabled && (config.min.size() != 3 || config.max.size() != 3)) {
    throw std::runtime_error(
        "ExplorationBoundingBox: min and max must each contain 3 values.");
  }
  if (config.enabled) {
    for (size_t i = 0; i < 3; ++i) {
      if (config.min.at(i) > config.max.at(i)) {
        throw std::runtime_error(
            "ExplorationBoundingBox: min values must not exceed max values.");
      }
    }
  }
  config_ = config;
}

const ExplorationBoundingBoxConfig& ExplorationBoundingBox::config() const {
  return config_;
}

bool ExplorationBoundingBox::contains(const Eigen::Vector3d& position) const {
  if (!config_.enabled) {
    return true;
  }

  for (int i = 0; i < 3; ++i) {
    if (position(i) < config_.min.at(i) || position(i) > config_.max.at(i)) {
      return false;
    }
  }
  return true;
}

}  // namespace hflex_eqa
