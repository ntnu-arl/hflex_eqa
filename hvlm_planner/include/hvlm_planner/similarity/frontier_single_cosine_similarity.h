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

#include "hvlm_planner/similarity/similarity_interface.h"

namespace hvlm_planner {
class FrontierSingleCosineSimilarity : public SimilarityInterface {
 public:
  enum class PoolingMode { AVERAGE, MAX };
  enum class WeightsMode { UNIFORM, LINEAR_DECAY, EXPONENTIAL_DECAY };
  struct Config {
    PoolingMode pooling_mode = PoolingMode::AVERAGE;
    WeightsMode weights_mode = WeightsMode::UNIFORM;
    float exponential_decay_rate =
        0.5f;  // Only used if weights_mode is EXPONENTIAL_DECAY, should be in (0, 1)
  } const config;

  explicit FrontierSingleCosineSimilarity(const Config& config);

  float compute(const FeatureVector& frontier_feature,
                const std::vector<FeatureVector>& frontier_features,
                const std::vector<FeatureVector>& reference_features) const override;

 protected:
  const std::vector<float> computeWeights(const size_t num_features) const;
};

void declare_config(FrontierSingleCosineSimilarity::Config& config);
}  // namespace hvlm_planner
