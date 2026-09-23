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
#include "hvlm_planner/similarity/frontier_single_cosine_similarity.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SimilarityInterface,
                                   FrontierSingleCosineSimilarity,
                                   FrontierSingleCosineSimilarity::Config>(
        "frontier_single_cosine_similarity");
}  // namespace

void declare_config(FrontierSingleCosineSimilarity::Config& config) {
  using namespace config;
  name("FrontierSingleSimilarityConfig");
  enum_field(config.pooling_mode,
             "pooling_mode",
             {{FrontierSingleCosineSimilarity::PoolingMode::AVERAGE, "average"},
              {FrontierSingleCosineSimilarity::PoolingMode::MAX, "max"}});
  enum_field(
      config.weights_mode,
      "weights_mode",
      {{FrontierSingleCosineSimilarity::WeightsMode::UNIFORM, "uniform"},
       {FrontierSingleCosineSimilarity::WeightsMode::LINEAR_DECAY, "linear_decay"},
       {FrontierSingleCosineSimilarity::WeightsMode::EXPONENTIAL_DECAY,
        "exponential_decay"}});
  field(config.exponential_decay_rate, "exponential_decay_rate");
}

FrontierSingleCosineSimilarity::FrontierSingleCosineSimilarity(const Config& config)
    : config(config::checkValid(config)) {}

float FrontierSingleCosineSimilarity::compute(
    const FeatureVector& frontier_feature,
    const std::vector<FeatureVector>&,
    const std::vector<FeatureVector>& reference_features) const {
  if (reference_features.empty() ||
      frontier_feature.size() != reference_features[0].size()) {
    return -1.0;
  }
  float max_similarity = -1.0;
  float overall_similarity = 0.0;
  const auto weights = computeWeights(reference_features.size());
  for (size_t i = 0; i < reference_features.size(); ++i) {
    const auto& reference_feature = reference_features[i];
    const auto similarity =
        math_utils::cosineSimilarity(frontier_feature, reference_feature);
    switch (config.pooling_mode) {
      case PoolingMode::AVERAGE: {
        overall_similarity += weights[i] * similarity;
        break;
      }
      case PoolingMode::MAX: {
        max_similarity = std::max(max_similarity, similarity);
        overall_similarity = max_similarity;
        break;
      }
    }
  }
  return overall_similarity;
}

const std::vector<float> FrontierSingleCosineSimilarity::computeWeights(
    const size_t num_features) const {
  std::vector<float> weights;
  weights.reserve(num_features);

  if (num_features == 0) return weights;

  switch (config.weights_mode) {
    case WeightsMode::UNIFORM: {
      weights.assign(num_features, 1.0f / static_cast<float>(num_features));
      break;
    }

    case WeightsMode::LINEAR_DECAY: {
      // w_i = (N - i) / (N*(N+1)/2)
      const float N = static_cast<float>(num_features);
      const float denom = N * (N + 1.0f) * 0.5f;

      for (size_t i = 0; i < num_features; ++i) {
        weights.push_back((N - static_cast<float>(i)) / denom);
      }
      break;
    }

    case WeightsMode::EXPONENTIAL_DECAY: {
      // w_i = r^i / sum(r^k)
      float sum = 0.0f;
      float value = 1.0f;

      for (size_t i = 0; i < num_features; ++i) {
        weights.push_back(value);
        sum += value;
        value *= config.exponential_decay_rate;
      }
      // normalize
      for (auto& w : weights) {
        w /= sum;
      }

      break;
    }
  }
  return weights;
}

}  // namespace hvlm_planner
