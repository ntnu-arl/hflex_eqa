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
#include "hvlm_planner/similarity/all_cosine_similarity.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SimilarityInterface,
                                   AllCosineSimilarity,
                                   AllCosineSimilarity::Config>("cosine_similarity");
}  // namespace

void declare_config(AllCosineSimilarity::Config& config) {
  using namespace config;
  name("AllCosineSimilarityConfig");
  base<FrontierSingleCosineSimilarity::Config>(config);
}

AllCosineSimilarity::AllCosineSimilarity(const Config& config)
    : FrontierSingleCosineSimilarity(config), config(config::checkValid(config)) {}

float AllCosineSimilarity::compute(
    const FeatureVector&,
    const std::vector<FeatureVector>& frontier_features,
    const std::vector<FeatureVector>& reference_features) const {
  if (frontier_features.empty() || reference_features.empty()) {
    return -1.0;
  }
  float max_similarity = -1.0;
  float overall_similarity = 0.0;
  const auto weights = computeWeights(reference_features.size());
  for (const auto& frontier_feature : frontier_features) {
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
  }
  if (config.pooling_mode == PoolingMode::AVERAGE) {
    overall_similarity /= static_cast<float>(frontier_features.size());
  }
  return overall_similarity;
}

}  // namespace hvlm_planner
