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
#include "hvlm_planner/similarity/simple_cosine_similarity.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SimilarityInterface,
                                   SimpleCosineSimilarity,
                                   SimpleCosineSimilarity::Config>(
        "simple_cosine_similarity");
}  // namespace

void declare_config(SimpleCosineSimilarity::Config&) {
  using namespace config;
  name("SimpleCosineSimilarityConfig");
}

SimpleCosineSimilarity::SimpleCosineSimilarity(const Config& config)
    : config(config::checkValid(config)) {}

float SimpleCosineSimilarity::compute(
    const FeatureVector& frontier_feature,
    const std::vector<FeatureVector>&,
    const std::vector<FeatureVector>& reference_features) const {
  if (reference_features.empty() ||
      frontier_feature.size() != reference_features[0].size()) {
    return -1.0;
  }
  return math_utils::cosineSimilarity(frontier_feature, reference_features[0]);
}
}  // namespace hvlm_planner
