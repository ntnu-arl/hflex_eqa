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
#include "hvlm_planner/low_level/find_room_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch,
                                   FindRoomSearch,
                                   FindRoomSearch::Config>("FindRoomSearch");

struct NodeScore {
  spark_dsg::NodeId node_id;
  double transition_score = 0.0;
  double progress_score = 0.0;
  double freespace_score = 0.0;
  double score = 0.0;
};

double maxPromptSimilarity(const SimilarityInterface& similarity,
                           const spark_dsg::GlobalFrontierNodeAttributes& attrs,
                           const std::vector<FeatureVector>& prompts) {
  double max_similarity = -1.0;
  for (const auto& prompt : prompts) {
    if (prompt.size() != attrs.semantic_feature.size()) {
      continue;
    }
    const std::vector<FeatureVector> reference_features{prompt};
    max_similarity =
        std::max(max_similarity,
                 static_cast<double>(similarity.compute(
                     attrs.semantic_feature, attrs.features, reference_features)));
  }
  return max_similarity;
}

std::vector<float> normalizeProgressWeights(const std::vector<float>& scores) {
  std::vector<float> weights = scores;
  float sum = 0.0f;
  for (auto& weight : weights) {
    weight = std::max(0.0f, weight);
    sum += weight;
  }
  if (sum <= 0.0f) {
    return weights;
  }
  for (auto& weight : weights) {
    weight /= sum;
  }
  return weights;
}

double weightedRoomTransitionSimilarity(
    const SimilarityInterface& similarity,
    const spark_dsg::GlobalFrontierNodeAttributes& attrs,
    const std::vector<FeatureVector>& room_embeddings,
    const std::vector<float>& progress_weights) {
  const size_t num_terms = std::min(room_embeddings.size(), progress_weights.size());
  if (num_terms == 0) {
    return 0.0;
  }

  double weighted_similarity = 0.0;
  for (size_t i = 0; i < num_terms; ++i) {
    if (progress_weights[i] <= 0.0f) {
      continue;
    }
    const auto& room_embedding = room_embeddings[i];
    if (room_embedding.size() != attrs.semantic_feature.size()) {
      continue;
    }
    const std::vector<FeatureVector> reference_features{room_embedding};
    const auto similarity_score = static_cast<double>(
        similarity.compute(attrs.semantic_feature, attrs.features, reference_features));
    weighted_similarity += static_cast<double>(progress_weights[i]) * similarity_score;
  }

  if (!std::isfinite(weighted_similarity)) {
    return 0.0;
  }
  return weighted_similarity;
}

void setGoalFromFrontierNode(const spark_dsg::SceneGraphNode& selected_node,
                             SearchOutput& output) {
  const auto& selected_node_attrs =
      selected_node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
  output.goal_pose = Eigen::Isometry3d::Identity();
  output.goal_pose.translation() = selected_node_attrs.position;
  double yaw =
      std::atan2(selected_node_attrs.direction.y(), selected_node_attrs.direction.x());
  output.goal_pose.linear() = math_utils::yawToRotationMatrix<double>(yaw);
  output.setSelectedFrontierPose(output.goal_pose);
}

void logNodeScores(const std::vector<NodeScore>& node_scores,
                   const SearchInput::Ptr& input,
                   double alpha,
                   double beta,
                   double gamma) {
  LOG(INFO) << "FindRoomSearch frontier scores:";
  for (const auto& node_score : node_scores) {
    LOG(INFO) << "  frontier_id=" << NodeSymbol(node_score.node_id)
              << " transition_score=" << node_score.transition_score
              << " alpha_term=" << alpha * node_score.transition_score
              << " progress_score=" << node_score.progress_score
              << " beta_term=" << beta * node_score.progress_score
              << " freespace_score=" << node_score.freespace_score
              << " gamma_term=" << gamma * node_score.freespace_score
              << " total_score=" << node_score.score;
  }
  std::ostringstream labels;
  labels << "[";
  for (size_t i = 0; i < input->floorplan_room_labels.size(); ++i) {
    if (i > 0) {
      labels << ", ";
    }
    labels << input->floorplan_room_labels[i] << ": ";
    labels << (i < input->floorplan_progress_scores.size()
                   ? input->floorplan_progress_scores[i]
                   : 0.0f);
  }
  labels << "]";
  LOG(INFO) << "FindRoomSearch progress labels=" << labels.str();
}

}  // namespace

void declare_config(FindRoomSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("FindRoomSearchConfig");
  field(config.similarity, "similarity");
  field(config.alpha, "alpha");
  field(config.beta, "beta");
  field(config.gamma, "gamma");
  field(config.debug_log_scores, "debug_log_scores");
}

FindRoomSearch::FindRoomSearch(const Config& config)
    : SemanticSearch(config),
      config(config::checkValid(config)),
      similarity_(config.similarity.create()) {
  assert(config.search_layer == spark_dsg::DsgLayers::FRONTIERS &&
         "FindRoomSearch only supports searching in the FRONTIERS layer.");
  assert(similarity_ != nullptr &&
         "Similarity component must be provided for FindRoomSearch.");
  CHECK_GE(config.alpha, 0.0) << "FindRoomSearch alpha must be non-negative.";
  CHECK_GE(config.beta, 0.0) << "FindRoomSearch beta must be non-negative.";
  CHECK_GE(config.gamma, 0.0) << "FindRoomSearch gamma must be non-negative.";
}

const SearchOutput FindRoomSearch::search(
    const SearchInput::Ptr& input,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const input::OccupancyGrid::Ptr& occupancy_grid) const {
  SearchOutput output;

  if (!dsg->hasLayer(config.search_layer)) {
    LOG(ERROR) << "DSG has no " << config.search_layer << " layer!";
    return output;
  }
  if (input->transition_prompt_embeddings.empty()) {
    LOG(ERROR) << "No transition prompt embeddings provided for FindRoomSearch!";
    return output;
  }
  if (input->floorplan_room_embeddings.empty() ||
      input->floorplan_progress_scores.empty()) {
    LOG(ERROR) << "No floorplan room embeddings/progress scores provided for "
                  "FindRoomSearch!";
    return output;
  }

  const auto& search_layer = dsg->getLayer(config.search_layer);
  std::vector<spark_dsg::NodeId> search_node_ids;
  for (const auto& [node_id, _] : search_layer.nodes()) {
    search_node_ids.push_back(node_id);
  }
  filterSearchNodesByBoundingBox(search_layer, search_node_ids);
  if (search_node_ids.empty()) {
    LOG(ERROR) << "No search nodes found in layer " << config.search_layer << "!";
    return output;
  }

  size_t max_free_space_points = 0;
  for (const auto& node_id : search_node_ids) {
    const auto& node = search_layer.getNode(node_id);
    const auto& attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    max_free_space_points =
        std::max(max_free_space_points, attrs.frontier_points.size());
  }

  const auto progress_weights =
      normalizeProgressWeights(input->floorplan_progress_scores);

  std::vector<NodeScore> node_scores;
  for (const auto& node_id : search_node_ids) {
    const auto& node = search_layer.getNode(node_id);
    const auto& attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    if (!attrs.validFeatures()) {
      VLOG(1) << "Node " << NodeSymbol(node_id).str()
              << " has no valid features, skipping.";
      continue;
    }

    NodeScore node_score;
    node_score.node_id = node_id;
    node_score.transition_score =
        maxPromptSimilarity(*similarity_, attrs, input->transition_prompt_embeddings);
    node_score.progress_score = weightedRoomTransitionSimilarity(
        *similarity_, attrs, input->floorplan_room_embeddings, progress_weights);
    node_score.freespace_score =
        max_free_space_points == 0 ? 0.0
                                   : static_cast<double>(attrs.frontier_points.size()) /
                                         static_cast<double>(max_free_space_points);
    node_score.score = config.alpha * node_score.transition_score +
                       config.beta * node_score.progress_score +
                       config.gamma * node_score.freespace_score;
    node_scores.push_back(node_score);
  }

  if (node_scores.empty()) {
    LOG(ERROR) << "No nodes with valid features found in layer " << config.search_layer
               << "!";
    return output;
  }

  std::sort(node_scores.begin(),
            node_scores.end(),
            [](const NodeScore& a, const NodeScore& b) { return a.score > b.score; });

  if (config.debug_log_scores) {
    logNodeScores(node_scores, input, config.alpha, config.beta, config.gamma);
  }

  if (input->direct_goal_mode) {
    const auto& selected_node = search_layer.getNode(node_scores.front().node_id);
    output.goal = node_scores.front().node_id;
    setGoalFromFrontierNode(selected_node, output);
    output.valid_goal = true;
    output.success = true;
    return output;
  }

  bool force_find = node_scores.size() == 1;
  for (const auto& node_score : node_scores) {
    const auto& node_id = node_score.node_id;
    const auto& node = search_layer.getNode(node_id);
    const auto& attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    const auto& connected_nav = attrs.connected_nav;
    if (!dsg->hasNode(connected_nav)) {
      VLOG(6) << "Node " << NodeSymbol(node_id).str()
              << " has no valid connected nav, skipping.";
      continue;
    }

    const auto frontier_cell_idx =
        occupancy_grid->worldToGrid(attrs.position.cast<float>().head<2>());
    if (occupancy_grid->grid.count(frontier_cell_idx) == 0) {
      continue;
    }
    VLOG(2) << "Node " << node_id << " at grid index (" << frontier_cell_idx.first
            << ", " << frontier_cell_idx.second << ") has cell state "
            << static_cast<int>(occupancy_grid->grid.at(frontier_cell_idx)) << ".";
    if (occupancy_grid->grid.at(frontier_cell_idx) != input::CellState::ObservedFree) {
      continue;
    }

    output.goal = node_id;
    setGoalFromFrontierNode(node, output);
    output.nav_graph_goal = connected_nav;
    if (findNavPath(dsg, occupancy_grid, output, force_find)) {
      output.success = true;
      output.valid_goal = true;
      if (output.nav_nodes.back() != output.nav_graph_goal) {
        LOG(WARNING) << "Shortest path did not reach the connected nav node of the "
                        "frontier.";
        output.goal = output.nav_nodes.back();
        output.goal_pose = output.nav_poses.back();
        output.nav_graph_goal = output.nav_nodes.back();
        output.nav_graph_goal_used = true;
      }
      return output;
    }
    LOG(INFO) << "Could not find a path from current pose to frontier node: "
              << NodeSymbol(node_id).str() << ". Checking other frontiers...";
  }

  LOG(ERROR) << "No valid frontier nodes with free connected navs found!";
  const auto& top_node = search_layer.getNode(node_scores.front().node_id);
  setGoalFromFrontierNode(top_node, output);
  output.valid_goal = true;
  return output;
}

}  // namespace hvlm_planner
