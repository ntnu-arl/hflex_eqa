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
#include "hvlm_planner/low_level/question_room_clip_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "hvlm_planner/common/global_info.h"
#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch,
                                   QuestionRoomClipSearch,
                                   QuestionRoomClipSearch::Config>(
        "QuestionRoomClipSearch");

struct NodeScore {
  spark_dsg::NodeId node_id;
  double object_logit;
  double object_probability = 0.0;
  double room_score;
  double score = 0.0;
  std::vector<double> frontier_room_probabilities;
};

std::vector<double> softmax(const std::vector<double>& logits) {
  std::vector<double> probabilities;
  probabilities.reserve(logits.size());
  if (logits.empty()) {
    return probabilities;
  }

  double exp_sum = 0.0;
  for (const auto logit : logits) {
    const double exp_value = std::exp(logit);
    probabilities.push_back(exp_value);
    exp_sum += exp_value;
  }

  if (exp_sum <= 0.0) {
    const double uniform_probability = 1.0 / static_cast<double>(logits.size());
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return probabilities;
  }

  for (auto& probability : probabilities) {
    probability /= exp_sum;
  }
  return probabilities;
}

bool computeRoomLogits(const FeatureVector& feature,
                       const std::vector<FeatureVector>& room_embeddings,
                       std::vector<double>& logits) {
  logits.clear();
  if (feature.size() == 0 || room_embeddings.empty()) {
    return false;
  }

  logits.reserve(room_embeddings.size());
  for (const auto& room_embedding : room_embeddings) {
    if (room_embedding.size() != feature.size()) {
      LOG(ERROR) << "Room embedding dimension (" << room_embedding.size()
                 << ") does not match feature dimension (" << feature.size() << ").";
      return false;
    }
    logits.push_back(math_utils::cosineSimilarity(feature, room_embedding));
  }
  return true;
}

double dotProduct(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return 0.0;
  }
  return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

std::string formatRoomTerms(const std::vector<std::string>& room_labels,
                            const std::vector<double>& frontier_room_probabilities,
                            const std::vector<double>& question_room_probabilities) {
  if (frontier_room_probabilities.empty() || question_room_probabilities.empty()) {
    return "[]";
  }

  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < frontier_room_probabilities.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    if (i < room_labels.size()) {
      stream << room_labels[i] << ": ";
    } else {
      stream << "room_" << i << ": ";
    }
    const double product =
        i < question_room_probabilities.size()
            ? frontier_room_probabilities[i] * question_room_probabilities[i]
            : 0.0;
    stream << "P(r|f)=" << frontier_room_probabilities[i] << " P(r|q)="
           << (i < question_room_probabilities.size() ? question_room_probabilities[i]
                                                      : 0.0)
           << " product=" << product;
  }
  stream << "]";
  return stream.str();
}

void logNodeScores(const std::vector<NodeScore>& node_scores,
                   const std::vector<std::string>& room_labels,
                   const std::vector<double>& question_room_probabilities,
                   double alpha,
                   double beta) {
  LOG(INFO) << "QuestionRoomClipSearch frontier scores:";
  for (const auto& node_score : node_scores) {
    const double object_term = alpha * node_score.object_probability;
    const double room_term = beta * node_score.room_score;
    LOG(INFO) << "  frontier_id=" << NodeSymbol(node_score.node_id)
              << " object_logit=" << node_score.object_logit
              << " P(f|objects)=" << node_score.object_probability
              << " alpha_object_term=" << object_term
              << " room_score=" << node_score.room_score
              << " beta_room_term=" << room_term << " total_score=" << node_score.score
              << " room_terms="
              << formatRoomTerms(room_labels,
                                 node_score.frontier_room_probabilities,
                                 question_room_probabilities);
  }
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

}  // namespace

void declare_config(QuestionRoomClipSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("QuestionRoomClipSearchConfig");
  field(config.similarity, "similarity");
  field(config.alpha, "alpha");
  field(config.beta, "beta");
  field(config.debug_log_scores, "debug_log_scores");
}

QuestionRoomClipSearch::QuestionRoomClipSearch(const Config& config)
    : SemanticSearch(config),
      config(config::checkValid(config)),
      similarity_(config.similarity.create()) {
  assert(config.search_layer == spark_dsg::DsgLayers::FRONTIERS &&
         "QuestionRoomClipSearch only supports searching in the FRONTIERS layer.");
  assert(similarity_ != nullptr &&
         "Similarity component must be provided for QuestionRoomClipSearch.");
  CHECK_GE(config.alpha, 0.0) << "QuestionRoomClipSearch alpha must be non-negative.";
  CHECK_GE(config.beta, 0.0) << "QuestionRoomClipSearch beta must be non-negative.";
  CHECK_LT(std::abs(config.alpha + config.beta - 1.0), 1.0e-6)
      << "QuestionRoomClipSearch alpha and beta must sum to 1.";
}

const SearchOutput QuestionRoomClipSearch::search(
    const SearchInput::Ptr& input,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const input::OccupancyGrid::Ptr& occupancy_grid) const {
  SearchOutput output;

  if (!dsg->hasLayer(config.search_layer)) {
    LOG(ERROR) << "DSG has no " << config.search_layer << " layer!";
    return output;
  }
  if (input->search_features.empty()) {
    LOG(ERROR) << "No search features provided!";
    return output;
  }

  const auto& search_layer = dsg->getLayer(config.search_layer);
  std::vector<spark_dsg::NodeId> search_node_ids;
  if (input->room_to_explore && dsg->hasNode(*input->room_to_explore)) {
    findSearchNodesInRoom(dsg, search_layer, *input->room_to_explore, search_node_ids);
  } else {
    for (const auto& [node_id, _] : search_layer.nodes()) {
      search_node_ids.push_back(node_id);
    }
    filterSearchNodesByBoundingBox(search_layer, search_node_ids);
  }
  if (search_node_ids.empty()) {
    LOG(ERROR) << "No search nodes found in layer " << config.search_layer << "!";
    return output;
  }

  std::vector<double> question_room_probabilities;
  const auto& global_info = GlobalInfo::instance();
  if (config.beta > 0.0) {
    std::vector<double> question_room_logits;
    if (!computeRoomLogits(global_info.getQuestionEmbedding(),
                           global_info.getRoomEmbeddings(),
                           question_room_logits)) {
      LOG(ERROR) << "Could not compute P(room | question).";
      return output;
    }
    question_room_probabilities = softmax(question_room_logits);
  }

  std::vector<NodeScore> node_scores;
  std::vector<double> object_logits;
  for (const auto& node_id : search_node_ids) {
    const auto& node = search_layer.getNode(node_id);
    const auto& node_attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    if (!node_attrs.validFeatures()) {
      VLOG(1) << "Node " << NodeSymbol(node_id).str()
              << " has no valid features, skipping.";
      continue;
    }

    const auto object_logit = static_cast<double>(similarity_->compute(
        node_attrs.semantic_feature, node_attrs.features, input->search_features));

    double room_score = 0.0;
    std::vector<double> frontier_room_probabilities;
    if (config.beta > 0.0) {
      std::vector<double> frontier_room_logits;
      if (!computeRoomLogits(node_attrs.semantic_feature,
                             global_info.getRoomEmbeddings(),
                             frontier_room_logits)) {
        VLOG(1) << "Could not compute P(room | frontier) for node "
                << NodeSymbol(node_id).str() << ", skipping.";
        continue;
      }
      frontier_room_probabilities = softmax(frontier_room_logits);
      room_score = dotProduct(frontier_room_probabilities, question_room_probabilities);
    }

    node_scores.push_back(
        {node_id, object_logit, 0.0, room_score, 0.0, frontier_room_probabilities});
    object_logits.push_back(object_logit);
  }
  if (node_scores.empty()) {
    LOG(ERROR) << "No nodes with valid features found in layer " << config.search_layer
               << "!";
    return output;
  }

  const auto object_probabilities = softmax(object_logits);
  for (size_t i = 0; i < node_scores.size(); ++i) {
    node_scores[i].object_probability = object_probabilities[i];
    node_scores[i].score = config.alpha * object_probabilities[i] +
                           config.beta * node_scores[i].room_score;
  }

  std::sort(node_scores.begin(),
            node_scores.end(),
            [](const NodeScore& a, const NodeScore& b) { return a.score > b.score; });

  if (config.debug_log_scores) {
    logNodeScores(node_scores,
                  global_info.getRoomLabels(),
                  question_room_probabilities,
                  config.alpha,
                  config.beta);
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
    const auto& node_attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    const auto& connected_nav = node_attrs.connected_nav;
    if (!dsg->hasNode(connected_nav)) {
      VLOG(6) << "Node " << NodeSymbol(node_id).str()
              << " has no valid connected nav, skipping.";
      continue;
    }
    const auto frontier_cell_idx =
        occupancy_grid->worldToGrid(node_attrs.position.cast<float>().head<2>());

    if (occupancy_grid->grid.count(frontier_cell_idx) > 0) {
      VLOG(2) << "Node " << node_id << " at grid index (" << frontier_cell_idx.first
              << ", " << frontier_cell_idx.second << ") has cell state "
              << static_cast<int>(occupancy_grid->grid.at(frontier_cell_idx)) << ".";
      if (occupancy_grid->grid.at(frontier_cell_idx) ==
          input::CellState::ObservedFree) {
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
        } else {
          LOG(INFO) << "Could not find a path from current pose to frontier node: "
                    << NodeSymbol(node_id).str() << ". Checking other frontiers...";
        }
      }
    }
  }

  LOG(ERROR) << "No valid frontier nodes with free connected navs found!";
  const auto& top_node = search_layer.getNode(node_scores.front().node_id);
  setGoalFromFrontierNode(top_node, output);
  output.valid_goal = true;
  return output;
}

}  // namespace hvlm_planner
