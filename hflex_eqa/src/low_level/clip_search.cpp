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
#include "hflex_eqa/low_level/clip_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch, ClipSearch, ClipSearch::Config>(
        "ClipSearch");
}  // namespace

void declare_config(ClipSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("ClipSearchConfig");
  field(config.similarity, "similarity");
}

ClipSearch::ClipSearch(const Config& config)
    : SemanticSearch(config),
      config(config::checkValid(config)),
      similarity_(config.similarity.create()) {
  assert(config.search_layer == spark_dsg::DsgLayers::FRONTIERS &&
         "ClipSearch only supports searching in the FRONTIERS layer.");
  assert(similarity_ != nullptr &&
         "Similarity component must be provided for ClipSearch.");
}

const SearchOutput ClipSearch::search(
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

  // Rank nodes in search layer based on cosine similarity of their features to the
  // search features
  std::vector<std::pair<spark_dsg::NodeId, double>> node_ranking;
  for (const auto& node_id : search_node_ids) {
    const auto& node = search_layer.getNode(node_id);
    const auto& node_attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    if (!node_attrs.validFeatures()) {
      VLOG(1) << "Node " << NodeSymbol(node_id).str()
              << " has no valid features, skipping.";
      continue;
    }
    const auto cos_sim = similarity_->compute(
        node_attrs.semantic_feature, node_attrs.features, input->search_features);
    node_ranking.emplace_back(std::make_pair(node_id, cos_sim));
  }
  if (node_ranking.empty()) {
    LOG(ERROR) << "No nodes with valid features found in layer " << config.search_layer
               << "!";
    return output;
  }

  // Sort nodes by cosine similarity in descending order
  std::sort(node_ranking.begin(),
            node_ranking.end(),
            [](const std::pair<spark_dsg::NodeId, double>& a,
               const std::pair<spark_dsg::NodeId, double>& b) {
              return a.second > b.second;
            });

  if (input->direct_goal_mode) {
    const auto& selected_node = search_layer.getNode(node_ranking.front().first);
    const auto& selected_node_attrs =
        selected_node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    output.goal = node_ranking.front().first;
    output.goal_pose = Eigen::Isometry3d::Identity();
    output.goal_pose.translation() = selected_node_attrs.position;
    double yaw = std::atan2(selected_node_attrs.direction.y(),
                            selected_node_attrs.direction.x());
    output.goal_pose.linear() = math_utils::yawToRotationMatrix<double>(yaw);
    output.setSelectedFrontierPose(output.goal_pose);
    output.valid_goal = true;
    output.success = true;
    return output;
  }

  bool force_find = node_ranking.size() == 1;
  for (const auto& [node_id, _] : node_ranking) {
    const auto& node = search_layer.getNode(node_id);
    const auto& node_attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    const auto& connected_nav_id =
        node.attributes<spark_dsg::GlobalFrontierNodeAttributes>().connected_nav;
    if (!dsg->hasNode(connected_nav_id)) {
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
        // Try to find a path to the frontier node through its connected nav node. If a
        // path can be found, select this node as the goal. If not, continue checking
        // other nodes in ranked order.
        const auto& selected_node = search_layer.getNode(node_id);
        output.goal = node_id;
        const auto& selected_node_attrs =
            selected_node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
        output.goal_pose = Eigen::Isometry3d::Identity();
        output.goal_pose.translation() = selected_node_attrs.position;
        double yaw = std::atan2(selected_node_attrs.direction.y(),
                                selected_node_attrs.direction.x());
        output.goal_pose.linear() = math_utils::yawToRotationMatrix<double>(yaw);
        output.setSelectedFrontierPose(output.goal_pose);
        output.nav_graph_goal = selected_node_attrs.connected_nav;
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
  // Set goal_pose to the position of the top-ranked node even if it is not free, to at
  // least give some direction to the planner
  const auto& top_node = search_layer.getNode(node_ranking.front().first);
  const auto& top_node_attrs =
      top_node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
  output.goal_pose = Eigen::Isometry3d::Identity();
  output.goal_pose.translation() = top_node_attrs.position;
  double yaw = std::atan2(top_node_attrs.direction.y(), top_node_attrs.direction.x());
  output.goal_pose.linear() = math_utils::yawToRotationMatrix<double>(yaw);
  output.setSelectedFrontierPose(output.goal_pose);
  output.valid_goal = true;
  return output;
}

}  // namespace hflex_eqa
