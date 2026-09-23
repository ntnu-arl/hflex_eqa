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
#include "hvlm_planner/low_level/closest_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>
#include <cassert>
#include <cmath>

#include "hvlm_planner/common/global_info.h"
#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch,
                                   ClosestSearch,
                                   ClosestSearch::Config>("ClosestSearch");

bool getCurrentAgentPose(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                         Pose& agent_pose) {
  const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!agent_layer_id) {
    LOG(ERROR) << "No agents layer key in DSG.";
    return false;
  }

  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
  if (!agent_layer) {
    LOG(ERROR) << "Missing layer '" << spark_dsg::DsgLayers::AGENTS
               << "' and partition " << prefix.key << " in DSG!";
    return false;
  }
  if (agent_layer->numNodes() == 0) {
    LOG(ERROR) << "No agent nodes in layer '" << spark_dsg::DsgLayers::AGENTS
               << "' and partition " << prefix.key << "!";
    return false;
  }

  spark_dsg::NodeSymbol pgmo_key(prefix.key, agent_layer->numNodes() - 1);
  if (!agent_layer->hasNode(pgmo_key)) {
    LOG(ERROR) << "Agent node with key " << pgmo_key << " does not exist in DSG!";
    return false;
  }

  const auto& agent_node = agent_layer->getNode(pgmo_key);
  agent_pose = Pose::Identity();
  agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                            .world_R_body.toRotationMatrix();
  agent_pose.translation() =
      agent_node.attributes<spark_dsg::NodeAttributes>().position;
  return true;
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

void declare_config(ClosestSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("ClosestSearchConfig");
}

ClosestSearch::ClosestSearch(const Config& config)
    : SemanticSearch(config), config(config::checkValid(config)) {
  assert(config.search_layer == spark_dsg::DsgLayers::FRONTIERS &&
         "ClosestSearch only supports searching in the FRONTIERS layer.");
}

const SearchOutput ClosestSearch::search(
    const SearchInput::Ptr& input,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const input::OccupancyGrid::Ptr& occupancy_grid) const {
  SearchOutput output;

  if (!dsg->hasLayer(config.search_layer)) {
    LOG(ERROR) << "DSG has no " << config.search_layer << " layer!";
    return output;
  }

  Pose agent_pose = Pose::Identity();
  if (!getCurrentAgentPose(dsg, agent_pose)) {
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

  // Rank frontier nodes by planar distance to the current agent pose.
  std::vector<std::pair<spark_dsg::NodeId, double>> node_ranking;
  const auto& agent_position = agent_pose.translation().head<2>();
  for (const auto& node_id : search_node_ids) {
    const auto& node = search_layer.getNode(node_id);
    const auto& node_attrs = node.attributes<spark_dsg::GlobalFrontierNodeAttributes>();
    const double distance = (node_attrs.position.head<2>() - agent_position).norm();
    node_ranking.emplace_back(node_id, distance);
  }
  if (node_ranking.empty()) {
    LOG(ERROR) << "No nodes found in layer " << config.search_layer << "!";
    return output;
  }

  std::sort(node_ranking.begin(),
            node_ranking.end(),
            [](const std::pair<spark_dsg::NodeId, double>& a,
               const std::pair<spark_dsg::NodeId, double>& b) {
              return a.second < b.second;
            });

  if (input->direct_goal_mode) {
    const auto& selected_node = search_layer.getNode(node_ranking.front().first);
    output.goal = node_ranking.front().first;
    setGoalFromFrontierNode(selected_node, output);
    output.valid_goal = true;
    output.success = true;
    return output;
  }

  bool force_find = node_ranking.size() == 1;
  for (const auto& [node_id, _] : node_ranking) {
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
  const auto& top_node = search_layer.getNode(node_ranking.front().first);
  setGoalFromFrontierNode(top_node, output);
  output.valid_goal = true;
  return output;
}

}  // namespace hvlm_planner
