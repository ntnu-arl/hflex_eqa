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
#include "hvlm_planner/low_level/semantic_search.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>

#include "hvlm_planner/common/exploration_bounding_box.h"
#include "hvlm_planner/common/global_info.h"

namespace hvlm_planner {

void declare_config(SemanticSearch::Config& config) {
  using namespace config;
  name("SemanticSearchConfig");
  field(config.shortest_path, "shortest_path");
  field(config.search_layer, "search_layer");
  field(config.nav_layer, "nav_layer");
  field(config.use_goal_orientation, "use_goal_orientation");
}

SemanticSearch::SemanticSearch(const Config& config)
    : config(config::checkValid(config)),
      shortest_path_(config.shortest_path.create()) {
  if (!shortest_path_) {
    throw std::runtime_error("SemanticSearch: Failed to create ShortestPath module.");
  }
}

bool SemanticSearch::findNavPath(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                 const input::OccupancyGrid::Ptr& occupancy_grid,
                                 SearchOutput& output,
                                 bool force) const {
  // Get current pose
  const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!agent_layer_id) {
    return false;
  }
  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
  if (!agent_layer) {
    LOG(ERROR) << "Missing layer '" << spark_dsg::DsgLayers::AGENTS << "and partition "
               << prefix.key << " in DSG!";
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
  output.start_pose = Eigen::Isometry3d::Identity();
  output.start_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                                   .world_R_body.toRotationMatrix();
  output.start_pose.translation() =
      agent_node.attributes<spark_dsg::NodeAttributes>().position;

  // Get closest nav_graph node to current pose
  const auto parent = agent_node.getParent();
  if (!parent) {
    LOG(ERROR) << "Agent node has no parent!";
    return false;
  }

  if (!dsg->hasLayer(config.nav_layer)) {
    LOG(ERROR) << "DSG has no " << config.nav_layer << " layer!";
    return false;
  }

  if (!dsg->getLayer(config.nav_layer).hasNode(parent.value())) {
    LOG(ERROR) << "Start node " << spark_dsg::NodeSymbol(parent.value()).str()
               << " does not exist in DSG!";
    return false;
  }

  if (!(shortest_path_->search(dsg,
                               occupancy_grid,
                               output.start_pose,
                               config.nav_layer,
                               parent.value(),
                               output.nav_graph_goal,
                               output.nav_nodes,
                               output.nav_poses,
                               force))) {
    return false;
  }
  // Compute desired rotations of nav poses: consider yaw only pointing to next nav node
  for (size_t i = 0; i < output.nav_poses.size() - 1; ++i) {
    Translation direction =
        output.nav_poses[i + 1].translation() - output.nav_poses[i].translation();
    direction.z() = 0.0;
    direction.normalize();
    double yaw = std::atan2(direction.y(), direction.x());
    Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());
    output.nav_poses[i].linear() = yaw_rot.toRotationMatrix();
  }
  // Use goal position for final orientation
  if (config.use_goal_orientation) {
    output.nav_poses.back().linear() = output.goal_pose.linear();
  } else {
    Translation final_direction =
        output.goal_pose.translation() -
        output.nav_poses[output.nav_poses.size() - 1].translation();
    final_direction.z() = 0.0;
    final_direction.normalize();
    double final_yaw = std::atan2(final_direction.y(), final_direction.x());
    Eigen::AngleAxisd final_yaw_rot(final_yaw, Eigen::Vector3d::UnitZ());
    output.nav_poses.back().linear() = final_yaw_rot.toRotationMatrix();
  }
  return true;
}

void SemanticSearch::findSearchNodesInRoom(
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const spark_dsg::DynamicSceneGraph::Layer& search_layer,
    const spark_dsg::NodeId& room_id,
    std::vector<spark_dsg::NodeId>& search_node_ids) const {
  search_node_ids.clear();
  spark_dsg::NodeId closest_search_node_id = 0;
  const auto room_center =
      dsg->getNode(room_id).attributes<spark_dsg::NodeAttributes>().position;
  double closest_search_node_dist = std::numeric_limits<double>::max();
  for (const auto& [node_id, node] : search_layer.nodes()) {
    const auto connected_nav =
        node->attributes<spark_dsg::GlobalFrontierNodeAttributes>().connected_nav;
    if (!dsg->hasNode(connected_nav)) {
      continue;
    }
    const auto& connected_nav_node = dsg->getNode(connected_nav);
    const auto& room_parent_id = connected_nav_node.getParent();
    if (room_parent_id && room_parent_id.value() == room_id) {
      search_node_ids.push_back(node_id);
    }
    if (const auto dist =
            (node->attributes<spark_dsg::NodeAttributes>().position.head<2>() -
             room_center.head<2>())
                .norm();
        dist < closest_search_node_dist) {
      closest_search_node_dist = dist;
      closest_search_node_id = node_id;
    }
  }
  if (search_node_ids.empty() &&
      closest_search_node_dist < std::numeric_limits<double>::max()) {
    search_node_ids.push_back(closest_search_node_id);
  }
  filterSearchNodesByBoundingBox(search_layer, search_node_ids);
}

bool SemanticSearch::isWithinBoundingBox(const Eigen::Vector3d& position) const {
  return ExplorationBoundingBox::instance().contains(position);
}

void SemanticSearch::filterSearchNodesByBoundingBox(
    const spark_dsg::DynamicSceneGraph::Layer& search_layer,
    std::vector<spark_dsg::NodeId>& search_node_ids) const {
  if (!ExplorationBoundingBox::instance().config().enabled) {
    return;
  }

  const auto before = search_node_ids.size();
  search_node_ids.erase(
      std::remove_if(search_node_ids.begin(),
                     search_node_ids.end(),
                     [&search_layer, this](const spark_dsg::NodeId& node_id) {
                       const auto& position =
                           search_layer.getNode(node_id)
                               .attributes<spark_dsg::NodeAttributes>()
                               .position;
                       return !isWithinBoundingBox(position);
                     }),
      search_node_ids.end());

  const auto filtered = before - search_node_ids.size();
  if (filtered > 0) {
    VLOG(2) << "Filtered " << filtered
            << " frontier search nodes outside exploration bounding box.";
  }
}

}  // namespace hvlm_planner
