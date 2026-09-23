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
#include <config_utilities/config_utilities.h>
#include <config_utilities/virtual_config.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "hvlm_planner/common/input.h"
#include "hvlm_planner/common/types.h"
#include "hvlm_planner/low_level/shortest_path.h"

namespace hvlm_planner {
struct SearchInput {
  using Ptr = std::shared_ptr<SearchInput>;
  std::vector<FeatureVector> search_features;
  std::vector<Label> search_labels;
  spark_dsg::NodeId start_id;
  spark_dsg::NodeId child_start_id;
  std::optional<NodeId> room_to_explore = std::nullopt;
  Pose child_start_pose = Pose::Identity();
  bool direct_goal_mode = false;
  std::vector<FeatureVector> transition_prompt_embeddings;
  std::vector<std::string> transition_prompts;
  std::vector<std::string> floorplan_room_labels;
  std::vector<float> floorplan_progress_scores;
  std::vector<FeatureVector> floorplan_room_embeddings;

  void fromLowLevelInput(const input::Input::Ptr& input) {
    search_features = input->getSearchFeatures();
    search_labels = input->getSearchLabels();
  }
};

struct SearchOutput {
  bool success = false;
  bool valid_goal = false;
  bool nav_graph_goal_used = false;
  bool has_selected_frontier_pose = false;
  spark_dsg::NodeId goal;
  spark_dsg::NodeId nav_graph_goal;
  Pose start_pose;
  Pose goal_pose;
  Pose selected_frontier_pose;
  std::vector<spark_dsg::NodeId> nav_nodes;
  std::vector<Pose> nav_poses;
  std::vector<Pose> extra_poses_start;
  std::vector<Pose> extra_poses_end;

  void setSelectedFrontierPose(const Pose& pose) {
    selected_frontier_pose = pose;
    has_selected_frontier_pose = true;
  }
};

class SemanticSearch {
 public:
  using Ptr = std::unique_ptr<SemanticSearch>;

  struct Config {
    config::VirtualConfig<ShortestPath> shortest_path;
    std::string search_layer = spark_dsg::DsgLayers::FRONTIERS;
    std::string nav_layer = spark_dsg::DsgLayers::PLACES;
    bool use_goal_orientation = false;
  } const config;

  explicit SemanticSearch(const Config& config);
  virtual ~SemanticSearch() = default;
  virtual const SearchOutput search(
      const SearchInput::Ptr& input,
      const spark_dsg::DynamicSceneGraph::Ptr& dsg,
      const input::OccupancyGrid::Ptr& occupancy_grid) const = 0;

 protected:
  bool findNavPath(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                   const input::OccupancyGrid::Ptr& occupancy_grid,
                   SearchOutput& output,
                   bool force = false) const;
  void findSearchNodesInRoom(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                             const spark_dsg::DynamicSceneGraph::Layer& search_layer,
                             const spark_dsg::NodeId& room_id,
                             std::vector<spark_dsg::NodeId>& search_node_ids) const;
  bool isWithinBoundingBox(const Eigen::Vector3d& position) const;
  void filterSearchNodesByBoundingBox(
      const spark_dsg::DynamicSceneGraph::Layer& search_layer,
      std::vector<spark_dsg::NodeId>& search_node_ids) const;
  ShortestPath::Ptr shortest_path_;
};

void declare_config(SemanticSearch::Config& config);
}  // namespace hvlm_planner
