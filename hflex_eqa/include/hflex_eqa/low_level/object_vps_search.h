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

#include <config_utilities/virtual_config.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>

#include "hflex_eqa/low_level/a_star.h"
#include "hflex_eqa/low_level/semantic_search.h"
#include "hflex_eqa/utils/nearest_neighbor.h"
#include "hflex_eqa/vp_selection/vp_selection_interface.h"

namespace hflex_eqa {

class ObjectVpsSearch : public SemanticSearch {
 public:
  struct Config : SemanticSearch::Config {
    config::VirtualConfig<vp_selection::VPSelectionInterface> vp_selection;
    AStarPlanner::Config a_star_config;
  } const config;

  explicit ObjectVpsSearch(const Config& config);

  const SearchOutput search(
      const SearchInput::Ptr& input,
      const spark_dsg::DynamicSceneGraph::Ptr& dsg,
      const input::OccupancyGrid::Ptr& occupancy_grid) const override;

 protected:
  void getSurroundingPts(const spark_dsg::ObjectNodeAttributes& object_attrs,
                         const spark_dsg::Mesh::Ptr& mesh,
                         const float& radius,
                         std::vector<Eigen::Vector3f>& surrounding_cloud) const;

  bool findBestReachableViewpoint(
      const std::vector<vp_selection::Viewpoint>& candidate_viewpoints,
      const input::OccupancyGrid::Ptr& occupancy_grid,
      const spark_dsg::DynamicSceneGraph::Ptr& dsg,
      SearchOutput& output,
      bool direct_goal_mode) const;

  vp_selection::VPSelectionInterface::Ptr vp_selection_;
  AStarPlanner::Ptr a_star_planner_;
  mutable nearest_neighbor::PointNeighborSearch::Ptr neighbor_search_;
  mutable nearest_neighbor::NearestNodeFinder::Ptr nearest_node_finder_;
};

void declare_config(ObjectVpsSearch::Config& config);

}  // namespace hflex_eqa
