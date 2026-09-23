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
#include "hvlm_planner/low_level/node_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>

#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch, NodeSearch, NodeSearch::Config>(
        "NodeSearch");
}  // namespace

void declare_config(NodeSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("NodeSearchConfig");
}

NodeSearch::NodeSearch(const Config& config)
    : SemanticSearch(config), config(config::checkValid(config)) {}

const SearchOutput NodeSearch::search(
    const SearchInput::Ptr& input,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const input::OccupancyGrid::Ptr& occupancy_grid) const {
  SearchOutput output;
  if (!dsg->hasNode(input->start_id)) {
    LOG(WARNING) << "Start node " << input->start_id << " not found in DSG.";
    return output;
  }
  if (!dsg->hasNode(input->child_start_id)) {
    LOG(WARNING) << "Child start node " << input->child_start_id
                 << " not found in DSG.";
    return output;
  }
  output.goal = input->child_start_id;
  output.goal_pose = input->child_start_pose;
  output.nav_graph_goal = input->start_id;
  if (input->direct_goal_mode) {
    output.success = true;
    output.valid_goal = true;
    return output;
  }
  output.success = findNavPath(dsg, occupancy_grid, output);
  return output;
}
}  // namespace hvlm_planner
