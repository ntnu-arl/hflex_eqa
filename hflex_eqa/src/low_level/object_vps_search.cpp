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
#include "hflex_eqa/low_level/object_vps_search.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/factory.h>
#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <spark_dsg/node_symbol.h>

#include "hflex_eqa/common/global_info.h"
#include "hflex_eqa/utils/dsg.h"
#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {

namespace {

static const auto registration =
    config::RegistrationWithConfig<SemanticSearch,
                                   ObjectVpsSearch,
                                   ObjectVpsSearch::Config>("ObjectVpsSearch");
}  // namespace

void declare_config(ObjectVpsSearch::Config& config) {
  using namespace config;
  base<SemanticSearch::Config>(config);
  name("ObjectVpsSearchConfig");
  field(config.vp_selection, "vp_selection");
  field(config.a_star_config, "a_star_config");
}

ObjectVpsSearch::ObjectVpsSearch(const Config& config)
    : SemanticSearch(config),
      config(config::checkValid(config)),
      vp_selection_(config.vp_selection.create()),
      a_star_planner_(std::make_unique<AStarPlanner>(config.a_star_config)) {
  assert(vp_selection_ != nullptr &&
         "VPSelectionInterface instance must be created successfully.");
}

const SearchOutput ObjectVpsSearch::search(
    const SearchInput::Ptr& input,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const input::OccupancyGrid::Ptr& occupancy_grid) const {
  SearchOutput output;

  // Check if start node exists in DSG
  if (!dsg->hasNode(input->start_id)) {
    LOG(WARNING) << "Object node " << input->start_id << " not found in DSG.";
    return output;
  }
  if (!dsg->hasMesh()) {
    LOG(WARNING) << "DSG does not contain a mesh.";
    return output;
  }

  // Get current pose
  Translation agent_node_pose;
  if (!dsg_utils::getAgentPose(dsg, agent_node_pose)) {
    LOG(WARNING) << "Failed to get agent pose from DSG.";
    return output;
  }
  output.valid_goal = true;

  // Get object node pointcloud
  const auto& object_node = dsg->getNode(input->start_id);
  const auto& object_attrs = object_node.attributes<spark_dsg::ObjectNodeAttributes>();
  const auto mesh = dsg->mesh();
  std::vector<Eigen::Vector3f> object_cloud;
  dsg_utils::getObjectPts(object_attrs, mesh, object_cloud);

  // Get object radius
  const auto opt_radius = math_utils::computeViewingRadiusFromBoundingBox(
      object_attrs.bounding_box,
      vp_selection_->config.camera_config.fx,
      vp_selection_->config.camera_config.fy,
      vp_selection_->config.camera_config.width,
      vp_selection_->config.camera_config.height);
  if (opt_radius <= 0) {
    LOG(WARNING) << "Computed non-positive optimal viewing radius: " << opt_radius;
    return output;
  }

  // Compute best viewpoints
  const auto viewpoints =
      vp_selection_->computeBestViewpoints(object_cloud,
                                           mesh,
                                           occupancy_grid,
                                           opt_radius,
                                           object_attrs.position.cast<float>(),
                                           object_attrs.mesh_connections,
                                           agent_node_pose.cast<float>().z());

  // Find best reachable viewpoint
  output.goal = input->start_id;
  if (!findBestReachableViewpoint(
          viewpoints, occupancy_grid, dsg, output, input->direct_goal_mode)) {
    LOG(WARNING) << "Failed to find a reachable viewpoint.";
  }

  return output;
}

void ObjectVpsSearch::getSurroundingPts(
    const spark_dsg::ObjectNodeAttributes& object_attrs,
    const spark_dsg::Mesh::Ptr& mesh,
    const float& radius,
    std::vector<Eigen::Vector3f>& surrounding_cloud) const {
  surrounding_cloud.clear();
  neighbor_search_.reset(new nearest_neighbor::PointNeighborSearch(mesh->points));
  std::vector<std::pair<size_t, float>> neighbors;
  neighbor_search_->findRadius(object_attrs.position.cast<float>(), radius, neighbors);
  for (const auto& [idx, _] : neighbors) {
    surrounding_cloud.push_back(mesh->pos(idx));
  }
}

bool ObjectVpsSearch::findBestReachableViewpoint(
    const std::vector<vp_selection::Viewpoint>& candidate_viewpoints,
    const input::OccupancyGrid::Ptr& occupancy_grid,
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    SearchOutput& output,
    bool direct_goal_mode) const {
  if (direct_goal_mode) {
    for (const auto& vp : candidate_viewpoints) {
      const auto grid_idx = occupancy_grid->worldToGrid(
          vp.position.translation().cast<float>().head<2>());
      if (occupancy_grid->grid.count(grid_idx) == 0) {
        continue;
      }
      if (occupancy_grid->grid.at(grid_idx) != input::CellState::ObservedFree) {
        continue;
      }

      output.goal_pose = vp.position.cast<double>();
      output.success = true;
      return true;
    }

    output.success = false;
    return false;
  }

  nearest_node_finder_.reset(new nearest_neighbor::NearestNodeFinder(
      dsg->getLayer(config.nav_layer),
      dsg_utils::getNodeIds(dsg->getLayer(config.nav_layer))));
  for (size_t i = 0; i < candidate_viewpoints.size(); ++i) {
    const auto& vp = candidate_viewpoints[i];
    // Find closest nav node to viewpoint
    std::vector<spark_dsg::NodeId> nearest_node_ids;
    if (!nearest_node_finder_->find(
            vp.position.translation().cast<double>(), 1, false, nearest_node_ids)) {
      VLOG(2) << "No nav nodes found near viewpoint " << i << " at position "
              << vp.position.translation().transpose();
      continue;
    }

    output.nav_graph_goal = nearest_node_ids[0];
    output.goal_pose = vp.position.cast<double>();
    // Find path in nav graph to closest nav node to viewpoint
    if (findNavPath(dsg, occupancy_grid, output, false)) {
      // Find path from nav node to viewpoint using A* and occupancy grid
      std::vector<Pose> placeholder_path;
      const auto clearance_segment =
          AStarPlanner::ClearanceSegment::TRAVERSABILITY_TO_GOAL;
      if (a_star_planner_->plan(occupancy_grid,
                                output.nav_poses.back(),
                                output.goal_pose,
                                placeholder_path,
                                clearance_segment)) {
        output.success = true;
        return true;
      } else {
        VLOG(2) << "No path found from closest nav node to viewpoint " << i;
      }
    } else {
      VLOG(2) << "No path found in nav graph to closest nav node for viewpoint " << i;
    }
  }
  output.success = false;
  return false;
}

}  // namespace hflex_eqa
