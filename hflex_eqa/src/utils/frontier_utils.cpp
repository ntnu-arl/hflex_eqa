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
#include "hflex_eqa/utils/frontier_utils.h"

#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {

void removeBlacklistedFrontiers(spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                const FrontierBlackList& blacklist,
                                const double& trans_tolerance,
                                const double& rot_tolerance,
                                const double& grid_resolution,
                                bool use_position_radius,
                                double position_radius_m) {
  const double trans_tolerance_grid = trans_tolerance * grid_resolution;
  const auto& frontiers_layer = dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS);
  std::vector<spark_dsg::NodeId> to_remove;
  for (const auto& [node_id, node] : frontiers_layer.nodes()) {
    const auto& pos = node->attributes<spark_dsg::NodeAttributes>().position;
    const auto& dir = node->attributes<spark_dsg::GlobalFrontierNodeAttributes>()
                          .direction;  // Eigen::Vector2d (encodes yaw of frontier)
    Pose frontier_pose = Pose::Identity();
    frontier_pose.translation() = pos;
    frontier_pose.linear() =
        math_utils::yawToRotationMatrix<double>(std::atan2(dir.y(), dir.x()));

    for (const auto& blacklisted_pos : blacklist) {
      if (use_position_radius &&
          (frontier_pose.translation() - blacklisted_pos.translation()).norm() <=
              position_radius_m) {
        to_remove.push_back(node_id);
        break;
      }

      const auto difference =
          math_utils::poseDifference(frontier_pose, blacklisted_pos, true);
      if (difference.translation <= trans_tolerance_grid &&
          difference.angles.z() <= rot_tolerance) {
        to_remove.push_back(node_id);
        break;
      }
    }
  }

  for (const auto& node_id : to_remove) {
    dsg->removeNode(node_id);
  }
}

}  // namespace hflex_eqa
