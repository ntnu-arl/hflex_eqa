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
#include "hflex_eqa/low_level/dijkstra.h"

#include <config_utilities/config.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include <algorithm>
#include <map>
#include <queue>
#include <vector>

namespace hflex_eqa {

void declare_config(Dijkstra::Config& config) {
  using namespace config;
  base<ShortestPath::Config>(config);
  name("Dijkstra::Config");
}

Dijkstra::Dijkstra(const Config& config) : ShortestPath(config), config(config) {}

bool Dijkstra::search(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                      const input::OccupancyGrid::Ptr& occupancy_grid,
                      const Pose& start_pose,
                      const std::string& layer_name,
                      const spark_dsg::NodeId& start_id,
                      const spark_dsg::NodeId& goal_id,
                      std::vector<spark_dsg::NodeId>& ids_path,
                      std::vector<Pose>& poses_path,
                      bool force) const {
  ids_path.clear();
  poses_path.clear();

  if (!dsg->hasLayer(layer_name)) {
    LOG(ERROR) << "Layer " << layer_name << " does not exist in the DSG.";
    return false;
  }

  const auto& layer = dsg->getLayer(layer_name);

  if (layer.numNodes() == 0) {
    LOG(ERROR) << "Layer " << layer_name << " has no nodes.";
    return false;
  }

  if (layer.numEdges() == 0) {
    LOG(ERROR) << "Layer " << layer_name << " has no edges.";
    return false;
  }

  if (!layer.hasNode(start_id)) {
    LOG(ERROR) << "Start node " << start_id << " does not exist in layer " << layer_name
               << ".";
    return false;
  }

  if (!layer.hasNode(goal_id)) {
    LOG(ERROR) << "Goal node " << goal_id << " does not exist in layer " << layer_name
               << ".";
    return false;
  }

  const auto& nodes = layer.nodes();
  const auto& edges = layer.edges();

  std::set<spark_dsg::NodeId> nodes_to_use;
  // Filter nodes to use based on occupancy grid if provided
  if (config.check_occupancy_grid) {
    if (!occupancy_grid || occupancy_grid->empty()) {
      LOG(ERROR) << "Occupancy grid is required for ground projection but is not "
                    "provided or empty.";
      return false;
    }
    VLOG(5) << "Occupancy grid size: " << occupancy_grid->grid.size();
    for (const auto& kv : nodes) {
      const auto& node_id = kv.first;
      const auto& node = kv.second;
      const auto& pos = node->attributes<spark_dsg::NodeAttributes>().position;
      const auto grid_idx =
          occupancy_grid->worldToGrid(input::Point2f(pos.x(), pos.y()));
      if (occupancy_grid->grid.count(grid_idx) == 0) {
        VLOG(5) << "Node " << node_id << " at position (" << pos.x() << ", " << pos.y()
                << ") is out of occupancy grid bounds.";
        continue;
      }
      VLOG(5) << "Node " << node_id << " at grid index (" << grid_idx.first << ", "
              << grid_idx.second << ") has cell state "
              << static_cast<int>(occupancy_grid->grid.at(grid_idx)) << ".";
      if (occupancy_grid->grid.at(grid_idx) != input::CellState::ObservedOccupied) {
        nodes_to_use.insert(node_id);
      }
    }
  } else {
    for (const auto& kv : nodes) {
      nodes_to_use.insert(kv.first);
    }
  }

  if (nodes_to_use.empty()) {
    LOG(ERROR)
        << "No nodes available for pathfinding after applying occupancy grid filter.";
    return false;
  }

  if (nodes_to_use.count(start_id) == 0) {
    LOG(ERROR) << "Start node " << start_id
               << " is not in free space according to the occupancy grid.";
    return false;
  }

  if (nodes_to_use.count(goal_id) == 0) {
    LOG(ERROR) << "Goal node " << goal_id
               << " is not in free space according to the occupancy grid.";
    return false;
  }

  // Distance and predecessor maps
  std::map<spark_dsg::NodeId, double> dist;
  std::map<spark_dsg::NodeId, spark_dsg::NodeId> prev;

  // Initialize distances
  for (const auto& id : nodes_to_use) {
    dist[id] = std::numeric_limits<double>::infinity();
  }
  dist[start_id] = 0.0;

  // Min-priority queue: (distance, node_id)
  std::priority_queue<QueueElem, std::vector<QueueElem>, std::greater<QueueElem>> queue;
  queue.emplace(0.0, start_id);

  while (!queue.empty()) {
    const auto [current_dist, u] = queue.top();
    queue.pop();

    // Skip outdated queue entries
    if (current_dist > dist[u]) {
      continue;
    }

    if (u == goal_id) {
      break;
    }

    const auto& u_pos = nodes.at(u)->attributes<spark_dsg::NodeAttributes>().position;

    // Iterate over all edges to find neighbors
    for (const auto& edge_kv : edges) {
      const auto& edge = edge_kv.first;

      spark_dsg::NodeId v;
      if (edge.k1 == u) {
        v = edge.k2;
      } else if (edge.k2 == u) {
        v = edge.k1;
      } else {
        continue;
      }
      if (nodes_to_use.count(v) == 0) {
        continue;
      }

      const auto& v_pos = nodes.at(v)->attributes<spark_dsg::NodeAttributes>().position;
      double weight = std::numeric_limits<double>::infinity();
      if (config.project_to_ground) {
        weight = (u_pos.head<2>() - v_pos.head<2>()).norm();
      } else {
        weight = (u_pos - v_pos).norm();
      }

      const double alt = dist[u] + weight;
      if (alt < dist[v]) {
        dist[v] = alt;
        prev[v] = u;
        queue.emplace(alt, v);
      }
    }
  }

  spark_dsg::NodeId final_goal_id = goal_id;
  // No path found
  if (dist[final_goal_id] == std::numeric_limits<double>::infinity()) {
    if (!force) {
      LOG(WARNING) << "No path found from " << NodeSymbol(start_id).str() << " to "
                   << NodeSymbol(final_goal_id).str();
      return false;
    }
    // If force is true, we find the path to the closest reachable node to the goal
    double closest_dist = std::numeric_limits<double>::infinity();
    const auto& goal_pos =
        nodes.at(final_goal_id)->attributes<spark_dsg::NodeAttributes>().position;
    for (const auto& id : nodes_to_use) {
      if (dist[id] < std::numeric_limits<double>::infinity()) {
        const auto& pos =
            nodes.at(id)->attributes<spark_dsg::NodeAttributes>().position;
        double goal_dist = config.project_to_ground
                               ? (pos.head<2>() - goal_pos.head<2>()).norm()
                               : (pos - goal_pos).norm();
        if (goal_dist < closest_dist) {
          closest_dist = goal_dist;
          final_goal_id = id;
        }
      }
    }
    if (closest_dist == std::numeric_limits<double>::infinity()) {
      LOG(WARNING) << "No reachable nodes found from " << start_id;
      return false;
    }
    if (final_goal_id == start_id) {
      LOG(WARNING) << "Start node " << start_id
                   << " is the closest reachable node to the goal, skipping this goal.";
      return false;
    }
    LOG(INFO) << "No path to goal, but closest reachable node is " << final_goal_id
              << " at distance " << closest_dist;
  }

  // Reconstruct path
  for (spark_dsg::NodeId at = final_goal_id;; at = prev[at]) {
    ids_path.push_back(at);
    if (at == start_id) {
      break;
    }
  }
  std::reverse(ids_path.begin(), ids_path.end());

  // Build pose path
  for (const auto& id : ids_path) {
    Pose pose = Pose::Identity();
    pose.translation() = nodes.at(id)->attributes<spark_dsg::NodeAttributes>().position;
    if (config.project_to_ground) {
      pose.translation().z() = start_pose.translation().z();
    }
    poses_path.push_back(pose);
  }

  return true;
}

}  // namespace hflex_eqa
