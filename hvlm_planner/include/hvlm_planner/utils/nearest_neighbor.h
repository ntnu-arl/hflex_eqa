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

#include <spark_dsg/scene_graph_layer.h>
#include <spark_dsg/scene_graph_types.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace hvlm_planner {
namespace nearest_neighbor {

class NearestNodeFinder {
 public:
  using Ptr = std::unique_ptr<NearestNodeFinder>;
  NearestNodeFinder(const spark_dsg::SceneGraphLayer& layer,
                    const std::vector<spark_dsg::NodeId>& nodes);

  NearestNodeFinder(const spark_dsg::SceneGraphLayer& layer,
                    const std::unordered_set<spark_dsg::NodeId>& nodes);

  virtual ~NearestNodeFinder();

  static Ptr fromLayer(const spark_dsg::SceneGraphLayer& layer);

  bool find(const Eigen::Vector3d& position,
            size_t num_to_find,
            bool skip_first,
            std::vector<spark_dsg::NodeId>& nearest_node_ids);

  bool findRadius(const Eigen::Vector3d& position,
                  double radius_m,
                  bool skip_first,
                  std::vector<spark_dsg::NodeId>& nearest_node_ids);

  const size_t num_nodes;

 private:
  struct Detail;
  std::unique_ptr<Detail> internals_;
};

class PointNeighborSearch {
 public:
  using Ptr = std::unique_ptr<PointNeighborSearch>;
  explicit PointNeighborSearch(const std::vector<Eigen::Vector3f>& points);
  virtual ~PointNeighborSearch();

  /**
   * @brief Find the nearest neighbor of the query point.
   * @param query_point The query point.
   * @param distance_squared The output squared distance to the nearest neighbor.
   * @param index The output index of the nearest neighbor in the tree data.
   * @returns True if a nearest neighbor was found, false otherwise.
   */
  bool search(const Eigen::Vector3f& query_point,
              float& distance_squared,
              size_t& index) const;

  /**
   * @brief Find all neighbors within a radius of the query point.
   * @param query_point The query point.
   * @param radius The search radius.
   * @param indices_and_distances The output vector of pairs of neighbor index and
   * squared distance.
   * @returns True if any neighbors were found, false otherwise.
   */
  bool findRadius(const Eigen::Vector3f& query_point,
                  const float radius,
                  std::vector<std::pair<size_t, float>>& indices_and_distances) const;

 private:
  struct Detail;
  std::unique_ptr<Detail> internals_;
};

}  // namespace nearest_neighbor
}  // namespace hvlm_planner
