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
#include "hflex_eqa/utils/nearest_neighbor.h"

#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph_layer.h>
#include <spark_dsg/scene_graph_types.h>

#include <nanoflann.hpp>

namespace hflex_eqa {
namespace nearest_neighbor {

using nanoflann::KDTreeSingleIndexAdaptor;
using nanoflann::KDTreeSingleIndexDynamicAdaptor;
using nanoflann::L2_Simple_Adaptor;

struct GraphKdTreeAdaptor {
  GraphKdTreeAdaptor(const spark_dsg::SceneGraphLayer& layer,
                     const std::vector<spark_dsg::NodeId>& nodes)
      : layer(layer), nodes(nodes) {}

  inline size_t kdtree_get_point_count() const { return nodes.size(); }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return layer.getNode(nodes[idx]).attributes().position(dim);
  }

  template <class T>
  bool kdtree_get_bbox(T&) const {
    return false;
  }

  const spark_dsg::SceneGraphLayer& layer;
  std::vector<spark_dsg::NodeId> nodes;
};

struct NearestNodeFinder::Detail {
  using Dist = L2_Simple_Adaptor<double, GraphKdTreeAdaptor>;
  using KDTree = KDTreeSingleIndexAdaptor<Dist, GraphKdTreeAdaptor, 3, size_t>;

  Detail(const spark_dsg::SceneGraphLayer& layer,
         const std::vector<spark_dsg::NodeId>& nodes)
      : adaptor(layer, nodes) {
    kdtree.reset(new KDTree(3, adaptor));
    kdtree->buildIndex();
  }

  ~Detail() = default;

  GraphKdTreeAdaptor adaptor;
  std::unique_ptr<KDTree> kdtree;
};

NearestNodeFinder::NearestNodeFinder(const spark_dsg::SceneGraphLayer& layer,
                                     const std::vector<spark_dsg::NodeId>& nodes)
    : num_nodes(nodes.size()), internals_(new Detail(layer, nodes)) {}

NearestNodeFinder::NearestNodeFinder(const spark_dsg::SceneGraphLayer& layer,
                                     const std::unordered_set<spark_dsg::NodeId>& nodes)
    : num_nodes(nodes.size()) {
  std::vector<spark_dsg::NodeId> node_vector(nodes.begin(), nodes.end());
  VLOG(10) << "Making node finder with " << nodes.size() << " nodes";
  internals_.reset(new Detail(layer, node_vector));
}

NearestNodeFinder::~NearestNodeFinder() {}

NearestNodeFinder::Ptr NearestNodeFinder::fromLayer(
    const spark_dsg::SceneGraphLayer& layer) {
  std::unordered_set<spark_dsg::NodeId> layer_nodes;
  for (const auto& [node_id, node] : layer.nodes()) {
    layer_nodes.insert(node_id);
  }

  if (layer_nodes.empty()) {
    return nullptr;
  }

  return std::make_unique<NearestNodeFinder>(layer, layer_nodes);
}

bool NearestNodeFinder::find(const Eigen::Vector3d& position,
                             size_t num_to_find,
                             bool skip_first,
                             std::vector<spark_dsg::NodeId>& nearest_node_ids) {
  const size_t limit = skip_first ? num_to_find + 1 : num_to_find;
  std::vector<size_t> nn_indices(limit);
  std::vector<double> distances(limit);

  size_t num_found = internals_->kdtree->knnSearch(
      position.data(), limit, nn_indices.data(), distances.data());

  size_t i = skip_first ? 1 : 0;
  for (; i < num_found; ++i) {
    nearest_node_ids.push_back(internals_->adaptor.nodes[nn_indices[i]]);
  }
  return nearest_node_ids.size() > 0;
}

bool NearestNodeFinder::findRadius(const Eigen::Vector3d& position,
                                   double radius,
                                   bool skip_first,
                                   std::vector<spark_dsg::NodeId>& nearest_node_ids) {
  std::vector<nanoflann::ResultItem<size_t, double>> neighbors;
  size_t num_found = internals_->kdtree->radiusSearch(
      position.data(), radius, neighbors, nanoflann::SearchParameters());

  size_t i = skip_first ? 1 : 0;
  for (; i < num_found; ++i) {
    const auto idx = neighbors[i].first;
    nearest_node_ids.push_back(internals_->adaptor.nodes[idx]);
  }
  return nearest_node_ids.size() > 0;
}

struct PointNeighborSearch::Detail {
  // Nanoflann interface.
  explicit Detail(const std::vector<Eigen::Vector3f>& points)
      : points_(points),
        tree_(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    tree_.buildIndex();
  }

  std::size_t kdtree_get_point_count() const { return points_.size(); }

  float kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return points_[idx].x();
    else if (dim == 1)
      return points_[idx].y();
    else
      return points_[idx].z();
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }

  const std::vector<Eigen::Vector3f>& points_;
  KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, Detail>, Detail, 3> tree_;
};

PointNeighborSearch::PointNeighborSearch(const std::vector<Eigen::Vector3f>& points) {
  internals_ = std::make_unique<Detail>(points);
}

PointNeighborSearch::~PointNeighborSearch(){};

bool PointNeighborSearch::search(const Eigen::Vector3f& query_point,
                                 float& distance_squared,
                                 size_t& index) const {
  nanoflann::KNNResultSet<float> resultSet(1);
  float query_point_arr[3] = {query_point.x(), query_point.y(), query_point.z()};
  resultSet.init(&index, &distance_squared);
  return internals_->tree_.findNeighbors(resultSet, &query_point_arr[0]);
}

bool PointNeighborSearch::findRadius(
    const Eigen::Vector3f& query_point,
    const float radius,
    std::vector<std::pair<size_t, float>>& indices_and_distances) const {
  float query_point_arr[3] = {query_point.x(), query_point.y(), query_point.z()};

  std::vector<nanoflann::ResultItem<unsigned int, float>> results;

  const bool found =
      internals_->tree_.radiusSearch(&query_point_arr[0], radius, results);

  indices_and_distances.clear();
  indices_and_distances.reserve(results.size());

  for (const auto& r : results) {
    indices_and_distances.emplace_back(static_cast<size_t>(r.first), r.second);
  }

  return found;
}

}  // namespace nearest_neighbor
}  // namespace hflex_eqa
