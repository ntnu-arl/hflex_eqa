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
#include "hflex_eqa/utils/dsg.h"

#include <glog/logging.h>

#include "hflex_eqa/common/global_info.h"

namespace hflex_eqa {
namespace dsg_utils {

bool getAgentPose(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                  Translation& agent_node_pose) {
  const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!agent_layer_id) {
    LOG(ERROR) << "DSG is missing layer '" << spark_dsg::DsgLayers::AGENTS << "'!";
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
  agent_node_pose = agent_node.attributes<spark_dsg::AgentNodeAttributes>().position;
  return true;
}

void getObjectPts(const spark_dsg::ObjectNodeAttributes& object_attrs,
                  const spark_dsg::Mesh::Ptr& mesh,
                  std::vector<Eigen::Vector3f>& object_cloud) {
  object_cloud.clear();
  for (const auto& v : object_attrs.mesh_connections) {
    object_cloud.push_back(mesh->pos(v));
  }
}

std::vector<spark_dsg::NodeId> getNodeIds(const spark_dsg::SceneGraphLayer& layer) {
  std::vector<spark_dsg::NodeId> node_ids;
  for (const auto& [node_id, _] : layer.nodes()) {
    node_ids.push_back(node_id);
  }
  return node_ids;
}

std::unique_ptr<open3d::geometry::TriangleMesh> convertToOpen3DMesh(
    const spark_dsg::Mesh::Ptr& mesh) {
  auto o3d_mesh = std::make_unique<open3d::geometry::TriangleMesh>();
  for (size_t i = 0; i < mesh->numVertices(); ++i) {
    o3d_mesh->vertices_.emplace_back(mesh->pos(i).cast<double>());
    if (mesh->has_colors) {
      o3d_mesh->vertex_colors_.emplace_back(
          Eigen::Vector3d(mesh->color(i).r, mesh->color(i).g, mesh->color(i).b) /
          255.0);
    }
  }
  for (size_t i = 0; i < mesh->numFaces(); ++i) {
    const auto& face = mesh->face(i);
    o3d_mesh->triangles_.emplace_back(face[0], face[1], face[2]);
  }
  return o3d_mesh;
}

}  // namespace dsg_utils
}  // namespace hflex_eqa
