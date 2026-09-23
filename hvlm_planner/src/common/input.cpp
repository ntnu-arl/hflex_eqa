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
#include "hvlm_planner/common/input.h"

using namespace spark_dsg;

namespace hvlm_planner {
namespace input {

void OccupancyGrid::GridBounds::update(const int& r, const int& c) {
  if (r < min_row) min_row = r;
  if (r > max_row) max_row = r;
  if (c < min_col) min_col = c;
  if (c > max_col) max_col = c;
  width = static_cast<size_t>(max_col - min_col + 1);
  height = static_cast<size_t>(max_row - min_row + 1);
}

void OccupancyGrid::clear() {
  grid.clear();
  bounds = GridBounds();
  resolution = 0.0f;
}

GridIndex OccupancyGrid::worldToGrid(const Point2f& p) const {
  int col = static_cast<int>(std::floor(p.x() / resolution));
  int row = static_cast<int>(std::floor(p.y() / resolution));
  return {row, col};
}

Point2f OccupancyGrid::gridToWorld(const GridIndex& idx) const {
  const int row = idx.first;
  const int col = idx.second;

  // Return cell center in world coordinates
  float x = (static_cast<float>(col) + 0.5f) * resolution;
  float y = (static_cast<float>(row) + 0.5f) * resolution;

  return Point2f(x, y);
}

bool OccupancyGrid::isFree(const Point2f& p) const {
  const auto idx = worldToGrid(p);
  return isFree(idx);
}

bool OccupancyGrid::isFree(const GridIndex& idx) const {
  if (grid.find(idx) == grid.end()) {
    return false;  // Unknown cells are treated as occupied
  }
  return grid.at(idx) == CellState::ObservedFree;
}

OccupancyGrid::Ptr OccupancyGrid::clone() const {
  OccupancyGrid::Ptr new_grid = std::make_shared<OccupancyGrid>();
  new_grid->grid = grid;
  new_grid->resolution = resolution;
  new_grid->bounds = bounds;
  return new_grid;
}

Input::Input(const uint64_t& timestamp_ns,
             const DynamicSceneGraph::Ptr& dsg,
             const OccupancyGrid::Ptr& occupancy_grid,
             const cv::Mat& view,
             const std::vector<FeatureVector>& search_features,
             const std::vector<Label>& search_labels)
    : timestamp_ns_(timestamp_ns),
      dsg_(dsg),
      occupancy_grid_(occupancy_grid),
      view_(view),
      search_features_(search_features),
      search_labels_(search_labels) {}

Input::Input(const uint64_t& timestamp_ns,
             const DynamicSceneGraph::Ptr& dsg,
             const OccupancyGrid::Ptr& occupancy_grid,
             const cv::Mat& view)
    : timestamp_ns_(timestamp_ns),
      dsg_(dsg),
      occupancy_grid_(occupancy_grid),
      view_(view) {}
Input::Input()
    : dsg_(std::make_shared<DynamicSceneGraph>()),
      occupancy_grid_(std::make_shared<OccupancyGrid>()) {}

void Input::setSearchFeatures(const std::vector<FeatureVector>& features) {
  search_features_ = features;
}
void Input::setSearchLabels(const std::vector<Label>& labels) {
  search_labels_ = labels;
}

void Input::setOccupancyGrid(const OccupancyGrid::Ptr& grid) { occupancy_grid_ = grid; }

bool Input::emptySearchData() const {
  return search_features_.empty() && search_labels_.empty();
}

}  // namespace input
}  // namespace hvlm_planner
