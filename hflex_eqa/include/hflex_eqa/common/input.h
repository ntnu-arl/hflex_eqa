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
#include <spark_dsg/dynamic_scene_graph.h>

#include <Eigen/Core>
#include <map>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

#include "hflex_eqa/common/types.h"

using namespace spark_dsg;

namespace hflex_eqa {
namespace input {

struct OccupancyGrid {
  using Ptr = std::shared_ptr<OccupancyGrid>;

  std::unordered_map<GridIndex, CellState, GridIndexHash> grid;
  float resolution = 0.0f;
  struct GridBounds {
    int min_row = std::numeric_limits<int>::max();
    int max_row = std::numeric_limits<int>::min();
    int min_col = std::numeric_limits<int>::max();
    int max_col = std::numeric_limits<int>::min();
    size_t width = 0;
    size_t height = 0;

    void update(const int& r, const int& c);
  } bounds;

  OccupancyGrid() = default;
  bool empty() const { return grid.empty(); }
  void clear();
  GridIndex worldToGrid(const Point2f& p) const;
  Point2f gridToWorld(const GridIndex& idx) const;
  bool isFree(const Point2f& p) const;
  bool isFree(const GridIndex& idx) const;
  OccupancyGrid::Ptr clone() const;
};

class Input {
 public:
  using Ptr = std::shared_ptr<Input>;
  Input(const uint64_t& timestamp_ns,
        const DynamicSceneGraph::Ptr& dsg,
        const OccupancyGrid::Ptr& occupancy_grid,
        const cv::Mat& view,
        const std::vector<FeatureVector>& search_features,
        const std::vector<Label>& search_labels);
  Input(const uint64_t& timestamp_ns,
        const DynamicSceneGraph::Ptr& dsg,
        const OccupancyGrid::Ptr& occupancy_grid,
        const cv::Mat& view);
  Input();
  virtual ~Input() = default;

  const uint64_t& getTimestamp() const { return timestamp_ns_; }
  DynamicSceneGraph::Ptr getDSG() const { return dsg_; }
  const OccupancyGrid::Ptr getOccupancyGrid() const { return occupancy_grid_; }
  const cv::Mat getView() const { return view_; }
  const std::vector<FeatureVector>& getSearchFeatures() const {
    return search_features_;
  }
  const std::vector<Label>& getSearchLabels() const { return search_labels_; }
  void setSearchFeatures(const std::vector<FeatureVector>& features);
  void setSearchLabels(const std::vector<Label>& labels);
  void setOccupancyGrid(const OccupancyGrid::Ptr& grid);
  void setTimestamp(const uint64_t& timestamp_ns) { timestamp_ns_ = timestamp_ns; }

  bool emptySearchData() const;

 protected:
  uint64_t timestamp_ns_;
  DynamicSceneGraph::Ptr dsg_;
  OccupancyGrid::Ptr occupancy_grid_;
  cv::Mat view_;
  std::vector<FeatureVector> search_features_;
  std::vector<Label> search_labels_;
};

struct EQAInput {
  using Ptr = std::shared_ptr<EQAInput>;

  Input::Ptr base_input;
  std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>> object_views;

  EQAInput() = default;
  EQAInput(const Input::Ptr& input) : base_input(input) {}
  EQAInput(const Input::Ptr& input,
           const std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>>& views)
      : base_input(input), object_views(views) {}
};

struct TriggerHLP {
  using Ptr = std::shared_ptr<TriggerHLP>;

  uint64_t timestamp_ns;
  bool answered = false;
  std::string answer;
  std::string reasoning;
  float confidence = 0.0f;
  bool valid = false;
};

struct HLPInput {
  using Ptr = std::shared_ptr<HLPInput>;

  Input::Ptr base_input;
  TriggerHLP::Ptr trigger;

  HLPInput() = default;
  HLPInput(const Input::Ptr& input, const TriggerHLP::Ptr& trigger)
      : base_input(input), trigger(trigger) {}
};

enum class LLPMode {
  EXPLORE,
  EXPLORE_ROOM,
  FIND_ROOM,
  GO_TO_OBJECTS_BASIC,
  GO_TO_OBJECTS,
  NONE
};
static const std::map<LLPMode, std::string> llp_modes_to_string = {
    {LLPMode::EXPLORE, "explore"},
    {LLPMode::EXPLORE_ROOM, "explore_room"},
    {LLPMode::FIND_ROOM, "find_room"},
    {LLPMode::GO_TO_OBJECTS_BASIC, "go_to_objects_basic"},
    {LLPMode::GO_TO_OBJECTS, "go_to_objects"},
    {LLPMode::NONE, "none"}};
static const std::map<std::string, LLPMode> string_to_llp_modes = {
    {"explore", LLPMode::EXPLORE},
    {"explore_room", LLPMode::EXPLORE_ROOM},
    {"find_room", LLPMode::FIND_ROOM},
    {"go_to_objects_basic", LLPMode::GO_TO_OBJECTS_BASIC},
    {"go_to_objects", LLPMode::GO_TO_OBJECTS},
    {"none", LLPMode::NONE}};

struct TriggerLLP {
  using Ptr = std::shared_ptr<TriggerLLP>;

  uint64_t timestamp_ns;
  bool answered = false;
  std::string answer;
  LLPMode mode;
  spark_dsg::NodeId target_room_id;
  std::string target_room_label;
  std::vector<spark_dsg::NodeId> target_object_ids;
  std::vector<FeatureVector> transition_prompt_embeddings;
  std::vector<std::string> transition_prompts;
  std::vector<std::string> floorplan_room_labels;
  std::vector<float> floorplan_progress_scores;
  std::vector<FeatureVector> floorplan_room_embeddings;
  std::string reasoning;
  float confidence = 0.0f;
  bool valid = false;
};

struct LLPInput {
  using Ptr = std::shared_ptr<LLPInput>;

  Input::Ptr base_input;
  TriggerLLP::Ptr trigger;

  LLPInput() = default;
  LLPInput(const Input::Ptr& input, const TriggerLLP::Ptr& trigger)
      : base_input(input), trigger(trigger) {}
};

}  // namespace input
}  // namespace hflex_eqa
