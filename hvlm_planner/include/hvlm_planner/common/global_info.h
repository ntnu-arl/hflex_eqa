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

#include <config_utilities/config.h>

#include <string>

#include "hvlm_planner/common/label_remapper.h"
#include "hvlm_planner/common/label_space_config.h"
#include "hvlm_planner/common/robot_prefix_config.h"
#include "hvlm_planner/common/semantic_color_map.h"
#include "hvlm_planner/common/types.h"

namespace hvlm_planner {

struct FrameConfig {
  //! Body frame for robot constructing the scene graph (see REP 105)
  std::string robot = "base_link";
  //! Frame odometry estimates are relative to (see REP 105)
  std::string odom = "odom";
  //! Frame that the optimized scene graph is in (see REP 105)
  std::string map = "map";
};

void declare_config(FrameConfig& config);

struct PipelineConfig {
  //! Frame information
  FrameConfig frames;
  //! Closed-set labelspace information
  LabelSpaceConfig label_space;
  //! Human readable category names for the labelspace
  std::map<uint32_t, std::string> label_names;
  //! Question to answer
  std::string question;
};

void declare_config(PipelineConfig& config);

class GlobalInfo {
 public:
  static GlobalInfo& instance();

  static GlobalInfo& init(const PipelineConfig& config, int robot_id = 0);

  // this invalidates any instances (mostly intended for testing)
  static void reset();

  static bool initialized() { return instance_ != nullptr; }

  void setForceShutdown(bool force_shutdown);

  bool force_shutdown() const;

  const PipelineConfig& getConfig() const;

  const FrameConfig& getFrames() const;

  const RobotPrefixConfig& getRobotPrefix() const;

  const std::map<uint32_t, std::string>& getLabelToNameMap() const;

  const std::map<std::string, uint32_t>& getLabelNameToIdMap() const {
    return label_name_to_id_map_;
  }

  bool getFirstFreeLabelId(uint32_t& label_id) const;

  bool addNewLabels(const std::vector<std::string>& new_labels);

  size_t getNumActiveLabels() const;

  const LabelSpaceConfig& getLabelSpaceConfig() const;

  size_t getTotalLabels() const;

  const LabelRemapper& getLabelRemapper() const;

  const SemanticColorMap* getSemanticColorMap() const;

  const std::vector<FeatureVector>& getSearchFeatures() const {
    return search_features_;
  }

  const std::vector<Label>& getSearchLabels() const { return search_labels_; }

  const std::vector<FeatureVector>& getRoomEmbeddings() const {
    return room_embeddings_;
  }

  const std::vector<std::string>& getRoomLabels() const { return room_labels_; }

  const FeatureVector& getQuestionEmbedding() const { return question_embedding_; }

  void setSearchFeatures(const std::vector<FeatureVector>& features) {
    search_features_ = features;
  }

  void setSearchLabels(const std::vector<Label>& labels) { search_labels_ = labels; }

  void setQuestion(const std::string& question);

  void setRoomEmbeddings(const std::vector<FeatureVector>& room_embeddings,
                         const std::vector<std::string>& room_labels) {
    room_embeddings_ = room_embeddings;
    room_labels_ = room_labels;
  };

  void setQuestionEmbedding(const FeatureVector& question_embedding) {
    question_embedding_ = question_embedding;
  }

  const std::string& getQuestion() const;

 private:
  GlobalInfo();

  void initFromConfig(const PipelineConfig& config, int robot_id);

 private:
  static std::unique_ptr<GlobalInfo> instance_;
  std::atomic<bool> force_shutdown_;

  PipelineConfig config_;
  RobotPrefixConfig robot_prefix_;
  LabelRemapper label_remapper_;
  std::shared_ptr<SemanticColorMap> label_colormap_;
  std::vector<FeatureVector> search_features_;
  std::vector<Label> search_labels_;
  std::vector<FeatureVector> room_embeddings_;
  std::vector<std::string> room_labels_;
  FeatureVector question_embedding_;
  std::map<std::string, Label> label_name_to_id_map_;
};

std::ostream& operator<<(std::ostream& out, const GlobalInfo& config);

}  // namespace hvlm_planner
