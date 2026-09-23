/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hflex_eqa/common/global_info.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <spark_dsg/labelspace.h>

#include <filesystem>
#include <fstream>

#include "hflex_eqa/common/semantic_color_map.h"

namespace hflex_eqa {

using ColorMapPtr = std::shared_ptr<SemanticColorMap>;

decltype(GlobalInfo::instance_) GlobalInfo::instance_;

struct LabelNameConversion {
  using YamlList = std::vector<std::map<std::string, std::string>>;
  using SourceMap = std::map<uint32_t, std::string>;

  static YamlList toIntermediate(const SourceMap& other, std::string&) {
    YamlList to_return;
    for (const auto& kv_pair : other) {
      std::map<std::string, std::string> value_map{
          {"label", std::to_string(kv_pair.first)}, {"name", kv_pair.second}};
      to_return.push_back(value_map);
    }

    return to_return;
  }

  static void fromIntermediate(const YamlList& other,
                               SourceMap& value,
                               std::string& error) {
    value.clear();
    for (const auto& value_map : other) {
      if (!value_map.count("name")) {
        error = "invalid format! missing key 'name'";
        break;
      }

      if (!value_map.count("label")) {
        error = "invalid format! missing key 'label'";
        break;
      }

      value[std::stoi(value_map.at("label"))] = value_map.at("name");
    }
  }
};

void declare_config(FrameConfig& frames) {
  using namespace config;
  name("FrameConfig");
  field(frames.robot, "robot_frame");
  field(frames.odom, "odom_frame");
  field(frames.map, "map_frame");
}

void declare_config(PipelineConfig& config) {
  using namespace config;
  name("PipelineConfig");
  field(config.frames, "frames", false);
  field(config.label_space, "label_space", false);
  field<LabelNameConversion>(config.label_names, "label_names");
  field(config.question, "question");
}

GlobalInfo::GlobalInfo() : force_shutdown_(false) {}

void GlobalInfo::initFromConfig(const PipelineConfig& config, int robot_id) {
  config_ = config::checkValid(config);
  robot_prefix_ = RobotPrefixConfig(robot_id);

  if (!config_.label_space.label_remap_filepath.empty()) {
    label_remapper_ = LabelRemapper(config_.label_space.label_remap_filepath);
  }

  if (!config_.label_space.colormap_filepath.empty()) {
    label_colormap_ = SemanticColorMap::fromCsv(config_.label_space.colormap_filepath);
  }

  if (label_colormap_) {
    VLOG(3) << "Loaded label space colors:" << std::endl << *label_colormap_;
  }
  for (const auto& kv : config_.label_names) {
    label_name_to_id_map_[kv.second] = kv.first;
  }
}

GlobalInfo& GlobalInfo::instance() {
  if (!instance_) {
    instance_.reset(new GlobalInfo());
  }

  return *instance_;
}

GlobalInfo& GlobalInfo::init(const PipelineConfig& config, int robot_id) {
  auto& curr = instance();
  curr.initFromConfig(config, robot_id);
  return curr;
}

void GlobalInfo::reset() { instance_.reset(new GlobalInfo()); }

void GlobalInfo::setForceShutdown(bool force_shutdown) {
  force_shutdown_ = force_shutdown;
}

bool GlobalInfo::force_shutdown() const { return force_shutdown_; }

const PipelineConfig& GlobalInfo::getConfig() const { return config_; }

const FrameConfig& GlobalInfo::getFrames() const { return config_.frames; }

const RobotPrefixConfig& GlobalInfo::getRobotPrefix() const { return robot_prefix_; }

const std::map<uint32_t, std::string>& GlobalInfo::getLabelToNameMap() const {
  return config_.label_names;
}

bool GlobalInfo::getFirstFreeLabelId(uint32_t& label_id) const {
  for (uint32_t id = 0; id < static_cast<uint32_t>(config_.label_space.total_labels);
       ++id) {
    if (config_.label_names.count(id) == 0) {
      continue;
    }
    if (config_.label_names.at(id) == "free") {
      label_id = id;
      return true;
    }
  }
  return false;
}

bool GlobalInfo::addNewLabels(const std::vector<std::string>& new_labels) {
  uint32_t next_label_id;
  if (!getFirstFreeLabelId(next_label_id)) {
    LOG(WARNING) << "No free label IDs available to add new labels.";
    return false;
  }
  if (next_label_id + new_labels.size() >
      static_cast<uint32_t>(config_.label_space.total_labels)) {
    LOG(WARNING) << "Not enough free label IDs available to add all new labels.";
    return false;
  }
  for (const auto& label_name : new_labels) {
    if (std::any_of(config_.label_names.begin(),
                    config_.label_names.end(),
                    [&](const auto& pair) { return pair.second == label_name; })) {
      continue;
    }
    config_.label_names[next_label_id] = label_name;
    label_name_to_id_map_[label_name] = next_label_id;
    ++next_label_id;
  }
  return true;
}

size_t GlobalInfo::getNumActiveLabels() const {
  size_t count = 0;
  for (const auto& kv : config_.label_names) {
    if (kv.second != "free" && kv.second != "unknown") {
      ++count;
    }
  }
  return count;
}

const LabelSpaceConfig& GlobalInfo::getLabelSpaceConfig() const {
  return config_.label_space;
}

size_t GlobalInfo::getTotalLabels() const { return config_.label_space.total_labels; }

const LabelRemapper& GlobalInfo::getLabelRemapper() const { return label_remapper_; }

const SemanticColorMap* GlobalInfo::getSemanticColorMap() const {
  return label_colormap_.get();
}

std::ostream& operator<<(std::ostream& out, const GlobalInfo& config) {
  const auto& config_data = config.getConfig();
  out << "================================ PipelineConfig "
         "================================\n";
  out << "Frames: \n";
  out << "  Robot frame: " << config_data.frames.robot << "\n";
  out << "  Odom frame: " << config_data.frames.odom << "\n";
  out << "  Map frame: " << config_data.frames.map << "\n";
  out << "Question: " << config_data.question << "\n";
  return out;
}

void GlobalInfo::setQuestion(const std::string& question) {
  config_.question = question;
}

const std::string& GlobalInfo::getQuestion() const { return config_.question; }

}  // namespace hflex_eqa
