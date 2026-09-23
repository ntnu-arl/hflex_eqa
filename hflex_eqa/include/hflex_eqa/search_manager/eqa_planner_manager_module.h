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
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_layer.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hflex_eqa/common/agent_state.h"
#include "hflex_eqa/common/global_info.h"
#include "hflex_eqa/common/input.h"
#include "hflex_eqa/common/module.h"
#include "hflex_eqa/common/output_sink.h"
#include "hflex_eqa/common/robot_prefix_config.h"
#include "hflex_eqa/common/types.h"
#include "hflex_eqa/search_manager/sinks.h"
#include "hflex_eqa/utils/frontier_utils.h"
#include "hflex_eqa/utils/nearest_neighbor.h"

namespace hflex_eqa {

class EQAPlannerManagerModule : public Module {
 public:
  using Ptr = std::shared_ptr<EQAPlannerManagerModule>;
  struct Config {
    double frontier_blacklist_trans_tolerance = 1.0;
    double frontier_blacklist_rot_tolerance = 1.0;
    bool frontier_blacklist_use_position_radius = false;
    double frontier_blacklist_position_radius_m = 1.0;
    uint64_t progress_timeout_ns = 30e9;         // 30 seconds
    double goal_reached_threshold_trans = 0.5;   // meters
    double goal_reached_threshold_angle = 10.0;  // degrees
    std::string eqa_planner_callable = "call_eqa_planner";
    std::string finish_callable = "call_finish";
    bool log_call = false;
    bool triggered = false;
    std::string nav_layer = spark_dsg::DsgLayers::TRAVERSABILITY;
    bool viewpoint_selection = true;
    double min_confidence = 0.7;
    std::vector<managers::Sink::Factory> sinks;
    std::vector<managers::MonitorSink::Factory> monitor_sinks;
  } const config;

  explicit EQAPlannerManagerModule(const Config& config);

  virtual ~EQAPlannerManagerModule() = default;

  EQAPlannerManagerModule& operator=(const EQAPlannerManagerModule&) = delete;

  void start() override;

  void stop() override;

  void save() override;

  std::string printInfo() const override;

  void spinEQAPlanner();

  void spinLLP();

  void stepEQAPlanner(input::EQAInput::Ptr input);

  void stepLLP(input::LLPInput::Ptr input);
  void stopImpl();

 protected:
  void spinOnceEQAPlanner(input::EQAInput::Ptr input);

  void spinOnceLLP(input::LLPInput::Ptr input);

  void monitorStateAndProgress(input::Input::Ptr& input);

  bool checkMissionComplete(const input::LLPInput::Ptr& input);

  bool handleObjectVisitFailure(input::Input::Ptr& input);

  bool handleNextObjectTrigger(input::Input::Ptr& input);

  void finishCurrentObjectVisit(input::Input::Ptr& input, bool reached_goal);

  void blacklistCurrentTargetFrontier(const std::string& reason);

  input::Input::Ptr getLatestInputForStop();

  void removeBlacklistedFrontiers(spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                  double grid_resolution) const;

  std::string getObjectClass(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                             spark_dsg::NodeId object_id) const;

  void addNewAgent(spark_dsg::DynamicSceneGraph::Ptr& current_dsg,
                   const spark_dsg::NodeSymbol& agent_id,
                   const spark_dsg::LayerId& agent_layer_id,
                   const RobotPrefixConfig& prefix,
                   const spark_dsg::SceneGraphNode& agent_node);

  input::Input::Ptr current_input_;
  input::Input::Ptr latest_input_;
  FrontierBlackList frontier_blacklist_;
  double last_distance_to_target_;
  uint64_t last_progress_time_ns_;
  size_t iteration_ = 0;
  std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>> visited_object_views_;
  std::vector<spark_dsg::NodeId> objects_to_visit_;
  input::LLPMode current_llp_mode_ = input::LLPMode::NONE;
  nearest_neighbor::NearestNodeFinder::Ptr node_finder_;

  std::unique_ptr<std::thread> llp_spin_thread_;
  std::unique_ptr<std::thread> eqa_planner_spin_thread_;
  std::atomic<bool> should_shutdown_{false};
  std::mutex mutex_;

  managers::Sink::List sinks_;
  managers::MonitorSink::List monitor_sinks_;
};

void declare_config(EQAPlannerManagerModule::Config& config);

}  // namespace hflex_eqa
