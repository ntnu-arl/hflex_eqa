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

#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hvlm_planner/common/agent_state.h"
#include "hvlm_planner/common/global_info.h"
#include "hvlm_planner/common/input.h"
#include "hvlm_planner/common/module.h"
#include "hvlm_planner/common/output_sink.h"
#include "hvlm_planner/common/types.h"
#include "hvlm_planner/search_manager/sinks.h"
#include "hvlm_planner/utils/frontier_utils.h"

namespace hvlm_planner {

class BaselineManagerModule : public Module {
 public:
  using Ptr = std::shared_ptr<BaselineManagerModule>;
  struct Config {
    double frontier_blacklist_trans_tolerance = 1.0;
    double frontier_blacklist_rot_tolerance = 1.0;
    uint64_t progress_timeout_ns = 30e9;         // 30 seconds
    double goal_reached_threshold_trans = 0.5;   // meters
    double goal_reached_threshold_angle = 10.0;  // degrees
    bool send_graph_to_eqa = true;
    bool send_graph_to_hlp = false;
    std::string eqa_callable = "call_eqa";
    std::string hlp_callable = "call_hlp";
    std::string finish_callable = "call_finish";
    bool log_eqa_calls = false;
    bool log_hlp_calls = false;
    bool triggered = false;
    std::vector<managers::Sink::Factory> sinks;
    std::vector<managers::MonitorSink::Factory> monitor_sinks;
  } const config;

  explicit BaselineManagerModule(const Config& config);

  virtual ~BaselineManagerModule() = default;

  BaselineManagerModule& operator=(const BaselineManagerModule&) = delete;

  void start() override;

  void stop() override;

  void save() override;

  std::string printInfo() const override;

  void spinHLP();

  void spinEQA();

  void spinLLP();

  void stepHLP(input::HLPInput::Ptr input);

  void stepEQA(input::EQAInput::Ptr input);

  void stepLLP(input::LLPInput::Ptr input);
  void stopImpl();

 protected:
  void spinOnceHLP(input::HLPInput::Ptr input);

  void spinOnceEQA(input::EQAInput::Ptr input);

  void spinOnceLLP(input::LLPInput::Ptr input);

  void monitorStateAndProgress(input::Input::Ptr& input);

  bool checkMissionComplete(const input::HLPInput::Ptr& input);

  input::Input::Ptr current_input_;
  FrontierBlackList frontier_blacklist_;
  double last_distance_to_target_;
  uint64_t last_progress_time_ns_;
  size_t iteration_ = 0;
  std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>> visited_object_views_;
  std::vector<spark_dsg::NodeId> objects_to_visit_;
  input::LLPMode current_llp_mode_ = input::LLPMode::NONE;

  std::unique_ptr<std::thread> spin_thread_;
  std::unique_ptr<std::thread> hlp_spin_thread_;
  std::unique_ptr<std::thread> eqa_spin_thread_;
  std::atomic<bool> should_shutdown_{false};
  std::mutex mutex_;

  managers::Sink::List sinks_;
  managers::MonitorSink::List monitor_sinks_;
};

void declare_config(BaselineManagerModule::Config& config);

}  // namespace hvlm_planner
