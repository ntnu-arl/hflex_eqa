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
#include <thread>
#include <unordered_set>
#include <vector>

#include "hflex_eqa/common/agent_state.h"
#include "hflex_eqa/common/global_info.h"
#include "hflex_eqa/common/input.h"
#include "hflex_eqa/common/module.h"
#include "hflex_eqa/common/output_sink.h"
#include "hflex_eqa/common/types.h"

namespace hflex_eqa {
class SearchManagerModule : public Module {
 public:
  using Ptr = std::shared_ptr<SearchManagerModule>;
  using FrontierBlackList = std::unordered_set<Pose, TranslationHash, TranslationEqual>;
  using Sink = OutputSink<const uint64_t,
                          const FrontierBlackList&,
                          const Pose&,
                          const spark_dsg::DynamicSceneGraph::Ptr&>;

  enum class Mode { AUTONOMOUS, TRIGGERED };

  struct Config {
    Mode mode = Mode::AUTONOMOUS;
    double frontier_blacklist_trans_tolerance = 1.0;
    double frontier_blacklist_rot_tolerance = 1.0;
    uint64_t progress_timeout_ns = 30e9;         // 30 seconds
    double goal_reached_threshold_trans = 0.5;   // meters
    double goal_reached_threshold_angle = 10.0;  // degrees
    std::vector<Sink::Factory> sinks;
  } const config;

  explicit SearchManagerModule(const Config& config);

  virtual ~SearchManagerModule() = default;

  SearchManagerModule& operator=(const SearchManagerModule&) = delete;

  void start() override;

  void stop() override;

  void save() override;

  std::string printInfo() const override;

  void spin();

  void step(input::LowLevelInput::Ptr& input);

  void stopImpl();

 protected:
  virtual void spinOnce(input::LowLevelInput::Ptr& input);

  virtual bool spinTriggered(input::LowLevelInput::Ptr& input,
                             const State& agent_state);

  virtual bool spinAutonomous(input::LowLevelInput::Ptr& input,
                              const State& agent_state);

  void removeBlacklistedFrontiers(spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                  const double& grid_resolution);

  std::unique_ptr<std::thread> spin_thread_;
  std::atomic<bool> should_shutdown_{false};
  std::mutex mutex_;

  FrontierBlackList frontier_blacklist_;
  double last_distance_to_target_;
  uint64_t last_progress_time_ns_;
  Sink::List sinks_;
};

void declare_config(SearchManagerModule::Config& config);

}  // namespace hflex_eqa
