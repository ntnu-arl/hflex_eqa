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
#include <config_utilities/virtual_config.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "hvlm_planner/common/agent_state.h"
#include "hvlm_planner/common/exploration_bounding_box.h"
#include "hvlm_planner/common/input.h"
#include "hvlm_planner/common/module.h"
#include "hvlm_planner/common/output_sink.h"
#include "hvlm_planner/low_level/a_star.h"
#include "hvlm_planner/low_level/low_level_output.h"
#include "hvlm_planner/low_level/planner.h"
#include "hvlm_planner/low_level/semantic_search.h"

namespace hvlm_planner {

class LowLevelModule : public Module {
 public:
  using Ptr = std::shared_ptr<LowLevelModule>;
  using Sink = OutputSink<LowLevelOutput::Ptr>;

  struct Config {
    config::VirtualConfig<SemanticSearch> semantic_search;
    config::VirtualConfig<SemanticSearch> node_search;
    config::VirtualConfig<SemanticSearch> object_search;
    config::VirtualConfig<SemanticSearch> find_room_search;
    config::VirtualConfig<Planner> planner;
    AStarPlanner::Config a_star_config;
    ExplorationBoundingBoxConfig bounding_box;
    std::vector<Sink::Factory> sinks;
    uint32_t wait_time_homing_us = 500000;  // 500 ms
    bool simulation = false;
    bool direct_goal_mode = false;
    bool collision_avoidance_start = false;
    bool collision_avoidance_goal = false;
  } const config;

  explicit LowLevelModule(const Config& config);

  virtual ~LowLevelModule();

  LowLevelModule& operator=(const LowLevelModule& other) = delete;

  void start() override;

  void stop() override;

  void save() override;

  std::string printInfo() const override;

  void spin();

  void spinHoming();

  const LowLevelOutput::Ptr step(const input::LLPInput::Ptr& input);

  void stopImpl();

  void addSink(const Sink::Ptr& sink);

 protected:
  virtual const LowLevelOutput::Ptr spinOnce(const input::LLPInput::Ptr& input);

  virtual LowLevelOutput::Ptr spinHomingOnce(const input::Input::Ptr& input);

  bool collisionAvoidance(const input::OccupancyGrid::Ptr& occupancy_grid,
                          const Pose& start_pose,
                          const Pose& end_pose,
                          std::vector<Pose>& nav_graph_poses,
                          AStarPlanner::ClearanceSegment clearance_segment) const;

  bool getCurrentAgentPose(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                           Pose& agent_pose) const;

  LowLevelOutput::Ptr finishWithDirectGoal(const input::Input::Ptr& input,
                                           const SearchOutput& output,
                                           const LowLevelOutput::Ptr& low_level_output,
                                           const State& state) const;

  bool explorationSearch(const SearchInput::Ptr& search_input,
                         const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                         const input::OccupancyGrid::Ptr& occupancy_grid,
                         SearchOutput& output) const;

  const SearchOutput goToNodeSearch(const input::Input::Ptr& input,
                                    const spark_dsg::NodeId& target_node_id,
                                    const Pose& target_position,
                                    LowLevelOutput::Ptr& low_level_output) const;

  void updateAgentState(const LowLevelOutput::Ptr& output, const State& state) const;

  SemanticSearch::Ptr semantic_search_;
  SemanticSearch::Ptr node_search_;
  SemanticSearch::Ptr object_search_;
  SemanticSearch::Ptr find_room_search_;
  Planner::Ptr planner_;
  AStarPlanner::Ptr a_star_planner_;

  Sink::List sinks_;
  std::unique_ptr<std::thread> spin_thread_;
  std::unique_ptr<std::thread> homing_thread_;
  std::atomic<bool> should_shutdown_{false};
  std::mutex mutex_;
};

void declare_config(LowLevelModule::Config& conf);

}  // namespace hvlm_planner
