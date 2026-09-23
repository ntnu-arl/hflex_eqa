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

#include <spark_dsg/scene_graph_types.h>

#include <memory>
#include <unordered_map>

#include "hvlm_planner/common/types.h"

namespace hvlm_planner {

enum class State { WAITING, READY, HLP, LLP, SEARCHING, STUCK, HOMING, FINISHED };

namespace state_utils {
static const std::unordered_map<State, std::string> state_to_string = {
    {State::WAITING, "WAITING"},
    {State::READY, "READY"},
    {State::HLP, "HLP"},
    {State::LLP, "LLP"},
    {State::SEARCHING, "SEARCHING"},
    {State::STUCK, "STUCK"},
    {State::HOMING, "HOMING"},
    {State::FINISHED, "FINISHED"}};

static const std::unordered_map<std::string, State> string_to_state = {
    {"WAITING", State::WAITING},
    {"READY", State::READY},
    {"HLP", State::HLP},
    {"LLP", State::LLP},
    {"SEARCHING", State::SEARCHING},
    {"STUCK", State::STUCK},
    {"HOMING", State::HOMING},
    {"FINISHED", State::FINISHED}};
}  // namespace state_utils

class AgentState {
 public:
  static AgentState& instance();

  static AgentState& init();
  static void reset();

  static bool initialized() { return instance_ != nullptr; }

  const State& getState() const;

  void setState(const State& new_state);

  const Pose& getTargetPosition() const;

  void setTargetPosition(const Pose& position);

  const Pose& getTargetFrontier() const;

  void setTargetFrontier(const Pose& frontier_pose);

  bool initializedTarget() const;

  void setInitializedTarget();

  void trySetHomePosition(const Pose& position, const spark_dsg::NodeId& node_id);

  const Pose& getHomePosition() const;

  const spark_dsg::NodeId& getHomeNodeId() const;

  bool initializedHome() const;

  void trySetReady();

  std::string stateString() const;

 private:
  AgentState();

  static std::unique_ptr<AgentState> instance_;
  State state_;
  Pose target_position_;
  Pose target_frontier_;
  bool initialized_target_;
  Pose home_position_;
  spark_dsg::NodeId home_node_id_;
  bool initialized_home_;
};

}  // namespace hvlm_planner
