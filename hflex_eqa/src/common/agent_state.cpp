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
#include "hflex_eqa/common/agent_state.h"

#include <glog/logging.h>

namespace hflex_eqa {

decltype(AgentState::instance_) AgentState::instance_;

AgentState::AgentState()
    : state_(State::WAITING),
      target_position_(Pose::Identity()),
      target_frontier_(Pose::Identity()),
      initialized_target_(false),
      home_position_(Pose::Identity()),
      initialized_home_(false) {}

AgentState& AgentState::instance() {
  if (!instance_) {
    instance_.reset(new AgentState());
  }
  return *instance_;
}

AgentState& AgentState::init() {
  auto& curr = instance();
  curr.state_ = State::WAITING;
  curr.target_position_ = Pose::Identity();
  curr.target_frontier_ = Pose::Identity();
  curr.initialized_target_ = false;
  curr.home_position_ = Pose::Identity();
  curr.initialized_home_ = false;
  return curr;
}

void AgentState::reset() { instance_.reset(new AgentState()); }

const State& AgentState::getState() const { return state_; }

void AgentState::setState(const State& new_state) { state_ = new_state; }

const Pose& AgentState::getTargetPosition() const { return target_position_; }

void AgentState::setTargetPosition(const Pose& position) {
  target_position_ = position;
}

const Pose& AgentState::getTargetFrontier() const { return target_frontier_; }

void AgentState::setTargetFrontier(const Pose& frontier_pose) {
  target_frontier_ = frontier_pose;
}

bool AgentState::initializedTarget() const { return initialized_target_; }

void AgentState::setInitializedTarget() { initialized_target_ = true; }

void AgentState::trySetHomePosition(const Pose& position,
                                    const spark_dsg::NodeId& node_id) {
  if (!initialized_home_) {
    home_position_ = position;
    home_node_id_ = node_id;
    initialized_home_ = true;
  }
}

const Pose& AgentState::getHomePosition() const { return home_position_; }

bool AgentState::initializedHome() const { return initialized_home_; }

const spark_dsg::NodeId& AgentState::getHomeNodeId() const { return home_node_id_; }

void AgentState::trySetReady() {
  if (state_ == State::WAITING) {
    state_ = State::READY;
    LOG(INFO) << "Agent state set to READY.";
  } else {
    LOG(WARNING) << "Received ready signal, but agent state is " << stateString()
                 << ". Ignoring.";
  }
}

std::string AgentState::stateString() const {
  switch (state_) {
    case State::WAITING:
      return "WAITING";
    case State::READY:
      return "READY";
    case State::SEARCHING:
      return "SEARCHING";
    case State::HOMING:
      return "HOMING";
    default:
      return "UNKNOWN";
  }
}

}  // namespace hflex_eqa
