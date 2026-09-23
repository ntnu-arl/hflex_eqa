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
#include "hflex_eqa/search_manager/baseline_manager_module.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include "hflex_eqa/common/global_info.h"
#include "hflex_eqa/common/pipeline_queues.h"
#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {

namespace {

static const auto registration =
    config::RegistrationWithConfig<Module,
                                   BaselineManagerModule,
                                   BaselineManagerModule::Config>(
        "BaselineManagerModule");
}  // namespace

void declare_config(BaselineManagerModule::Config& config) {
  using namespace config;
  name("BaselineManagerModuleConfig");
  field(config.frontier_blacklist_trans_tolerance,
        "frontier_blacklist_trans_tolerance",
        "grid_resolution_multiplier");
  field(config.frontier_blacklist_rot_tolerance,
        "frontier_blacklist_rot_tolerance",
        "degrees");
  field(config.progress_timeout_ns, "progress_timeout_ns");
  field(config.goal_reached_threshold_trans, "goal_reached_threshold_trans", "m");
  field(config.goal_reached_threshold_angle, "goal_reached_threshold_angle", "deg");
  field(config.send_graph_to_eqa, "send_graph_to_eqa");
  field(config.send_graph_to_hlp, "send_graph_to_hlp");
  field(config.eqa_callable, "eqa_callable");
  field(config.hlp_callable, "hlp_callable");
  field(config.log_eqa_calls, "log_eqa_calls");
  field(config.log_hlp_calls, "log_hlp_calls");
  field(config.triggered, "triggered");
  field(config.sinks, "sinks");
  field(config.monitor_sinks, "monitor_sinks");
}

BaselineManagerModule::BaselineManagerModule(const Config& config)
    : config(config::checkValid(config)),
      last_distance_to_target_(std::numeric_limits<double>::infinity()),
      last_progress_time_ns_(0),
      sinks_(managers::Sink::instantiate(config.sinks)),
      monitor_sinks_(managers::MonitorSink::instantiate(config.monitor_sinks)) {
  LOG(INFO) << "Initialized with:\n" << printInfo();
}

void BaselineManagerModule::start() {
  hlp_spin_thread_.reset(new std::thread(&BaselineManagerModule::spinHLP, this));
  eqa_spin_thread_.reset(new std::thread(&BaselineManagerModule::spinEQA, this));
  spin_thread_.reset(new std::thread(&BaselineManagerModule::spinLLP, this));
  LOG(INFO) << "Search manager module started!";
}

void BaselineManagerModule::stopImpl() {
  should_shutdown_ = true;

  if (spin_thread_) {
    VLOG(2) << "Joining spin thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "Stopped spin thread!";
  }
  if (eqa_spin_thread_) {
    VLOG(2) << "Joining EQA spin thread and stopping";
    eqa_spin_thread_->join();
    eqa_spin_thread_.reset();
    VLOG(2) << "Stopped EQA spin thread!";
  }
  if (hlp_spin_thread_) {
    VLOG(2) << "Joining HLP spin thread and stopping";
    hlp_spin_thread_->join();
    hlp_spin_thread_.reset();
    VLOG(2) << "Stopped HLP spin thread!";
  }
}

void BaselineManagerModule::stop() { stopImpl(); }

void BaselineManagerModule::save() { LOG(ERROR) << "Save not implemented yet."; }

std::string BaselineManagerModule::printInfo() const {
  return config::toString(config);
}

void BaselineManagerModule::stepEQA(input::EQAInput::Ptr input) { spinOnceEQA(input); }

void BaselineManagerModule::stepHLP(input::HLPInput::Ptr input) { spinOnceHLP(input); }

void BaselineManagerModule::stepLLP(input::LLPInput::Ptr input) { spinOnceLLP(input); }

void BaselineManagerModule::monitorStateAndProgress(input::Input::Ptr& input) {
  auto& agent_state_instance = AgentState::instance();
  if (agent_state_instance.getState() != State::SEARCHING) {
    return;
  }

  auto dsg = input->getDSG();
  const bool initialized_target = agent_state_instance.initializedTarget();
  const auto& target_pose = agent_state_instance.getTargetPosition();
  // const auto& target_frontier = agent_state_instance.getTargetFrontier();

  if (!initialized_target) {
    LOG(ERROR) << "Agent state is SEARCHING but target not initialized, skipping "
                  "progress monitoring.";
    return;
  }

  // Get agent position from DSG
  const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!agent_layer_id) {
    LOG(ERROR) << "No agents layer key in DSG, cannot monitor progress towards target.";
    return;
  }
  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
  if (!agent_layer) {
    LOG(ERROR) << "No agents layer in DSG, cannot monitor progress towards target.";
    return;
  }
  spark_dsg::NodeSymbol pgmo_key(prefix.key, agent_layer->numNodes() - 1);
  const auto& agent_node = agent_layer->getNode(pgmo_key);
  Pose agent_pose = Pose::Identity();
  agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                            .world_R_body.toRotationMatrix();
  agent_pose.translation() =
      agent_node.attributes<spark_dsg::NodeAttributes>().position;

  const auto distance_to_target = math_utils::poseDifference(agent_pose, target_pose);

  // 1. Check for progress towards target and if no progress for some time,
  // blacklist frontier and set state to READY
  if (last_distance_to_target_ > distance_to_target.translation) {
    last_progress_time_ns_ = input->getTimestamp();
    last_distance_to_target_ = distance_to_target.translation;
  }
  bool set_ready = false;
  bool reached_goal = false;
  if (input->getTimestamp() - last_progress_time_ns_ > config.progress_timeout_ns) {
    LOG(INFO) << "No progress towards target for " << config.progress_timeout_ns
              << " ns, blacklisting frontier";
    set_ready = true;
  } else if (distance_to_target.angles.z() <= config.goal_reached_threshold_angle &&
             distance_to_target.translation <= config.goal_reached_threshold_trans) {
    // 2. Check if we have reached the target and if so, blacklist frontier and set
    // state to READY
    LOG(INFO) << "Reached target frontier";
    set_ready = true;
    reached_goal = true;
  }

  if (set_ready && objects_to_visit_.size() > 0) {
    visited_object_views_[objects_to_visit_.back()] = std::nullopt;
    if (reached_goal) {
      visited_object_views_[objects_to_visit_.back()] = input->getView();
    }
    objects_to_visit_.pop_back();
    if (objects_to_visit_.size() > 0) {
      LOG(INFO) << "Going to next object to visit";
      agent_state_instance.setState(State::LLP);
      auto llp_trigger = std::make_shared<input::TriggerLLP>();
      llp_trigger->timestamp_ns = input->getTimestamp();
      llp_trigger->mode = input::LLPMode::GO_TO_OBJECTS;
      llp_trigger->target_object_ids = objects_to_visit_;
      llp_trigger->valid = true;
      // Add new agent node before spinning LLP
      auto current_dsg = current_input_->getDSG();

      spark_dsg::NodeSymbol next_key(prefix.key, agent_layer->numNodes());
      auto agent_attrs =
          agent_node.attributes<spark_dsg::AgentNodeAttributes>().clone();
      current_dsg->emplaceNode(
          agent_layer_id->layer, next_key, std::move(agent_attrs), prefix.key);
      current_input_->setOccupancyGrid(input->getOccupancyGrid());
      current_input_->setTimestamp(input->getTimestamp());

      spinOnceLLP(
          input::LLPInput::Ptr(new input::LLPInput(current_input_, llp_trigger)));
    } else {
      LOG(INFO) << "No more objects to visit, setting state to READY.";
      agent_state_instance.setState(State::READY);
    }
  } else if (set_ready) {
    agent_state_instance.setState(State::READY);
  }
}

void BaselineManagerModule::spinEQA() {
  bool should_shutdown = false;
  auto& global_info = GlobalInfo::instance();
  auto& state = AgentState::instance();
  while (!should_shutdown) {
    auto& queue = PipelineQueues::instance().main_input_queue;
    bool has_data = queue.poll();
    if (global_info.force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }
    if (!has_data) {
      continue;
    }
    auto input = queue.pop();
    monitorStateAndProgress(input);
    auto& trigger_queue = PipelineQueues::instance().trigger_step_queue;
    bool state_ready = state.getState() == State::READY;
    bool has_trigger = trigger_queue.poll();
    if (state_ready) {
      if (!config.triggered || has_trigger) {
        ++iteration_;
      }
    }
    managers::MonitorSink::callAll(
        monitor_sinks_, input->getTimestamp(), state.getState(), iteration_);

    if (!state_ready) {
      VLOG(3) << "Agent state is not READY, skipping EQA spin.";
      continue;
    }
    if (config.triggered) {
      if (!has_trigger) {
        VLOG(3) << "No trigger to plan, skipping EQA spin.";
        continue;
      }
      trigger_queue.pop();
    }
    LOG(INFO) << "New input received, sending to EQA and setting state to HLP.";
    input->setSearchFeatures(global_info.getSearchFeatures());
    input->setSearchLabels(global_info.getSearchLabels());
    current_input_ = input;
    objects_to_visit_.clear();

    state.setState(State::HLP);
    spinOnceEQA(
        input::EQAInput::Ptr(new input::EQAInput(input, visited_object_views_)));
    visited_object_views_.clear();
  }
}

void BaselineManagerModule::spinHLP() {
  bool should_shutdown = false;
  auto& global_info = GlobalInfo::instance();
  auto& state = AgentState::instance();
  while (!should_shutdown) {
    if (global_info.force_shutdown() || !current_input_) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }
    if (!current_input_) {
      continue;
    }
    if (state.getState() != State::HLP) {
      VLOG(3) << "Agent state is not HLP, skipping HLP spin.";
      continue;
    }

    auto& hlp_queue = PipelineQueues::instance().hlp_input_queue;

    bool hlp_triggered = hlp_queue.poll();
    if (!hlp_triggered) {
      VLOG(3) << "No trigger to plan, skipping HLP spin.";
      continue;
    }
    auto trigger = hlp_queue.pop();
    if (!trigger->valid) {
      LOG(WARNING) << "Received invalid HLP trigger, skipping.";
      continue;
    }
    state.setState(State::LLP);
    spinOnceHLP(input::HLPInput::Ptr(new input::HLPInput(current_input_, trigger)));
  }
}

void BaselineManagerModule::spinLLP() {
  bool should_shutdown = false;
  auto& global_info = GlobalInfo::instance();
  auto& state = AgentState::instance();
  while (!should_shutdown) {
    if (global_info.force_shutdown() || !current_input_) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }
    if (!current_input_) {
      continue;
    }
    if (state.getState() != State::LLP) {
      VLOG(3) << "Agent state is not LLP, skipping LLP spin.";
      continue;
    }

    auto& llp_queue = PipelineQueues::instance().trigger_llp_queue;

    bool llp_triggered = llp_queue.poll();
    if (!llp_triggered) {
      VLOG(3) << "No trigger to plan, skipping LLP spin.";
      continue;
    }
    auto trigger = llp_queue.pop();
    if (!trigger->valid) {
      LOG(WARNING) << "Received invalid LLP trigger, skipping.";
      state.setState(State::READY);
      continue;
    }
    spinOnceLLP(input::LLPInput::Ptr(new input::LLPInput(current_input_, trigger)));
  }
}

void BaselineManagerModule::spinOnceHLP(input::HLPInput::Ptr input) {
  if (checkMissionComplete(input)) {
    LOG(INFO) << "Mission complete, setting state to FINISHED.";
    LOG(INFO) << "Question: " << GlobalInfo::instance().getQuestion();
    LOG(INFO) << "Reasoning: " << input->trigger->reasoning;
    LOG(INFO) << "Answer: " << input->trigger->answer;
    AgentState::instance().setState(State::FINISHED);

    bool called = false;
    if (function_map_.count(config.finish_callable) > 0) {
      auto* callable = dynamic_cast<FunctionWrapper<void(const bool)>*>(
          function_map_[config.finish_callable].get());
      if (callable) {
        callable->call(true);
        called = true;
      }
    }
    if (!called) {
      LOG(ERROR) << config.finish_callable
                 << " is not set or has wrong signature, cannot call finish.";
    }
    return;
  }
  if (config.log_eqa_calls) {
    LOG(INFO) << "EQA called with question: " << GlobalInfo::instance().getQuestion();
    LOG(INFO) << "EQA call reasoning: " << input->trigger->reasoning;
    LOG(INFO) << "EQA call confidence: " << input->trigger->confidence;
  }

  auto dsg = input->base_input->getDSG();
  removeBlacklistedFrontiers(dsg,
                             frontier_blacklist_,
                             config.frontier_blacklist_trans_tolerance,
                             config.frontier_blacklist_rot_tolerance,
                             input->base_input->getOccupancyGrid()->resolution);
  // Check if call_hlp_ is set before calling to avoid potential null pointer
  // dereference
  bool called = false;
  if (function_map_.count(config.hlp_callable) > 0) {
    auto* callable =
        dynamic_cast<FunctionWrapper<void(const input::HLPInput::Ptr&, bool)>*>(
            function_map_[config.hlp_callable].get());
    if (callable) {
      callable->call(input, config.send_graph_to_hlp);
      called = true;
    }
  }
  if (!called) {
    LOG(ERROR) << config.hlp_callable
               << " is not set or has wrong signature, cannot call HLP.";
  }
}

void BaselineManagerModule::spinOnceEQA(input::EQAInput::Ptr input) {
  auto dsg = input->base_input->getDSG();
  removeBlacklistedFrontiers(dsg,
                             frontier_blacklist_,
                             config.frontier_blacklist_trans_tolerance,
                             config.frontier_blacklist_rot_tolerance,
                             input->base_input->getOccupancyGrid()->resolution);
  // Check if call_eqa_ is set before calling to avoid potential null pointer
  // dereference
  bool called = false;
  if (function_map_.count(config.eqa_callable) > 0) {
    auto* callable =
        dynamic_cast<FunctionWrapper<void(const input::EQAInput::Ptr&, bool)>*>(
            function_map_[config.eqa_callable].get());

    if (callable) {
      callable->call(input, config.send_graph_to_eqa);
      called = true;
    }
  }
  if (!called) {
    LOG(ERROR) << config.eqa_callable
               << " is not set or has wrong signature, cannot call EQA.";
  }
}

void BaselineManagerModule::spinOnceLLP(input::LLPInput::Ptr input) {
  if (config.log_hlp_calls) {
    LOG(INFO) << "HLP called with question: " << GlobalInfo::instance().getQuestion();
    LOG(INFO) << "HLP call mode: "
              << input::llp_modes_to_string.at(input->trigger->mode);
    LOG(INFO) << "HLP call target room id: " << input->trigger->target_room_id;
    LOG(INFO) << "HLP call target object ids: ";
    for (const auto& obj_id : input->trigger->target_object_ids) {
      LOG(INFO) << "  " << obj_id;
    }
    LOG(INFO) << "HLP call reasoning: " << input->trigger->reasoning;
    LOG(INFO) << "HLP call confidence: " << input->trigger->confidence;
  }
  current_llp_mode_ = input->trigger->mode;
  objects_to_visit_ = input->trigger->target_object_ids;
  auto dsg = input->base_input->getDSG();
  auto& agent_state_instance = AgentState::instance();
  // const auto& target_pose = agent_state_instance.getTargetPosition();
  const auto& target_frontier = agent_state_instance.getTargetFrontier();
  const bool initialized_target = agent_state_instance.initializedTarget();

  if (initialized_target) {
    frontier_blacklist_.insert(target_frontier);
  }
  removeBlacklistedFrontiers(dsg,
                             frontier_blacklist_,
                             config.frontier_blacklist_trans_tolerance,
                             config.frontier_blacklist_rot_tolerance,
                             input->base_input->getOccupancyGrid()->resolution);
  if (dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS).numNodes() == 0 &&
      (current_llp_mode_ == input::LLPMode::EXPLORE ||
       current_llp_mode_ == input::LLPMode::EXPLORE_ROOM)) {
    LOG(INFO) << "No frontiers in READY mode, triggering homing.";
    if (agent_state_instance.initializedHome()) {
      auto& homing_queue = PipelineQueues::instance().homing_queue;
      if (!homing_queue.push(input->base_input)) {
        LOG(WARNING) << "Search manager could not push to homing queue, queue full?";
      }
    } else {
      LOG(WARNING) << "Home position not initialized, cannot trigger homing.";
    }
    return;
  }
  last_distance_to_target_ = std::numeric_limits<double>::infinity();
  last_progress_time_ns_ = input->base_input->getTimestamp();
  VLOG(1) << "Trigger received, sending to low level and setting state to SEARCHING.";
  managers::Sink::callAll(sinks_,
                          input->base_input->getTimestamp(),
                          frontier_blacklist_,
                          AgentState::instance().getTargetFrontier(),
                          dsg);
  if (!PipelineQueues::instance().low_level_input_queue.push(input)) {
    LOG(WARNING)
        << "Search manager could not push to low level input queue, queue full?";
  }
}

bool BaselineManagerModule::checkMissionComplete(const input::HLPInput::Ptr& input) {
  // For simplicity, we just check if the answer contains the question as a
  return input->trigger->answered || AgentState::instance().getState() == State::HOMING;
}

}  // namespace hflex_eqa
