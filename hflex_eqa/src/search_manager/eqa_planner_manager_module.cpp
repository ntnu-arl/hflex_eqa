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
#include "hflex_eqa/search_manager/eqa_planner_manager_module.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include "hflex_eqa/common/global_info.h"
#include "hflex_eqa/common/pipeline_queues.h"
#include "hflex_eqa/utils/dsg.h"
#include "hflex_eqa/utils/math.h"

namespace hflex_eqa {

namespace {

static const auto registration =
    config::RegistrationWithConfig<Module,
                                   EQAPlannerManagerModule,
                                   EQAPlannerManagerModule::Config>(
        "EQAPlannerManagerModule");
}  // namespace

void declare_config(EQAPlannerManagerModule::Config& config) {
  using namespace config;
  name("EQAPlannerManagerModuleConfig");
  field(config.frontier_blacklist_trans_tolerance,
        "frontier_blacklist_trans_tolerance",
        "grid_resolution_multiplier");
  field(config.frontier_blacklist_rot_tolerance,
        "frontier_blacklist_rot_tolerance",
        "degrees");
  field(config.frontier_blacklist_use_position_radius,
        "frontier_blacklist_use_position_radius");
  field(config.frontier_blacklist_position_radius_m,
        "frontier_blacklist_position_radius_m",
        "m");
  field(config.progress_timeout_ns, "progress_timeout_ns");
  field(config.goal_reached_threshold_trans, "goal_reached_threshold_trans", "m");
  field(config.goal_reached_threshold_angle, "goal_reached_threshold_angle", "deg");
  field(config.eqa_planner_callable, "eqa_planner_callable");
  field(config.log_call, "log_call");
  field(config.triggered, "triggered");
  field(config.nav_layer, "nav_layer");
  field(config.viewpoint_selection, "viewpoint_selection");
  field(config.min_confidence, "min_confidence");
  field(config.sinks, "sinks");
  field(config.monitor_sinks, "monitor_sinks");
  checkCondition(config.frontier_blacklist_position_radius_m >= 0.0,
                 "frontier_blacklist_position_radius_m must be non-negative.");
}

EQAPlannerManagerModule::EQAPlannerManagerModule(const Config& config)
    : config(config::checkValid(config)),
      last_distance_to_target_(std::numeric_limits<double>::infinity()),
      last_progress_time_ns_(0),
      sinks_(managers::Sink::instantiate(config.sinks)),
      monitor_sinks_(managers::MonitorSink::instantiate(config.monitor_sinks)) {
  LOG(INFO) << "Initialized with:\n" << printInfo();
}

void EQAPlannerManagerModule::start() {
  llp_spin_thread_.reset(new std::thread(&EQAPlannerManagerModule::spinLLP, this));
  eqa_planner_spin_thread_.reset(
      new std::thread(&EQAPlannerManagerModule::spinEQAPlanner, this));
  LOG(INFO) << "EQAPlannerManagerModule started.";
}

void EQAPlannerManagerModule::stopImpl() {
  should_shutdown_ = true;
  if (llp_spin_thread_) {
    VLOG(2) << "Joining LLP spin thread and stopping.";
    llp_spin_thread_->join();
    llp_spin_thread_.reset();
    VLOG(2) << "LLP spin thread stopped.";
  }
  if (eqa_planner_spin_thread_) {
    VLOG(2) << "Joining EQAPlanner spin thread and stopping.";
    eqa_planner_spin_thread_->join();
    eqa_planner_spin_thread_.reset();
    VLOG(2) << "EQAPlanner spin thread stopped.";
  }
}

void EQAPlannerManagerModule::stop() { stopImpl(); }

void EQAPlannerManagerModule::save() {
  LOG(ERROR) << "Save not implemented for EQAPlannerManagerModule.";
}

std::string EQAPlannerManagerModule::printInfo() const {
  return config::toString(config);
}

void EQAPlannerManagerModule::stepEQAPlanner(input::EQAInput::Ptr input) {
  spinOnceEQAPlanner(input);
}

void EQAPlannerManagerModule::stepLLP(input::LLPInput::Ptr input) {
  spinOnceLLP(input);
}

void EQAPlannerManagerModule::monitorStateAndProgress(input::Input::Ptr& input) {
  auto& agent_state_instance = AgentState::instance();
  if (agent_state_instance.getState() != State::SEARCHING) {
    return;
  }

  auto dsg = input->getDSG();
  const bool initialized_target = agent_state_instance.initializedTarget();
  const auto& target_pose = agent_state_instance.getTargetPosition();
  const auto& target_frontier = agent_state_instance.getTargetFrontier();

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
  const bool current_target_is_frontier =
      current_llp_mode_ == input::LLPMode::EXPLORE ||
      current_llp_mode_ == input::LLPMode::EXPLORE_ROOM ||
      current_llp_mode_ == input::LLPMode::FIND_ROOM;
  const bool target_is_projected_from_frontier =
      current_target_is_frontier &&
      (target_pose.translation() - target_frontier.translation()).norm() >
          config.goal_reached_threshold_trans;
  const bool reached_target_position =
      distance_to_target.translation <= config.goal_reached_threshold_trans;
  const bool reached_target_orientation =
      distance_to_target.angles.z() <= config.goal_reached_threshold_angle;
  if (input->getTimestamp() - last_progress_time_ns_ > config.progress_timeout_ns) {
    LOG(INFO) << "No progress towards target for " << config.progress_timeout_ns
              << " ns, blacklisting frontier";
    set_ready = true;
  } else if (reached_target_position &&
             (target_is_projected_from_frontier || reached_target_orientation)) {
    // 2. Check if we have reached the target and if so, blacklist frontier and set
    // state to READY
    LOG(INFO) << "Reached target"
              << (target_is_projected_from_frontier
                      ? " at projected traversability point"
                      : " frontier");
    set_ready = true;
    reached_goal = true;
  }

  if (set_ready) {
    blacklistCurrentTargetFrontier(reached_goal ? "reached target frontier"
                                                : "progress timeout");
  }

  if (set_ready && objects_to_visit_.size() > 0) {
    finishCurrentObjectVisit(input, reached_goal);
  } else if (set_ready) {
    agent_state_instance.setState(State::READY);
  }
}

void EQAPlannerManagerModule::blacklistCurrentTargetFrontier(
    const std::string& reason) {
  if (current_llp_mode_ != input::LLPMode::EXPLORE &&
      current_llp_mode_ != input::LLPMode::EXPLORE_ROOM &&
      current_llp_mode_ != input::LLPMode::FIND_ROOM) {
    return;
  }

  auto& agent_state_instance = AgentState::instance();
  if (!agent_state_instance.initializedTarget()) {
    return;
  }

  const auto& target_frontier = agent_state_instance.getTargetFrontier();
  frontier_blacklist_.insert(target_frontier);
  LOG(INFO) << "Blacklisted target frontier after " << reason << " at "
            << target_frontier.translation().transpose()
            << ". Total blacklisted frontiers: " << frontier_blacklist_.size();
}

void EQAPlannerManagerModule::removeBlacklistedFrontiers(
    spark_dsg::DynamicSceneGraph::Ptr& dsg, double grid_resolution) const {
  hflex_eqa::removeBlacklistedFrontiers(
      dsg,
      frontier_blacklist_,
      config.frontier_blacklist_trans_tolerance,
      config.frontier_blacklist_rot_tolerance,
      grid_resolution,
      config.frontier_blacklist_use_position_radius,
      config.frontier_blacklist_position_radius_m);
}

input::Input::Ptr EQAPlannerManagerModule::getLatestInputForStop() {
  auto& main_input_queue = PipelineQueues::instance().main_input_queue;
  {
    std::lock_guard<std::mutex> lock(main_input_queue.mutex);
    if (!main_input_queue.queue.empty()) {
      return main_input_queue.queue.back();
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  return latest_input_ ? latest_input_ : current_input_;
}

std::string EQAPlannerManagerModule::getObjectClass(
    const spark_dsg::DynamicSceneGraph::Ptr& dsg, spark_dsg::NodeId object_id) const {
  if (!dsg || !dsg->hasNode(object_id)) {
    return "UNKNOWN";
  }

  const auto& node = dsg->getNode(object_id);
  const auto attrs = node.tryAttributes<spark_dsg::SemanticNodeAttributes>();
  if (!attrs) {
    return "UNKNOWN";
  }

  const auto& label_to_name = GlobalInfo::instance().getLabelToNameMap();
  const auto iter = label_to_name.find(attrs->semantic_label);
  if (iter == label_to_name.end()) {
    return "UNKNOWN";
  }
  return iter->second;
}

void EQAPlannerManagerModule::finishCurrentObjectVisit(input::Input::Ptr& input,
                                                       bool reached_goal) {
  auto& agent_state_instance = AgentState::instance();
  const auto current_object_id = objects_to_visit_.back();
  visited_object_views_[current_object_id] = std::nullopt;
  if (reached_goal) {
    visited_object_views_[current_object_id] = input->getView().clone();
  }
  objects_to_visit_.pop_back();

  if (objects_to_visit_.empty()) {
    LOG(INFO) << "No more objects to visit, setting state to READY.";
    agent_state_instance.setState(State::READY);
    current_llp_mode_ = input::LLPMode::NONE;
    return;
  }

  LOG(INFO) << "Going to next object to visit";
  auto& global_info = GlobalInfo::instance();
  input->setSearchFeatures(global_info.getSearchFeatures());
  input->setSearchLabels(global_info.getSearchLabels());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_input_ = input;
  }
  agent_state_instance.setState(State::LLP);
  auto llp_trigger = std::make_shared<input::TriggerLLP>();
  llp_trigger->timestamp_ns = input->getTimestamp();
  llp_trigger->mode = config.viewpoint_selection ? input::LLPMode::GO_TO_OBJECTS
                                                 : input::LLPMode::GO_TO_OBJECTS_BASIC;
  llp_trigger->target_object_ids = objects_to_visit_;
  llp_trigger->valid = true;
  spinOnceLLP(input::LLPInput::Ptr(new input::LLPInput(input, llp_trigger)));
}

bool EQAPlannerManagerModule::handleObjectVisitFailure(input::Input::Ptr& input) {
  auto& agent_state_instance = AgentState::instance();
  if (agent_state_instance.getState() != State::READY ||
      (current_llp_mode_ != input::LLPMode::GO_TO_OBJECTS &&
       current_llp_mode_ != input::LLPMode::GO_TO_OBJECTS_BASIC) ||
      objects_to_visit_.empty()) {
    return false;
  }

  LOG(INFO) << "Current object goal failed before reaching the target; "
               "skipping to next object.";
  finishCurrentObjectVisit(input, false);
  return true;
}

bool EQAPlannerManagerModule::handleNextObjectTrigger(input::Input::Ptr& input) {
  auto& queue = PipelineQueues::instance().trigger_next_object_queue;
  if (!queue.poll()) {
    return false;
  }
  queue.clear();

  auto& agent_state_instance = AgentState::instance();
  if (current_llp_mode_ != input::LLPMode::GO_TO_OBJECTS &&
      current_llp_mode_ != input::LLPMode::GO_TO_OBJECTS_BASIC) {
    LOG(WARNING) << "Go to next object requested, but current mode is not "
                    "GO_TO_OBJECTS.";
    return false;
  }
  if (agent_state_instance.getState() != State::SEARCHING) {
    LOG(WARNING) << "Go to next object requested, but robot is not currently "
                    "going to an object.";
    return false;
  }
  if (objects_to_visit_.size() <= 1) {
    LOG(WARNING) << "Go to next object requested, but no more objects are "
                    "available in the queue.";
    return false;
  }

  LOG(INFO) << "Go to next object requested, skipping current object.";
  finishCurrentObjectVisit(input, false);
  return true;
}

void EQAPlannerManagerModule::addNewAgent(
    spark_dsg::DynamicSceneGraph::Ptr& current_dsg,
    const spark_dsg::NodeSymbol& agent_id,
    const spark_dsg::LayerId& agent_layer_id,
    const RobotPrefixConfig& prefix,
    const spark_dsg::SceneGraphNode& agent_node) {
  auto agent_attrs = agent_node.attributes<spark_dsg::AgentNodeAttributes>().clone();
  Eigen::Vector3d agent_position = agent_attrs->position;
  current_dsg->emplaceNode(
      agent_layer_id, agent_id, std::move(agent_attrs), prefix.key);

  // Find parents
  std::vector<spark_dsg::NodeId> nearest_node_ids;
  if (current_dsg->hasLayer(config.nav_layer)) {
    node_finder_.reset(new nearest_neighbor::NearestNodeFinder(
        current_dsg->getLayer(config.nav_layer),
        dsg_utils::getNodeIds(current_dsg->getLayer(config.nav_layer))));
    if (node_finder_->find(agent_position, 1, false, nearest_node_ids)) {
      current_dsg->addOrUpdateEdge(nearest_node_ids.front(), agent_id, nullptr, false);
    }
  }
}

void EQAPlannerManagerModule::spinEQAPlanner() {
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
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_input_ = input;
    }
    if (handleNextObjectTrigger(input)) {
      continue;
    }
    monitorStateAndProgress(input);
    if (handleObjectVisitFailure(input)) {
      continue;
    }
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
    LOG(INFO) << "New input received, sending to EQA and setting state to LLP.";
    input->setSearchFeatures(global_info.getSearchFeatures());
    input->setSearchLabels(global_info.getSearchLabels());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_input_ = input;
    }
    objects_to_visit_.clear();
    state.setState(State::LLP);
    spinOnceEQAPlanner(
        input::EQAInput::Ptr(new input::EQAInput(input, visited_object_views_)));
    visited_object_views_.clear();
  }
}

void EQAPlannerManagerModule::spinLLP() {
  bool should_shutdown = false;
  auto& global_info = GlobalInfo::instance();
  auto& state = AgentState::instance();
  while (!should_shutdown) {
    input::Input::Ptr available_input;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      available_input = current_input_ ? current_input_ : latest_input_;
    }
    if (global_info.force_shutdown() || !available_input) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }
    if (!available_input) {
      continue;
    }

    auto& llp_queue = PipelineQueues::instance().trigger_llp_queue;
    bool llp_triggered = llp_queue.poll();
    if (!llp_triggered) {
      VLOG(3) << "No trigger to plan, skipping LLP spin.";
      continue;
    }

    auto trigger = llp_queue.pop();
    const bool stop_trigger = trigger && trigger->mode == input::LLPMode::NONE;
    if (state.getState() != State::LLP && !stop_trigger) {
      VLOG(3) << "Agent state is not LLP, dropping stale LLP trigger.";
      continue;
    }

    if (!trigger) {
      LOG(WARNING) << "Received null LLP trigger, skipping.";
      state.setState(State::READY);
      continue;
    }
    if (!trigger->valid) {
      LOG(WARNING) << "Received invalid LLP trigger, skipping.";
      state.setState(State::READY);
      continue;
    }
    if (stop_trigger) {
      LOG(INFO) << "Stop trigger received, sending hold path and setting state to "
                   "READY.";
      objects_to_visit_.clear();
      current_llp_mode_ = input::LLPMode::NONE;
      state.setState(State::READY);
    }
    input::Input::Ptr llp_input;
    if (stop_trigger) {
      llp_input = getLatestInputForStop();
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      llp_input = current_input_;
    }
    if (!llp_input) {
      LOG(WARNING) << "No input available for LLP trigger, skipping.";
      state.setState(State::READY);
      continue;
    }
    spinOnceLLP(input::LLPInput::Ptr(new input::LLPInput(llp_input, trigger)));
  }
}

void EQAPlannerManagerModule::spinOnceEQAPlanner(input::EQAInput::Ptr input) {
  auto dsg = input->base_input->getDSG();
  removeBlacklistedFrontiers(dsg, input->base_input->getOccupancyGrid()->resolution);
  // Check if call_eqa_ is set before calling to avoid potential null pointer
  // dereference
  bool called = false;
  if (function_map_.count(config.eqa_planner_callable) > 0) {
    auto* callable = dynamic_cast<FunctionWrapper<void(const input::EQAInput::Ptr&)>*>(
        function_map_[config.eqa_planner_callable].get());

    if (callable) {
      callable->call(input);
      called = true;
    }
  }
  if (!called) {
    LOG(ERROR) << config.eqa_planner_callable
               << " is not set or has wrong signature, cannot call EQA.";
  }
}

void EQAPlannerManagerModule::spinOnceLLP(input::LLPInput::Ptr input) {
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
  if (config.log_call) {
    LOG(INFO) << "EQA Planner called with question: "
              << GlobalInfo::instance().getQuestion();
    LOG(INFO) << "EQA Planner call mode: "
              << input::llp_modes_to_string.at(input->trigger->mode);
    LOG(INFO) << "EQA Planner call target room id: " << input->trigger->target_room_id;
    LOG(INFO) << "EQA Planner call target room label: "
              << input->trigger->target_room_label;
    LOG(INFO) << "EQA Planner call transition prompt embeddings: "
              << input->trigger->transition_prompt_embeddings.size();
    LOG(INFO) << "EQA Planner call floorplan room labels: "
              << input->trigger->floorplan_room_labels.size();
    LOG(INFO) << "EQA Planner call floorplan progress scores: "
              << input->trigger->floorplan_progress_scores.size();
    LOG(INFO) << "EQA Planner call floorplan room embeddings: "
              << input->trigger->floorplan_room_embeddings.size();
    LOG(INFO) << "EQA Planner call target object ids: ";
    const auto log_dsg = input->base_input->getDSG();
    for (const auto& obj_id : input->trigger->target_object_ids) {
      LOG(INFO) << "  " << obj_id << " (class: " << getObjectClass(log_dsg, obj_id)
                << ")";
    }
    LOG(INFO) << "EQA Planner call reasoning: " << input->trigger->reasoning;
    LOG(INFO) << "EQA Planner call confidence: " << input->trigger->confidence;
  }

  current_llp_mode_ = input->trigger->mode;
  objects_to_visit_ = input->trigger->target_object_ids;
  auto dsg = input->base_input->getDSG();
  auto& agent_state_instance = AgentState::instance();
  if (current_llp_mode_ == input::LLPMode::NONE) {
    last_distance_to_target_ = std::numeric_limits<double>::infinity();
    last_progress_time_ns_ = input->base_input->getTimestamp();
    agent_state_instance.setState(State::READY);
    if (!PipelineQueues::instance().low_level_input_queue.push(input)) {
      LOG(WARNING)
          << "Search manager could not push stop input to low level queue, queue full?";
    }
    return;
  }

  const bool initialized_target = agent_state_instance.initializedTarget();

  if (initialized_target) {
    blacklistCurrentTargetFrontier("new low-level trigger");
  }
  removeBlacklistedFrontiers(dsg, input->base_input->getOccupancyGrid()->resolution);
  if (dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS).numNodes() == 0 &&
      current_llp_mode_ == input::LLPMode::EXPLORE) {
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

bool EQAPlannerManagerModule::checkMissionComplete(const input::LLPInput::Ptr& input) {
  // For simplicity, we just check if the answer contains the question as a
  return (input->trigger->answered &&
          input->trigger->confidence > config.min_confidence) ||
         AgentState::instance().getState() == State::HOMING;
}

}  // namespace hflex_eqa
