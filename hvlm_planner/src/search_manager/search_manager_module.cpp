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
#include "hvlm_planner/search_manager/search_manager_module.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include "hvlm_planner/common/global_info.h"
#include "hvlm_planner/common/pipeline_queues.h"
#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<Module,
                                   SearchManagerModule,
                                   SearchManagerModule::Config>("SearchManagerModule");
}  // namespace

void declare_config(SearchManagerModule::Config& config) {
  using namespace config;
  name("SearchManagerModuleConfig");
  enum_field(config.mode,
             "mode",
             {{SearchManagerModule::Mode::AUTONOMOUS, "autonomous"},
              {SearchManagerModule::Mode::TRIGGERED, "triggered"}});
  field(config.frontier_blacklist_trans_tolerance,
        "frontier_blacklist_trans_tolerance",
        "grid_resolution_multiplier");
  field(config.frontier_blacklist_rot_tolerance,
        "frontier_blacklist_rot_tolerance",
        "degrees");
  field(config.progress_timeout_ns, "progress_timeout_ns");
  field(config.goal_reached_threshold_trans, "goal_reached_threshold_trans", "m");
  field(config.goal_reached_threshold_angle, "goal_reached_threshold_angle", "deg");
  field(config.sinks, "sinks");
}

SearchManagerModule::SearchManagerModule(const Config& config)
    : config(config::checkValid(config)),
      last_distance_to_target_(std::numeric_limits<double>::infinity()),
      last_progress_time_ns_(0),
      sinks_(Sink::instantiate(config.sinks)) {
  LOG(INFO) << "Initialized with:\n" << printInfo();
}

void SearchManagerModule::start() {
  spin_thread_.reset(new std::thread(&SearchManagerModule::spin, this));
  LOG(INFO) << "Search manager module started!";
}

void SearchManagerModule::stopImpl() {
  should_shutdown_ = true;

  if (spin_thread_) {
    VLOG(2) << "Joining spin thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "Stopped spin thread!";
  }
}

void SearchManagerModule::stop() { stopImpl(); }

void SearchManagerModule::save() { LOG(ERROR) << "Save not implemented yet."; }

std::string SearchManagerModule::printInfo() const { return config::toString(config); }

void SearchManagerModule::step(input::LowLevelInput::Ptr& input) { spinOnce(input); }

void SearchManagerModule::spin() {
  bool should_shutdown = false;
  auto& global_info = GlobalInfo::instance();
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
    const auto& search_features = global_info.getSearchFeatures();
    const auto& search_labels = global_info.getSearchLabels();
    if (!search_features.empty()) {
      input->setSearchFeatures(search_features);
    }
    if (!search_labels.empty()) {
      input->setSearchLabels(search_labels);
    }

    if (input->emptySearchData()) {
      VLOG(4) << "Received input with empty search data, skipping search manager spin.";
      continue;
    }
    spinOnce(input);
  }
}

void SearchManagerModule::spinOnce(input::LowLevelInput::Ptr& input) {
  VLOG(5) << "Search manager spinning with input timestamp: " << input->getTimestamp();
  if (!input->getDSG()->hasLayer(spark_dsg::DsgLayers::FRONTIERS)) {
    LOG(WARNING) << "No frontiers layer in DSG, cannot run search manager.";
    return;
  }
  const auto& agent_state = AgentState::instance().getState();
  if (agent_state == State::WAITING) {
    VLOG(3) << "Agent state is WAITING, skipping search manager spin.";
    return;
  }
  bool send_to_low_level = false;
  switch (config.mode) {
    case Mode::AUTONOMOUS:
      send_to_low_level = spinAutonomous(input, agent_state);
      break;

    case Mode::TRIGGERED:
      send_to_low_level = spinTriggered(input, agent_state);
      break;
  }
  Sink::callAll(sinks_,
                input->getTimestamp(),
                frontier_blacklist_,
                AgentState::instance().getTargetFrontier(),
                input->getDSG());
  if (!send_to_low_level) {
    VLOG(4) << "Search manager decided not to send to low level";
    return;
  }
  VLOG(2) << "Search manager decided to send to low level";
  if (!PipelineQueues::instance().low_level_input_queue.push(input)) {
    LOG(WARNING)
        << "Search manager could not push to low level input queue, queue full?";
  }
}

bool SearchManagerModule::spinTriggered(input::LowLevelInput::Ptr& input,
                                        const State& agent_state) {
  auto dsg = input->getDSG();
  auto& agent_state_instance = AgentState::instance();
  const auto& target_pose = agent_state_instance.getTargetPosition();
  const auto& target_frontier = agent_state_instance.getTargetFrontier();
  const bool initialized_target = agent_state_instance.initializedTarget();

  VLOG(3) << "Agent state: " << agent_state_instance.stateString()
          << ", initialized target: " << initialized_target;

  switch (agent_state) {
    case State::READY: {
      auto& trigger_queue = PipelineQueues::instance().trigger_planner_queue;
      if (!trigger_queue.poll()) {
        VLOG(4) << "No trigger to plan, skipping triggered search manager spin.";
        return false;
      }
      trigger_queue.pop();
      if (initialized_target) {
        VLOG(1) << "Blacklisted frontiers: " << frontier_blacklist_.size();
        for (const auto& blacklisted_pos : frontier_blacklist_) {
          VLOG(1) << "    Blacklisted frontier at t = "
                  << blacklisted_pos.translation().transpose() << " and q: "
                  << Eigen::Quaterniond(blacklisted_pos.linear()).coeffs().transpose();
        }
        frontier_blacklist_.insert(target_frontier);
        VLOG(1) << "Blacklisting frontier at t = "
                << target_frontier.translation().transpose() << " and q: "
                << Eigen::Quaterniond(target_frontier.linear()).coeffs().transpose();
        VLOG(1) << "Total blacklisted frontiers: " << frontier_blacklist_.size();
      }
      removeBlacklistedFrontiers(dsg, input->getOccupancyGrid()->resolution);
      if (dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS).numNodes() == 0) {
        LOG(INFO) << "No frontiers in READY mode, triggering homing.";
        if (agent_state_instance.initializedHome()) {
          auto& homing_queue = PipelineQueues::instance().homing_queue;
          if (!homing_queue.push(input)) {
            LOG(WARNING)
                << "Search manager could not push to homing queue, queue full?";
          }
        } else {
          LOG(WARNING) << "Home position not initialized, cannot trigger homing.";
        }
        return false;
      }
      last_distance_to_target_ = std::numeric_limits<double>::infinity();
      last_progress_time_ns_ = input->getTimestamp();
      VLOG(1)
          << "Trigger received, sending to low level and setting state to SEARCHING.";
      return true;
      break;
    }
    case State::SEARCHING: {
      if (!initialized_target) {
        LOG(ERROR) << "Agent state is SEARCHING but target not initialized, skipping "
                      "progress monitoring.";
        return true;
      }

      // Get agent position from DSG
      const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
      if (!agent_layer_id) {
        LOG(ERROR)
            << "No agents layer key in DSG, cannot monitor progress towards target.";
        return false;
      }
      const auto& prefix = GlobalInfo::instance().getRobotPrefix();
      const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
      if (!agent_layer) {
        LOG(ERROR) << "No agents layer in DSG, cannot monitor progress towards target.";
        return false;
      }
      spark_dsg::NodeSymbol pgmo_key(prefix.key, agent_layer->numNodes() - 1);
      const auto& agent_node = agent_layer->getNode(pgmo_key);
      Pose agent_pose = Pose::Identity();
      agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                                .world_R_body.toRotationMatrix();
      agent_pose.translation() =
          agent_node.attributes<spark_dsg::NodeAttributes>().position;

      const auto distance_to_target =
          math_utils::poseDifference(agent_pose, target_pose);

      // 1. Check for progress towards target and if no progress for some time,
      // blacklist frontier and set state to READY
      if (last_distance_to_target_ > distance_to_target.translation) {
        last_progress_time_ns_ = input->getTimestamp();
        last_distance_to_target_ = distance_to_target.translation;
      }
      if (input->getTimestamp() - last_progress_time_ns_ > config.progress_timeout_ns) {
        LOG(INFO) << "No progress towards target for " << config.progress_timeout_ns
                  << " ns, blacklisting frontier setting state to READY.";
        agent_state_instance.setState(State::READY);
        return false;
      }

      // 2. Check if we have reached the target and if so, blacklist frontier and set
      // state to READY
      if (distance_to_target.angles.z() <= config.goal_reached_threshold_angle &&
          distance_to_target.translation <= config.goal_reached_threshold_trans) {
        LOG(INFO) << "Reached target frontier, setting state to READY.";
        agent_state_instance.setState(State::READY);
        return false;
      }
      break;
    }
    default:
      return false;
  }

  return false;
}

bool SearchManagerModule::spinAutonomous(input::LowLevelInput::Ptr& input,
                                         const State& agent_state) {
  // Get agent state info and DSG
  auto dsg = input->getDSG();
  auto& agent_state_instance = AgentState::instance();
  const auto& target_pose = agent_state_instance.getTargetPosition();
  const auto& target_frontier = agent_state_instance.getTargetFrontier();
  const bool initialized_target = agent_state_instance.initializedTarget();

  VLOG(3) << "Agent state: " << agent_state_instance.stateString()
          << ", initialized target: " << initialized_target;

  switch (agent_state) {
    case State::READY: {
      if (initialized_target) {
        frontier_blacklist_.insert(target_frontier);
      }
      removeBlacklistedFrontiers(dsg, input->getOccupancyGrid()->resolution);
      if (dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS).numNodes() == 0) {
        LOG(INFO) << "No frontiers in READY mode, triggering homing.";
        if (agent_state_instance.initializedHome()) {
          auto& homing_queue = PipelineQueues::instance().homing_queue;
          if (!homing_queue.push(input)) {
            LOG(WARNING)
                << "Search manager could not push to homing queue, queue full?";
          }
        } else {
          LOG(WARNING) << "Home position not initialized, cannot trigger homing.";
        }
        return false;
      }
      last_distance_to_target_ = std::numeric_limits<double>::infinity();
      last_progress_time_ns_ = input->getTimestamp();
      VLOG(1)
          << "Trigger received, sending to low level and setting state to SEARCHING.";
      return true;
      break;
    }
    case State::SEARCHING: {
      if (!initialized_target) {
        LOG(ERROR) << "Agent state is SEARCHING but target not initialized, skipping "
                      "progress monitoring.";
        return true;
      }

      // Get agent position from DSG
      const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
      if (!agent_layer_id) {
        LOG(ERROR)
            << "No agents layer key in DSG, cannot monitor progress towards target.";
        return false;
      }
      const auto& prefix = GlobalInfo::instance().getRobotPrefix();
      const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
      if (!agent_layer) {
        LOG(ERROR) << "No agents layer in DSG, cannot monitor progress towards target.";
        return false;
      }
      spark_dsg::NodeSymbol pgmo_key(prefix.key, agent_layer->numNodes() - 1);
      const auto& agent_node = agent_layer->getNode(pgmo_key);
      Pose agent_pose = Pose::Identity();
      agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                                .world_R_body.toRotationMatrix();
      agent_pose.translation() =
          agent_node.attributes<spark_dsg::NodeAttributes>().position;

      const auto distance_to_target =
          math_utils::poseDifference(agent_pose, target_pose);

      // 1. Check for progress towards target and if no progress for some time,
      // blacklist frontier and set state to READY
      if (last_distance_to_target_ > distance_to_target.translation) {
        last_progress_time_ns_ = input->getTimestamp();
        last_distance_to_target_ = distance_to_target.translation;
      }
      if (input->getTimestamp() - last_progress_time_ns_ > config.progress_timeout_ns) {
        LOG(INFO) << "No progress towards target for " << config.progress_timeout_ns
                  << " ns, blacklisting frontier setting state to READY.";
        agent_state_instance.setState(State::READY);
        return false;
      }

      // 2. Check if we have reached the target and if so, blacklist frontier and set
      // state to READY
      if (distance_to_target.angles.z() <= config.goal_reached_threshold_angle &&
          distance_to_target.translation <= config.goal_reached_threshold_trans) {
        LOG(INFO) << "Reached target frontier, setting state to READY.";
        agent_state_instance.setState(State::READY);
        return false;
      }
      break;
    }
    default:
      return false;
  }

  return false;
}

void SearchManagerModule::removeBlacklistedFrontiers(
    spark_dsg::DynamicSceneGraph::Ptr& dsg, const double& grid_resolution) {
  const double tolerance = config.frontier_blacklist_trans_tolerance * grid_resolution;
  const auto& frontiers_layer = dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS);
  std::vector<NodeId> to_remove;
  for (const auto& [node_id, node] : frontiers_layer.nodes()) {
    const auto& pos = node->attributes<spark_dsg::NodeAttributes>().position;
    const auto& dir = node->attributes<spark_dsg::GlobalFrontierNodeAttributes>()
                          .direction;  // Eigen::Vector2d (encodes yaw of frontier)
    Pose frontier_pose = Pose::Identity();
    frontier_pose.translation() = pos;
    frontier_pose.linear() =
        math_utils::yawToRotationMatrix<double>(std::atan2(dir.y(), dir.x()));

    for (const auto& blacklisted_pos : frontier_blacklist_) {
      const auto difference =
          math_utils::poseDifference(frontier_pose, blacklisted_pos, true);
      if (difference.translation <= tolerance &&
          difference.angles.z() <= config.frontier_blacklist_rot_tolerance) {
        to_remove.push_back(node_id);
        break;
      }
    }
  }

  for (const auto& node_id : to_remove) {
    dsg->removeNode(node_id);
  }
}

}  // namespace hvlm_planner
