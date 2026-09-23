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
#include "hvlm_planner/low_level/low_level_module.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "hvlm_planner/common/agent_state.h"
#include "hvlm_planner/common/global_info.h"
#include "hvlm_planner/common/pipeline_queues.h"
#include "hvlm_planner/utils/math.h"

namespace hvlm_planner {

namespace {

static const auto registration =
    config::RegistrationWithConfig<Module, LowLevelModule, LowLevelModule::Config>(
        "LowLevelModule");
}  // namespace

void declare_config(LowLevelModule::Config& config) {
  using namespace config;
  name("LowLevelModuleConfig");
  field(config.semantic_search, "semantic_search");
  field(config.node_search, "node_search");
  field(config.object_search, "object_search");
  field(config.find_room_search, "find_room_search");
  field(config.planner, "planner");
  field(config.a_star_config, "a_star_config");
  field(config.bounding_box, "bounding_box");
  field(config.sinks, "sinks");
  field(config.wait_time_homing_us, "wait_time_homing_us", "microseconds");
  field(config.simulation, "simulation");
  field(config.direct_goal_mode, "direct_goal_mode");
  field(config.collision_avoidance_start, "collision_avoidance_start");
  field(config.collision_avoidance_goal, "collision_avoidance_goal");
}

LowLevelModule::LowLevelModule(const Config& config)
    : config(config::checkValid(config)),
      semantic_search_(config.semantic_search.create()),
      node_search_(config.node_search.create()),
      object_search_(config.object_search.create()),
      find_room_search_(config.find_room_search.create()),
      planner_(config.planner.create()),
      a_star_planner_(std::make_unique<AStarPlanner>(config.a_star_config)),
      sinks_(Sink::instantiate(config.sinks)) {
  ExplorationBoundingBox::instance().setConfig(this->config.bounding_box);
  assert(semantic_search_ && "LowLevelModule: Failed to create SemanticSearch module.");
  assert(node_search_ && "LowLevelModule: Failed to create NodeSearch module.");
  assert(object_search_ && "LowLevelModule: Failed to create ObjectSearch module.");
  assert(find_room_search_ &&
         "LowLevelModule: Failed to create FindRoomSearch module.");
  assert(planner_ && "LowLevelModule: Failed to create Planner module.");
  LOG(INFO) << "Initialized with:\n" << printInfo();
}

LowLevelModule::~LowLevelModule() { stopImpl(); }

void LowLevelModule::start() {
  spin_thread_.reset(new std::thread(&LowLevelModule::spin, this));
  homing_thread_.reset(new std::thread(&LowLevelModule::spinHoming, this));
  LOG(INFO) << "Low-level planning module started!";
}

void LowLevelModule::stopImpl() {
  should_shutdown_ = true;

  if (spin_thread_) {
    VLOG(2) << "Joining spin thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "Stopped spin thread!";
  }
  if (homing_thread_) {
    VLOG(2) << "Joining homing thread and stopping";
    homing_thread_->join();
    homing_thread_.reset();
    VLOG(2) << "Stopped homing thread!";
  }
}

void LowLevelModule::stop() { stopImpl(); }

void LowLevelModule::save() { LOG(ERROR) << "Save not implemented yet."; }

std::string LowLevelModule::printInfo() const {
  return config::toString(config) + "\n" + Sink::printSinks(sinks_);
}

void LowLevelModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    auto& queue = PipelineQueues::instance().low_level_input_queue;
    bool has_data = queue.poll();
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinOnce(queue.pop());
  }
}

void LowLevelModule::spinHoming() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    auto& queue = PipelineQueues::instance().homing_queue;
    bool has_data = queue.poll(config.wait_time_homing_us);
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinHomingOnce(queue.pop());
  }
}

const LowLevelOutput::Ptr LowLevelModule::step(const input::LLPInput::Ptr& input) {
  return spinOnce(input);
}

void LowLevelModule::addSink(const Sink::Ptr& sink) {
  if (sink) {
    sinks_.push_back(sink);
  }
}

const LowLevelOutput::Ptr LowLevelModule::spinOnce(const input::LLPInput::Ptr& input) {
  auto& agent_state = AgentState::instance();
  if (!input || !input->base_input) {
    LOG(ERROR) << "Input is null";
    agent_state.setState(State::READY);
    return nullptr;
  }

  VLOG(4) << "LowLevelModule processing input packet";
  auto low_level_output = std::make_shared<LowLevelOutput>();
  low_level_output->timestamp_ns = input->base_input->getTimestamp();
  const auto& dsg = input->base_input->getDSG();
  const auto& occupancy_grid = input->base_input->getOccupancyGrid();

  SearchInput::Ptr search_input = std::make_shared<SearchInput>();
  search_input->fromLowLevelInput(input->base_input);
  search_input->direct_goal_mode = config.direct_goal_mode;
  SearchOutput output;

  switch (input->trigger->mode) {
    case input::LLPMode::EXPLORE: {
      VLOG(2) << "LLP Mode: EXPLORE";
      bool exploration_success =
          explorationSearch(search_input, dsg, occupancy_grid, output);
      low_level_output->fromSearchOutput(output);
      if (!exploration_success) {
        if ((config.simulation || config.direct_goal_mode) && output.valid_goal) {
          return finishWithDirectGoal(
              input->base_input, output, low_level_output, State::SEARCHING);
        }
        Sink::callAll(sinks_, low_level_output);
        return low_level_output;
      }
      break;
    }
    case input::LLPMode::EXPLORE_ROOM: {
      VLOG(2) << "LLP Mode: EXPLORE_ROOM";
      search_input->room_to_explore = input->trigger->target_room_id;
      bool exploration_success =
          explorationSearch(search_input, dsg, occupancy_grid, output);
      low_level_output->fromSearchOutput(output);
      if (!exploration_success) {
        if ((config.simulation || config.direct_goal_mode) && output.valid_goal) {
          return finishWithDirectGoal(
              input->base_input, output, low_level_output, State::SEARCHING);
        }
        Sink::callAll(sinks_, low_level_output);
        return low_level_output;
      }
      break;
    }
    case input::LLPMode::FIND_ROOM: {
      VLOG(2) << "LLP Mode: FIND_ROOM";
      search_input->transition_prompt_embeddings =
          input->trigger->transition_prompt_embeddings;
      search_input->transition_prompts = input->trigger->transition_prompts;
      search_input->floorplan_room_labels = input->trigger->floorplan_room_labels;
      search_input->floorplan_progress_scores =
          input->trigger->floorplan_progress_scores;
      search_input->floorplan_room_embeddings =
          input->trigger->floorplan_room_embeddings;
      output = find_room_search_->search(search_input, dsg, occupancy_grid);
      low_level_output->fromSearchOutput(output);
      if (!output.success) {
        LOG(ERROR) << "Find-room search failed.";
        if ((config.simulation || config.direct_goal_mode) && output.valid_goal) {
          return finishWithDirectGoal(
              input->base_input, output, low_level_output, State::SEARCHING);
        }
        agent_state.setState(State::READY);
        Sink::callAll(sinks_, low_level_output);
        return low_level_output;
      }
      break;
    }
    case input::LLPMode::GO_TO_OBJECTS_BASIC: {
      VLOG(2) << "LLP Mode: GO_TO_OBJECTS_BASIC";
      // TODO: Implement go to multiple objects in one search, for now just go to first
      // object in list
      if (input->trigger->target_object_ids.empty()) {
        LOG(ERROR)
            << "GO_TO_OBJECTS_BASIC mode triggered but no target object IDs provided.";
        agent_state.setState(State::READY);
        return low_level_output;
      }
      spark_dsg::NodeId target_object_id = input->trigger->target_object_ids.back();
      if (!dsg->hasNode(target_object_id)) {
        LOG(ERROR) << "Target object ID " << target_object_id
                   << " not found in DSG, cannot run GO_TO_OBJECTS search.";
        if (input->trigger->target_object_ids.size() > 1) {
          agent_state.setState(State::READY);
          LOG(INFO)
              << "GO_TO_OBJECTS_BASIC mode triggered with multiple target object IDs, "
                 "but first ID not found in DSG. "
                 "Skipping to next object ID.";
        } else {
          agent_state.setState(State::READY);
          LOG(ERROR)
              << "GO_TO_OBJECTS_BASIC mode triggered but no valid target object IDs "
                 "found in DSG. Setting state to READY.";
        }
        return low_level_output;
      }
      const auto& target_object_node = dsg->getNode(target_object_id);
      const auto& target_object_attrs =
          target_object_node.attributes<spark_dsg::NodeAttributes>();
      Pose target_position = Pose::Identity();
      target_position.translation() = target_object_attrs.position;
      const auto nav_parent = target_object_node.getParent();
      if (nav_parent) {
        const auto& nav_parent_node = dsg->getNode(nav_parent.value());
        const auto nav_parent_position =
            nav_parent_node.attributes<spark_dsg::NodeAttributes>().position;
        // Compute yaw to face the object from the parent node
        Eigen::Vector2d to_object =
            target_position.translation().head<2>() - nav_parent_position.head<2>();
        double yaw = std::atan2(to_object.y(), to_object.x());
        target_position.linear() = math_utils::yawToRotationMatrix<double>(yaw);
      }
      output = goToNodeSearch(
          input->base_input, target_object_id, target_position, low_level_output);
      if (!output.success) {
        LOG(ERROR) << "Node search for GO_TO_OBJECTS failed.";
        if (config.simulation || config.direct_goal_mode) {
          LOG(INFO) << "Semantic search did not find a path, but found a valid goal. "
                       "Using direct goal execution.";
          output.valid_goal = true;
          output.goal_pose = target_position;
          if (nav_parent) {
            output.goal_pose.translation() =
                dsg->getNode(nav_parent.value())
                    .attributes<spark_dsg::NodeAttributes>()
                    .position;
          }
          return finishWithDirectGoal(
              input->base_input, output, low_level_output, State::SEARCHING);
        }
        if (input->trigger->mode == input::LLPMode::GO_TO_OBJECTS_BASIC &&
            input->trigger->target_object_ids.size() > 1) {
          LOG(INFO) << "GO_TO_OBJECTS mode triggered with multiple target object IDs, "
                       "but search for first ID failed. "
                       "Skipping to next object ID.";
          agent_state.setState(State::READY);
        } else {
          LOG(ERROR) << "GO_TO_OBJECTS mode triggered but search for target object "
                        "failed. Setting state to STUCK.";
          agent_state.setState(State::STUCK);
        }
        return low_level_output;
      }

      break;
    }
    case input::LLPMode::GO_TO_OBJECTS: {
      VLOG(2) << "LLP Mode: GO_TO_OBJECTS";
      if (input->trigger->target_object_ids.empty()) {
        LOG(ERROR) << "GO_TO_OBJECTS mode triggered but no target object IDs provided.";
        agent_state.setState(State::READY);
        return low_level_output;
      }
      search_input->start_id = input->trigger->target_object_ids.back();
      output = object_search_->search(search_input, dsg, occupancy_grid);
      if (!output.success) {
        LOG(ERROR) << "Object search for GO_TO_OBJECTS failed.";
        if ((config.simulation || config.direct_goal_mode) && output.valid_goal) {
          LOG(INFO) << "Semantic search did not find a path, but found a valid goal. "
                       "Using direct goal execution.";
          return finishWithDirectGoal(
              input->base_input, output, low_level_output, State::SEARCHING);
        }
        agent_state.setState(State::READY);
        return low_level_output;
      }
      low_level_output->fromSearchOutput(output);
      break;
    }
    case input::LLPMode::NONE: {
      VLOG(2) << "LLP Mode: NONE";
      Pose current_pose = Pose::Identity();
      if (!getCurrentAgentPose(dsg, current_pose)) {
        LOG(ERROR) << "Could not get current agent pose for stop path.";
        agent_state.setState(State::READY);
        return low_level_output;
      }

      low_level_output->goal_pose = current_pose;
      low_level_output->start_pose = current_pose;
      low_level_output->waypoints = {current_pose, current_pose};
      low_level_output->success = true;
      low_level_output->valid_goal = true;
      agent_state.setTargetPosition(current_pose);
      agent_state.setTargetFrontier(current_pose);
      agent_state.setInitializedTarget();
      agent_state.setState(State::READY);
      Sink::callAll(sinks_, low_level_output);
      return low_level_output;
    }
    default:
      LOG(WARNING) << "Unknown LLP mode";
      agent_state.setState(State::READY);
      return low_level_output;
      break;
  }

  if (config.direct_goal_mode) {
    return finishWithDirectGoal(
        input->base_input, output, low_level_output, State::SEARCHING);
  }

  // Colllision avoidance
  if (config.collision_avoidance_start) {
    LOG(INFO) << "Performing collision avoidance at start.";
    const auto clearance_segment =
        AStarPlanner::ClearanceSegment::START_TO_TRAVERSABILITY;
    if (!collisionAvoidance(input->base_input->getOccupancyGrid(),
                            output.start_pose,
                            output.nav_poses.front(),
                            output.extra_poses_start,
                            clearance_segment)) {
      LOG(ERROR) << "Collision avoidance at start failed, setting state to STUCK.";
      agent_state.setState(State::STUCK);
      return low_level_output;
    }
  }
  if (config.collision_avoidance_goal && !output.nav_graph_goal_used) {
    LOG(INFO) << "Performing collision avoidance at goal.";
    const auto clearance_segment =
        AStarPlanner::ClearanceSegment::TRAVERSABILITY_TO_GOAL;
    if (!collisionAvoidance(input->base_input->getOccupancyGrid(),
                            output.nav_poses.back(),
                            output.goal_pose,
                            output.extra_poses_end,
                            clearance_segment)) {
      LOG(ERROR) << "Collision avoidance at goal failed, using nav graph goal pose.";
      output.goal = output.nav_nodes.back();
      output.goal_pose = output.nav_poses.back();
      output.nav_graph_goal = output.nav_nodes.back();
      output.nav_graph_goal_used = true;
    }
  } else if (config.collision_avoidance_goal) {
    LOG(INFO) << "Skipping collision avoidance at goal because nav graph goal is "
                 "already being used.";
  }
  // Plan path (interpolation + possible collision checking)
  if (!planner_->plan(
          output, input->base_input->getOccupancyGrid(), low_level_output->waypoints)) {
    LOG(ERROR) << "Planner failed to generate a path.";
    if (input->trigger->mode == input::LLPMode::GO_TO_OBJECTS &&
        input->trigger->target_object_ids.size() > 1) {
      LOG(INFO) << "GO_TO_OBJECTS mode triggered with multiple target object IDs, but "
                   "planner failed for first ID. "
                   "Skipping to next object ID.";
      agent_state.setState(State::READY);
    } else {
      LOG(ERROR)
          << "Planner failed to generate a path for target. Setting state to STUCK.";
      agent_state.setState(State::STUCK);
    }
    low_level_output->success = false;
    return low_level_output;
  }
  low_level_output->success = true;
  updateAgentState(low_level_output, State::SEARCHING);
  Sink::callAll(sinks_, low_level_output);
  return low_level_output;
}

LowLevelOutput::Ptr LowLevelModule::spinHomingOnce(const input::Input::Ptr& input) {
  auto low_level_output = std::make_shared<LowLevelOutput>();
  low_level_output->timestamp_ns = input->getTimestamp();
  if (!input) {
    LOG(ERROR) << "Input is null";
    return low_level_output;
  }
  auto& agent_state = AgentState::instance();
  if (!agent_state.initializedHome()) {
    LOG(ERROR) << "Home position not initialized, cannot run homing search.";
    return low_level_output;
  }
  VLOG(4) << "LowLevelModule homing spin processing input packet";
  const auto& home_position = agent_state.getHomePosition();
  const auto& home_node_id = agent_state.getHomeNodeId();

  // Find path to home node through node search
  const auto output =
      goToNodeSearch(input, home_node_id, home_position, low_level_output);
  if (!output.success) {
    LOG(ERROR) << "Node search for homing failed.";
    agent_state.setState(State::STUCK);
    return low_level_output;
  }

  if (config.direct_goal_mode) {
    return finishWithDirectGoal(input, output, low_level_output, State::HOMING);
  }

  // Plan path (interpolation + possible collision checking)
  if (!planner_->plan(output, input->getOccupancyGrid(), low_level_output->waypoints)) {
    LOG(ERROR) << "Homing planner failed to generate a path.";
    low_level_output->success = false;
    agent_state.setState(State::STUCK);
    return low_level_output;
  }
  low_level_output->success = true;
  VLOG(2) << "Homing search and planning successful, updating agent state to HOMING.";
  updateAgentState(low_level_output, State::HOMING);
  Sink::callAll(sinks_, low_level_output);
  return low_level_output;
}

bool LowLevelModule::getCurrentAgentPose(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                         Pose& agent_pose) const {
  if (!dsg) {
    LOG(ERROR) << "Cannot get current agent pose from null DSG.";
    return false;
  }

  const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  if (!agent_layer_id) {
    LOG(ERROR) << "No agents layer key in DSG, cannot determine current pose.";
    return false;
  }

  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
  if (!agent_layer || agent_layer->numNodes() == 0) {
    LOG(ERROR)
        << "No agent layer or agent nodes in DSG, cannot determine current pose.";
    return false;
  }

  spark_dsg::NodeSymbol pgmo_key(prefix.key, agent_layer->numNodes() - 1);
  if (!agent_layer->hasNode(pgmo_key)) {
    LOG(ERROR) << "Latest agent node " << pgmo_key << " not found in DSG.";
    return false;
  }

  const auto& agent_node = agent_layer->getNode(pgmo_key);
  agent_pose = Pose::Identity();
  agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                            .world_R_body.toRotationMatrix();
  agent_pose.translation() =
      agent_node.attributes<spark_dsg::NodeAttributes>().position;
  return true;
}

LowLevelOutput::Ptr LowLevelModule::finishWithDirectGoal(
    const input::Input::Ptr& input,
    const SearchOutput& output,
    const LowLevelOutput::Ptr& low_level_output,
    const State& state) const {
  if (!low_level_output) {
    LOG(ERROR) << "LowLevelOutput is null, cannot finish direct goal execution.";
    return low_level_output;
  }

  low_level_output->fromSearchOutput(output);
  low_level_output->nav_graph_nodes.clear();
  low_level_output->nav_graph_poses.clear();
  low_level_output->waypoints.clear();

  Pose start_pose = output.start_pose;
  if (input) {
    getCurrentAgentPose(input->getDSG(), start_pose);
  }

  if ((start_pose.translation() - output.goal_pose.translation()).norm() > 1e-6 ||
      !start_pose.linear().isApprox(output.goal_pose.linear())) {
    low_level_output->waypoints.push_back(start_pose);
  }
  low_level_output->waypoints.push_back(output.goal_pose);
  low_level_output->success = config.direct_goal_mode ? false : true;

  updateAgentState(low_level_output, state);
  Sink::callAll(sinks_, low_level_output);
  return low_level_output;
}

bool LowLevelModule::collisionAvoidance(
    const input::OccupancyGrid::Ptr& occupancy_grid,
    const Pose& start_pose,
    const Pose& end_pose,
    std::vector<Pose>& nav_graph_poses,
    AStarPlanner::ClearanceSegment clearance_segment) const {
  if ((start_pose.translation() - end_pose.translation()).norm() <
      occupancy_grid->resolution) {
    LOG(INFO)
        << "Collision avoidance: start position is close to end position, no need "
           "for collision avoidance.";
    return true;
  }
  if (!a_star_planner_->plan(
          occupancy_grid, start_pose, end_pose, nav_graph_poses, clearance_segment) ||
      nav_graph_poses.empty()) {
    LOG(ERROR) << "Collision avoidance: A* planner failed to find a path.";
    return false;
  }
  // Insert the collision avoidance path into the nav graph poses, either at the
  // beginning (if start) or end (if goal)
  LOG(INFO) << "Collision avoidance: Added " << nav_graph_poses.size()
            << " waypoints to avoid collision.";
  return true;
}

bool LowLevelModule::explorationSearch(const SearchInput::Ptr& search_input,
                                       const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                       const input::OccupancyGrid::Ptr& occupancy_grid,
                                       SearchOutput& output) const {
  auto& agent_state = AgentState::instance();
  output = semantic_search_->search(search_input, dsg, occupancy_grid);
  if (!output.success) {
    LOG(ERROR) << "Semantic search failed to find a path for room exploration.";
    if ((config.simulation || config.direct_goal_mode) && output.valid_goal) {
      LOG(INFO) << "Semantic search did not find a path, but found a valid goal. "
                   "Using direct goal execution.";
      agent_state.setTargetPosition(output.goal_pose);
      agent_state.setTargetFrontier(output.goal_pose);
      agent_state.setInitializedTarget();
      agent_state.setState(State::SEARCHING);
      return false;
    }
    LOG(ERROR) << "Semantic search failed to find a path and did not find a valid "
                  "goal for room exploration. Setting state to STUCK.";
    agent_state.setState(State::STUCK);
    return false;
  }
  return true;
}

const SearchOutput LowLevelModule::goToNodeSearch(
    const input::Input::Ptr& input,
    const spark_dsg::NodeId& target_node_id,
    const Pose& target_position,
    LowLevelOutput::Ptr& low_level_output) const {
  const auto& dsg = input->getDSG();
  if (!dsg->hasNode(target_node_id)) {
    LOG(ERROR) << "Target node ID " << target_node_id
               << " not found in DSG, cannot run node search.";
    return SearchOutput{};
  }

  const auto& target_node = dsg->getNode(target_node_id);
  const auto target_node_parent = target_node.getParent();
  if (!target_node_parent) {
    LOG(ERROR) << "Target node has no parent!";
    return SearchOutput{};
  }

  // Plan homing path to home position
  SearchInput::Ptr search_input = std::make_shared<SearchInput>();
  search_input->child_start_id = target_node_id;
  search_input->child_start_pose = target_position;
  search_input->start_id = target_node_parent.value();
  search_input->direct_goal_mode = config.direct_goal_mode;
  const auto output =
      node_search_->search(search_input, input->getDSG(), input->getOccupancyGrid());
  if (!output.success) {
    LOG(ERROR) << "Node search failed to find a path to node "
               << spark_dsg::NodeSymbol(target_node_id).str();
    return SearchOutput{};
  }
  low_level_output->fromSearchOutput(output);
  return output;
}

void LowLevelModule::updateAgentState(const LowLevelOutput::Ptr& output,
                                      const State& state) const {
  if (!output) {
    LOG(ERROR) << "LowLevelOutput is null, cannot update agent state.";
    return;
  }

  if (config.direct_goal_mode) {
    if (!output->success && !output->valid_goal) {
      LOG(ERROR) << "LowLevelOutput indicates failure, not updating agent state.";
      return;
    }
  } else {
    if ((!config.simulation && !output->success) ||
        (config.simulation && !output->valid_goal && !output->success)) {
      LOG(ERROR) << "LowLevelOutput indicates failure, not updating agent state.";
      return;
    }
  }
  auto& agent_state = AgentState::instance();
  agent_state.setTargetPosition(output->waypoints.back());
  agent_state.setTargetFrontier(output->has_selected_frontier_pose
                                    ? output->selected_frontier_pose
                                    : output->goal_pose);
  agent_state.setInitializedTarget();
  agent_state.setState(state);
  VLOG(2) << "Setting agent target position to ("
          << output->waypoints.back().translation().x() << ", "
          << output->waypoints.back().translation().y() << ", "
          << output->waypoints.back().translation().z() << ") and target frontier to ("
          << agent_state.getTargetFrontier().translation().x() << ", "
          << agent_state.getTargetFrontier().translation().y() << ", "
          << agent_state.getTargetFrontier().translation().z() << ") and state to "
          << agent_state.stateString();
}

}  // namespace hvlm_planner
