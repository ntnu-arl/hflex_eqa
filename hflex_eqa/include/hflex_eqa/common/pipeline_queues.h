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

#include <memory>
#include <variant>

#include "hflex_eqa/common/input.h"
#include "hflex_eqa/common/message_queue.h"

namespace hflex_eqa {

namespace input {
class Input;
}  // namespace input

class PipelineQueues {
 public:
  ~PipelineQueues();
  static PipelineQueues& instance();
  void clear();

  //! Main input queue
  input::MessageQueue<input::Input::Ptr> main_input_queue;
  //! Low level input queue
  input::MessageQueue<input::LLPInput::Ptr> low_level_input_queue;
  //! Homing queue
  input::MessageQueue<input::Input::Ptr> homing_queue;
  //! HLP input queue
  input::MessageQueue<input::TriggerHLP::Ptr> hlp_input_queue;
  //! Trigger LLP queue
  input::MessageQueue<input::TriggerLLP::Ptr> trigger_llp_queue;
  //! Trigger step queue
  input::MessageQueue<std::monostate> trigger_step_queue;
  //! Trigger next object queue
  input::MessageQueue<std::monostate> trigger_next_object_queue;

 private:
  PipelineQueues();

  inline static std::unique_ptr<PipelineQueues> s_instance_;
};

}  // namespace hflex_eqa
