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
#include "hflex_eqa/common/hflex_eqa_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/printing.h>
#include <config_utilities/settings.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hflex_eqa/common/agent_state.h>

namespace hflex_eqa {

HflexEqaPipeline::HflexEqaPipeline(const PipelineConfig& pipeline_config,
                                   int robot_id,
                                   int config_verbosity)
    : config_verbosity_(config_verbosity) {
  const auto& config = GlobalInfo::init(pipeline_config, robot_id);
  AgentState::init();
  LOG(INFO) << "[HflexEqa] Initialized pipeline with:\n" << config;
}

void HflexEqaPipeline::init() {}

HflexEqaPipeline::~HflexEqaPipeline() {}

std::string makeBanner(const std::string& message,
                       size_t print_width,
                       char fill,
                       bool with_header = true,
                       bool with_footer = false) {
  std::stringstream ss;
  if (with_header) {
    ss << std::string(print_width, fill) << std::endl;
  }
  const auto msg_size = message.size() + 3;
  const auto spacing = msg_size >= print_width ? 0 : print_width - msg_size;
  ss << fill << " " << message << (spacing ? std::string(spacing, ' ') + fill : "")
     << std::endl;
  if (with_footer) {
    ss << std::string(print_width, fill) << std::endl;
  }
  return ss.str();
}

std::string HflexEqaPipeline::getModuleInfo(const std::string& name,
                                            const Module* mod) const {
  const auto print_width = config::Settings().printing.width;
  std::stringstream ss;
  ss << makeBanner(name, print_width, '*', true, true);
  if (!mod) {
    ss << "UNITIALIZED MODULE!" << std::endl;
  } else {
    const auto info = mod->printInfo();
    if (!info.empty()) {
      ss << info << std::endl;
    }
  }
  ss << std::string(print_width, '*') << std::endl;
  return ss.str();
}

void HflexEqaPipeline::showModules() const {
  const auto print_width = config::Settings().printing.width;
  std::stringstream ss;
  ss << std::endl << makeBanner("Modules", print_width, '=', true, true);
  for (auto&& [name, mod] : modules_) {
    ss << std::endl << getModuleInfo(name, mod.get());
  }
  VLOG(config_verbosity_) << ss.str();
}

void HflexEqaPipeline::start() {
  showModules();

  for (auto&& [name, mod] : modules_) {
    if (!mod) {
      LOG(FATAL) << "Found unitialized module: " << name;
      continue;
    }

    mod->start();
  }
}

void HflexEqaPipeline::stop() {
  for (auto&& [name, mod] : modules_) {
    if (!mod) {
      LOG(FATAL) << "Found unitialized module: " << name;
      continue;
    }

    mod->stop();
  }
}

}  // namespace hflex_eqa
