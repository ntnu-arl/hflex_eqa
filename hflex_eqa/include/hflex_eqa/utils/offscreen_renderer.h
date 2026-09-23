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

#include <open3d/Open3D.h>
#include <open3d/visualization/gui/Application.h>
#include <open3d/visualization/gui/Gui.h>
#include <open3d/visualization/rendering/Open3DScene.h>
#include <open3d/visualization/rendering/filament/FilamentEngine.h>
#include <open3d/visualization/rendering/filament/FilamentRenderer.h>

namespace open3d {
namespace visualization {
namespace rendering {

class OffscreenRenderer {
 public:
  OffscreenRenderer(int width, int height);
  ~OffscreenRenderer();

  Open3DScene* GetScene();

  std::shared_ptr<open3d::geometry::Image> RenderToDepthImage(
      bool z_in_view_space = false);

  std::shared_ptr<open3d::geometry::Image> RenderToImage();

  void SetupCamera(const Eigen::Matrix3d& intrinsic, const Eigen::Matrix4d& extrinsic);

 private:
  int width_;
  int height_;
  FilamentRenderer* renderer_;
  Open3DScene* scene_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
