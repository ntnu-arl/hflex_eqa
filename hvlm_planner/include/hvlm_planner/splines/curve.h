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

#include <cassert>
#include <vector>

#include "hvlm_planner/splines/vector.h"

namespace hvlm_planner {
namespace splines {

class Curve {
 public:
  Curve();
  virtual ~Curve();

 protected:
  std::vector<Vector> _way_points;

 public:
  void addWayPoint(const Vector& point);
  virtual void finalize() {}
  void clear();

 protected:
  void addNode(const Vector& node);
  virtual void _onWayPointAdded() = 0;

 protected:
  std::vector<Vector> _nodes;
  std::vector<double> _distances;

 public:
  Vector node(int i) const { return _nodes[i]; }
  double lengthFromStartingPoint(int i) const { return _distances[i]; }
  double lengthBetweenNodes(int i, int j) const {
    return std::abs(_distances[j] - _distances[i]);
  }
  bool hasNextNode(int i) const { return static_cast<int>(_nodes.size()) > i; }
  size_t nodeCount() const { return _nodes.size(); }
  bool isEmpty() const { return _nodes.empty(); }
  double totalLength() const {
    assert(!_distances.empty());
    return _distances[_distances.size() - 1];
  }

 protected:
  int _steps;

 public:
  void incrementSteps(int steps) { _steps += steps; }
  void setSteps(int steps) { _steps = steps; }
};

}  // namespace splines
}  // namespace hvlm_planner
