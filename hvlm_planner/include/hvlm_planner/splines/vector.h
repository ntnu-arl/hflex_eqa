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

#include <string>

namespace hvlm_planner {
namespace splines {

class Vector {
 public:
  double x;
  double y;
  double z;

 public:
  Vector();
  Vector(double _x, double _y, double _z);
  Vector(const Vector& rhs);
  Vector& operator=(const Vector& rhs);

 public:
  Vector cross(const Vector& rhs) const;
  Vector normalize() const;
  double dot(const Vector& rhs) const;
  double lengthSq() const;

 public:
  Vector Truncate(double max_value) const;

 public:
  Vector& operator-=(const Vector& rhs);
  Vector& operator+=(const Vector& rhs);
  Vector& operator*=(double value);
  Vector& operator/=(double value);

 public:
  bool operator<=(const Vector& rhs) const;
  bool operator>=(const Vector& rhs) const;
  bool operator==(const Vector& rhs) const;
  bool operator!=(const Vector& rhs) const;

 public:
  void reset() {
    x = 0;
    y = 0;
    z = 0;
  }

 public:
  std::string toString() const;

 public:
  double length() const;
};

Vector operator+(const Vector& v1, const Vector& v2);
Vector operator-(const Vector& v1, const Vector& v2);
Vector operator*(const Vector& v1, double value);
Vector operator*(double value, const Vector& v1);
Vector operator/(const Vector& v1, double value);

}  // namespace splines
}  // namespace hvlm_planner
