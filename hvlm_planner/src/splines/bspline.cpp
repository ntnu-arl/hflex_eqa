#include "hvlm_planner/splines/bspline.h"

#include <iostream>

namespace hvlm_planner {
namespace splines {

BSpline::BSpline() : Curve() {}

BSpline::~BSpline() {}

void BSpline::_onWayPointAdded() {
  if (_way_points.size() == 1) {
    // Clamp start by duplicating the first waypoint 3 times
    _way_points.push_back(_way_points[0]);
    _way_points.push_back(_way_points[0]);
    _way_points.push_back(_way_points[0]);
  }
  if (_way_points.size() < 4) {
    return;
  }

  int new_control_point_index = static_cast<int>(_way_points.size()) - 1;

  int pt = new_control_point_index - 3;

  for (int i = 0; i <= _steps; i++) {
    double u = (double)i / (double)_steps;

    addNode(interpolate(u,
                        _way_points[pt],
                        _way_points[pt + 1],
                        _way_points[pt + 2],
                        _way_points[pt + 3]));
  }
}

void BSpline::finalize() {
  if (_way_points.size() >= 4) {
    const Vector& last = _way_points.back();
    _way_points.push_back(last);
  }
  _onWayPointAdded();
}

Vector BSpline::interpolate(
    double u, const Vector& P0, const Vector& P1, const Vector& P2, const Vector& P3) {
  Vector point;
  point = u * u * u * ((-1) * P0 + 3 * P1 - 3 * P2 + P3) / 6;
  point += u * u * (3 * P0 - 6 * P1 + 3 * P2) / 6;
  point += u * ((-3) * P0 + 3 * P2) / 6;
  point += (P0 + 4 * P1 + P2) / 6;

  return point;
}

}  // namespace splines
}  // namespace hvlm_planner
