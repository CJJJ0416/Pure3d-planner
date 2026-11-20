#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scan_planner {
// A geometric reference has arc length, not an execution clock or spline state.
class RollingReference {
 public:
  struct Projection { double arc{0}, distance{std::numeric_limits<double>::infinity()}; };
  std::vector<Eigen::Vector3d> points;
  std::vector<double> arc;
  void set(const std::vector<Eigen::Vector3d>& input) {
    std::vector<Eigen::Vector3d> clean;
    for (const auto& p:input) {
      if (!p.allFinite()) throw std::invalid_argument("Nonfinite reference");
      if (clean.empty() || (p-clean.back()).norm()>1e-8) clean.push_back(p);
    }
    if (clean.size()<2) throw std::invalid_argument("Degenerate reference");
    points=std::move(clean);arc.assign(points.size(),0.);
    for (size_t i=1;i<points.size();++i) arc[i]=arc[i-1]+(points[i]-points[i-1]).norm();
  }
  bool empty() const { return points.size()<2; }
  double length() const { return arc.empty()?0.:arc.back(); }
  Projection project(const Eigen::Vector3d& p) const {
    Projection best;
    for (size_t i=1;i<points.size();++i) {
      const Eigen::Vector3d d=points[i]-points[i-1];
      const double t=std::clamp((p-points[i-1]).dot(d)/d.squaredNorm(),0.,1.);
      const double distance=(points[i-1]+t*d-p).norm();
      if (distance<best.distance) best={arc[i-1]+t*(arc[i]-arc[i-1]),distance};
    }
    return best;
  }
  Eigen::Vector3d at(double s) const {
    if (empty()) throw std::logic_error("Empty reference");
    s=std::clamp(s,0.,length());
    const auto it=std::upper_bound(arc.begin(),arc.end(),s);
    const size_t i=std::min(points.size()-1,std::max(size_t(1),size_t(it-arc.begin())));
    return points[i-1]+(points[i]-points[i-1])*(s-arc[i-1])/(arc[i]-arc[i-1]);
  }
  double remainingHorizontal(const Eigen::Vector3d& body) const {
    if (empty()) return 0.;
    const double s=project(body).arc;
    Eigen::Vector3d previous=at(s);
    double remaining=0.;
    for (size_t i=1;i<points.size();++i) {
      if (arc[i]<=s) continue;
      remaining+=(points[i]-previous).head<2>().norm();
      previous=points[i];
    }
    return remaining;
  }
  bool sharesNearPrefix(const RollingReference& next,const Eigen::Vector3d& body,
                        double tolerance=.08) const {
    if (empty() || next.empty()) return false;
    const auto a=project(body), b=next.project(body);
    const double keep=std::min({.4,length()-a.arc,next.length()-b.arc});
    if (a.distance>.5 || b.distance>.5 || keep<.15) return false;
    for (double s=0;s<=keep+1e-9;s+=.025)
      if ((at(a.arc+s)-next.at(b.arc+s)).norm()>tolerance) return false;
    return true;
  }
};

// Reserve distance for input latency and braking before the reference endpoint.
inline double referenceSpeedLimit(double remaining, double maximum, double deceleration,
                                  double reaction_time, double stop_margin,double preview_time=0.) {
  const double distance=std::max(0.,remaining-stop_margin);
  const double reaction=deceleration*reaction_time;
  const double brake=std::sqrt(reaction*reaction+2.*deceleration*distance)-reaction;
  return std::min({maximum,brake,preview_time>0.?distance/preview_time:maximum});
}
}  // namespace scan_planner
