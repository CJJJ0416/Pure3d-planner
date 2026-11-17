#pragma once
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace scan_planner {
class WallTimeBudget {
 public:
  explicit WallTimeBudget(double seconds) : started_(Clock::now()) {
    if (!std::isfinite(seconds) || seconds <= 0.)
      throw std::invalid_argument("Replan timeout must be positive and finite");
    deadline_ = started_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  }
  bool expired() const { return Clock::now() >= deadline_; }
  double elapsed() const { return std::chrono::duration<double>(Clock::now() - started_).count(); }
 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point started_, deadline_;
};
}
