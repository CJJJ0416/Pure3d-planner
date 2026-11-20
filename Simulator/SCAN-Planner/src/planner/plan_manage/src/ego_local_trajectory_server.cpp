#include <rclcpp/create_timer.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Eigen>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/scheduled_spline.hpp>
#include <scan_planner_msgs/msg/spline_progress.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.hpp>

#include "scan_bspline_opt/uniform_bspline.h"
#include "scan_plan_env/grid_map.h"

namespace scan_planner
{
class EgoLocalTrajectoryServer : public rclcpp::Node
{
public:
  EgoLocalTrajectoryServer() : Node("ego_local_trajectory_server")
  {
    time_forward_ = declare_parameter<double>("time_forward", 0.8);
    heading_error_threshold_ = declare_parameter<double>("heading_error_threshold", 0.8);
    kp_pos_ = declare_parameter<double>("kp_pos", 0.8);
    kp_yaw_ = declare_parameter<double>("kp_yaw", 1.5);
    max_vx_ = declare_parameter<double>("max_vx", 0.75);
    max_vy_ = declare_parameter<double>("max_vy", 0.35);
    max_vyaw_ = std::min(declare_parameter<double>("max_vyaw", 1.0), kMaxVYawLimit);
    finish_dist_ = declare_parameter<double>("finish_dist", 0.15);
    max_tracking_acc_ = declare_parameter<double>("interface.max_tracking_acc", 0.5);
    if (!std::isfinite(heading_error_threshold_) || heading_error_threshold_ <= 0.0)
      throw std::invalid_argument("heading_error_threshold must be positive");
    if (!std::isfinite(max_tracking_acc_) || max_tracking_acc_ <= 0.0)
      throw std::invalid_argument("interface.max_tracking_acc must be positive");

    map_.initMap(this);
    scheduled_sub_ = create_subscription<scan_planner_msgs::msg::ScheduledSpline>(
        "planning/scheduled_spline", 10,
        std::bind(&EgoLocalTrajectoryServer::scheduleSpline, this, std::placeholders::_1));
    progress_pub_ = create_publisher<scan_planner_msgs::msg::SplineProgress>("planning/spline_progress", 10);
    active_spline_pub_ = create_publisher<scan_planner_msgs::msg::Bspline>("planning/bspline", 10);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "body_pose", rclcpp::SensorDataQoS(),
        std::bind(&EgoLocalTrajectoryServer::odomCallback, this, std::placeholders::_1));
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 20);
    execution_frozen_pub_ = create_publisher<std_msgs::msg::Bool>("planning/go2_execution_frozen", 10);
    navigation_hold_sub_ = create_subscription<std_msgs::msg::Bool>("/entry/navigation/hold", rclcpp::QoS(1).transient_local(),
        [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
          if (msg->data && !navigation_hold_) queued_.reset();
          navigation_hold_=msg->data;
          navigation_heartbeat_=std::chrono::steady_clock::now();
          if (navigation_hold_) receive_traj_=false;
        });
    cmd_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(10),
                                   std::bind(&EgoLocalTrajectoryServer::cmdCallback, this));
    last_update_time_ = now();
    RCLCPP_INFO(get_logger(), "EGO local trajectory server ready");
  }

private:
  using ScheduledSpline = scan_planner_msgs::msg::ScheduledSpline;
  using Steady = std::chrono::steady_clock;

  struct QueuedSpline {
    ScheduledSpline request;
    std::vector<UniformBspline> curve;
  };

  static constexpr double kMaxVYawLimit = 1.0;

  static double normalizeAngle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  static Eigen::Vector2d clampNorm(const Eigen::Vector2d &value, double max_norm)
  {
    const double norm = value.norm();
    return (norm <= max_norm || norm < 1e-6) ? value : value / norm * max_norm;
  }

  double estimateDesiredYaw(double t_cur, const Eigen::Vector3d &pos_des) const
  {
    const double t_look = std::min(traj_duration_, t_cur + time_forward_);
    Eigen::Vector3d direction = traj_[0].evaluateDeBoorT(t_look) - pos_des;
    if (direction.head<2>().squaredNorm() < 1e-4)
      direction = traj_[1].evaluateDeBoorT(t_cur);
    return direction.head<2>().squaredNorm() < 1e-4
        ? odom_yaw_ : std::atan2(direction.y(), direction.x());
  }

  void publishStop(double yaw_rate = 0.0)
  {
    last_planar_world_.setZero();
    last_yaw_cmd_ = 0.;
    geometry_msgs::msg::Twist cmd;
    cmd.angular.z = std::clamp(yaw_rate, -max_vyaw_, max_vyaw_);
    cmd_vel_pub_->publish(cmd);
  }

  void publishExecutionFrozen(bool frozen)
  {
    std_msgs::msg::Bool msg;
    msg.data = frozen;
    execution_frozen_pub_->publish(msg);
  }

  static std::vector<UniformBspline> decodeSpline(const scan_planner_msgs::msg::Bspline &msg)
  {
    const size_t n = msg.pos_pts.size();
    if (msg.order < 1) return {};
    const auto degree = static_cast<size_t>(msg.order);
    if (n <= degree || msg.knots.size() != n+degree+1) return {};
    Eigen::MatrixXd points(3, n);
    for (size_t i=0; i<n; ++i) points.col(i) << msg.pos_pts[i].x, msg.pos_pts[i].y, msg.pos_pts[i].z;
    const auto knots = Eigen::Map<const Eigen::VectorXd>(msg.knots.data(), msg.knots.size());
    if (!points.allFinite() || !knots.allFinite() || !std::is_sorted(msg.knots.begin(), msg.knots.end()) ||
        knots(degree) >= knots(n)) return {};
    for (size_t derivative=1; derivative<=3; ++derivative)
      for (size_t i=0; i+derivative<n; ++i)
        if (knots(i+degree+1)<=knots(i+derivative)) return {};
    UniformBspline position(points, msg.order, .1);
    position.setKnot(knots);
    auto velocity=position.getDerivative();
    auto acceleration=velocity.getDerivative();
    return {position, velocity, acceleration, acceleration.getDerivative()};
  }

  void activate(std::vector<UniformBspline> curve, const scan_planner_msgs::msg::Bspline &msg)
  {
    traj_=std::move(curve);
    traj_duration_=traj_[0].getTimeSum();
    traj_id_=msg.traj_id;
    exec_time_ = 0.0;
    last_update_time_ = now();
    receive_traj_ = true;
  }

  bool mapReady() const
  {
    return have_odom_ && map_.occupancyFresh(.6) &&
           std::chrono::duration<double>(Steady::now()-last_odom_wall_).count() < .6;
  }

  bool collision(UniformBspline &curve, double start)
  {
    const double end=curve.getTimeSum();
    const int samples=std::max(1, static_cast<int>(std::ceil((end-start)/.03)));
    for (int i=0; i<=samples; ++i) {
      const double t=start+(end-start)*i/samples;
      const Eigen::Vector3d tangent=curve.evaluateDeBoorT(std::min(end,t+.04))-
                                    curve.evaluateDeBoorT(std::max(0.,t-.04));
      const double yaw=tangent.head<2>().squaredNorm()>1e-8 ?
                       std::atan2(tangent.y(),tangent.x()) : odom_yaw_;
      const Eigen::Vector3d p=curve.evaluateDeBoorT(t);
      if (!p.allFinite() || map_.getInflateOccupancy(p,yaw)!=0) return true;
    }
    return false;
  }

  void result(int64_t id, bool accepted, const std::string &reason)
  {
    result_id_=id; result_accepted_=accepted; result_reason_=reason;
  }

  void scheduleSpline(const ScheduledSpline::ConstSharedPtr msg)
  {
    const auto id=msg->spline.traj_id;
    if (id<=0 || id <= last_requested_id_) return;
    last_requested_id_=id;
    if (msg->spline.order!=5 || msg->spline.pos_pts.size()>256) {
      result(id,false,"invalid_curve"); return;
    }
    auto curve=decodeSpline(msg->spline);
    const auto &times=msg->cap_times;
    const auto &caps=msg->xy_speed_caps;
    if (curve.empty() || curve[0].getTimeSum()>60. || times.size()<2 || times.size()!=caps.size() || times.size()>10000 ||
        !std::is_sorted(times.begin(), times.end()) || std::abs(times.front())>1e-8 ||
        std::abs(times.back()-curve[0].getTimeSum())>1e-6 ||
        !std::all_of(times.begin(),times.end(),[](double t){return std::isfinite(t);}) ||
        !std::all_of(caps.begin(),caps.end(),[](double v){return std::isfinite(v)&&v>0.&&v<=1.;})) {
      result(id,false,"invalid_curve"); return;
    }
    if (!mapReady()) { result(id,false,"map_or_odom_stale"); return; }
    if (msg->predecessor_id==0) {
      if (receive_traj_ || (curve[0].evaluateDeBoorT(0.)-odom_pos_).norm()>.10) {
        result(id,false,"start_state_changed"); return;
      }
    } else {
      const double at=msg->predecessor_time;
      if (!std::isfinite(at) || !receive_traj_ || traj_id_!=msg->predecessor_id ||
          at<exec_time_ || at>traj_duration_+1e-8) {
        result(id,false,"handoff_expired"); return;
      }
      for (size_t derivative=0; derivative<4; ++derivative)
        if ((traj_[derivative].evaluateDeBoorT(at)-curve[derivative].evaluateDeBoorT(0.)).norm()>1e-5) {
          result(id,false,"handoff_state_mismatch"); return;
        }
    }
    if (collision(curve[0],0.)) { result(id,false,"occupied_candidate"); return; }
    queued_=QueuedSpline{*msg,std::move(curve)};
    result(id,true,"queued");
  }

  void publishProgress(bool frozen)
  {
    if (Steady::now()-last_progress_wall_ < std::chrono::milliseconds(40)) return;
    last_progress_wall_=Steady::now();
    scan_planner_msgs::msg::SplineProgress msg;
    msg.active_id=receive_traj_ ? traj_id_ : 0;
    msg.elapsed=receive_traj_ ? exec_time_ : 0.;
    msg.duration=receive_traj_ ? traj_duration_ : 0.;
    msg.queued_id=queued_ ? queued_->request.spline.traj_id : 0;
    msg.result_id=result_id_;msg.accepted=result_accepted_;msg.reason=result_reason_;
    msg.ready=mapReady();msg.frozen=frozen;
    progress_pub_->publish(msg);
  }

  void stopActive(const std::string &reason)
  {
    if (queued_) result(queued_->request.spline.traj_id,false,reason);
    else if (receive_traj_) result(traj_id_,false,reason);
    queued_.reset();
    receive_traj_=false;
  }

  void tryHandoff(double &dt)
  {
    if (!queued_) return;
    const auto &request=queued_->request;
    if (request.predecessor_id!=0 &&
        (request.predecessor_id!=traj_id_ || exec_time_+dt<request.predecessor_time)) return;
    if (collision(queued_->curve[0],0.)) {
      result(request.spline.traj_id,false,"occupied_handoff");
      queued_.reset();
      return;
    }
    dt=request.predecessor_id==0 ? 0. : std::max(0.,exec_time_+dt-request.predecessor_time);
    activate(std::move(queued_->curve),request.spline);
    cap_times_=request.cap_times;
    xy_caps_=request.xy_speed_caps;
    auto msg=request.spline;
    msg.start_time=now();
    active_spline_pub_->publish(msg);
    result(msg.traj_id,true,"active");
    queued_.reset();
  }

  double speedCap(double t) const
  {
    if (xy_caps_.empty()) return std::max(max_vx_,max_vy_);
    const auto at=std::lower_bound(cap_times_.begin(),cap_times_.end(),t);
    const size_t i=std::min(cap_times_.size()-1,size_t(at-cap_times_.begin()));
    double cap = std::min({std::max(max_vx_,max_vy_),xy_caps_[i],xy_caps_[i==0 ? 0 : i-1]});
    if (queued_ && queued_->request.predecessor_id == traj_id_) {
      const double remaining = std::max(0., queued_->request.predecessor_time - t);
      cap = std::min(cap, queued_->request.xy_speed_caps.front() + max_tracking_acc_ * remaining);
    }
    return cap;
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    odom_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    odom_yaw_ = tf2::getYaw(msg->pose.pose.orientation);
    have_odom_ = odom_pos_.allFinite() && std::isfinite(odom_yaw_);
    last_odom_wall_=Steady::now();
  }

  void cmdCallback()
  {
    if (!mapReady()) {
      stopActive("map_or_odom_stale");
      publishProgress(true);
      publishStop();
      last_update_time_=now();
      return;
    }
    const auto current_time = now();
    if (navigation_hold_ || Steady::now() - navigation_heartbeat_ > std::chrono::milliseconds(600)) {
      if (!navigation_hold_) stopActive("hold_heartbeat_expired");
      publishProgress(true);
      publishExecutionFrozen(true);
      publishStop();
      last_update_time_ = current_time;
      return;
    }
    if (!receive_traj_ && queued_) {
      double initial_dt = 0.;
      tryHandoff(initial_dt);
    }
    if (!receive_traj_ || !have_odom_)
    {
      publishProgress(false);
      publishExecutionFrozen(false);
      publishStop();
      return;
    }
    double dt = (current_time - last_update_time_).seconds();
    if (dt < 0.0 || dt > 0.2) dt = 0.0;
    const double control_dt = std::min(dt, .05);
    if (Steady::now()-last_safety_wall_>=std::chrono::milliseconds(50)) {
      last_safety_wall_=Steady::now();
      if (map_.getInflateOccupancy(odom_pos_,odom_yaw_)!=0 || collision(traj_[0],exec_time_)) {
        stopActive("occupied_execution");
        publishProgress(true);
        publishStop();
        last_update_time_=current_time;
        return;
      }
    }
    const double t_eval = std::min(exec_time_, traj_duration_);
    Eigen::Vector3d pos_des = traj_[0].evaluateDeBoorT(t_eval);
    const double yaw_error = normalizeAngle(estimateDesiredYaw(t_eval, pos_des) - odom_yaw_);
    // EGO trajectory-server sampling: evaluate PVA on the committed spline,
    // then transform the planar command into the quadruped body heading.
    const double yaw_target = std::clamp(kp_yaw_ * yaw_error, -max_vyaw_, max_vyaw_);
    const double yaw_command = last_yaw_cmd_ + std::clamp(
        yaw_target - last_yaw_cmd_, -2.5 * control_dt, 2.5 * control_dt);
    const Eigen::Vector2d tangent = traj_[1].evaluateDeBoorT(t_eval).head<2>();
    const double reference_speed = tangent.norm();
    double phase_rate = 1.;
    if (reference_speed > 1e-4) {
      const double lag = (pos_des - odom_pos_).head<2>().dot(tangent / reference_speed);
      phase_rate = std::clamp((.40 - lag) / .28, .35, 1.);
    }
    const double alignment = std::clamp(1. - .65 * std::abs(yaw_error) / heading_error_threshold_, .35, 1.);
    const double tracking_rate = phase_rate * alignment;
    publishExecutionFrozen(false);
    dt *= tracking_rate;
    tryHandoff(dt);
    exec_time_ = std::min(traj_duration_, exec_time_ + dt);
    last_update_time_ = current_time;
    pos_des = traj_[0].evaluateDeBoorT(exec_time_);
    const Eigen::Vector3d vel_des = traj_[1].evaluateDeBoorT(exec_time_);
    const Eigen::Vector2d pos_error(pos_des.x() - odom_pos_.x(), pos_des.y() - odom_pos_.y());
    const double speed_cap = speedCap(exec_time_);
    Eigen::Vector2d vel_world = clampNorm(tracking_rate * vel_des.head<2>() + alignment * kp_pos_ * pos_error, speed_cap);
    vel_world = last_planar_world_ + clampNorm(vel_world - last_planar_world_, max_tracking_acc_ * control_dt);
    vel_world = clampNorm(vel_world, speed_cap);
    const double c = std::cos(odom_yaw_);
    const double s = std::sin(odom_yaw_);
    geometry_msgs::msg::Twist command;
    command.linear.x = std::clamp(c * vel_world.x() + s * vel_world.y(), -max_vx_, max_vx_);
    command.linear.y = std::clamp(-s * vel_world.x() + c * vel_world.y(), -max_vy_, max_vy_);
    command.angular.z = yaw_command;
    if (exec_time_ >= traj_duration_ && pos_error.norm() < finish_dist_)
      command = geometry_msgs::msg::Twist();
    last_planar_world_ = Eigen::Vector2d(c * command.linear.x - s * command.linear.y,
                                        s * command.linear.x + c * command.linear.y);
    last_yaw_cmd_ = command.angular.z;
    cmd_vel_pub_->publish(command);
    publishProgress(false);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr execution_frozen_pub_;
  rclcpp::Subscription<ScheduledSpline>::SharedPtr scheduled_sub_;
  rclcpp::Publisher<scan_planner_msgs::msg::SplineProgress>::SharedPtr progress_pub_;
  rclcpp::Publisher<scan_planner_msgs::msg::Bspline>::SharedPtr active_spline_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr navigation_hold_sub_;
  bool navigation_hold_{true};
  std::chrono::steady_clock::time_point navigation_heartbeat_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  bool receive_traj_{false};
  bool have_odom_{false};
  double max_tracking_acc_{0.5};
  Eigen::Vector2d last_planar_world_{Eigen::Vector2d::Zero()};
  double last_yaw_cmd_{0.};
  GridMap map_;
  std::optional<QueuedSpline> queued_;
  std::vector<double> cap_times_, xy_caps_;
  Steady::time_point last_odom_wall_{}, last_progress_wall_{}, last_safety_wall_{};
  int64_t last_requested_id_{0}, result_id_{0};
  bool result_accepted_{false};
  std::string result_reason_;
  std::vector<UniformBspline> traj_;
  double traj_duration_{0.0};
  std::int64_t traj_id_{0};
  Eigen::Vector3d odom_pos_{Eigen::Vector3d::Zero()};
  double odom_yaw_{0.0};
  double exec_time_{0.0};
  rclcpp::Time last_update_time_{0, 0, RCL_ROS_TIME};
  double time_forward_, heading_error_threshold_, kp_pos_, kp_yaw_;
  double max_vx_, max_vy_, max_vyaw_, finish_dist_;
};
}  // namespace scan_planner

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<scan_planner::EgoLocalTrajectoryServer>());
  rclcpp::shutdown();
  return 0;
}
