#include <rclcpp/create_timer.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Eigen>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/spline_progress.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.hpp>

#include "scan_bspline_opt/uniform_bspline.h"
#include "plan_manage/rolling_reference.hpp"

namespace scan_planner
{
class ClosedLoopController : public rclcpp::Node
{
public:
  ClosedLoopController() : Node("closed_loop_controller")
  {
    time_forward_ = declare_parameter<double>("time_forward", 0.8);
    heading_error_threshold_ = declare_parameter<double>("heading_error_threshold", 0.8);
    kp_pos_ = declare_parameter<double>("kp_pos", 0.8);
    kp_yaw_ = declare_parameter<double>("kp_yaw", 1.5);
    max_vx_ = declare_parameter<double>("max_vx", 0.75);
    max_vy_ = declare_parameter<double>("max_vy", 0.35);
    max_vyaw_ = std::min(declare_parameter<double>("max_vyaw", 1.0), kMaxVYawLimit);
    finish_dist_ = declare_parameter<double>("finish_dist", 0.15);
    adaptive_speed_ = declare_parameter<bool>("adaptive_reference_speed", false);
    speed_acceleration_ = declare_parameter<double>("reference_speed_acceleration", 0.5);
    speed_reaction_time_ = declare_parameter<double>("reference_speed_reaction_time", 0.5);
    speed_stop_margin_ = declare_parameter<double>("reference_speed_stop_margin", 0.15);
    speed_preview_time_ = declare_parameter<double>("reference_speed_preview_time", 3.0);
    height_tolerance_ = declare_parameter<double>("reference_height_tolerance", 0.5);
    reference_body_height_ = declare_parameter<double>("reference_body_height", 0.4);
    if (!std::isfinite(speed_acceleration_) || speed_acceleration_<=0. ||
        !std::isfinite(speed_reaction_time_) || speed_reaction_time_<0. ||
        !std::isfinite(speed_stop_margin_) || speed_stop_margin_<0. ||
        !std::isfinite(speed_preview_time_) || speed_preview_time_<=0. ||
        !std::isfinite(height_tolerance_) || height_tolerance_<=0. ||
        !std::isfinite(reference_body_height_))
      throw std::invalid_argument("Invalid adaptive reference speed parameters");

    reference_sub_ = create_subscription<nav_msgs::msg::Path>(
        "planning/reference_accepted", 10,
        [this](const nav_msgs::msg::Path::ConstSharedPtr msg) {
          const auto stamp=rclcpp::Time(msg->header.stamp).nanoseconds();
          if (stamp<=last_cancel_stamp_ || stamp<reference_stamp_) return;
          if (!adaptive_speed_) return;
          std::vector<Eigen::Vector3d> points;
          for (const auto& pose:msg->poses) {
            const auto& p=pose.pose.position;
            points.emplace_back(p.x,p.y,p.z+reference_body_height_);
          }
          try { reference_.set(points); reference_frame_=msg->header.frame_id;reference_stamp_=stamp; }
          catch (const std::invalid_argument&) { reference_=RollingReference(); }
        });

    pure_path_sub_=create_subscription<nav_msgs::msg::Path>("/pure_path",rclcpp::QoS(1).transient_local(),
        [this](const nav_msgs::msg::Path::ConstSharedPtr msg) {
          if (!adaptive_speed_) return;
          const auto stamp=rclcpp::Time(msg->header.stamp).nanoseconds();
          if (stamp<=last_cancel_stamp_ || stamp<pure_stamp_) return;
          pure_stamp_=stamp;pure_frame_=msg->header.frame_id;
          std::vector<Eigen::Vector3d> points;
          for (const auto& p:msg->poses) points.emplace_back(p.pose.position.x,p.pose.position.y,p.pose.position.z);
          try { pure_reference_.set(points); }
          catch (const std::invalid_argument&) { pure_reference_=RollingReference(); }
        });
    speed_status_pub_=create_publisher<std_msgs::msg::String>("/scan/planning/reference_speed",10);

    bspline_sub_ = create_subscription<scan_planner_msgs::msg::Bspline>(
        "planning/bspline", 10,
        std::bind(&ClosedLoopController::bsplineCallback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "body_pose", rclcpp::SensorDataQoS(),
        std::bind(&ClosedLoopController::odomCallback, this, std::placeholders::_1));
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 20);
    execution_frozen_pub_ = create_publisher<std_msgs::msg::Bool>("planning/go2_execution_frozen", 10);
    progress_pub_ = create_publisher<scan_planner_msgs::msg::SplineProgress>("planning/spline_progress", 10);
    navigation_hold_sub_ = create_subscription<std_msgs::msg::Bool>("/entry/navigation/hold", rclcpp::QoS(1).transient_local(),
        [this](const std_msgs::msg::Bool::ConstSharedPtr msg) {
          navigation_controlled_=true;
          navigation_hold_=msg->data;
          navigation_heartbeat_=std::chrono::steady_clock::now();
          if (navigation_hold_) receive_traj_=false;
        });
    cancel_goal_sub_=create_subscription<std_msgs::msg::Header>("/scan/planning/cancel_goal",rclcpp::QoS(1).transient_local(),
        [this](const std_msgs::msg::Header::ConstSharedPtr msg) {
          const auto stamp=rclcpp::Time(msg->stamp).nanoseconds();
          if (stamp<last_cancel_stamp_ || (receive_traj_ && stamp<traj_start_stamp_)) return;
          last_cancel_stamp_=stamp;receive_traj_=false;exec_time_=0.;
          reference_=RollingReference();reference_frame_.clear();pure_reference_=RollingReference();pure_frame_.clear();
          publishExecutionFrozen(true);publishStop(0.,"cancelled");
        });
    cmd_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(10),
                                   std::bind(&ClosedLoopController::cmdCallback, this));
    last_update_time_ = now();
    RCLCPP_INFO(get_logger(), "Closed-loop controller ready");
  }

private:
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

  void publishStop(double yaw_rate = 0.0,const char* reason="stopped")
  {
    last_velocity_world_.setZero();
    geometry_msgs::msg::Twist cmd;
    cmd.angular.z = std::clamp(yaw_rate, -max_vyaw_, max_vyaw_);
    cmd_vel_pub_->publish(cmd);
    publishSpeedStatus(0.,0.,0.,cmd,reason);
  }

  void publishSpeedStatus(double accepted,double pending,double limit,
                         const geometry_msgs::msg::Twist& command,const char* reason) {
    if (!adaptive_speed_) return;
    const auto stamp=now().nanoseconds();
    if (stamp>=speed_status_stamp_ && stamp-speed_status_stamp_<100000000 && speed_status_reason_==reason) return;
    speed_status_stamp_=stamp;speed_status_reason_=reason;
    std::ostringstream value;
    value<<"{\"accepted_remaining_m\":"<<accepted<<",\"pure_remaining_m\":"<<pending
         <<",\"effective_remaining_m\":"<<std::min(accepted,pending)<<",\"speed_limit_mps\":"<<limit
         <<",\"vx\":"<<command.linear.x<<",\"vy\":"<<command.linear.y
         <<",\"command_xy_mps\":"<<std::hypot(command.linear.x,command.linear.y)
         <<",\"preview_time_s\":"<<speed_preview_time_<<",\"reason\":\""<<reason<<"\"}";
    std_msgs::msg::String message;message.data=value.str();speed_status_pub_->publish(message);
  }

  void publishExecutionFrozen(bool frozen)
  {
    std_msgs::msg::Bool msg;
    msg.data = frozen;
    execution_frozen_pub_->publish(msg);
    scan_planner_msgs::msg::SplineProgress progress;
    progress.active_id=traj_id_;
    progress.elapsed=exec_time_;
    progress.duration=traj_duration_;
    progress.ready=receive_traj_ && have_odom_;
    progress.frozen=frozen;
    progress_pub_->publish(progress);
  }

  void bsplineCallback(const scan_planner_msgs::msg::Bspline::ConstSharedPtr msg)
  {
    if (rclcpp::Time(msg->start_time).nanoseconds()<=last_cancel_stamp_) return;
    if (navigation_hold_) return;
    if (msg->pos_pts.empty() || msg->knots.empty() || msg->order <= 0)
    {
      RCLCPP_WARN(get_logger(), "Ignoring invalid B-spline");
      return;
    }
    Eigen::MatrixXd points(3, msg->pos_pts.size());
    for (size_t i = 0; i < msg->pos_pts.size(); ++i)
      points.col(i) << msg->pos_pts[i].x, msg->pos_pts[i].y, msg->pos_pts[i].z;
    Eigen::VectorXd knots(msg->knots.size());
    for (size_t i = 0; i < msg->knots.size(); ++i) knots(i) = msg->knots[i];
    UniformBspline position(points, msg->order, 0.1);
    position.setKnot(knots);
    traj_ = {position, position.getDerivative()};
    traj_.push_back(traj_[1].getDerivative());
    traj_duration_ = traj_[0].getTimeSum();
    traj_id_ = msg->traj_id;
    traj_start_stamp_=rclcpp::Time(msg->start_time).nanoseconds();
    exec_time_ = 0.0;
    last_update_time_ = now();
    receive_traj_ = true;
    RCLCPP_INFO(get_logger(), "Received trajectory %lld, duration %.3fs",
                static_cast<long long>(traj_id_), traj_duration_);
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    odom_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    odom_yaw_ = tf2::getYaw(msg->pose.pose.orientation);
    odom_frame_ = msg->header.frame_id;
    have_odom_ = true;
  }

  void cmdCallback()
  {
    if (navigation_controlled_ && (navigation_hold_ ||
        std::chrono::duration<double>(std::chrono::steady_clock::now()-navigation_heartbeat_).count()>2.0)) {
      publishExecutionFrozen(true);
      publishStop(0.,"navigation_hold");
      last_update_time_=now();
      return;
    }
    if (!receive_traj_ || !have_odom_ ||
        (adaptive_speed_ && (reference_.empty() || pure_reference_.empty() ||
          reference_frame_!=odom_frame_ || pure_frame_!=odom_frame_ || !odom_pos_.allFinite())))
    {
      publishExecutionFrozen(adaptive_speed_);
      publishStop(0.,"waiting_reference");
      return;
    }
    const auto current_time = now();
    double dt = (current_time - last_update_time_).seconds();
    if (dt < 0.0 || dt > 0.2) dt = 0.0;
    const double control_dt=dt;
    const double t_eval = std::min(exec_time_, traj_duration_);
    double speed_cap=std::max(max_vx_,max_vy_);double accepted_remaining=0.,pure_remaining=0.;
    if (adaptive_speed_) {
      const auto accepted_projection=reference_.project(odom_pos_);
      const auto pure_projection=pure_reference_.project(odom_pos_);
      // The reference may start at an observed support anchor beyond the LIO
      // blind zone. Horizontal snap distance is not a tracking failure.
      if (std::abs(reference_.at(accepted_projection.arc).z()-odom_pos_.z())>height_tolerance_ ||
          std::abs(pure_reference_.at(pure_projection.arc).z()-odom_pos_.z())>height_tolerance_) {
        publishExecutionFrozen(true);publishStop(0.,"reference_height_mismatch");last_update_time_=current_time;return;
      }
      accepted_remaining=reference_.remainingHorizontal(odom_pos_);
      pure_remaining=pure_reference_.remainingHorizontal(odom_pos_);
      const double remaining=std::min(accepted_remaining,pure_remaining);
      speed_cap=referenceSpeedLimit(remaining,speed_cap,speed_acceleration_,
                                    speed_reaction_time_,speed_stop_margin_,speed_preview_time_);
      const double nominal_speed=traj_[1].evaluateDeBoorT(t_eval).head<2>().norm();
      dt*=std::min(1.,speed_cap/std::max(nominal_speed,0.05));
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1000,
          "Adaptive reference speed: accepted %.3f m, pure %.3f m, cap %.3f m/s",accepted_remaining,pure_remaining,speed_cap);
    }
    Eigen::Vector3d pos_des = traj_[0].evaluateDeBoorT(t_eval);
    const double yaw_error = normalizeAngle(estimateDesiredYaw(t_eval, pos_des) - odom_yaw_);
    const double yaw_command = std::clamp(kp_yaw_ * yaw_error, -max_vyaw_, max_vyaw_);
    if (std::abs(yaw_error) > heading_error_threshold_)
    {
      publishExecutionFrozen(true);
      publishStop(yaw_command,"heading_alignment");
      last_update_time_ = current_time;
      return;
    }

    publishExecutionFrozen(false);
    exec_time_ = std::min(traj_duration_, exec_time_ + dt);
    last_update_time_ = current_time;
    pos_des = traj_[0].evaluateDeBoorT(exec_time_);
    const Eigen::Vector3d vel_des = traj_[1].evaluateDeBoorT(exec_time_);
    const Eigen::Vector2d pos_error(pos_des.x() - odom_pos_.x(), pos_des.y() - odom_pos_.y());
    Eigen::Vector2d vel_world = clampNorm(
        Eigen::Vector2d(vel_des.x(), vel_des.y()) + kp_pos_ * pos_error,
        speed_cap);
    if (adaptive_speed_) {
      vel_world=last_velocity_world_+clampNorm(vel_world-last_velocity_world_,speed_acceleration_*control_dt);
      vel_world=clampNorm(vel_world,speed_cap);
    }
    const double c = std::cos(odom_yaw_);
    const double s = std::sin(odom_yaw_);
    geometry_msgs::msg::Twist command;
    Eigen::Vector2d body_velocity(c*vel_world.x()+s*vel_world.y(),-s*vel_world.x()+c*vel_world.y());
    if (adaptive_speed_) {
      const double scale=std::min({1.,max_vx_/std::max(std::abs(body_velocity.x()),1e-9),
                                      max_vy_/std::max(std::abs(body_velocity.y()),1e-9)});
      body_velocity*=scale;
      last_velocity_world_=Eigen::Vector2d(c*body_velocity.x()-s*body_velocity.y(),s*body_velocity.x()+c*body_velocity.y());
    } else {
      body_velocity.x()=std::clamp(body_velocity.x(),-max_vx_,max_vx_);
      body_velocity.y()=std::clamp(body_velocity.y(),-max_vy_,max_vy_);
    }
    command.linear.x=body_velocity.x();command.linear.y=body_velocity.y();
    command.angular.z = yaw_command;
    if (exec_time_ >= traj_duration_ && pos_error.norm() < finish_dist_)
    {
      command = geometry_msgs::msg::Twist();
      last_velocity_world_.setZero();
    }
    cmd_vel_pub_->publish(command);
    publishSpeedStatus(accepted_remaining,pure_remaining,speed_cap,command,"path_length");
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr execution_frozen_pub_;
  rclcpp::Publisher<scan_planner_msgs::msg::SplineProgress>::SharedPtr progress_pub_;
  rclcpp::Subscription<scan_planner_msgs::msg::Bspline>::SharedPtr bspline_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr reference_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr pure_path_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr speed_status_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr navigation_hold_sub_;
  bool navigation_hold_{false}, navigation_controlled_{false};
  std::chrono::steady_clock::time_point navigation_heartbeat_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  bool receive_traj_{false};
  bool have_odom_{false};
  std::vector<UniformBspline> traj_;
  double traj_duration_{0.0};
  std::int64_t traj_id_{0};
  std::int64_t last_cancel_stamp_{-1},traj_start_stamp_{-1};
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr cancel_goal_sub_;
  Eigen::Vector3d odom_pos_{Eigen::Vector3d::Zero()};
  double odom_yaw_{0.0};
  double exec_time_{0.0};
  rclcpp::Time last_update_time_{0, 0, RCL_ROS_TIME};
  double time_forward_, heading_error_threshold_, kp_pos_, kp_yaw_;
  double max_vx_, max_vy_, max_vyaw_, finish_dist_;
  bool adaptive_speed_{false};
  double speed_acceleration_,speed_reaction_time_,speed_stop_margin_,reference_body_height_,speed_preview_time_,height_tolerance_;
  RollingReference reference_,pure_reference_;
  std::string reference_frame_,odom_frame_,pure_frame_,speed_status_reason_;
  std::int64_t reference_stamp_{-1},pure_stamp_{-1},speed_status_stamp_{-1};
  Eigen::Vector2d last_velocity_world_{Eigen::Vector2d::Zero()};
};
}  // namespace scan_planner

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<scan_planner::ClosedLoopController>());
  rclcpp::shutdown();
  return 0;
}
