// glocal-nav2 interface only: keep SCAN's planning and tracking algorithms.
#include <cmath>
#include <memory>
#include <interactive_markers/interactive_marker_server.hpp>
#include <rclcpp/create_timer.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class ScanWorkspaceBridge : public rclcpp::Node {
 public:
  ScanWorkspaceBridge() : Node("scan_workspace_bridge") {
    frame_ = declare_parameter<std::string>("planning_frame", "map");
    sensor_frame_ = declare_parameter<std::string>("sensor_frame", "laser_livox");
    body_height_ = declare_parameter("path_body_height", 0.4);
    path_sample_distance_ = declare_parameter("path_sample_distance", 0.8);
    if (!std::isfinite(path_sample_distance_) || path_sample_distance_ < 0.0)
      throw std::invalid_argument("path_sample_distance must be finite and nonnegative");
    if (declare_parameter("manual_markers", false)) {
      server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>("basic_controls", this);
      display_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(50),
          std::bind(&ScanWorkspaceBridge::displayMarkers, this));
      goal_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(500),
          std::bind(&ScanWorkspaceBridge::publishManualGoal, this));
    }
    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);
    const auto qos = rclcpp::SensorDataQoS();
    body_pub_ = create_publisher<nav_msgs::msg::Odometry>("body_pose", qos);
    sensor_pub_ = create_publisher<nav_msgs::msg::Odometry>("sensor_pose", qos);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud", qos);
    goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("goal", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("initial_path", rclcpp::QoS(1).transient_local());
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>("input_odom", qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { odom(*msg); });
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("input_cloud", qos,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { cloud(*msg); });
    path_sub_ = create_subscription<nav_msgs::msg::Path>("input_path", rclcpp::QoS(1).transient_local(),
        [this](nav_msgs::msg::Path::ConstSharedPtr msg) { path(*msg); });
    RCLCPP_INFO(get_logger(), "SCAN input bridge: poses/clouds -> %s; body twist retained", frame_.c_str());
  }

 private:
  bool poseInMap(geometry_msgs::msg::PoseStamped &pose) {
    if (pose.header.frame_id.empty()) return false;
    const auto &p = pose.pose.position;
    auto &q = pose.pose.orientation;
    Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
    if (!Eigen::Vector3d(p.x,p.y,p.z).allFinite() || !rotation.coeffs().allFinite() ||
        !std::isfinite(rotation.norm()) || rotation.norm() < 1e-6) return false;
    rotation.normalize();
    q.x=rotation.x(); q.y=rotation.y(); q.z=rotation.z(); q.w=rotation.w();
    if (pose.header.frame_id != frame_) {
      pose = buffer_->transform(pose, frame_, tf2::durationFromSec(0.1));
    }
    return true;
  }

  void odom(const nav_msgs::msg::Odometry &input) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header=input.header; pose.pose=input.pose.pose;
    try {
      if (!poseInMap(pose)) return;
      auto output=input;
      output.header=pose.header; output.pose.pose=pose.pose;
      // ROS Odometry twist is still in child_frame_id; the SCAN FSM rotates it to map.
      body_pub_->publish(output);
      robot_pose_ = pose.pose;
      robot_frame_ = input.child_frame_id;
      have_body_ = true;
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "SCAN odometry TF: %s", e.what());
    }
  }

  void cloud(const sensor_msgs::msg::PointCloud2 &input) {
    if (input.header.frame_id.empty() || input.width*input.height == 0) return;
    try {
      const auto sensor = buffer_->lookupTransform(frame_,
          input.header.frame_id == frame_ ? sensor_frame_ : input.header.frame_id,
          rclcpp::Time(input.header.stamp), rclcpp::Duration::from_seconds(0.1));
      nav_msgs::msg::Odometry pose;
      pose.header=input.header; pose.header.frame_id=frame_;
      pose.child_frame_id = input.header.frame_id == frame_ ? sensor_frame_ : input.header.frame_id;
      pose.pose.pose.position.x=sensor.transform.translation.x;
      pose.pose.pose.position.y=sensor.transform.translation.y;
      pose.pose.pose.position.z=sensor.transform.translation.z;
      pose.pose.pose.orientation=sensor.transform.rotation;
      sensor_pub_->publish(pose);
      auto output=input;
      if (input.header.frame_id != frame_) {
        pcl::PointCloud<pcl::PointXYZ> source, world;
        pcl::fromROSMsg(input, source);
        const auto &p=sensor.transform.translation;
        const auto &q=sensor.transform.rotation;
        Eigen::Affine3f transform=Eigen::Translation3f(p.x,p.y,p.z)*
            Eigen::Quaternionf(q.w,q.x,q.y,q.z).normalized();
        pcl::transformPointCloud(source, world, transform);
        pcl::toROSMsg(world, output);
      }
      output.header=input.header; output.header.frame_id=frame_;
      cloud_pub_->publish(output);
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "SCAN cloud TF: %s", e.what());
    }
  }

  static bool changed(const geometry_msgs::msg::Pose &a, const geometry_msgs::msg::Pose &b) {
    const Eigen::Vector3d pa(a.position.x,a.position.y,a.position.z), pb(b.position.x,b.position.y,b.position.z);
    const Eigen::Vector4d qa(a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w);
    const Eigen::Vector4d qb(b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w);
    return (pa-pb).norm() >= 0.001 || std::min((qa-qb).norm(),(qa+qb).norm()) >= 0.001;
  }

  void makeMarker(const std::string &name, const geometry_msgs::msg::Pose &pose) {
    using Control = visualization_msgs::msg::InteractiveMarkerControl;
    visualization_msgs::msg::InteractiveMarker marker;
    marker.header.frame_id=frame_; marker.name=name; marker.pose=pose;
    marker.description = name == "start_pos" ? "Robot: " + robot_frame_ : "end_pos 6-DOF Controller";
    visualization_msgs::msg::Marker sphere;
    sphere.type=visualization_msgs::msg::Marker::SPHERE;
    sphere.pose.orientation.w=1.0;
    sphere.scale.x=sphere.scale.y=sphere.scale.z=0.4;
    sphere.color.b=sphere.color.a=1.0;
    Control visible;
    visible.always_visible=true; visible.markers.push_back(sphere);
    marker.controls.push_back(visible);
    if (name == "end_pos") {
      const std::string axes[] = {"x", "z", "y"};
      for (int i=0; i<3; ++i) {
        for (bool rotate : {true,false}) {
          Control c;
          c.name=(rotate ? "rotate_" : "move_")+axes[i];
          c.interaction_mode=rotate ? Control::ROTATE_AXIS : Control::MOVE_AXIS;
          c.orientation_mode=Control::FIXED;
          c.orientation.w=std::sqrt(0.5);
          if (i==0) c.orientation.x=std::sqrt(0.5);
          if (i==1) c.orientation.y=std::sqrt(0.5);
          if (i==2) c.orientation.z=std::sqrt(0.5);
          marker.controls.push_back(c);
        }
      }
    }
    server_->insert(marker, std::bind(&ScanWorkspaceBridge::feedback, this, std::placeholders::_1));
  }

  geometry_msgs::msg::Pose endDisplayPose() const {
    auto pose=manual_goal_.pose;
    pose.position.z+=0.5;
    pose.orientation=geometry_msgs::msg::Quaternion();
    pose.orientation.w=1.0;
    return pose;
  }

  void displayMarkers() {
    if (!have_body_) return;
    const auto current=now();
    if (!markers_created_) {
      makeMarker("start_pos",robot_pose_);
      manual_goal_.header.frame_id=frame_;
      manual_goal_.pose=robot_pose_;
      manual_goal_.pose.position.x+=1.0;
      manual_goal_.pose.orientation=geometry_msgs::msg::Quaternion();
      manual_goal_.pose.orientation.w=1.0;
      makeMarker("end_pos",endDisplayPose());
      markers_created_=true;
      last_start_pose_=robot_pose_;
      last_start_time_=current;
    } else if (current<last_start_time_ || (current-last_start_time_).seconds()>=0.1) {
      last_start_time_=current;
      if (changed(robot_pose_,last_start_pose_)) {
        server_->setPose("start_pos",robot_pose_);
        last_start_pose_=robot_pose_;
      }
    }
    // One publisher flushes at 20Hz; odometry/feedback never publishes markers directly.
    server_->applyChanges();
  }

  void feedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr &input) {
    using Feedback=visualization_msgs::msg::InteractiveMarkerFeedback;
    if (input->event_type!=Feedback::POSE_UPDATE) return;
    if (input->marker_name=="start_pos") {
      server_->setPose("start_pos",robot_pose_);
      return;
    }
    if (input->marker_name!="end_pos") return;
    auto reset=[this]() {
      std_msgs::msg::Header header; header.frame_id=frame_;
      server_->setPose("end_pos",endDisplayPose(),header);
    };
    geometry_msgs::msg::PoseStamped target;
    target.header=input->header;
    target.pose=input->pose;
    try {
      if (!poseInMap(target)) { reset(); return; }
      target.pose.position.z-=0.5;
      if (target.pose.position.z < -0.1) { reset(); return; }
      target.pose.orientation=geometry_msgs::msg::Quaternion();
      target.pose.orientation.w=1.0;
      manual_goal_=target;
      have_manual_goal_=true;
      if (input->header.frame_id!=frame_) reset();
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),3000,"SCAN marker TF: %s",e.what());
      reset();
    }
  }

  void publishManualGoal() {
    if (!have_body_ || !have_manual_goal_) return;
    const auto &p=manual_goal_.pose.position;
    const Eigen::Vector3d end(p.x,p.y,p.z);
    if (have_attempt_ && (end-last_end_).norm()<0.01) return;
    last_end_=end; have_attempt_=true;
    manual_goal_.header.stamp=now();
    goal_pub_->publish(manual_goal_);
  }

  void path(const nav_msgs::msg::Path &input) {
    if (input.poses.empty()) return;
    auto output=input;
    try {
      for (auto &point : output.poses) {
        if (point.header.frame_id.empty()) point.header=input.header;
        if (!poseInMap(point)) {
          RCLCPP_WARN(get_logger(), "Ignoring PCT path: invalid pose or missing frame");
          return;
        }
        // PCT already publishes planning height. SCAN adds body_height to terrain paths.
        point.pose.position.z -= body_height_;
      }
      output.header.frame_id=frame_;
      // Match the quadruped EGO interface: retain points at least 0.8m apart in XYZ.
      // Always retain the exact final pose, including its heading and height.
      std::vector<geometry_msgs::msg::PoseStamped> samples;
      samples.reserve(output.poses.size());
      for (const auto &point : output.poses) {
        const auto &p=point.pose.position;
        if (samples.empty()) {
          samples.push_back(point);
          continue;
        }
        const auto &last=samples.back().pose.position;
        if (Eigen::Vector3d(p.x-last.x,p.y-last.y,p.z-last.z).norm() >= path_sample_distance_)
          samples.push_back(point);
      }
      const auto &end=output.poses.back();
      const auto &p=end.pose.position;
      const auto &last=samples.back().pose.position;
      if (Eigen::Vector3d(p.x-last.x,p.y-last.y,p.z-last.z).norm() > 1e-6)
        samples.push_back(end);
      else
        samples.back()=end;
      output.poses=std::move(samples);
      path_pub_->publish(output);
      RCLCPP_INFO(get_logger(), "PCT path: %zu -> %zu points (spacing %.2f m), published initial_path in %s",
                  input.poses.size(), output.poses.size(), path_sample_distance_, frame_.c_str());
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN(get_logger(), "SCAN path TF: %s", e.what());
    }
  }

  std::unique_ptr<interactive_markers::InteractiveMarkerServer> server_;
  rclcpp::TimerBase::SharedPtr display_timer_, goal_timer_;
  geometry_msgs::msg::Pose robot_pose_, last_start_pose_;
  geometry_msgs::msg::PoseStamped manual_goal_;
  Eigen::Vector3d last_end_=Eigen::Vector3d::Zero();
  rclcpp::Time last_start_time_{0,0,RCL_ROS_TIME};
  bool have_body_=false, markers_created_=false, have_manual_goal_=false, have_attempt_=false;
  std::string robot_frame_;
  std::string frame_, sensor_frame_;
  double body_height_, path_sample_distance_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::shared_ptr<tf2_ros::TransformListener> listener_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr body_pub_, sensor_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<ScanWorkspaceBridge>());
  rclcpp::shutdown();
  return 0;
}
