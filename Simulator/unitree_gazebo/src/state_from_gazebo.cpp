// Gazebo truth owns world -> base. LIO owns the separate estimated body chain.
#include <memory>
#include <string>
#include "gazebo_msgs/msg/link_states.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_ros/transform_broadcaster.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("state_from_gazebo");
  const auto entity = node->declare_parameter<std::string>("entity_name", "a1");
  const auto frame = node->declare_parameter<std::string>("truth_frame", "world");
  auto publisher = node->create_publisher<nav_msgs::msg::Odometry>("/Odometry_gazebo", 1);
  auto broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);
  rclcpp::Time last(0, 0, RCL_ROS_TIME);
  bool have_last = false;
  auto subscription = node->create_subscription<gazebo_msgs::msg::LinkStates>(
      "/link_states", 10, [&](const gazebo_msgs::msg::LinkStates::SharedPtr msg) {
        const auto stamp = node->now();
        if (have_last && stamp >= last && (stamp - last).seconds() < 1. / 500.) return;
        last = stamp; have_last = true;
        size_t index = 0;
        while (index < msg->name.size() && msg->name[index] != entity + "::base") ++index;
        if (index >= msg->name.size() || index >= msg->pose.size() || index >= msg->twist.size()) return;
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = frame;
        odom.child_frame_id = "base";
        odom.pose.pose = msg->pose[index];
        const auto &q = msg->pose[index].orientation;
        tf2::Quaternion rotation(q.x, q.y, q.z, q.w);
        const auto &twist = msg->twist[index];
        const auto linear = tf2::Matrix3x3(rotation).transpose() *
            tf2::Vector3(twist.linear.x, twist.linear.y, twist.linear.z);
        const auto angular = tf2::Matrix3x3(rotation).transpose() *
            tf2::Vector3(twist.angular.x, twist.angular.y, twist.angular.z);
        odom.twist.twist.linear.x = linear.x();
        odom.twist.twist.linear.y = linear.y();
        odom.twist.twist.linear.z = linear.z();
        odom.twist.twist.angular.x = angular.x();
        odom.twist.twist.angular.y = angular.y();
        odom.twist.twist.angular.z = angular.z();
        geometry_msgs::msg::TransformStamped transform;
        transform.header = odom.header;
        transform.child_frame_id = "base";
        transform.transform.translation.x = odom.pose.pose.position.x;
        transform.transform.translation.y = odom.pose.pose.position.y;
        transform.transform.translation.z = odom.pose.pose.position.z;
        transform.transform.rotation = odom.pose.pose.orientation;
        broadcaster->sendTransform(transform);
        publisher->publish(odom);
      });
  RCLCPP_INFO(node->get_logger(), "Gazebo truth: %s -> base; entity=%s", frame.c_str(), entity.c_str());
  rclcpp::spin(node);
  subscription.reset(); broadcaster.reset(); publisher.reset(); node.reset();
  rclcpp::shutdown();
  return 0;
}
