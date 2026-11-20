// ROS2 port of Mid360_imu_sim, MIT (c) 2021 livox.
#pragma once
#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "livox_ode_multiray_shape.h"
namespace gazebo {
struct AviaRotateInfo { double azimuth; double zenith; };
class LivoxPointsPlugin : public SensorPlugin {
public:
 ~LivoxPointsPlugin() override;
 void Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) override;
private:
 void OnUpdate();
 void SetRays();
 gazebo_ros::Node::SharedPtr ros_node_;
 rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_, alias_pub_;
 rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr custom_pub_;
 sensors::RaySensorPtr sensor_;
 physics::WorldPtr world_;
 physics::CollisionPtr collision_;
 boost::shared_ptr<physics::LivoxOdeMultiRayShape> rays_;
 event::ConnectionPtr connection_;
 std::vector<AviaRotateInfo> pattern_;
 std::string frame_;
 size_t start_ = 0, samples_ = 24000, downsample_ = 1;
 double min_range_ = .1, max_range_ = 40;
};
}
