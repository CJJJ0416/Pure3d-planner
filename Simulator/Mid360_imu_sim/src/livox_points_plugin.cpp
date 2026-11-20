// ROS2 port of Mid360_imu_sim, MIT (c) 2021 livox.
#include "livox_laser_simulation/livox_points_plugin.h"
#include "livox_laser_simulation/csv_reader.hpp"
#include <gazebo/physics/PhysicsEngine.hh>
#include <gazebo/physics/physics.hh>
#include <sensor_msgs/msg/point_field.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <boost/thread/recursive_mutex.hpp>
namespace gazebo {
GZ_REGISTER_SENSOR_PLUGIN(LivoxPointsPlugin)
LivoxPointsPlugin::~LivoxPointsPlugin() {
 connection_.reset();
 if (rays_) rays_->Fini();
 rays_.reset(); collision_.reset();
 custom_pub_.reset(); alias_pub_.reset(); cloud_pub_.reset(); ros_node_.reset();
}
void LivoxPointsPlugin::Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) {
 ros_node_ = gazebo_ros::Node::Get(sdf);
 if (!ros_node_) return;
 sensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(parent);
 if (!sensor_) { RCLCPP_ERROR(ros_node_->get_logger(), "MID360 requires a ray sensor"); return; }
 world_ = physics::get_world(parent->WorldName());
 if (!world_ || world_->Physics()->GetType() != "ode") {
   RCLCPP_ERROR(ros_node_->get_logger(), "MID360 requires ODE physics"); return;
 }
 const auto csv = sdf->Get<std::string>("csv_file_name");
 std::vector<std::vector<double>> data;
 if (!CsvReader::ReadCsvFile(csv, data)) { RCLCPP_ERROR(ros_node_->get_logger(), "Cannot load %s", csv.c_str()); return; }
 for (const auto& row : data) if (row.size()==3 && std::isfinite(row[1]) && std::isfinite(row[2]))
   pattern_.push_back({row[1]*M_PI/180, row[2]*M_PI/180-M_PI_2});
 const auto count = sdf->Get<int>("samples"); const auto stride = sdf->Get<int>("downsample");
 if (pattern_.empty() || count < 1 || stride < 1 || count > 1000000) {
   RCLCPP_ERROR(ros_node_->get_logger(), "Invalid MID360 pattern/sampling parameters"); return;
 }
 samples_=count; downsample_=stride; frame_=sdf->Get<std::string>("frame_name");
 const auto range=sdf->GetElement("ray")->GetElement("range");
 min_range_=range->Get<double>("min"); max_range_=range->Get<double>("max");
 if (!(min_range_>=0 && max_range_>min_range_)) { RCLCPP_ERROR(ros_node_->get_logger(), "Invalid ranges"); return; }
 collision_=world_->Physics()->CreateCollision("multiray", parent->ParentName());
 collision_->SetName("mid360_pattern_collision");
 collision_->SetRelativePose(parent->Pose()); collision_->SetInitialRelativePose(parent->Pose());
 rays_.reset(new physics::LivoxOdeMultiRayShape(collision_));
 collision_->SetShape(rays_); rays_->Load(sdf); rays_->Init();
 for (size_t j=0;j<samples_;j+=downsample_) {
   const auto& info=pattern_[j%pattern_.size()];
   ignition::math::Quaterniond q(0,info.zenith,info.azimuth);
   const auto axis=parent->Pose().Rot()*q*ignition::math::Vector3d::UnitX;
   rays_->AddRay(parent->Pose().Pos()+axis*min_range_,parent->Pose().Pos()+axis*max_range_);
 }
 const auto qos=rclcpp::SensorDataQoS().reliable();
 cloud_pub_=ros_node_->create_publisher<sensor_msgs::msg::PointCloud2>("/scan",qos);
 alias_pub_=ros_node_->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar",qos);
 custom_pub_=ros_node_->create_publisher<livox_ros_driver2::msg::CustomMsg>("/livox/lidar2",qos);
 connection_=parent->ConnectUpdated(std::bind(&LivoxPointsPlugin::OnUpdate,this));
 parent->SetActive(true);
 RCLCPP_INFO(ros_node_->get_logger(), "MID360 ROS2 loaded: %zu pattern rows, %zu rays/frame, frame=%s",pattern_.size(),rays_->RayShapes().size(),frame_.c_str());
}
void LivoxPointsPlugin::SetRays() {
 size_t ray=0; const auto pose=sensor_->Pose();
 for (size_t j=0;j<samples_;j+=downsample_,++ray) {
   const auto& info=pattern_[(start_+j)%pattern_.size()];
   ignition::math::Quaterniond q(0,info.zenith,info.azimuth);
   const auto axis=pose.Rot()*q*ignition::math::Vector3d::UnitX;
   rays_->RayShapes()[ray]->SetPoints(pose.Pos()+axis*min_range_,pose.Pos()+axis*max_range_);
 }
}
void LivoxPointsPlugin::OnUpdate() {
 if (!rays_ || !rclcpp::ok()) return;
 // Freeze geometry for this instantaneous scan; release before ROS serialization.
 boost::recursive_mutex::scoped_lock physics_lock(*world_->Physics()->GetPhysicsUpdateMutex());
 SetRays(); rays_->Update();
 const auto stamp=world_->SimTime();
 livox_ros_driver2::msg::CustomMsg custom;
 custom.header.stamp.sec=stamp.sec; custom.header.stamp.nanosec=stamp.nsec;
 custom.header.frame_id=frame_; custom.timebase=static_cast<uint64_t>(stamp.sec)*1000000000ULL+stamp.nsec;
 custom.lidar_id=1; custom.points.reserve(rays_->RayShapes().size());
 for(size_t j=0,idx=0;j<samples_;j+=downsample_,++idx) {
   const auto range=rays_->GetRange(idx);
   if (!std::isfinite(range) || range<=min_range_ || range>=max_range_) continue;
   const auto& info=pattern_[(start_+j)%pattern_.size()];
   const auto pt=ignition::math::Quaterniond(0,info.zenith,info.azimuth)*ignition::math::Vector3d::UnitX*range;
   livox_ros_driver2::msg::CustomPoint point;
   point.x=pt.X();point.y=pt.Y();point.z=pt.Z();
   point.reflectivity=static_cast<uint8_t>(std::clamp(rays_->GetRetro(idx),0.0,255.0));
   point.tag=0x10; point.line=0; point.offset_time=0; // All rays measured at this one simulation instant.
   custom.points.push_back(point);
 }
 physics_lock.unlock();
 start_=(start_+samples_)%pattern_.size(); custom.point_num=custom.points.size();
 sensor_msgs::msg::PointCloud2 cloud;
 cloud.header=custom.header; cloud.height=1;cloud.width=custom.point_num;
 cloud.is_bigendian=false;cloud.is_dense=true;cloud.point_step=28;cloud.row_step=cloud.point_step*cloud.width;
 const char* names[]={"x","y","z","intensity","time","offset_time","line","tag","reflectivity"};
 const uint32_t offsets[]={0,4,8,12,16,20,24,25,26};
 for(size_t k=0;k<9;++k) {
  sensor_msgs::msg::PointField f;f.name=names[k];f.offset=offsets[k];f.count=1;
  f.datatype=k<5?sensor_msgs::msg::PointField::FLOAT32:k==5?sensor_msgs::msg::PointField::UINT32:sensor_msgs::msg::PointField::UINT8;
  cloud.fields.push_back(f);
 }
 cloud.data.resize(cloud.row_step,0);
 for(size_t k=0;k<custom.points.size();++k) {
  const auto& p=custom.points[k];float values[]={p.x,p.y,p.z,static_cast<float>(p.reflectivity),0.f};
  auto* out=cloud.data.data()+k*cloud.point_step;std::memcpy(out,values,sizeof(values));out[24]=p.line;out[25]=p.tag;out[26]=p.reflectivity;
 }
 cloud_pub_->publish(cloud);alias_pub_->publish(cloud);custom_pub_->publish(custom);
}
}
