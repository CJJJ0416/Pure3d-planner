// Adapted from glocal-nav State_RL_test.cpp (Unitree Robotics, 2020-2023).
#include "unitree_guide_controller/FSM/State_RL_test.h"
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace unitree_guide_controller {
namespace {
constexpr size_t kDofs = 12, kObs = 45, kHistory = 5;
// Training order FL, FR, RL, RR, unlike the guide controller FR, FL, RR, RL.
const std::array<std::string, kDofs> kPolicyJoints = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
constexpr std::array<float, kDofs> kDefault = {
    -.15f, .55f, -1.5f, .15f, .55f, -1.5f,
    -.15f, .7f, -1.5f, .15f, .7f, -1.5f};
struct VelocityCommand {
    std::array<float, 3> value{};
    int64_t stamp = 0;
    uint64_t epoch = 0;  // 按键门控代次，防止恢复时使用缓停前缓存的速度。
};
struct Snapshot { std::array<float, kObs> obs{}; int64_t stamp = 0; };
struct Result { std::array<float, kDofs> action{}; int64_t stamp = 0; };
}

struct StateRL::Impl {
    CtrlInterfaces& interfaces;
    rclcpp_lifecycle::LifecycleNode::SharedPtr node;
    std::array<size_t, kDofs> policy_to_controller{};
    torch::jit::Module model;
    torch::Device device{torch::kCPU};
    realtime_tools::RealtimeBuffer<VelocityCommand> velocity;
    // 【用户指定改动】偶数=接收，奇数=缓停/保持零；回调与控制线程通过原子代次隔离。
    std::atomic<uint64_t> velocity_epoch{0};
    std::array<float, 3> effective_velocity{}, soft_stop_initial{};
    double soft_stop_elapsed = 0, soft_stop_duration;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr trace;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable ready;
    Snapshot pending;
    Result result;
    bool has_pending = false, has_result = false, stopping = false;
    std::atomic<bool> failed{false};
    std::array<float, kDofs> last_action{};
    std::array<double, kDofs> hold_position{};
    double elapsed = 0, inference_period, cmd_timeout, inference_timeout, kp, kd;
    int64_t enter_time = -1, result_time = -1;

    Impl(CtrlInterfaces& i, rclcpp_lifecycle::LifecycleNode::SharedPtr n,
         const std::vector<std::string>& joints) : interfaces(i), node(std::move(n)) {
        for (size_t p = 0; p < kDofs; ++p) {
            const auto it = std::find(joints.begin(), joints.end(), kPolicyJoints[p]);
            if (it == joints.end()) throw std::runtime_error("RL policy requires A1 joint " + kPolicyJoints[p]);
            policy_to_controller[p] = std::distance(joints.begin(), it);
        }
        const auto path = node->get_parameter("rl.model_path").as_string();
        const auto device_name = node->get_parameter("rl.device").as_string();
        if (device_name != "cpu" && device_name != "cuda") throw std::runtime_error("rl.device must be cpu or cuda");
        if (device_name == "cuda") device = torch::Device(torch::kCUDA);
        inference_period = 1.0 / node->get_parameter("rl.inference_rate").as_double();
        cmd_timeout = node->get_parameter("rl.cmd_timeout").as_double();
        soft_stop_duration = node->get_parameter("rl.soft_stop_duration").as_double();
        inference_timeout = node->get_parameter("rl.inference_timeout").as_double();
        kp = node->get_parameter("rl.kp").as_double();
        kd = node->get_parameter("rl.kd").as_double();
        if (!std::isfinite(inference_period) || inference_period <= 0 ||
            cmd_timeout <= 0 || inference_timeout <= inference_period || kp < 0 || kd < 0 ||
            !std::isfinite(soft_stop_duration) || soft_stop_duration <= 0)
            throw std::runtime_error("Invalid RL timing or gains");
        at::set_num_threads(1);
        torch::NoGradGuard guard;
        model = torch::jit::load(path, device);
        model.eval();
        auto warmup = model.get_method("act_inference")({torch::zeros({1, kHistory * kObs},
            torch::TensorOptions().dtype(torch::kFloat32).device(device))}).toTensor();
        if (warmup.numel() != kDofs || !torch::isfinite(warmup).all().item<bool>())
            throw std::runtime_error("Policy act_inference must return 12 finite actions for [1,225]");
        subscription = node->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 1, [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
                const auto epoch = velocity_epoch.load(std::memory_order_acquire);
                if (epoch % 2 != 0) return;  // 缓停后即使 EGO 继续发布，也不更新目标速度。
                VelocityCommand command;
                command.value = {static_cast<float>(msg->linear.x), static_cast<float>(msg->linear.y),
                                 static_cast<float>(msg->angular.z)};
                for (float v : command.value) if (!std::isfinite(v)) return;
                command.stamp = node->now().nanoseconds();
                command.epoch = epoch;
                velocity.writeFromNonRT(command);
            });
        if (node->get_parameter("rl.publish_trace").as_bool())
            trace = node->create_publisher<std_msgs::msg::Float32MultiArray>("~/rl_trace", 10);
        RCLCPP_INFO(node->get_logger(), "RL policy loaded: %s (%s), input [1,225], output 12, %.1f Hz",
                    path.c_str(), device_name.c_str(), 1.0 / inference_period);
    }

    void infer() {
        try {
            torch::NoGradGuard guard;
            auto history = torch::zeros({kHistory, kObs}, torch::kFloat32);
            bool first = true;
            while (true) {
                Snapshot sample;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock, [this] { return stopping || has_pending; });
                    if (stopping) break;
                    sample = pending;
                    has_pending = false;
                }
                auto obs = torch::from_blob(sample.obs.data(), {1, kObs}, torch::kFloat32).clone();
                if (first) { history = obs.repeat({kHistory, 1}); first = false; }
                else history = torch::cat({history.slice(0, 1, kHistory), obs}, 0);
                auto actions = model.get_method("act_inference")(
                    {history.reshape({1, kHistory * kObs}).to(device)}).toTensor()
                    .to(torch::kCPU).to(torch::kFloat32).contiguous().reshape({-1});
                if (actions.numel() != kDofs || !torch::isfinite(actions).all().item<bool>())
                    throw std::runtime_error("Policy produced invalid actions");
                Result output;
                output.stamp = sample.stamp;
                std::copy_n(actions.data_ptr<float>(), kDofs, output.action.begin());
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    result = output;
                    has_result = true;
                }
                if (trace) {
                    // [sample_sim_time, flattened history (225), unscaled actions (12)].
                    std_msgs::msg::Float32MultiArray msg;
                    msg.data.reserve(238);
                    msg.data.push_back(sample.stamp * 1e-9);
                    msg.data.insert(msg.data.end(), history.data_ptr<float>(), history.data_ptr<float>() + kHistory*kObs);
                    msg.data.insert(msg.data.end(), output.action.begin(), output.action.end());
                    trace->publish(msg);
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node->get_logger(), "RL inference failed: %s", e.what());
            failed = true;
        }
    }

    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_one();
        if (worker.joinable()) worker.join();
    }
};

StateRL::StateRL(CtrlInterfaces& interfaces, rclcpp_lifecycle::LifecycleNode::SharedPtr node,
                 const std::vector<std::string>& joints)
    : FSMState(FSMStateName::RL, "RL", interfaces), impl_(std::make_unique<Impl>(interfaces, std::move(node), joints)) {}
StateRL::~StateRL() { impl_->stop(); }

void StateRL::enter() {
    impl_->stop();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = false;
        impl_->has_pending = impl_->has_result = false;
    }
    impl_->failed = false;
    // 重新进入 RL 时不复用旧速度；P 的暂停状态保留，必须再次按 P 才恢复。
    impl_->velocity_epoch.fetch_add(2, std::memory_order_acq_rel);
    impl_->effective_velocity.fill(0);
    impl_->soft_stop_initial.fill(0);
    impl_->soft_stop_elapsed = impl_->soft_stop_duration;
    impl_->last_action.fill(0);
    impl_->elapsed = impl_->inference_period;
    impl_->enter_time = impl_->result_time = -1;
    for (size_t j = 0; j < kDofs; ++j) {
        impl_->hold_position[j] = ctrl_interfaces_.joint_position_state_interface_[j].get().get_value();
        ctrl_interfaces_.joint_position_command_interface_[j].get().set_value(impl_->hold_position[j]);
        ctrl_interfaces_.joint_velocity_command_interface_[j].get().set_value(0);
        ctrl_interfaces_.joint_torque_command_interface_[j].get().set_value(0);
        ctrl_interfaces_.joint_kp_command_interface_[j].get().set_value(impl_->kp);
        ctrl_interfaces_.joint_kd_command_interface_[j].get().set_value(impl_->kd);
    }
    ctrl_interfaces_.control_inputs_.command = 0;
    impl_->worker = std::thread(&Impl::infer, impl_.get());
}

void StateRL::run(const rclcpp::Time& time, const rclcpp::Duration& period) {
    if (impl_->enter_time < 0) impl_->enter_time = time.nanoseconds();
    {
        std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
        if (lock.owns_lock() && impl_->has_result) {
            impl_->last_action = impl_->result.action;
            impl_->result_time = impl_->result.stamp;
            impl_->has_result = false;
        }
    }
    const auto last_time = impl_->result_time >= 0 ? impl_->result_time : impl_->enter_time;
    if ((time.nanoseconds() - last_time) * 1e-9 > impl_->inference_timeout) {
        if (!impl_->failed.exchange(true)) RCLCPP_ERROR(impl_->node->get_logger(), "RL inference timed out; switching to passive");
    }
    if (impl_->failed) return;
    // 【用户指定改动：非阻塞缓停】每个控制周期只推进一步，for 处理 vx/vy/yaw 三个分量。
    // 不在这里用 for+sleep 连续执行整个减速过程，关节控制与 RL 推理需要持续运行。
    if (impl_->velocity_epoch.load(std::memory_order_acquire) % 2 != 0) {
        impl_->soft_stop_elapsed = std::min(impl_->soft_stop_duration,
            impl_->soft_stop_elapsed + std::max(0.0, period.seconds()));
        const float scale = static_cast<float>(1.0 - impl_->soft_stop_elapsed / impl_->soft_stop_duration);
        for (size_t j = 0; j < 3; ++j)
            impl_->effective_velocity[j] = impl_->soft_stop_initial[j] * scale;
    } else {
        const auto command = *impl_->velocity.readFromRT();
        const double command_age = (time.nanoseconds() - command.stamp) * 1e-9;
        const bool fresh = command.epoch == impl_->velocity_epoch.load(std::memory_order_acquire) &&
            command_age >= 0 && command_age <= impl_->cmd_timeout;
        for (size_t j = 0; j < 3; ++j)
            impl_->effective_velocity[j] = fresh ? command.value[j] : 0.f;
    }
    impl_->elapsed += period.seconds();
    if (impl_->elapsed >= impl_->inference_period) {
        Snapshot sample;
        sample.stamp = time.nanoseconds();
        // IMU gyro is already expressed in the body frame. Original p3d gyro was world-frame.
        Eigen::Quaterniond q(
            ctrl_interfaces_.imu_state_interface_[0].get().get_value(),
            ctrl_interfaces_.imu_state_interface_[1].get().get_value(),
            ctrl_interfaces_.imu_state_interface_[2].get().get_value(),
            ctrl_interfaces_.imu_state_interface_[3].get().get_value());
        if (!q.coeffs().allFinite() || q.norm() < .5) { impl_->failed = true; return; }
        const auto gravity = q.normalized().conjugate() * Eigen::Vector3d(0, 0, -1);
        for (size_t j = 0; j < 3; ++j) {
            sample.obs[j] = ctrl_interfaces_.imu_state_interface_[4+j].get().get_value() * .25f;
            sample.obs[3+j] = gravity[j];
            sample.obs[6+j] = impl_->effective_velocity[j] * (j == 2 ? .25f : 2.f);
        }
        for (size_t p = 0; p < kDofs; ++p) {
            const auto j = impl_->policy_to_controller[p];
            sample.obs[9+p] = ctrl_interfaces_.joint_position_state_interface_[j].get().get_value() - kDefault[p];
            sample.obs[21+p] = ctrl_interfaces_.joint_velocity_state_interface_[j].get().get_value() * .05f;
            sample.obs[33+p] = impl_->last_action[p];
        }
        for (float v : sample.obs) if (!std::isfinite(v)) { impl_->failed = true; return; }
        std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
        if (lock.owns_lock()) {
            impl_->pending = sample;
            impl_->has_pending = true;
            impl_->elapsed = std::fmod(impl_->elapsed, impl_->inference_period);
            impl_->ready.notify_one();
        }
    }
    if (impl_->result_time >= 0) {
        for (size_t p = 0; p < kDofs; ++p) {
            const auto j = impl_->policy_to_controller[p];
            ctrl_interfaces_.joint_position_command_interface_[j].get().set_value(kDefault[p] + .25f * impl_->last_action[p]);
        }
    }
}

void StateRL::exit() { impl_->stop(); }
void StateRL::toggleVelocityReception() {
    // 只由 FSM 控制线程调用。推进代次后，任何在途旧回调的消息也无法进入策略。
    const auto epoch = impl_->velocity_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (epoch % 2 != 0) {
        impl_->soft_stop_initial = impl_->effective_velocity;
        impl_->soft_stop_elapsed = 0;
        RCLCPP_INFO(impl_->node->get_logger(), "P: block /cmd_vel; ramp vx/vy/yaw to zero in %.2f s.",
                    impl_->soft_stop_duration);
    } else {
        RCLCPP_INFO(impl_->node->get_logger(), "P: resume /cmd_vel; waiting for a fresh velocity command.");
    }
}
FSMStateName StateRL::checkChange() {
    if (impl_->failed || ctrl_interfaces_.control_inputs_.command == 1) return FSMStateName::PASSIVE;
    if (ctrl_interfaces_.control_inputs_.command == 2) return FSMStateName::FIXEDSTAND;
    return FSMStateName::RL;
}
}
