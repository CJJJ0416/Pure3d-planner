// Policy adapter for the State_RL_test.cpp supplied by glocal-nav.
// Copyright (c) 2020-2023 Unitree Robotics. Original policy contract retained.
#pragma once

#include <controller_common/FSM/FSMState.h>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <memory>
#include <vector>

namespace unitree_guide_controller {
class StateRL final : public FSMState {
public:
    StateRL(CtrlInterfaces& interfaces,
            rclcpp_lifecycle::LifecycleNode::SharedPtr node,
            const std::vector<std::string>& joints);
    ~StateRL() override;
    void enter() override;
    void run(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void exit() override;
    FSMStateName checkChange() override;
    // 【用户指定改动】由 FSM 在控制线程中响应 P 键：切换缓停 / 恢复接收速度。
    void toggleVelocityReception();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
