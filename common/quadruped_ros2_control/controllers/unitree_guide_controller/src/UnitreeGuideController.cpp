//
// Created by tlab-uav on 24-9-6.
//

#include "unitree_guide_controller/UnitreeGuideController.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>

#include <unitree_guide_controller/gait/WaveGenerator.h>
#include "unitree_guide_controller/robot/QuadrupedRobot.h"

namespace unitree_guide_controller
{
    using config_type = controller_interface::interface_configuration_type;

    controller_interface::InterfaceConfiguration UnitreeGuideController::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

        conf.names.reserve(joint_names_.size() * command_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : command_interface_types_)
            {
                if (!command_prefix_.empty())
                {
                    conf.names.push_back(command_prefix_ + "/" + joint_name + "/" += interface_type);
                }
                else
                {
                    conf.names.push_back(joint_name + "/" += interface_type);
                }
            }
        }

        return conf;
    }

    controller_interface::InterfaceConfiguration UnitreeGuideController::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

        conf.names.reserve(joint_names_.size() * state_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : state_interface_types_)
            {
                conf.names.push_back(joint_name + "/" += interface_type);
            }
        }

        for (const auto& interface_type : imu_interface_types_)
        {
            conf.names.push_back(imu_name_ + "/" += interface_type);
        }

        return conf;
    }

    controller_interface::return_type UnitreeGuideController::
    update(const rclcpp::Time& time, const rclcpp::Duration& period)
    {
        // auto now = std::chrono::steady_clock::now();
        // std::chrono::duration<double> time_diff = now - last_update_time_;
        // last_update_time_ = now;
        //
        // // Calculate the frequency
        // update_frequency_ = 1.0 / time_diff.count();
        // RCLCPP_INFO(get_node()->get_logger(), "Update frequency: %f Hz", update_frequency_);

        if (ctrl_component_.robot_model_ == nullptr)
        {
            return controller_interface::return_type::OK;
        }

        const auto event = *input_buffer_.readFromRT();
        if (event.sequence != consumed_sequence_) {
            // Analog updates/neutral messages cannot erase an unconsumed mode key.
            const auto command = ctrl_interfaces_.control_inputs_.command;
            ctrl_interfaces_.control_inputs_ = event.input;
            ctrl_interfaces_.control_inputs_.command = command;
            consumed_sequence_ = event.sequence;
        }
        const auto mode_event = *mode_buffer_.readFromRT();
        if (mode_event.sequence != consumed_mode_sequence_) {
            ctrl_interfaces_.control_inputs_.command = mode_event.input.command;
            consumed_mode_sequence_ = mode_event.sequence;
        }
        // 【用户指定改动：P 缓停/启动】11 是独立的速度门控事件，不切换 RL/FSM 状态。
        // 事件只消费一次，后续键盘 command=0 不会重复切换或覆盖未消费的按键。
        if (ctrl_interfaces_.control_inputs_.command == 11) {
            ctrl_interfaces_.control_inputs_.command = 0;
            if (mode_ == FSMMode::NORMAL && current_state_->state_name == FSMStateName::RL) {
                state_list_.rl->toggleVelocityReception();
            } else {
                RCLCPP_INFO(get_node()->get_logger(), "P soft-stop/resume is available after entering RL (key 7).");
            }
        }
        ctrl_component_.robot_model_->update();
        ctrl_component_.wave_generator_->update();
        ctrl_component_.estimator_->update();

        if (mode_ == FSMMode::NORMAL)
        {
            current_state_->run(time, period);
            next_state_name_ = current_state_->checkChange();
            if (next_state_name_ != current_state_->state_name)
            {
                mode_ = FSMMode::CHANGE;
                next_state_ = getNextState(next_state_name_);
                RCLCPP_INFO(get_node()->get_logger(), "Switched from %s to %s",
                            current_state_->state_name_string.c_str(), next_state_->state_name_string.c_str());
            }
        }
        else if (mode_ == FSMMode::CHANGE)
        {
            current_state_->exit();
            current_state_ = next_state_;

            current_state_->enter();
            mode_ = FSMMode::NORMAL;
        }

        return controller_interface::return_type::OK;
    }

    controller_interface::CallbackReturn UnitreeGuideController::on_init()
    {
        try
        {
            joint_names_ = auto_declare<std::vector<std::string>>("joints", joint_names_);
            command_interface_types_ =
                auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);
            state_interface_types_ =
                auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);

            // imu sensor
            imu_name_ = auto_declare<std::string>("imu_name", imu_name_);
            base_name_ = auto_declare<std::string>("base_name", base_name_);
            imu_interface_types_ = auto_declare<std::vector<std::string>>("imu_interfaces", state_interface_types_);
            command_prefix_ = auto_declare<std::string>("command_prefix", command_prefix_);
            feet_names_ =
                auto_declare<std::vector<std::string>>("feet_names", feet_names_);

            auto_declare<std::string>("rl.model_path",
                ament_index_cpp::get_package_share_directory("unitree_guide_controller") +
                "/policies/policy_act_inference_stair.pt");
            auto_declare<std::string>("rl.device", "cpu");
            auto_declare<double>("rl.inference_rate", 50.0);
            auto_declare<double>("rl.cmd_timeout", 0.5);
            // 从已送入策略的 vx/vy/yaw 指令线性降到零的时间，单位 s。
            auto_declare<double>("rl.soft_stop_duration", 1.0);
            auto_declare<double>("rl.inference_timeout", 0.3);
            auto_declare<double>("rl.kp", 80.0);
            auto_declare<double>("rl.kd", 1.0);
            auto_declare<bool>("rl.publish_trace", false);

            // pose parameters
            down_pos_ = auto_declare<std::vector<double>>("down_pos", down_pos_);
            stand_pos_ = auto_declare<std::vector<double>>("stand_pos", stand_pos_);
            stand_kp_ = auto_declare<double>("stand_kp", stand_kp_);
            stand_kd_ = auto_declare<double>("stand_kd", stand_kd_);

            get_node()->get_parameter("update_rate", ctrl_interfaces_.frequency_);
            RCLCPP_INFO(get_node()->get_logger(), "Controller Manager Update Rate: %d Hz", ctrl_interfaces_.frequency_);

            ctrl_component_.estimator_ = std::make_shared<Estimator>(ctrl_interfaces_, ctrl_component_);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
            return controller_interface::CallbackReturn::ERROR;
        }

        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn UnitreeGuideController::on_configure(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        control_input_subscription_ = get_node()->create_subscription<control_input_msgs::msg::Inputs>(
            "/control_input", 10, [this](const control_input_msgs::msg::Inputs::SharedPtr msg)
            {
                InputEvent event;
                event.sequence = ++input_sequence_;
                event.input = *msg;
                input_buffer_.writeFromNonRT(event);
                if (msg->command != 0) {
                    event.sequence = ++mode_sequence_;
                    mode_buffer_.writeFromNonRT(event);
                }
            });

        robot_description_subscription_ = get_node()->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(),
            [this](const std_msgs::msg::String::SharedPtr msg)
            {
                ctrl_component_.robot_model_ = std::make_shared<QuadrupedRobot>(
                    ctrl_interfaces_, msg->data, feet_names_, base_name_);
                ctrl_component_.balance_ctrl_ = std::make_shared<BalanceCtrl>(ctrl_component_.robot_model_);
            });

        try {
            state_list_.rl = std::make_shared<StateRL>(ctrl_interfaces_, get_node(), joint_names_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_node()->get_logger(), "RL configuration failed: %s", e.what());
            return CallbackReturn::ERROR;
        }
        ctrl_component_.wave_generator_ = std::make_shared<WaveGenerator>(0.45, 0.5, Vec4(0, 0.5, 0.5, 0));

        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn
    UnitreeGuideController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        // clear out vectors in case of restart
        ctrl_interfaces_.clear();

        // Bind by full name; neither loaned-interface order nor map iteration is a policy contract.
        for (const auto& joint : joint_names_) {
            for (const auto& type : command_interface_types_) {
                const auto full = (command_prefix_.empty() ? "" : command_prefix_ + "/") + joint + "/" + type;
                const auto it = std::find_if(command_interfaces_.begin(), command_interfaces_.end(),
                    [&full](const auto& interface) { return interface.get_name() == full; });
                if (it == command_interfaces_.end() || !command_interface_map_.count(type)) return CallbackReturn::ERROR;
                command_interface_map_.at(type)->push_back(*it);
            }
            for (const auto& type : state_interface_types_) {
                const auto full = joint + "/" + type;
                const auto it = std::find_if(state_interfaces_.begin(), state_interfaces_.end(),
                    [&full](const auto& interface) { return interface.get_name() == full; });
                if (it == state_interfaces_.end() || !state_interface_map_.count(type)) return CallbackReturn::ERROR;
                state_interface_map_.at(type)->push_back(*it);
            }
        }
        for (const auto& type : imu_interface_types_) {
            const auto full = imu_name_ + "/" + type;
            const auto it = std::find_if(state_interfaces_.begin(), state_interfaces_.end(),
                [&full](const auto& interface) { return interface.get_name() == full; });
            if (it == state_interfaces_.end()) return CallbackReturn::ERROR;
            ctrl_interfaces_.imu_state_interface_.push_back(*it);
        }
        if (joint_names_.size() != 12 || ctrl_interfaces_.imu_state_interface_.size() != 10)
            return CallbackReturn::ERROR;

        // Create FSM List
        state_list_.passive = std::make_shared<StatePassive>(ctrl_interfaces_);
        state_list_.fixedDown = std::make_shared<StateFixedDown>(ctrl_interfaces_, down_pos_, stand_kp_, stand_kd_);
        state_list_.fixedStand = std::make_shared<StateFixedStand>(ctrl_interfaces_, stand_pos_, stand_kp_, stand_kd_);
        state_list_.swingTest = std::make_shared<StateSwingTest>(ctrl_interfaces_, ctrl_component_);
        state_list_.freeStand = std::make_shared<StateFreeStand>(ctrl_interfaces_, ctrl_component_);
        state_list_.balanceTest = std::make_shared<StateBalanceTest>(ctrl_interfaces_, ctrl_component_);
        state_list_.trotting = std::make_shared<StateTrotting>(ctrl_interfaces_, ctrl_component_);

        // Initialize FSM
        current_state_ = state_list_.passive;
        consumed_mode_sequence_ = mode_buffer_.readFromRT()->sequence;
        current_state_->enter();
        next_state_ = current_state_;
        next_state_name_ = current_state_->state_name;
        mode_ = FSMMode::NORMAL;

        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn UnitreeGuideController::on_deactivate(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        if (current_state_) current_state_->exit();
        if (state_list_.passive) state_list_.passive->enter();
        release_interfaces();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn
    UnitreeGuideController::on_cleanup(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        if (state_list_.rl) state_list_.rl->exit();
        state_list_.rl.reset();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn
    UnitreeGuideController::on_error(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        if (state_list_.rl) state_list_.rl->exit();
        return CallbackReturn::SUCCESS;
    }

    controller_interface::CallbackReturn
    UnitreeGuideController::on_shutdown(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        if (state_list_.rl) state_list_.rl->exit();
        return CallbackReturn::SUCCESS;
    }

    std::shared_ptr<FSMState> UnitreeGuideController::getNextState(const FSMStateName stateName) const
    {
        switch (stateName)
        {
        case FSMStateName::RL:
            return state_list_.rl;
        case FSMStateName::INVALID:
            return state_list_.invalid;
        case FSMStateName::PASSIVE:
            return state_list_.passive;
        case FSMStateName::FIXEDDOWN:
            return state_list_.fixedDown;
        case FSMStateName::FIXEDSTAND:
            return state_list_.fixedStand;
        case FSMStateName::FREESTAND:
            return state_list_.freeStand;
        case FSMStateName::TROTTING:
            return state_list_.trotting;
        case FSMStateName::SWINGTEST:
            return state_list_.swingTest;
        case FSMStateName::BALANCETEST:
            return state_list_.balanceTest;
        default:
            return state_list_.invalid;
        }
    }
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(unitree_guide_controller::UnitreeGuideController, controller_interface::ControllerInterface);
