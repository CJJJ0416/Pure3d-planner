//
// Created by biao on 24-9-11.
//


#ifndef KEYBOARDINPUT_H
#define KEYBOARDINPUT_H
#include <rclcpp/rclcpp.hpp>
#include <control_input_msgs/msg/inputs.hpp>
#include <termios.h>
#include <geometry_msgs/msg/twist.hpp>


template <typename T1, typename T2>
T1 max(const T1 a, const T2 b) {
    return (a > b ? a : b);
}

template <typename T1, typename T2>
T1 min(const T1 a, const T2 b) {
    return (a < b ? a : b);
}


class KeyboardInput final : public rclcpp::Node {
public:
    KeyboardInput();

    ~KeyboardInput() override {
        if (terminal_configured_) tcsetattr(STDIN_FILENO, TCSANOW, &old_tio_);
    }

private:
    void timer_callback();

    void check_command(char key);
    void check_rl_velocity(char key);
    void publish_rl_velocity();
    void check_value(char key);

    static bool kbhit();

    control_input_msgs::msg::Inputs inputs_;
    rclcpp::Publisher<control_input_msgs::msg::Inputs>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::TimerBase::SharedPtr velocity_timer_;
    geometry_msgs::msg::Twist velocity_command_;
    bool publish_cmd_vel_ = false, rl_keyboard_mode_ = false, velocity_active_ = false;
    bool terminal_configured_ = false;
    double forward_speed_ = 0.5, lateral_speed_ = 0.25, turn_speed_ = 0.5;

    bool just_published_ = false;
    int reset_count_ = 0;

    float sensitivity_left_ = 0.05;
    float sensitivity_right_ = 0.05;
    termios old_tio_{}, new_tio_{};
};


#endif //KEYBOARDINPUT_H
