//
// Created by biao on 24-9-11.
//

#include <keyboard_input/KeyboardInput.h>
#include <cmath>
#include <stdexcept>
#include <unistd.h>
#include <cctype>

KeyboardInput::KeyboardInput() : Node("keyboard_input_node") {
    publisher_ = create_publisher<control_input_msgs::msg::Inputs>("control_input", 10);
    timer_ = create_wall_timer(std::chrono::milliseconds(10), std::bind(&KeyboardInput::timer_callback, this));
    inputs_ = control_input_msgs::msg::Inputs();

    publish_cmd_vel_ = declare_parameter("publish_cmd_vel", false);
    forward_speed_ = declare_parameter("forward_speed", 0.5);
    lateral_speed_ = declare_parameter("lateral_speed", 0.25);
    turn_speed_ = declare_parameter("turn_speed", 0.5);
    for (double value : {forward_speed_, lateral_speed_, turn_speed_})
        if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Keyboard speeds must be finite and positive");
    if (publish_cmd_vel_) {
        velocity_publisher_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        velocity_timer_ = create_wall_timer(std::chrono::milliseconds(50),
            std::bind(&KeyboardInput::publish_rl_velocity, this));
    }
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &old_tio_) != 0)
        throw std::runtime_error("Keyboard input requires an interactive terminal");
    new_tio_ = old_tio_;
    new_tio_.c_lflag &= (~ICANON & ~ECHO);
    if (tcsetattr(STDIN_FILENO, TCSANOW, &new_tio_) != 0)
        throw std::runtime_error("Cannot configure keyboard terminal");
    terminal_configured_ = true;
    RCLCPP_INFO(get_logger(), "Keyboard input node started.");
    RCLCPP_INFO(get_logger(), "Press 1-0 to switch between different modes");
    RCLCPP_INFO(get_logger(), "Traditional states: WASD / IJKL. Keys: 2 crouch/stand, 7 RL, 1 passive.");
    RCLCPP_INFO(get_logger(), "P (in RL): soft-stop and block /cmd_vel; press P again to resume fresh commands.");
    if (publish_cmd_vel_)
        RCLCPP_INFO(get_logger(), "After 7 (RL): W/S change forward speed by %.2f m/s, Q/E change turn speed by %.2f rad/s, C/D change lateral speed by %.2f m/s. Speed holds until SPACE or a mode key. Ctrl+C expires within RL cmd_timeout.", forward_speed_, turn_speed_, lateral_speed_);
    else
        RCLCPP_INFO(get_logger(), "RL velocity keyboard disabled; external /cmd_vel only. Enable: --ros-args -p publish_cmd_vel:=true");
    RCLCPP_INFO(get_logger(), "Please input keys, press Ctrl+C to quit.");
}

void KeyboardInput::timer_callback() {
    if (kbhit()) {
        char key;
        const auto count = ::read(STDIN_FILENO, &key, 1);
        if (count == 0) { rclcpp::shutdown(); return; }
        if (count < 0) return;
        check_rl_velocity(key);
        check_command(key);
        if (inputs_.command == 0) check_value(key);
        else {
            inputs_.lx = 0;
            inputs_.ly = 0;
            inputs_.rx = 0;
            inputs_.ry = 0;
            reset_count_ = 1;
        }
        publisher_->publish(inputs_);
        reset_count_ = 1;
        just_published_ = true;
    } else {
        if (just_published_) {
            reset_count_ -= 1;
            if (reset_count_ == 0) {
                just_published_ = false;
                if (inputs_.command != 0) {
                    inputs_.command = 0;
                    publisher_->publish(inputs_);
                }
            }
        }
    }
}

void KeyboardInput::publish_rl_velocity() {
    // Publish only while the keyboard owns a nonzero RL command; staying open
    // without movement keys does not stream zeros over a navigation publisher.
    if (publish_cmd_vel_ && rl_keyboard_mode_ && velocity_active_)
        velocity_publisher_->publish(velocity_command_);
}

void KeyboardInput::check_rl_velocity(const char key) {
    if (key == 'p' || key == 'P') {
        // P 只发送门控事件：释放键盘的持续速度，不抢先发布零而破坏控制器的缓停起始值。
        velocity_command_ = geometry_msgs::msg::Twist();
        velocity_active_ = false;
        return;
    }
    if (!publish_cmd_vel_) return;
    if ((key >= '0' && key <= '9') || key == ' ') {
        if (velocity_active_) velocity_publisher_->publish(geometry_msgs::msg::Twist());
        velocity_command_ = geometry_msgs::msg::Twist();
        velocity_active_ = false;
        if (key != ' ') rl_keyboard_mode_ = key == '7';
        return;
    }
    if (!rl_keyboard_mode_) return;
    switch (std::tolower(static_cast<unsigned char>(key))) {
    case 'w':
        velocity_command_.linear.x = std::min(
            std::max(velocity_command_.linear.x + forward_speed_, -1.0),
            1.0);
        break;

    case 's':
        velocity_command_.linear.x = std::min(
            std::max(velocity_command_.linear.x - forward_speed_, -1.0),
            1.0);
        break;

    case 'c':
        velocity_command_.linear.y = std::min(
            std::max(velocity_command_.linear.y + lateral_speed_, -1.0),
            1.0);
        break;

    case 'd':
        velocity_command_.linear.y = std::min(
            std::max(velocity_command_.linear.y - lateral_speed_, -1.0),
            1.0);
        break;

    case 'q':
        velocity_command_.angular.z = std::min(
            std::max(velocity_command_.angular.z + turn_speed_, -1.5),
            1.5);
        break;

    case 'e':
        velocity_command_.angular.z = std::min(
            std::max(velocity_command_.angular.z - turn_speed_, -1.5),
            1.5);
        break;

    default:
        return;
    }

    velocity_active_ = true;
    publish_rl_velocity();
    RCLCPP_INFO(get_logger(), "RL /cmd_vel: x=%.2f y=%.2f yaw=%.2f (SPACE to stop)",
                velocity_command_.linear.x, velocity_command_.linear.y, velocity_command_.angular.z);
}

void KeyboardInput::check_command(const char key) {
    switch (key) {
        case 'p':
        case 'P':
            inputs_.command = 11;  // 【用户指定改动】FSM 的 RL 缓停/恢复事件。
            break;
        case '1':
            inputs_.command = 1; // L2_B
            break;
        case '2':
            inputs_.command = 2; // L2_A
            break;
        case '3':
            inputs_.command = 3; // L2_X
            break;
        case '4':
            inputs_.command = 4; // L2_Y
            break;
        case '5':
            inputs_.command = 5; // L1_A
            break;
        case '6':
            inputs_.command = 6; // L1_B
            break;
        case '7':
            inputs_.command = 7; // L1_X
            break;
        case '8':
            inputs_.command = 8; // L1_Y
            break;
        case '9':
            inputs_.command = 9;
            break;
        case '0':
            inputs_.command = 10;
        break;
        case ' ':
            inputs_.lx = 0;
            inputs_.ly = 0;
            inputs_.rx = 0;
            inputs_.ry = 0;
            inputs_.command = 0;
            break;
        default:
            inputs_.command = 0;
            break;
    }
}

void KeyboardInput::check_value(char key) {
    switch (key) {
        case 'w':
        case 'W':
            inputs_.ly = min<float>(inputs_.ly + sensitivity_left_, 1.0);
            break;
        case 's':
        case 'S':
            inputs_.ly = max<float>(inputs_.ly - sensitivity_left_, -1.0);
            break;
        case 'd':
        case 'D':
            inputs_.lx = min<float>(inputs_.lx + sensitivity_left_, 1.0);
            break;
        case 'a':
        case 'A':
            inputs_.lx = max<float>(inputs_.lx - sensitivity_left_, -1.0);
            break;

        case 'i':
        case 'I':
            inputs_.ry = min<float>(inputs_.ry + sensitivity_right_, 1.0);
            break;
        case 'k':
        case 'K':
            inputs_.ry = max<float>(inputs_.ry - sensitivity_right_, -1.0);
            break;
        case 'l':
        case 'L':
            inputs_.rx = min<float>(inputs_.rx + sensitivity_right_, 1.0);
            break;
        case 'j':
        case 'J':
            inputs_.rx = max<float>(inputs_.rx - sensitivity_right_, -1.0);
            break;
        default:
            break;
    }
}

bool KeyboardInput::kbhit() {
    timeval tv = {0L, 0L};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<KeyboardInput>();
        spin(node);
        node.reset();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("keyboard_input"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
