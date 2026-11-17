# Keyboard Input

This node will read the keyboard input and publish a control_input_msgs/Input message.

Tested environment:
* Ubuntu 24.04
  * ROS2 Jazzy
* Ubuntu 22.04
  * ROS2 Humble

### Build Command
```bash
cd ~/ros2_ws
colcon build --packages-up-to keyboard_input
```

### Launch Command
```bash
source ~/ros2_ws/install/setup.bash
ros2 run keyboard_input keyboard_input
```

## 1. Use Instructions for Unitree Guide
### 1.1 Control Mode
* Passive Mode: Keyboard 1
* Fixed Stand: Keyboard 2
    * Free Stand: Keyboard 3
    * Trot: Keyboard 4
    * SwingTest: Keyboard 5
    * Balance: Keyboard 6
### 1.2 Control Input
* WASD IJKL: Move robot
* Space: Reset Speed Input

## EOLP RL keyboard bridge (Humble)

Run `keyboard.sh` in glocal-nav2, or `ros2 run keyboard_input keyboard_input --ros-args -p publish_cmd_vel:=true`. After key 7, W/S changes x by +/-0.5 m/s, C/D changes y by +/-0.25 m/s, Q/E changes yaw by +/-0.5 rad/s. Commands hold at 20 Hz until SPACE or a digit mode key. Default publish_cmd_vel=false preserves external navigation ownership. forward_speed, lateral_speed and turn_speed are configurable positive parameters. Traditional control_input behavior remains available.

## P：RL 缓停与恢复接收速度

进入 RL 后按 `P`（或 `p`），发送 `/control_input` 的 `command=11`。
控制器 FSM 消费一次事件后屏蔽 `/cmd_vel`，包括 EGO 与键盘的速度指令；
从当前 RL 使用的三轴速度指令开始，在 `rl.soft_stop_duration`（默认 1 秒）内线性减到零。
控制循环持续运行，减速完成后仍在 RL 中维持零速度输入。
再次按 `P` 恢复接收，旧缓存速度不会被使用，必须有新的 `/cmd_vel` 才会启动。
该按键在 RL 之外只提示先进入 RL；RL 暂停状态不会被退出/重入自动解除。

`P` 会释放键盘的持续速度发布，但不抢先发布零速度，以便控制器从当前指令缓停。
仅通过 EGO 导航时运行 `./keyboard.sh -p publish_cmd_vel:=false`：模式键和 P 键仍可用，
键盘不发布速度。这里修改的是下游速度接收接口，EGO 算法继续运行。
