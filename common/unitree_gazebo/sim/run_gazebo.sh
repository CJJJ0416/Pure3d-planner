#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PURE3D_SIM_WS"
if [[ ! -f "$PURE3D_SIM_ROOT/install/unitree_gazebo/share/unitree_gazebo/package.xml" ]]; then
  echo 'Build first: ./build_gazebo.sh' >&2
  exit 1
fi
case "${PURE3D_SIM_SCENE:-building}" in
  building)
    sim_building_default=/home/cjh/EOLP_ws/src/sru_a1_bringup/models/Building-no-rails
    if [[ ! -f "$sim_building_default/model.sdf" ]]; then
      sim_building_default="$PURE3D_UNITREE_ROOT/models/Building-no-rails"
    fi
    sim_building="${PURE3D_BUILDING_MODEL:-$sim_building_default}"
    sim_building="$(realpath -e -- "$sim_building")"
    sim_world="$(dirname -- "$(dirname -- "$sim_building")")/worlds/Building-no-rails.world"
    for sim_required in "$sim_world" "$sim_building/model.sdf" "$sim_building/meshes/Building-no-rails.dae"; do
      if [[ ! -f "$sim_required" ]]; then
        echo "Missing Building-no-rails asset: $sim_required" >&2
        exit 1
      fi
    done
    echo "Gazebo world: $sim_world"
    echo "Building model: $sim_building"
    exec /opt/ros/humble/bin/ros2 launch unitree_gazebo gazebo.launch.py \
      "world:=$sim_world" "scene_models_path:=$(dirname -- "$sim_building")" \
      x:=-5.5 y:=6.0 height:=0.5 yaw:=0.0 \
      odom_x:=0.0 odom_y:=0.0 odom_z:=0.0 planner:=scan flight_type:=1 device:=cpu "$@"
    ;;
  terrain)
    sim_scene="${PURE3D_TERRAIN_SCENE:-$PURE3D_SIM_WS/datasets/simulation/isaac_lio_20261117/scene}"
    sim_terrain="${PURE3D_TERRAIN_OUTPUT:-$PURE3D_SIM_WS/datasets/simulation/gazebo_terrain}"
    /usr/bin/python3 common/unitree_gazebo/sim/prepare_world.py --scene "$sim_scene" --output "$sim_terrain" --floor "${PURE3D_TERRAIN_FLOOR:-0}"
    exec /opt/ros/humble/bin/ros2 launch unitree_gazebo terrain.launch.py "terrain_dir:=$sim_terrain" "$@"
    ;;
  *)
    echo 'PURE3D_SIM_SCENE must be building or terrain' >&2
    exit 1
    ;;
esac
