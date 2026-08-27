# leader /dev/tty.usbmodem5B3D0469991
# follower /dev/tty.usbmodem5B3D0470181


import time 
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5B3D0470181",
    id="follower",
    use_degrees=True,
)

robot = SO101Follower(config)
robot.connect()

obs = robot.get_observation()
current = {k: obs[k] for k in obs if k.endswith(".pos")}

target = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos": 25.0,
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 3.0,
}

duration = 2.0
steps = 60
for i in range(steps):
    alpha = i / steps
    action = {}
    for joint in target:
        action[joint] = current[joint] * (1 - alpha) + target[joint] * alpha
    robot.send_action(action)
    time.sleep(duration / steps)

time.sleep(100)

robot.disconnect()