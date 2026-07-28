import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
from collections import deque
from lerobot.common.constants import HF_LEROBOT_HOME
import pandas as pd

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from dataset import Dataset

# =========================
# User Inputs
# =========================
checkpoint = "t_follow_hand_delay_reduced_actions_352/20000"
SRC_REPO = "clara/handasync"
PREDICTION_HORIZON = 20

# =========================
# Policy Setup
# =========================
print(policy_config.__file__)
config = _config.get_config("pi05_xarm_finetune")
checkpoint_dir = download.maybe_download(
    "/home/admin/openpi/checkpoints/pi05_xarm_finetune/"+checkpoint
)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(policy._is_pytorch_model)

# =========================
# Test Dataset Setup
# =========================
src_dataset = Dataset(SRC_REPO)
# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("R.txt")
t = np.loadtxt("t.txt")
# Compile hand poses
hand_images = []
depth_images = []
xyzgrips = []

for parquet_path in sorted((HF_LEROBOT_HOME /SRC_REPO /"data"/ "chunk-000").glob("*.parquet")):
    print(f"Processing {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    # Every 10th frame
    for i in range(2, len(df), 10):
        hand_img = src_dataset.decode_img(df.at[i, "hand_camera"])
        extra_img = src_dataset.decode_img(df.at[i, "extra_camera"])

        xyzgrip = src_dataset.fix_vec(df.at[i, "actions"], src_dataset.robot_dof)
        hand_images.append(hand_img)
        depth_images.append(extra_img)
        xyzgrips.append(xyzgrip)

# =========================
# Global variables
# =========================
obs = None
curr_pose = None
random_frame = None

# =========================
# Desired Action
# =========================
def get_desired_action():
    global curr_pose, random_frame
    xyzgrip = xyzgrips[random_frame]

    xyz = xyzgrip[0:3]
    grip_action = xyzgrip[-1]

    # Command robot towards the hand position
    increment = 20  # mm
    goal_pose = R @ np.array(xyz)*1000 + t
    distance = np.linalg.norm(goal_pose - curr_pose)
    delta_pose = ( goal_pose - curr_pose ) / distance * increment

    action = np.concatenate((curr_pose+delta_pose,np.array([grip_action],dtype=np.float32)))
    for i in range(PREDICTION_HORIZON):
        curr_pose += delta_pose
        next_action = np.concatenate((curr_pose,np.array([grip_action],dtype=np.float32)))
        action = np.vstack((action, next_action))

    action_horizon = action

    return action_horizon

# =========================
# Actual Action
# =========================
def get_actual_action():
    global curr_pose, random_frame

    hand_img = hand_images[random_frame]
    depth_colormap = depth_images[random_frame]
    robot_img = np.zeros_like(hand_img)

    observation = {
        "observation/exterior_image_1_left": robot_img.copy(),
        "observation/exterior_image_2_left": depth_colormap.copy(),
        "observation/wrist_image_left": hand_img.copy(),
        "observation/gripper_position": np.random.uniform(0,1),
        "observation/joint_position": curr_pose,
        "prompt": "Follow the hand",
    }

    v_pi = np.array(policy.infer(observation)["actions"])
    action_horizon = v_pi[:PREDICTION_HORIZON, :]

    return action_horizon


for i in range(1):
    random_frame = np.random.choice(len(xyzgrips))

    # Randomly initialize "robot position"
    robot_x_bounds = [485,160]
    robot_y_bounds = [345,-210]
    robot_z_bounds = [600,160]

    x_init = np.random.uniform(robot_x_bounds[1], robot_x_bounds[0])
    y_init = np.random.uniform(robot_y_bounds[1], robot_y_bounds[0])
    z_init = np.random.uniform(robot_z_bounds[1], robot_z_bounds[0])

    curr_pose = np.array([x_init, y_init, z_init])

    desired_action = get_desired_action()
    actual_action = get_actual_action()

    # =========================
    # Plot desired vs actual xyz trajectory
    # =========================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        desired_action[:, 0], desired_action[:, 1], desired_action[:, 2],
        color="#2a78d6", linewidth=2, marker="o", markersize=4,
        label="Desired action",
    )
    ax.plot(
        actual_action[:, 0], actual_action[:, 1], actual_action[:, 2],
        color="#eb6834", linewidth=2, marker="o", markersize=4,
        label="Actual action",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Desired vs Actual End-Effector Trajectory")
    ax.legend()
    plt.show()
