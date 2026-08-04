import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
from collections import deque
from lerobot.common.constants import HF_LEROBOT_HOME
import pandas as pd
import cv2

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from dataset import Dataset

# =========================
# User Inputs
# =========================
checkpoint = "t_handasync_20mm_20chunk_depthcircle/20000"
SRC_REPO = "clara/handasync"
PREDICTION_HORIZON = 20

# =========================
# Policy Setup
# =========================
print(policy_config.__file__)
config = _config.get_config("pi05_xarm_finetune")
checkpoint_dir = download.maybe_download(
    "/home/admin/new/src/openpi/checkpoints/pi05_xarm_finetune/"+checkpoint
)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(policy._is_pytorch_model)

# =========================
# Test Dataset Setup
# =========================
src_dataset = Dataset(SRC_REPO)
# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("/home/admin/new/src/handteleop/R.txt")
t = np.loadtxt("/home/admin/new/src/handteleop/t.txt")
# Compile hand poses
hand_images = []
depth_images = []
xyzgrips = []

for parquet_path in sorted((HF_LEROBOT_HOME /SRC_REPO /"data"/ "chunk-000").glob("*.parquet")):
    print(f"Processing {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    # Every 10th frame
    for i in range(3, len(df), 10):
        hand_img = src_dataset.decode_img(df.at[i, "hand_camera"])
        extra_img = src_dataset.decode_img(df.at[i, "extra_camera"])

        xyzgrip = src_dataset.fix_vec(df.at[i, "actions"], src_dataset.robot_dof)
        z = xyzgrip[2]

        # Map z (depth) -> circle radius, drawn in the corner of hand_img
        HAND_IMG_SIZE = (320, 240)  # (width, height) after resize
        Z_CIRCLE_RADIUS_RANGE = (3, 30)  # pixels
        Z_CIRCLE_COLOR = (255, 0, 0)  # RGB red (resized_hand_img is RGB, not BGR)
        Z_CIRCLE_MARGIN = 2  # pixels of padding between circle and image edge
        Z_CIRCLE_THICKNESS = 2  # pixels; outline thickness (hollow circle)

        # Fixed center, inset by the largest possible radius + margin, so the
        # circle never gets clipped by the image border at any z value.
        _max_radius = Z_CIRCLE_RADIUS_RANGE[1]
        Z_CIRCLE_CENTER = (
            _max_radius + Z_CIRCLE_MARGIN,
            HAND_IMG_SIZE[1] - _max_radius - Z_CIRCLE_MARGIN,
        )

        z_values = [xyzgrip[2] for xyzgrip in xyzgrips]
        z_min, z_max = min(z_values), max(z_values)

        def z_to_radius(z):
            if z_max == z_min:
                return Z_CIRCLE_RADIUS_RANGE[0]
            frac = (z - z_min) / (z_max - z_min)
            r_min, r_max = Z_CIRCLE_RADIUS_RANGE
            return int(round(r_min + frac * (r_max - r_min)))

        cv2.circle(hand_img, Z_CIRCLE_CENTER, z_to_radius(z), Z_CIRCLE_COLOR, Z_CIRCLE_THICKNESS)

        hand_images.append(hand_img)
        depth_images.append(extra_img)
        xyzgrips.append(xyzgrip)

# =========================
# Global variables
# =========================
obs = None
curr_pose_actual = None
curr_pose_desired = None
random_frame = None

# =========================
# Desired Action
# =========================
def get_desired_action():
    global curr_pose_desired, random_frame
    xyzgrip = xyzgrips[random_frame]

    xyz = xyzgrip[0:3]
    grip_action = xyzgrip[-1]

    # Command robot towards the hand position
    increment = 5  # mm
    goal_pose = R @ np.array(xyz)*1000 + t
    distance = np.linalg.norm(goal_pose - curr_pose_desired)
    delta_pose = ( goal_pose - curr_pose_desired ) / distance * increment

    curr_pose_desired += delta_pose
    action = np.concatenate((curr_pose_desired,np.array([grip_action],dtype=np.float32)))
    for i in range(PREDICTION_HORIZON-1):
        curr_pose_desired += delta_pose
        next_action = np.concatenate((curr_pose_desired,np.array([grip_action],dtype=np.float32)))
        action = np.vstack((action, next_action))

    action_horizon = action

    return action_horizon

# =========================
# Actual Action
# =========================
def get_actual_action():
    global curr_pose_actual, random_frame

    hand_img = hand_images[random_frame]
    depth_colormap = depth_images[random_frame]
    robot_img = np.zeros_like(hand_img)

    observation = {
        "observation/exterior_image_1_left": robot_img.copy(),
        "observation/exterior_image_2_left": depth_colormap.copy(),
        "observation/wrist_image_left": hand_img.copy(),
        "observation/gripper_position":  np.array([np.random.uniform(0, 1)], dtype=np.float32),
        "observation/joint_position": curr_pose_actual.copy(),
        "prompt": "Follow the hand",
    }

    v_pi = np.array(policy.infer(observation)["actions"])
    action_horizon = v_pi[:PREDICTION_HORIZON, :]

    curr_pose_actual = action_horizon[-1, :3]

    return action_horizon


num_interations = 10
total_traj_desired = np.zeros((0, 3))
total_traj_actual = np.zeros((0, 3))
error = np.zeros(num_interations)
gripper_actions = np.zeros((num_interations, 2))

# Randomly initialize "robot position"
robot_x_bounds = [485,160]
robot_y_bounds = [345,-210]
robot_z_bounds = [600,160]

x_init = np.random.uniform(robot_x_bounds[1], robot_x_bounds[0])
y_init = np.random.uniform(robot_y_bounds[1], robot_y_bounds[0])
z_init = np.random.uniform(robot_z_bounds[1], robot_z_bounds[0])

curr_pose = np.array([x_init, y_init, z_init], dtype=np.float32)
curr_pose_actual = curr_pose.copy()
curr_pose_desired = curr_pose.copy()

for i in range(num_interations):
    random_frame = np.random.choice(len(xyzgrips))
    cv2.imwrite(f"hand_img_{i+1}.png", cv2.cvtColor(hand_images[random_frame], cv2.COLOR_RGB2BGR))
    desired_action = get_desired_action()
    actual_action = get_actual_action()
    gripper_actions[i,0] = desired_action[-1, 3]
    gripper_actions[i,1] = actual_action[-1, 3]

    error[i] = np.linalg.norm(desired_action[-1, :3] - actual_action[-1, :3])

    total_traj_desired = np.vstack((total_traj_desired, desired_action[:, :3]))
    total_traj_actual = np.vstack((total_traj_actual, actual_action[:, :3]))

# =========================
# Plot error
# =========================
plt.figure(figsize=(8, 6))
plt.plot(range(num_interations), error, marker="o", color="#2a78d6", label="Error")
plt.xlabel("Iteration")
plt.ylabel("Error (mm)")
plt.title("Error between Desired and Actual End-Effector Position")
plt.grid()
plt.legend()

# =========================
# Plot gripper actions
# =========================
plt.figure(figsize=(8, 6))
plt.plot(range(num_interations), gripper_actions[:, 0], marker="o", color="#2a78d6", label="Desired Gripper Action")
plt.plot(range(num_interations), gripper_actions[:, 1], marker="o", color="#eb6834", label=" Actual Gripper Action")
plt.xlabel("Iteration")
plt.ylabel("Gripper Action")
plt.title("Desired vs Actual Gripper Action")
plt.grid()
plt.legend()

# =========================
# Plot desired vs actual xyz trajectory
# =========================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    total_traj_desired[:, 0], total_traj_desired[:, 1], total_traj_desired[:, 2],
    color="#2a78d6", linewidth=2, marker="o", markersize=4,
    label="Desired action",
)
ax.plot(
    total_traj_actual[:, 0], total_traj_actual[:, 1], total_traj_actual[:, 2],
    color="#eb6834", linewidth=2, marker="o", markersize=4,
    label="Actual action",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Desired vs Actual End-Effector Trajectory")
ax.set_xlim(robot_x_bounds[1], robot_x_bounds[0])
ax.set_ylim(robot_y_bounds[1], robot_y_bounds[0])
ax.set_zlim(robot_z_bounds[1], robot_z_bounds[0])
ax.legend()
plt.show()
