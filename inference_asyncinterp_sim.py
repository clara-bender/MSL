import numpy as np
import pandas as pd
import cv2
import time
import threading
from collections import deque

from lerobot.common.constants import HF_LEROBOT_HOME

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from dataset import Dataset

# =========================
# User inputs
# =========================
SRC_REPO = "clara/handasync"  # dataset to sample observation images from
FPS = 30.0
DT = 1.0 / FPS
CONTROL_HZ = 50.0 # multiple of 10
PREDICTION_HORIZON = 20
MIN_EXECUTION_HORIZON = 10
ROBOT_DOF = 4

mutex = threading.Lock()
condition_variable = threading.Condition(mutex)

delay_init = 5
buffer_size = 5

print(f"Robot DOF: {ROBOT_DOF}")

# =========================
# Policy Setup
# =========================
print(policy_config.__file__)
config = _config.get_config("pi05_xarm_finetune")
checkpoint = "t_follow_hand_delay_reduced_actions_352/20000"
checkpoint_dir = download.maybe_download(
    "/home/admin/openpi/checkpoints/pi05_xarm_finetune/"+checkpoint
)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(policy._is_pytorch_model)

# =========================
# Source Dataset Setup (replaces live cameras)
# =========================
WIDTH, HEIGHT = 640, 480

src_dataset = Dataset(SRC_REPO)

hand_images = []
robot_images = []
depth_images = []

for parquet_path in sorted((HF_LEROBOT_HOME / SRC_REPO / "data" / "chunk-000").glob("*.parquet")):
    print(f"Loading {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    # Every 10th frame, to keep the in-memory pool a manageable size
    for i in range(0, len(df), 10):
        hand_img = src_dataset.decode_img(df.at[i, "hand_camera"])
        robot_img = src_dataset.decode_img(df.at[i, "robot_camera"])
        depth_img = src_dataset.decode_img(df.at[i, "extra_camera"])

        hand_images.append(cv2.resize(hand_img, (WIDTH, HEIGHT)))
        robot_images.append(cv2.resize(robot_img, (WIDTH, HEIGHT)))
        depth_images.append(cv2.resize(depth_img, (WIDTH, HEIGHT)))

print(f"Loaded {len(hand_images)} frames from source dataset '{SRC_REPO}'")

def sample_images():
    idx = np.random.randint(len(hand_images))
    return robot_images[idx], depth_images[idx], hand_images[idx]

# =========================
# Simulated Robot State
# (assumes the arm executes every commanded pose perfectly, in place of a real xArm)
# =========================
sim_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, roll, pitch, yaw -- mirrors arm.get_position()[1]
sim_gripper = 850.0  # raw gripper position -- mirrors arm.get_gripper_position()[1]

# =========================
# Observation
# =========================
def get_observation():
    robot_img, depth_colormap, hand_img = sample_images()

    pose = sim_pose.copy()
    g_p = sim_gripper

    # Convert [-180, 180] to [0, 360] for roll and yaw
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360

    # Convert angles from degrees to radians for roll, pitch, and yaw (array --> list)
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()

    if ROBOT_DOF == 4:
        state = np.array(pose[:3], dtype=np.float32)
    else:
        state = np.array(pose[:3] + angles_rad, dtype=np.float32)

    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": robot_img.copy(),
        "observation/exterior_image_2_left": depth_colormap.copy(),
        "observation/wrist_image_left": hand_img.copy(),
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Follow the hand",
    }

    return observation

# =========================
# Command Interpolation
# =========================
def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

    speed = np.linalg.norm(delta_increment[:3]) * CONTROL_HZ  # mm/s
    print(f"Robot speed: {speed:.2f} mm/s")

    for i in range(int(DT * CONTROL_HZ)):
        start_time = time.perf_counter()
        state += delta_increment
        command = state.copy()

        # Convert roll and yaw from [0 360] to [-180 180]
        if ROBOT_DOF == 7:
            command[3] = (command[3]+ 180) % 360 -180
            command[5] = (command[5]+ 180) % 360 -180

        # Hard-code roll, pitch, and yaw
        elif ROBOT_DOF == 4:
            command = np.concatenate((command, np.array([180, 0, 0])))

        # Simulate perfect execution: the arm instantaneously reaches the commanded pose
        sim_pose[:] = command.tolist()

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start_time)
        time.sleep(max(time_left,0))

# =========================
# Action Getter
# =========================
def get_action(observation_next):
    global t, observation_curr

    with condition_variable:
        t += 1

        observation_curr = observation_next
        condition_variable.notify()

        action = action_curr[t - 1, :].copy()

    return action

# =========================
# Guided Inference
# =========================
def guided_inference(policy, observation, action_prev, delay, time_since_last_inference):
    H = PREDICTION_HORIZON
    i = np.arange(delay, H - time_since_last_inference)
    c = (H - time_since_last_inference - i) / (H - time_since_last_inference - delay + 1)

    W = np.ones(H)
    W[0:delay] = 1.0
    W[delay:H - time_since_last_inference] = c * (np.exp(c) - 1) / (np.exp(1) - 1)
    W[H - time_since_last_inference:] = 0.0

    T, robot_dof = action_prev.shape
    if T < H:
        action_prev = np.pad(action_prev, ((0, H - T), (0, 0)), mode='constant')

    v_pi = np.array(policy.infer(observation)["actions"])
    v_pi = v_pi[:H, :]  # ensure correct shape

    A = action_prev.copy()
    action_estimate = A*W[:,None] + v_pi*(1-W[:, None])

    return action_estimate[:H, :]


# =========================
# Inference Loop
# =========================
def inference_loop():
    global t, action_curr, observation_curr

    Q = deque([delay_init], maxlen=buffer_size)

    while True:
        with condition_variable:
            while t < MIN_EXECUTION_HORIZON:
                condition_variable.wait()

            time_since_last_inference = t

            # Remove actions that have already been executed
            action_prev = action_curr[
                time_since_last_inference:PREDICTION_HORIZON
            ].copy()

            delay = max(Q)
            print("Delay: ", delay)
            obs = observation_curr.copy()

        # ---- lock released ----

        action_new = guided_inference(
            policy,
            obs,
            action_prev,
            delay,
            time_since_last_inference
        )

        action_curr[:action_new.shape[0], :] = action_new
        t = t - time_since_last_inference
        Q.append(t)


# =========================
# Execution Loop
# =========================
def execution_loop():
    global t, sim_gripper

    while True:
        # print("t:", t)
        t0 = time.perf_counter()

        observation = get_observation()
        command = get_action(observation)

        current_pose = sim_pose.copy()

        # Convert roll and yaw from [-180, 180] to [0, 360] for interpolation
        current_pose[3] = current_pose[3] % 360
        current_pose[5] = current_pose[5] % 360

        # Convert roll, pitch, and yaw from radians to degrees for the robot
        if ROBOT_DOF == 7:
            cmd_pose = command[:6].copy()
            cmd_pose[3:6] = cmd_pose[3:6] / np.pi * 180
            cmd_gripper = command[6]* -860 + 850 # unnormalize the gripper action
        elif ROBOT_DOF == 4:
            cmd_pose = command[:3].copy()
            cmd_gripper = command[3]* -860 + 850 # unnormalize the gripper action
            start_pose = np.array(current_pose[0:3], dtype=np.float32)

        # execute smooth motion to target via interpolation
        interpolate_action(start_pose, cmd_pose)

        # simulate the gripper reaching its commanded position perfectly
        sim_gripper = cmd_gripper

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0))


# =========================
# Shared State
# =========================
t = 0
observation_curr = get_observation()
action_curr = np.array(policy.infer(observation_curr)["actions"], dtype=np.float32)

# =========================
# Thread Startup
# =========================
if __name__ == "__main__":
    print("Starting simulated closed-loop control system...")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    while True:
        time.sleep(1)
