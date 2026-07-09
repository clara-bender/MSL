import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time
import threading

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

FPS = 20.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
ACTION_ROLLOUT = 40

# Shared variables for threading
next_action = None
action_ready = threading.Event()

arm = XArmAPI('192.168.1.222')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

config = _config.get_config("pi05_xarm_finetune")
checkpoint_dir = download.maybe_download("/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)
# check serials for which camera is which, second one is currently the external viewer
pipelines = []
configs = []

# Enable streams
for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    # Enable streams (color + depth if you want)
    config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    pipeline.start(config)
    pipelines.append(pipeline)
    configs.append(config)

def inference_thread(observation,policy):
    global next_action
    inference = policy.infer(observation)
    next_action = np.array(inference["actions"])
    action_ready.set()

def get_observation():
    frames_wrist = pipelines[1].wait_for_frames()
    frames_exterior = pipelines[0].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    a = np.asanyarray(wrist.get_data())
    b = np.asanyarray(exterior.get_data())

    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)
    code, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Pick up the bag and place it on the blue x",
    }
    return observation

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

    for i in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        command = state + delta_increment
        command[3] = (command[3]+ 180) % 360 -180
        command[5] = (command[5]+ 180) % 360 -180

        x, y, z, roll, pitch, yaw = command
        print(x, y, z, roll, pitch, yaw)

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start)
        time.sleep(max(time_left,0))
       

# Do first inference before moving
observation = get_observation()
inference_thread(observation,policy)
action_ready.clear()
current_action = next_action.copy()
rollout_count = 0
rollout_start = 0
first_time = True

print(ACTION_ROLLOUT//3)

while rollout_count < ACTION_ROLLOUT:
        # grab current state
        t0 = time.perf_counter()
        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)
        
        # get the target angles
        cmd_joint_pose = np.array(current_action[rollout_count,:6])
        cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

        # print command
        print("command")
        
        # execute smooth motion to target via interpolation
        interpolate_action(state, cmd_joint_pose)
        cmd_gripper_pose = (current_action[rollout_count,6]) * -860 + 850 # unnormalize the gripper action
        arm.set_gripper_position(cmd_gripper_pose)

        if rollout_count == ACTION_ROLLOUT//3 and first_time:
            # Start second inference in a separate thread halfway through the rollout
            print("Started second inference")
            observation = get_observation()
            threading.Thread(target=inference_thread, args=(observation, policy)).start()
            rollout_start = rollout_count
            first_time = False

        # Swap in next action if ready
        if action_ready.is_set():
            print("Swapping in next action")
            current_action[0:ACTION_ROLLOUT-rollout_start] = next_action[rollout_start:ACTION_ROLLOUT]
            action_ready.clear()
            observation = get_observation()
            threading.Thread(target=inference_thread, args=(observation, policy)).start()
            rollout_start = rollout_count
            rollout_count = -1

        rollout_count += 1
        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left,0))
print("Exited while loop :(")