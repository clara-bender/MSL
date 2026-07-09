import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time
import threading
import queue
from collections import deque
import torch

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

import time
from camera import Camera

# User inputs
FPS = 20.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 50.0 # keep as a multiple of 10
PREDICTION_HORIZON = 20
MIN_EXECUTION_HORIZON = 10
ROBOT_DOF = 4
TASK = "Follow the hand"

mutex = threading.Lock()
condition_variable = threading.Condition(mutex)
delay_init = 5
buffer_size = 5
trajectory_blend_steps = 10
max_guidance_weight = 1

config = _config.get_config("pi05_xarm_finetune")
checkpoint_dir = download.maybe_download("/home/admin/new/src/openpi/checkpoints/pi05_xarm_finetune/t_follow_hand_delay_reduced_actions_352/24999")

# XArm Setup
arm = XArmAPI('192.168.1.222')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Connect to cameras
HAND_CAMERA_SERIAL = "317222072257"
ROBOT_CAMERA_SERIAL = "243522071742"
WIDTH, HEIGHT, FPS = 640, 480, 60

hand_camera = Camera(HAND_CAMERA_SERIAL, WIDTH, HEIGHT, FPS)
time.sleep(3)
robot_camera = Camera(ROBOT_CAMERA_SERIAL, WIDTH, HEIGHT, FPS)
time.sleep(3)

# Functions
def get_observation():
    hand_img, depth_img = hand_camera.get_image(True)
    robot_img, _ = robot_camera.get_image(False)

    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3], dtype=np.float32)
    code, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": robot_img,
        "observation/exterior_image_2_left": depth_img,
        "observation/wrist_image_left": hand_img,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": TASK,
    }
    return observation

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

    for i in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        state += delta_increment
        command = state.copy()
        # command[3] = (command[3]+ 180) % 360 -180
        # command[5] = (command[5]+ 180) % 360 -180

        # x, y, z, roll, pitch, yaw = command
        # print(x, y, z, roll, pitch, yaw)

        command = np.concatenate((command, np.array([180, 0, 0])))

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start)
        time.sleep(max(time_left,0))

def get_action(observation_next):
    with mutex:
        global t, observation_curr
        t +=1
        observation_curr = observation_next
        condition_variable.notify()
        return action_curr[t-1,:].copy()

def guided_inference(policy,observation,action_prev,delay,time_since_last_inference):
    #1. Compute time-weighting W (Eq. 5 style)
    H = PREDICTION_HORIZON
    i = np.arange(delay, H - time_since_last_inference)
    c = (H - time_since_last_inference - i) / (H - time_since_last_inference - delay + 1)
    W = np.ones(H)
    W[0:delay] = 1.0
    W[delay:H - time_since_last_inference] = c * (np.exp(c) - 1) / (np.exp(1) - 1)
    W[H - time_since_last_inference:] = 0.0
    # 2. Right-pad previous trajectory
    T, robot_dof = action_prev.shape
    if T < H:
        action_prev = np.pad(action_prev,((0, H - T), (0, 0)),mode='constant')
    # 3. Compute policy chunk ONCE (π₀₅ does not depend on A or τ)
    v_pi = np.array(policy.infer(observation)["actions"])  # shape (H, robot_dof)
    print(f"INFERENCE SIZE: {v_pi.size}")
    # 4. Initialize trajectory (smooth real-time choice)
    A = action_prev.copy()
    # 5. Refinement loop (formerly denoising loop)
    for tau in np.linspace(0, 1, trajectory_blend_steps):
        # Estimated trajectory after policy push
        action_estimate = A + (1 - tau) * v_pi
        # Weighted correction toward previous chunk
        weighted_error = (action_prev - action_estimate) * W[:, None]
        # Identity Jacobian for π₀₅ → g = weighted_error
        g = weighted_error
        # Guidance scaling (clipped)
        r_squared = (1 - tau)**2 / (tau**2 + (1 - tau)**2 + 1e-8)
        scaling = min(
            max_guidance_weight,(1 - tau) / (tau * r_squared + 1e-8))
        # Integration step
        A = A + (1 / trajectory_blend_steps) * (v_pi + scaling * g)
    return A
        
# Procedure: Initialize shared state
t = 0
action_curr = np.array(policy.infer(get_observation())["actions"], dtype=np.float32)  # shape (H, robot_dof)
observation_curr = None

# Procedure: Inference loop
def inference_loop():
    global t, action_curr, observation_curr
    mutex.acquire()
    Q = deque([delay_init], maxlen=buffer_size)  # Holds past delays
    while True:
        with condition_variable:  # wait until enough actions have been executed
            while t < MIN_EXECUTION_HORIZON:
                condition_variable.wait()
            # record number of executed actions since last inference
            time_since_last_inference = t
            # slice the remaining actions
            action_prev = action_curr[time_since_last_inference:PREDICTION_HORIZON].copy()
            # estimate delay conservatively
            delay = max(Q)
        
        # with M released
        mutex.release()
        # run guided inference
        action_new = guided_inference(policy, observation_curr, action_prev, delay, time_since_last_inference)

        # swap to new chunk as soon as it is available
        action_curr[:action_new.shape[0], :] = action_new
        # reset t so that it indexes into new trajectory
        t = t - time_since_last_inference
        # record the observed delay
        Q.append(t)

def execution_loop():
    global t, action_curr, observation_curr

    while True:
        t0 = time.perf_counter()
        # 1. Get the latest observation from cameras
        observation = get_observation()
        # 2. Get the next action safely (returns a copy)
        command = get_action(observation)
        # 3. Split into Cartesian joints and gripper
        cmd_joint_pose = command[:3].copy()
        # cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180  # radians -> degrees
        # 4. Read current robot pose
        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)
        # 5. Smoothly interpolate toward target joint pose
        #interpolate_action(state, cmd_joint_pose)
        # 6. Execute gripper action
        cmd_gripper_pose = np.clip(command[3] * -860 + 850, 0, 850)
        #arm.set_gripper_position(cmd_gripper_pose)

        # print command
        # print("command")
        # print(pose)
        # print(cmd_joint_pose)

        time_left = DT - (time.perf_counter() - t0)
        
        time.sleep(max(time_left,0))

if __name__ == "__main__":
    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    infer_thread.join()
    exec_thread.join()