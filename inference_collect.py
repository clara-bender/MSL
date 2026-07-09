from tkinter import messagebox

import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time
import threading
import os
import select
import sys
import subprocess
import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
from collections import deque
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

class InferenceCollector:
    def __init__(self,root):
        self.root = root
        self.root.title("Inference Collector")

        # =========================
        # Policy Setup
        # =========================
        config = _config.get_config("pi05_xarm_finetune")
        checkpoint_dir = download.maybe_download(
            "/home/admin/openpi/checkpoints/pi05_xarm_finetune/clara_training1/25000"
        )
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # =========================
        # Collection inputs
        # =========================
        self.REPO_NAME = "clara/with_rollouts"
        self.FPS_COLLECT = 20
        self.DT_COLLECT = 1.0 / self.FPS_COLLECT # multiple of 10
        ARM_IP = "192.168.1.222"

        self.TASK_DESCRIPTION = "Pick up the bag and place it on the blue x"

        # =========================
        # Inference inputs
        # =========================
        self.FPS = 60.0 # still finding upper limit on this
        self.DT = 1.0 / self.FPS
        self.CONTROL_HZ = 150.0 # multiple of 10
        self.PREDICTION_HORIZON = 20
        self.MIN_EXECUTION_HORIZON = 10
        self.ROBOT_DOF = 7
        self.delay_init = 5
        self.buffer_size = 5

        # =========================
        # Initializations
        # =========================
        self.mutex = threading.Lock()
        self.condition_variable = threading.Condition(self.mutex)
        self.t = 0
        self.observation_curr = None
        self.action_curr = np.zeros((self.PREDICTION_HORIZON, self.ROBOT_DOF), dtype=np.float32)
        self.frames_recorded = 0
        self.prev_data = None

        self.latest_wrist_frame = None      # For live display
        self.latest_tripod_frame = None      # For live display
        self.recorded_frames = []             # For playback after stopping demo
        self.playback_index = 0
        self.mode = "LIVE"
        self.gripper_change = 0.0
        self.joint_change = 0.0
        self.GRIPPER_EPSILON = 5.0
        self.JOINT_EPSILON = 2.0 # degrees
        self.reward = 0
        self.last_gripper_pos = 0.0
        self.failed = False

        self.stop = threading.Event()

        # =========================
        # GUI Setup
        # =========================
        self.label_1 = tk.Label(root)
        self.label_1.pack(side='left')

        self.label_2 = tk.Label(root)
        self.label_2.pack(side='right')

        # Create the buttons and link them to their respective actions
        start_button = tk.Button(root, text="Start", command=self.start_demo)
        start_button.pack(side='left')  # Add some padding to the button

        stop_button = tk.Button(root, text="Stop", command=self.stop_demo)
        stop_button.pack(side='right')

        self.failure_button = tk.Button(root, text="Failure", command=self.set_reward)
        self.failure_button.pack(side='top')

        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(side="bottom")
        self.playback_slider = tk.Scale(
            self.root,
            from_=0,
            to=0,  # will update later when frames are loaded
            orient=tk.HORIZONTAL,
            label="Playback Frame",
            command=self.on_slider_move
        )
        self.playback_slider.pack(fill="x")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_gui()  # Start the GUI update loop
        # =========================
        # XArm Setup
        # =========================
        self.arm = XArmAPI(ARM_IP)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)


        # =========================
        # RealSense Camera Setup
        # =========================
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) < 2:
            raise RuntimeError("Need at least two RealSense cameras connected")

        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        print("Found cameras:", serials)

        self.pipelines = []
        self.configs = []

        for serial in serials:
            pipeline = rs.pipeline()
            config_rs = rs.config()
            config_rs.enable_device(serial)
            config_rs.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30) # check these values...
            pipeline.start(config_rs)
            self.pipelines.append(pipeline)
            self.configs.append(config_rs)

        # =========================
        # Data Collection Setup
        # =========================

        self.dataset_path = HF_LEROBOT_HOME / self.REPO_NAME

        if self.dataset_path.exists(): 
            self.dataset = LeRobotDataset(
                root=self.dataset_path,
                repo_id=self.REPO_NAME,
            )
            print("Adding to existing dataset, waiting for signal.")
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.REPO_NAME,
                robot_type="xarm",
                fps=self.FPS_COLLECT,
                features={
                    "exterior_image_1_left": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "exterior_image_2_left": { # this one is not used, put it as zeros or something
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image_left": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "joint_position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["joint_position"],
                    },
                    "gripper_position": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["gripper_position"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                        "names": ["actions"],
                    },
                },
            )
            print("Dataset created, waiting for start signal.")


    # =========================
    # Observation
    # =========================
    def get_observation(self):
        frames_wrist = self.pipelines[1].wait_for_frames()
        frames_exterior = self.pipelines[0].wait_for_frames()

        wrist = frames_wrist.get_color_frame()
        exterior = frames_exterior.get_color_frame()

        a = np.asanyarray(wrist.get_data())
        b = np.asanyarray(exterior.get_data())

        pose = self.arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
        state = np.array(pose[:3] + angles_rad, dtype=np.float32)

        _, g_p = self.arm.get_gripper_position()
        g_p = np.array((g_p - 850) / -860)

        observation = {
            "observation/exterior_image_1_left": b,
            "observation/wrist_image_left": a,
            "observation/gripper_position": g_p,
            "observation/joint_position": state,
            "prompt": "Pick up the bag and place it on the blue x",
        }

        return observation

    # =========================
    # Get Camera Frames
    # =========================
    def read_cameras(self):
        frames_wrist = self.pipelines[1].wait_for_frames()
        frames_exterior = self.pipelines[0].wait_for_frames()
        wrist = frames_wrist.get_color_frame()
        exterior = frames_exterior.get_color_frame()
        wrist = np.asanyarray(wrist.get_data())
        exterior = np.asanyarray(exterior.get_data())
        if self.reward == 0:
            exterior2 = np.zeros_like(exterior)
        else:
            exterior2 = np.ones_like(exterior) * 255

        return wrist, exterior, exterior2
    
    # =========================
    # Display Camera Images on GUI
    # =========================

    def on_slider_move(self, value):
        self.playback_index = int(value)
    def update_gui(self):
        if self.mode == "LIVE":
            if self.latest_wrist_frame is not None:
                frame_1 = self.latest_wrist_frame
                frame_2 = self.latest_tripod_frame
            else:
                self.root.after(self.FPS_COLLECT, self.update_gui)
                return
        elif self.mode == "PLAYBACK":
            if len(self.recorded_frames) == 0:
                return
            frame_1, frame_2, info = self.recorded_frames[self.playback_index]
            info_text = f"""Reward: {info['reward']:.2f}"""# | Gripper Change: {info['gripper_change']:.2f} | Joint Change: {info['joint_change']:.2f}"""
            self.info_label.configure(text=info_text)

        img_1 = Image.fromarray(frame_1)
        imgtk_1 = ImageTk.PhotoImage(image=img_1)

        img_2 = Image.fromarray(frame_2)
        imgtk_2 = ImageTk.PhotoImage(image=img_2)

        self.label_1.imgtk = imgtk_1  # prevent garbage collection
        self.label_1.configure(image=imgtk_1)
        self.label_2.imgtk = imgtk_2  # prevent garbage collection
        self.label_2.configure(image=imgtk_2)

        self.root.after(self.FPS_COLLECT, self.update_gui)  # ~20 FPS

    def set_reward(self):
        # Get the current text of the failure button
        current_text = self.failure_button.cget("text")
        
        # Toggle the button's text based on its current value
        if current_text == "Failure":
            self.failed = True
            self.reward = 1
            self.failure_button.config(text="Success")
        else:
            self.failed = False
            self.reward = 0
            self.failure_button.config(text="Failure")

    # =========================
    # Command Interpolation
    # =========================
    def interpolate_action(self, state, goal):
        delta_increment = (goal - state) / (self.DT * self.CONTROL_HZ * 6)

        for i in range(int(self.DT * self.CONTROL_HZ)):
            start_time = time.perf_counter()
            state += delta_increment
            command = state.copy()
            command[3] = (command[3]+ 180) % 360 -180
            command[5] = (command[5]+ 180) % 360 -180

            x, y, z, roll, pitch, yaw = command
            print("Interpolation:")
            print(x, y, z, roll, pitch, yaw)

            self.arm.set_servo_cartesian(command, speed=100, mvacc=1000)

            time_left = (1 / self.CONTROL_HZ) - (time.perf_counter() - start_time)
            time.sleep(max(time_left,0))

    # =========================
    # Action Getter
    # =========================
    def get_action(self, observation_next):
        with self.condition_variable:
            self.t += 1

            self.observation_curr = observation_next
            self.condition_variable.notify()

            action = self.action_curr[self.t - 1, :].copy()

        return action

    # =========================
    # Guided Inference
    # =========================
    def guided_inference(self, policy, observation, action_prev, delay, time_since_last_inference):
        H = self.PREDICTION_HORIZON
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
        v_pi = v_pi[:H, :self.ROBOT_DOF]  # ensure correct shape

        A = action_prev.copy()
        action_estimate = A*W[:,None] + v_pi*(1-W[:, None])

        return action_estimate[:H, :self.ROBOT_DOF]


    # =========================
    # Inference Loop
    # =========================
    def inference_loop(self):

        Q = deque([self.delay_init], maxlen=self.buffer_size)

        while not self.stop.is_set():
            with self.condition_variable:
                while (
                    self.t < self.MIN_EXECUTION_HORIZON
                    and not self.stop.is_set()
                ):
                    self.condition_variable.wait(timeout=0.1)

                if self.stop.is_set():
                    break

                time_since_last_inference = self.t
                # Remove actions that have already been executed
                action_prev = self.action_curr[
                    time_since_last_inference:self.PREDICTION_HORIZON
                ].copy() 

                delay = max(Q)
                # print("Delay: ", delay)
                obs = self.observation_curr.copy()

            # ---- lock released ----

            action_new = self.guided_inference(
                self.policy,
                obs,
                action_prev,
                delay,
                time_since_last_inference
            )

            self.action_curr[:action_new.shape[0], :] = action_new
            self.t = self.t - time_since_last_inference
            Q.append(self.t)


    # =========================
    # Execution Loop
    # =========================
    def execution_loop(self):
        
        while not self.stop.is_set():
            # print("t:", self.t)
            t0 = time.perf_counter()

            observation = self.get_observation()
            command = self.get_action(observation)

            cmd_joint_pose = command[:6].copy()
            cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180

            pose = self.arm.get_position()[1]
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            state = np.array(pose, dtype=np.float32)

            print("Current pose:")
            print(pose)
            print("Command pose:")
            print(cmd_joint_pose)

            # execute smooth motion to target via interpolation
            self.interpolate_action(state, cmd_joint_pose)
            cmd_gripper_pose = (command[6]) * -860 + 850 # unnormalize the gripper action
            self.arm.set_gripper_position(cmd_gripper_pose)

            self.gripper_change = abs(cmd_gripper_pose - self.last_gripper_pos)
            self.joint_change = np.linalg.norm(cmd_joint_pose - state[:6])

            self.last_gripper_pos = cmd_gripper_pose

            # if not self.failed:
            #     if self.gripper_change < self.GRIPPER_EPSILON and self.joint_change < self.JOINT_EPSILON:
            #         self.reward = 1
            #     else:
            #         self.reward = 0

            time_left = self.DT - (time.perf_counter() - t0)
            time.sleep(max(time_left, 0))

    # =========================
    # Collection Loop
    # =========================
    def collection_loop(self):
        while not self.stop.is_set():
            start = time.perf_counter()

            # 1. Capture CURRENT state (Time t+1 relative to prev_data)
            #joints = arm.get_servo_angle(is_radian=True)[1][:6]
            pose = self.arm.get_position()[1]
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            # ensure roll and yaw are continuous, also make sure pitch doesn't exceed 90 deg
            # when collecting demos
            angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
        
            gripper = (self.arm.get_gripper_position()[1] - 850) / -860
            curr_state = np.array(pose[:3] + angles_rad + [gripper], dtype=np.float32)
            
            wrist, base, base2 = self.read_cameras()

            # Save latest frame for GUI (thread-safe shallow swap)
            self.latest_wrist_frame = wrist.copy()
            self.latest_tripod_frame = base.copy()

            self.recorded_frames.append((wrist.copy(), base.copy(),
                                         {"reward": self.reward
                                        #   "gripper_change": self.gripper_change, 
                                        #   "joint_change": self.joint_change
                                          }))

            # 2. If we have a previous observation, record it with CURRENT state as the action
            if self.prev_data is not None:
                self.dataset.add_frame(
                    {
                        "joint_position": self.prev_data["joints"],
                        "gripper_position": self.prev_data["gripper"],
                        "actions": curr_state,  # This is the "future" state reached
                        "exterior_image_1_left": self.prev_data["base"],
                        "exterior_image_2_left": self.prev_data["base2"],
                        "wrist_image_left": self.prev_data["wrist"],
                        "task": self.TASK_DESCRIPTION,
                    }
                )
                self.frames_recorded += 1

            # 3. Store current observations to be paired with the next frame's state
            self.prev_data = {
                "joints": curr_state[:6],
                "gripper": curr_state[-1:],
                "wrist": wrist,
                "base": base,
                "base2": base2,
                "reward": self.reward
            }

            # ---- Timing ----
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, self.DT_COLLECT - elapsed))

    # =========================
    # Start Demo
    # =========================
    def start_demo(self):
        print("Starting demo")
        self.stop.clear()
        self.prev_data = None
        self.t = 0
        self.frames_recorded = 0
        self.mode = "LIVE"
        self.recorded_frames = []

        self.infer_thread = threading.Thread(target=self.inference_loop,daemon=True)
        self.exec_thread = threading.Thread(target=self.execution_loop,daemon=True)
        self.collect_thread = threading.Thread(target=self.collection_loop,daemon=True)

        self.observation_curr = self.get_observation()
        self.action_curr = np.array(self.policy.infer(self.observation_curr)["actions"], dtype=np.float32)
        self.infer_thread.start()
        self.exec_thread.start()
        self.collect_thread.start()
        self.prev_data = None

    # =========================
    # End Demo
    # =========================
    def stop_demo(self):
        self.stop.set()
        self.mode = "PLAYBACK"
        self.playback_slider.config(to=len(self.recorded_frames) - 1)
        self.infer_thread.join()
        self.exec_thread.join()
        self.collect_thread.join()
        time.sleep(0.1)
        answer = messagebox.askyesno(
            "Save Episode",
            "Would you like to save this episode?"
        )
        if answer:
            self.dataset.save_episode()
            print("Episode saved")
        else:
            print("Episode discarded")
            # HARD RESET: clears the in-memory episode buffer
            self.dataset = LeRobotDataset( root=self.dataset_path, repo_id=self.REPO_NAME, )
        
        self.frames_recorded = 0
        self.t = 0
        self.go_home()


    def on_close(self):
            self.stop.set()
            self.infer_thread.join()
            self.exec_thread.join()
            self.collect_thread.join()

            self.root.destroy()

    # =========================
    # Go Home
    # =========================
    def go_home(self):
            """
            Same motion logic as in reset_callback, to move the arm home.
            """
            self.is_resetting = True
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_gripper_enable(enable=True)
            self.arm.set_gripper_mode(0)
            self.arm.set_state(0)
            cmd_joint_pose = [0.0, -90.4, -24.0, 0.0, 61.3, 180.0] 
            cmd_gripper_pose = 850.0
            self.arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=False, wait=True) 
            self.arm.set_gripper_position(cmd_gripper_pose, wait=True)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(1)
            self.arm.set_state(0)
            self.is_resetting = False

# =========================
# Startup
# =========================
root = tk.Tk()
root.title("Inference Collector")
app = InferenceCollector(root)
root.mainloop()