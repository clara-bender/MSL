from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
from pathlib import Path
import shutil

class Dataset():
    def __init__(self, repo_name, fps_collect=20, img_height=240, img_width=320, robot_dof=7, robot_type="xarm"):
        self.task = None
        self.fps = fps_collect
        self.img_height = img_height
        self.img_width = img_width
        self.robot_dof = robot_dof
        self.robot_type = robot_type

        dataset_path = HF_LEROBOT_HOME / repo_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        if Path(dataset_path/"meta").exists() and Path(dataset_path/"data").exists():
            ds = LeRobotDataset(
                root=dataset_path,
                repo_id=repo_name,
            )
            self.fps = ds.fps
            self.robot_type = ds.meta.robot_type
            features = ds.features
            self.img_height, self.img_width, _ = features["robot_camera"]["shape"]
            self.robot_dof = features["actions"]["shape"][0]
            # confirm exact structure of ds.meta.tasks before relying on this:
            self.task = list(ds.meta.tasks.values())

            self.dataset = ds
            
            print(f"Loaded existing dataset, repo: {repo_name}")
        else:
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_name,
                robot_type=robot_type,
                fps=fps_collect,
                features={
                    "robot_camera": {
                        "dtype": "image",
                        "shape": (img_height, img_width, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "extra_camera": { # this one is not used, put it as zeros or something
                        "dtype": "image",
                        "shape": (img_height, img_width, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "hand_camera": {
                        "dtype": "image",
                        "shape": (img_height, img_width, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "eef_position": {
                        "dtype": "float32",
                        "shape": (robot_dof-1,),
                        "names": ["joint_position"],
                    },
                    "gripper_position": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["gripper_position"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (robot_dof,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                        "names": ["actions"],
                    },
                },
            )
            print(f"Created new dataset, repo: {repo_name}")

    def collect(self, observation_queue, task):

        self.task = task

        while not observation_queue.empty():
            obs = observation_queue.get()
            tripod_camera = obs["observation/exterior_image_1_left"]
            extra_camera = obs["observation/exterior_image_2_left"]
            wrist_camera = obs["observation/wrist_image_left"]
            gripper_pos = obs["observation/gripper_position"]
            servo_state = obs["observation/joint_position"]

            total_state = np.concatenate((servo_state,np.array([gripper_pos],dtype=np.float32)))

            if self.prev_data is not None:
                self.dataset.add_frame(
                    {
                        "joint_position": self.prev_data["joints"],
                        "gripper_position": self.prev_data["gripper"],
                        "actions": total_state,  # This is the "future" state reached
                        "exterior_image_1_left": self.prev_data["base"],
                        "exterior_image_2_left": self.prev_data["base2"],
                        "wrist_image_left": self.prev_data["wrist"],
                        "task": task,
                    }
                )
                self.frames_recorded += 1

            self.prev_data = {
                "joints": total_state[:self.robot_dof-1],
                "gripper": total_state[-1:],
                "wrist": wrist_camera,
                "base": tripod_camera,
                "base2": extra_camera,
            }

        self.dataset.save_episode()