from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
from pathlib import Path
import shutil
import cv2

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

            self.my_dataset = ds
            
            print(f"Loaded existing dataset, repo: {repo_name}")
        else:
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            self.my_dataset = LeRobotDataset.create(
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
        prev_obs = None
        frames_recorded = 0

        while not observation_queue.empty():
            
            obs = observation_queue.get()
            

            if prev_obs is not None:
                prev_obs["actions"] = obs["actions"]
                self.my_dataset.add_frame(prev_obs)
                frames_recorded += 1

            prev_obs = obs

        self.my_dataset.save_episode()
        return frames_recorded

    def decode_img(x):
        if isinstance(x, dict):
            x = x.get("bytes", None)

        if x is None:
            return None

        arr = np.frombuffer(x, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def fix_vec(x, dim=None):
        if isinstance(x, dict):
            x = x.get("array", x.get("value", x))

        x = np.array(x, dtype=np.float32)

        if dim is not None:
            x = x.reshape(dim)

        return x