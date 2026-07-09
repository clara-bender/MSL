import time
import numpy as np
import pyrealsense2 as rs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from xarm.wrapper import XArmAPI
from pathlib import Path

# ------------------------
# Config
# ------------------------
REPO_NAME = "clara/xarm_demos_pickandplace"
FPS = 8.0
DT = 1.0 / FPS
ARM_IP = "192.168.1.222"

TASK_DESCRIPTION = "Pick up the coffee bag and place it on the blue x"

START_FLAG = Path("/tmp/start_demo")
STOP_FLAG  = Path("/tmp/stop_demo")

# ------------------------
# Init robot + cameras
# ------------------------
arm = XArmAPI(ARM_IP)
arm.connect()

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)
# check serials for which camera is which, second one is currently the external viewer