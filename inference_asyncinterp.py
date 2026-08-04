import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time
import threading
from collections import deque

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from camera import Camera
import warnings

# =========================
# User inputs
# =========================
debug = False
robot_active = True
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

print(f"Debug mode: {debug}")
print(f"Robot DOF: {ROBOT_DOF}")


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")


# =========================
# Policy Setup
# =========================
print(policy_config.__file__)
config = _config.get_config("pi05_xarm_finetune")
checkpoint = "t_handasync_5mmrandgrip/24999"
checkpoint_dir = download.maybe_download(
    "/home/admin/new/src/openpi/checkpoints/pi05_xarm_finetune/"+checkpoint
)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(policy._is_pytorch_model)

# =========================
# XArm Setup
# =========================
robot_x_bounds = [485,160]
robot_y_bounds = [345,-210]
robot_z_bounds = [600,160]
boundary = robot_x_bounds + robot_y_bounds + robot_z_bounds

if not debug:
    arm = XArmAPI('192.168.1.222')

    # Start up robot
    """Initialize XArm for control with safety limits."""
    code, state = arm.get_state()
    if state != 0:
        arm.clean_error()
        time.sleep(0.5)

    arm.motion_enable(enable=True)
    arm.set_mode(7)
    arm.set_state(0)
    arm.set_gripper_enable(enable=True)
    arm.set_gripper_mode(0)
    print(boundary)
    print('XArm initialized with safety limits')

# =========================
# RealSense Camera Setup
# =========================
HAND_CAMERA_SERIAL = "317222072257"
ROBOT_CAMERA_SERIAL = "243522071742"
WIDTH, HEIGHT, FPS = 640, 480, 60

if not debug:
    hand_camera = Camera(HAND_CAMERA_SERIAL, WIDTH, HEIGHT, FPS)
    time.sleep(3)
    robot_camera = Camera(ROBOT_CAMERA_SERIAL, WIDTH, HEIGHT, FPS)
    time.sleep(3)
    mpHands, hands, mpDraw = Camera.initialize_hands()

# Rotation and translation matrix (hand xyz pos --> eef xyz pos)
R = np.loadtxt("/home/admin/new/src/handteleop/R.txt")
trans = np.loadtxt("/home/admin/new/src/handteleop/t.txt")
# =========================
# Observation
# =========================
def get_observation():
    if not debug:
        hand_img, depth_colormap = hand_camera.get_image(True)
        robot_img, _ = robot_camera.get_image(False)

        pose = arm.get_position()[1]
        _, g_p = arm.get_gripper_position()
    else:
        hand_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        depth_colormap = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        robot_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        g_p = 0.0

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
        "observation/exterior_image_1_left": np.zeros_like(robot_img.copy()),
        "observation/exterior_image_2_left": depth_colormap.copy(),
        "observation/wrist_image_left": hand_img.copy(),
        "observation/gripper_position": np.random.uniform(0,1),
        "observation/joint_position": state,
        "prompt": "Follow the hand",
    }

    return observation

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

def get_teleop():
    pinch_threshold = 0.06
    hand_img, depth_colormap = hand_camera.get_image(True)
    hand_results = hands.process(hand_img)

    if hand_results.multi_hand_landmarks and hand_results.multi_hand_world_landmarks:
        for handLms, worldLms in zip(hand_results.multi_hand_landmarks,
                                    hand_results.multi_hand_world_landmarks):
            
            if ROBOT_DOF == 4:
                uv_middle_knuckle = hand_camera.to_pixel(handLms.landmark[5]) # middle knuckle (pixel)
                xyz_cam = hand_camera.convert_2d_to_3d(uv_middle_knuckle)
                if len(xyz_cam) != 3:
                    print("not all points detected")
                    return None
                xyz_robot = R @ np.array(xyz_cam)*1000 + trans

                # --- world coordinates (meters) ---
                thumb_w = worldLms.landmark[4]
                index_w = worldLms.landmark[8]
                thumb_3d = np.array([thumb_w.x, thumb_w.y, thumb_w.z])
                index_3d = np.array([index_w.x, index_w.y, index_w.z])
                d = np.linalg.norm(thumb_3d - index_3d)

                code, grip_curr = arm.get_gripper_position()
                if d <= pinch_threshold:
                    grip_cmd = np.max([0, grip_curr-100])
                else:
                    grip_cmd = np.min([850, grip_curr+100])

                return np.concatenate((xyz_robot, [grip_cmd]))
    else:
        print("No hand detection")
        return None


# =========================
# Execution Loop
# =========================
def execution_loop():
    global t
    
    while True:
        t0 = time.time()
        observation = get_observation()
        command_inf = get_action(observation)
        command_teleop = get_teleop()
        if command_teleop is not None:
            command = command_teleop
        else:
            command = command_inf
        print(f"command: {command}")

        cmd_gripper = command[-1]*-860 + 850 # unnormalize the gripper action

        # Hard-code roll, pitch, and yaw
        if ROBOT_DOF == 4:
            command = np.concatenate((command[0:3], np.array([np.pi, 0, 0]), [cmd_gripper]))

        if not debug and robot_active:
            arm.set_gripper_position(cmd_gripper)
            arm.set_position(x=command[0], y=command[1], z=command[2], roll=command[3],
                              pitch=command[4],yaw=command[5], is_radian=True, wait=False)
            
        time_passed = time.time() - t0
        print(f"Recording freq: {1/time_passed}")


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
    print("Starting control system...")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    exec_thread = threading.Thread(target=execution_loop, daemon=True)

    infer_thread.start()
    exec_thread.start()

    while True:
        time.sleep(1)