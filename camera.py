import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

class Camera():
    def __init__(self,serial,width,height,fps):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps

        self.cfg = rs.config()
        self.cfg.enable_device(self.serial)
        self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.pipe = rs.pipeline()
        self.align = rs.align(rs.stream.color)

        try:
            profile = self.pipe.start(self.cfg)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        except RuntimeError as e:
            print(f"\nFAILED START for {self.serial}")
            print("ERROR:", e)
            self.pipe = None

    def get_image(self,depth_bool=False):
        try:
            frameset = self.pipe.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            print("Frame timeout, skipping...")
            return None,None

        if depth_bool:
            frameset = self.align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            depth_img = np.asanyarray(depth_frame.get_data())

            self.depth_frame = depth_frame

            # --- DEPTH VIS ---
            depth_m = depth_img * self.depth_scale
            MAX_DEPTH_METERS = 2.0
            depth_clipped = np.clip(depth_m, 0, MAX_DEPTH_METERS)
            depth_norm = (depth_clipped / MAX_DEPTH_METERS * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        else:
            depth_colormap = None

        color_frame = frameset.get_color_frame()

        img = np.asanyarray(color_frame.get_data())

        return img, depth_colormap
    
    def convert_2d_to_3d(self, point2d):
        u = point2d[0]
        v = point2d[1]
        point3d = None
        if 0 <= u < self.width and 0 <= v < self.height:
            depth = self.depth_frame.get_distance(u, v)
            if depth > 0:
                intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
                point3d = rs.rs2_deproject_pixel_to_point(intrin, [u, v], depth)
        
        return point3d
    
    def to_pixel(self, lm):
        cx = int(np.clip(lm.x * self.width, 0, self.width - 1))
        cy = int(np.clip(lm.y * self.height, 0, self.height - 1))
        return [cx, cy]

    def initialize_hands():
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
                static_image_mode=False,
                max_num_hands=1
            )
        mpDraw = mp.solutions.drawing_utils
        return mpHands, hands, mpDraw