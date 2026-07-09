import pyrealsense2 as rs
import numpy as np
import cv2

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

            # --- DEPTH VIS ---
            depth_m = depth_img * self.depth_scale
            MAX_DEPTH_METERS = 2.0
            depth_clipped = np.clip(depth_m, 0, MAX_DEPTH_METERS)
            depth_norm = (depth_clipped / MAX_DEPTH_METERS * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        else: depth_colormap = None

        color_frame = frameset.get_color_frame()

        img = np.asanyarray(color_frame.get_data())

        return img, depth_colormap