import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time

def test_device():
    try:
        # Initialize the device with the same configuration as Visualizer-base.py
        k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        
        # Try to start the device
        k4a.start()
        print("Device started successfully!")
        
        # Try to get a capture
        capture = k4a.get_capture()
        if capture.color is not None:
            print("Successfully captured color image")
            print(f"Color image shape: {capture.color.shape}")
        if capture.depth is not None:
            print("Successfully captured depth image")
            print(f"Depth image shape: {capture.depth.shape}")
        if capture.transformed_depth is not None:
            print("Successfully captured transformed depth image")
            print(f"Transformed depth shape: {capture.transformed_depth.shape}")
            
        # Stop the device
        k4a.stop()
        print("Device stopped successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_device() 