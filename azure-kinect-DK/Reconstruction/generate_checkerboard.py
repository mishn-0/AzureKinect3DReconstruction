import numpy as np
import cv2
import os

def generate_checkerboard(output_path="checkerboard.png", size=(10, 7), square_size=30, border_size=2):
    """
    Generate a checkerboard pattern for camera calibration.
    
    Args:
        output_path (str): Path to save the checkerboard image
        size (tuple): Number of internal corners (width, height)
        square_size (int): Size of each square in pixels
        border_size (int): Size of the black border in pixels
    """
    # Calculate total image size (including border)
    width = size[0] * square_size + 2 * border_size
    height = size[1] * square_size + 2 * border_size
    
    # Create black image
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Draw checkerboard pattern
    for i in range(size[1] + 1):
        for j in range(size[0] + 1):
            if (i + j) % 2 == 0:
                x1 = border_size + j * square_size
                y1 = border_size + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"[INFO] Checkerboard pattern saved to {output_path}")
    print(f"[INFO] Pattern size: {size[0]}x{size[1]} internal corners")
    print(f"[INFO] Square size: {square_size} pixels")
    print(f"[INFO] Total size: {width}x{height} pixels")
    print(f"[INFO] Border size: {border_size} pixels")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("calibration_patterns", exist_ok=True)
    
    # Generate different sizes of checkerboard patterns
    patterns = [
        ("calibration_patterns/checkerboard_10x7.png", (10, 7), 30),  # Default size
        ("calibration_patterns/checkerboard_10x7_large.png", (10, 7), 50),  # Larger squares
        ("calibration_patterns/checkerboard_10x7_small.png", (10, 7), 20),  # Smaller squares
    ]
    
    for output_path, size, square_size in patterns:
        generate_checkerboard(output_path, size, square_size, border_size=2) 