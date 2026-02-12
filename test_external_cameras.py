#!/usr/bin/env python
"""
Script to identify and test external webcams, excluding the built-in laptop camera.

This script will:
1. Find all available cameras
2. Display their information
3. Test reading frames from each camera
4. Help you identify which cameras are your external webcams

Run this script to determine which camera indices correspond to your external webcams.
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv_sam import OpenCVCamera
from lerobot.cameras.configs import ColorMode
import cv2
import numpy as np
import contextlib
import io
import os

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

def display_frame_preview(frame, camera_id, window_name):
    """Display a preview window for the camera frame."""
    # Resize if too large for display
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    # Convert RGB to BGR for OpenCV display
    if len(frame.shape) == 3:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        display_frame = frame
    
    cv2.imshow(window_name, display_frame)
    # Use longer wait time (30ms) and check for key press
    key = cv2.waitKey(30) & 0xFF
    return key

def main():
    # Suppress stderr warnings during camera discovery
    with contextlib.redirect_stderr(io.StringIO()):
        print("=" * 60)
        print("Finding all available cameras...")
        print("=" * 60)
        
        # Find all cameras
        all_cameras = OpenCVCamera.find_cameras()
    
    if not all_cameras:
        print("No cameras found! Make sure your cameras are connected.")
        return
    
    print(f"\nFound {len(all_cameras)} camera(s):\n")
    
    # Display camera information
    for i, cam_info in enumerate(all_cameras):
        print(f"Camera #{i}:")
        print(f"  ID: {cam_info['id']}")
        print(f"  Name: {cam_info['name']}")
        print(f"  Backend: {cam_info['backend_api']}")
        profile = cam_info['default_stream_profile']
        print(f"  Resolution: {profile['width']}x{profile['height']}")
        print(f"  FPS: {profile['fps']}")
        # Add hint about which might be built-in
        if cam_info['id'] == 0 and profile['fps'] < 10:
            print(f"  ⚠️  Likely built-in laptop camera (low FPS)")
        print()
    
    print("\n" + "=" * 60)
    print("Testing each camera")
    print("=" * 60)
    print("\nIMPORTANT: Click on the camera window to make it active for keyboard input!")
    print("  - Press 'q' or 'Q' to skip to next camera")
    print("  - Press 'ESC' to exit")
    print("\nTIP: The built-in laptop camera is usually index 0 on macOS (often lower FPS).")
    print("     Your external webcams will be higher indices (1, 2, etc.) with higher FPS.\n")
    
    # Test each camera
    for cam_info in all_cameras:
        cam_id = cam_info['id']
        window_name = f"Camera {cam_id} - Press 'q' to skip, 'ESC' to exit"
        
        print(f"\n{'='*60}")
        print(f"Testing Camera {cam_id} ({cam_info['name']})")
        print(f"{'='*60}")
        
        try:
            # Create camera config
            config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            
            # Connect to camera
            camera = OpenCVCamera(config)
            camera.connect()
            
            print(f"✓ Successfully connected to camera {cam_id}")
            print("  Displaying live preview...")
            print("  Click the window and press 'q' to skip, 'ESC' to exit")
            
            # Read and display frames
            frame_count = 0
            max_frames = 300  # Auto-skip after ~10 seconds at 30fps
            while frame_count < max_frames:
                try:
                    frame = camera.read()
                    frame_count += 1
                    
                    # Display frame
                    key = display_frame_preview(frame, cam_id, window_name)
                    
                    # Check for 'q' or 'Q' key
                    if key == ord('q') or key == ord('Q'):
                        print(f"\n  ✓ Skipping camera {cam_id}...")
                        break
                    elif key == 27:  # ESC key
                        print("\n  Exiting...")
                        camera.disconnect()
                        cv2.destroyAllWindows()
                        return
                    
                    # Print frame info every 60 frames (less spam)
                    if frame_count % 60 == 0:
                        print(f"  Read {frame_count} frames (shape: {frame.shape}) - Press 'q' to skip")
                        
                except KeyboardInterrupt:
                    print("\n  Interrupted by user")
                    camera.disconnect()
                    cv2.destroyAllWindows()
                    return
                except Exception as e:
                    print(f"  Error reading frame: {e}")
                    break
            
            if frame_count >= max_frames:
                print(f"\n  ✓ Auto-skipped camera {cam_id} after {max_frames} frames")
            
            camera.disconnect()
            cv2.destroyWindow(window_name)
            print(f"✓ Camera {cam_id} test completed")
            
        except Exception as e:
            print(f"✗ Failed to use camera {cam_id}: {e}")
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Camera Testing Complete!")
    print("=" * 60)
    
    # Identify likely external cameras (not index 0, or index 0 with high FPS)
    external_cameras = []
    for cam_info in all_cameras:
        cam_id = cam_info['id']
        fps = cam_info['default_stream_profile']['fps']
        # Assume external if not index 0, or if index 0 has high FPS (unusual for built-in)
        if cam_id != 0:
            external_cameras.append(cam_id)
        elif cam_id == 0 and fps >= 10:  # Index 0 with decent FPS might be external
            external_cameras.append(cam_id)
    
    if external_cameras:
        print(f"\nBased on the cameras found, your external webcams are likely:")
        for cam_id in external_cameras:
            print(f"  - Camera {cam_id}")
        print("\nTo use only your external webcams for training, use:")
        if len(external_cameras) >= 2:
            print('  --robot.cameras="{')
            print(f'    front: {{type: opencv, index_or_path: {external_cameras[0]}, width: 640, height: 480, fps: 30}},')
            print(f'    side: {{type: opencv, index_or_path: {external_cameras[1]}, width: 640, height: 480, fps: 30}}')
            print('  }"')
        else:
            print('  --robot.cameras="{')
            print(f'    front: {{type: opencv, index_or_path: {external_cameras[0]}, width: 640, height: 480, fps: 30}}')
            print('  }"')
    else:
        print("\nTo use only your external webcams for training, use camera indices")
        print("that are NOT your built-in laptop camera (usually index 0).")
        print("\nExample configuration for two external webcams:")
        print('  --robot.cameras="{')
        print('    front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30},')
        print('    side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}')
        print('  }"')
        print("\nReplace indices 1 and 2 with the actual indices of your external webcams.")

if __name__ == "__main__":
    main()
