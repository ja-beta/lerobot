from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv_sam import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# First, find all available cameras using the improved find_cameras method
print("Finding all available cameras...")
all_cameras = OpenCVCamera.find_cameras()
print(f"\nFound {len(all_cameras)} cameras that can read frames:\n")

for i, cam_info in enumerate(all_cameras):
    print(f"Camera #{i}:")
    print(f"  ID: {cam_info['id']}")
    print(f"  Name: {cam_info['name']}")
    print(f"  Backend: {cam_info['backend_api']}")
    print(f"  Resolution: {cam_info['default_stream_profile']['width']}x{cam_info['default_stream_profile']['height']}")
    print(f"  FPS: {cam_info['default_stream_profile']['fps']}")
    print()

# Test reading from each detected camera
for cam_info in all_cameras:
    cam_id = cam_info['id']
    print(f"\n{'='*50}")
    print(f"Testing camera {cam_id} ({cam_info['name']})")
    print(f"{'='*50}")
    
    try:
        # Use default settings from the camera
        config = OpenCVCameraConfig(
            index_or_path=cam_id,
            color_mode=ColorMode.RGB,
        )
        
        camera = OpenCVCamera(config)
        camera.connect()
        
        # Try reading a few frames
        print(f"Successfully connected to camera {cam_id}")
        for i in range(5):
            try:
                frame = camera.read()
                print(f"  Frame {i+1}: shape {frame.shape}, dtype {frame.dtype}")
            except Exception as e:
                print(f"  Error reading frame {i+1}: {e}")
        
        camera.disconnect()
        print(f"Camera {cam_id} test completed successfully!")
        
    except Exception as e:
        print(f"Failed to use camera {cam_id}: {e}")