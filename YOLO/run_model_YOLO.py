from pypylon import pylon
from ultralytics import YOLO  # Import the YOLO model
import cv2
import os

MODEL_PATH = "./models/yolo8n_200_epoch.onnx"  # Change this to your actual trained weights path!
current_dir = os.path.dirname(os.path.abspath(__file__))
DEBUG = True  # Set to True to print debug information aka FPS and time taken for inference

# MODEL PARAMETERS
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3

# 1. Load the PyTorch YOLO model
print(f"Loading model {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")


def get_camera():
    # Attempt to initialize Basler Camera
    try:
        print("Attempting to connect to Basler camera...")
        tl_factory = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        camera.Open()
        camera.PixelFormat.SetValue("BGR8")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"Connected to: {camera.GetDeviceInfo().GetModelName()}")
        return camera, "basler"
    except Exception as e:
        print(f"Basler connection failed: {e}")
        print("Falling back to onboard camera...")
        # Fallback to default webcam
        cap = cv2.VideoCapture(0)
        return cap, "opencv"

def speed_to_string(results):
    preprocess_time = results[0].speed['preprocess']
    inference_time = results[0].speed['inference']
    postprocess_time = results[0].speed['postprocess']
    total_time = preprocess_time + inference_time + postprocess_time
    
    speed_str = (
        f"Preprocess: {preprocess_time:.4f}, "
        f"Inference: {inference_time:.4f}, "
        f"Postprocess: {postprocess_time:.4f},"
        f"Total: {total_time:.4f}"
    )

    return speed_str

cam_obj, cam_type = get_camera()

# I DID NOT KNOW ABOUT YIELD, THIS IS SUPER COOL GEMINI
def get_frame_generator(cam_obj, cam_type):
    """Generator that yields a frame regardless of camera type."""
    if cam_type == "basler":
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        while cam_obj.IsGrabbing():
            grab = cam_obj.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = converter.Convert(grab)
                yield img.GetArray()
            grab.Release()
    elif cam_type == "opencv":
        while True:
            ret, frame = cam_obj.read()
            if not ret: break
            yield frame
    else:
        raise ValueError("Unsupported camera type")

# --- Main Logic (Clean and Dry) ---
for frame in get_frame_generator(cam_obj, cam_type):
    # 2. Inference
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    if DEBUG:
        print(speed_to_string(results))
        
    # 3. Annotate & Display
    annotated_frame = results[0].plot()
    cv2.imshow("Camera Frame", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
