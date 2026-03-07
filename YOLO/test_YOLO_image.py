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

frame =  cv2.imread("./YOLO/test.png")  # Change this to your actual test image path!
# --- Main Logic (Clean and Dry) ---
while True:
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
