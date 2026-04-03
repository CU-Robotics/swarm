#convert pt file to onnx

from ultralytics import YOLO

model = YOLO("models/yolo26s_300_fakeplates.pt", task="detect")  # load a custom model
# Export the model to ONNX format
model.export(format="onnx", imgsz= [480, 768])