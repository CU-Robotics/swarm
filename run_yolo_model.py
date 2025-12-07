from pypylon import pylon
import cv2 
import ultralytics
import os


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
print("Device:", camera.GetDeviceInfo().GetModelName())

camera.Open()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.onnx")

# Optional: set pixel format
camera.PixelFormat.SetValue("BGR8")

camera.StartGrabbing()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

# load in yolov8 model
model = ultralytics.YOLO(model_path)

while camera.IsGrabbing():
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab.GrabSucceeded():
        img = converter.Convert(grab)
        frame = img.GetArray()

        results = model(frame)

        # show results overlaid on image
        annotated_frame = results[0].plot()

        cv2.imshow("Annotated Image", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        
    else:
        print("Grab failed")

    grab.Release()
