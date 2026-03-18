from pypylon import pylon
import cv2 
import os
import time

#s/n = 40688134

# --- Configuration ---
target_fps = 5  # Set how many frames you want to save per second
folder_name = "blue_red_down"  # Name of the output folder
# ---------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, folder_name)

# Create the output directory if it doesn't exist yet
os.makedirs(output_dir, exist_ok=True)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
print("Device:", camera.GetDeviceInfo().GetModelName())

camera.Open()

# Optional: set pixel format
camera.PixelFormat.SetValue("BGR8")

camera.StartGrabbing()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

# --- Timing and Tracking Variables ---
save_interval = 1.0 / target_fps
last_save_time = time.time()
saved_image_counter = 0

while camera.IsGrabbing():
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab.GrabSucceeded():
        img = converter.Convert(grab)
        frame = img.GetArray()

        # Show the raw camera stream
        # change size to smaller from 1920x1200
        cv2.imshow("Camera Stream", cv2.resize(frame, (960, 600)))

        # --- Capture and Save Logic ---
        current_time = time.time()
        if (current_time - last_save_time) >= save_interval:
            # Format filename to include folder name and a 5-digit counter
            filename = f"{folder_name}_{saved_image_counter:05d}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # Save the raw frame
            cv2.imwrite(save_path, frame)
            print(f"Saved: {filename}")
            
            last_save_time = current_time
            saved_image_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Grab failed")

    grab.Release()

# Clean up hardware resources and windows
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()