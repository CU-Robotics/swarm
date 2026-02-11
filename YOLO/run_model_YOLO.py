import cv2
import torch
import os
import numpy as np
import yaml
import pathlib
import os
import logging
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# import yolo model
parent_dir = pathlib.Path(__file__).resolve().parent
model_dir = parent_dir / 'best.pt'

yolo_model = YOLO(model_dir)



target_x_pixels = 100
target_y_pixels = 100

threshold = 0.9

def main():
    # cap = cv2.VideoCapture("/dev/stereo-cam-right-video", cv2.CAP_V4L2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # fmt = "UYVY"
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    assert cap.isOpened()

    guesses = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
    queue = []
    while True:
        ret, image = cap.read()
        
        if not ret:
            raise RuntimeError("no frame")

        classes = {1:'1', 2:'2', 3:'3', 4:'4', 5:'sentry', 6:'base', 7:'tower'}
        # classes = ["1", "2", "3", "4", "sentry", "base", "tower"]


        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.sqrt()),
        ])

        plates = yolo_model.predict(source=image, conf=0.5, save=False, verbose=False)[0].boxes.xyxy.cpu().numpy()
        # print(len(plates))
        # cv2.imshow("test", image)
        # cv2.waitKey(1)
        # print("plates:", plates)
        # # print(len(plates))
        # # cs = []
        # print("plates:", plates)
        for i, plate in enumerate(plates):
            print("looping")
            # get bounding box
            x, y, w, h = plate
            x = int(x)
            y = int(y)
            w = int(w) - x
            h = int(h) - y
            print(x, y, w, h)
            if w*h < 10:
                print("too small, skipping")
                continue
            # print(w, h)

            # find out largest side, if the width is greater, we want to make the height match
            # If the height is greater, we want to keep as usual
            l_side = max(w, h)

            # crop image to bounding box
            cropped = image[(y + h//2 - l_side//2):(y + h //2 + l_side//2), x:(x + w)]

            # resize image to target size
            cropped = cv2.resize(cropped, (target_x_pixels, target_y_pixels))

            bw = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            # cs.append(bw)
            cv2.imshow("test", cropped)
            cv2.waitKey(1)
            # cv2.imwrite("TEST.png", bw, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # return
            # bw = bw.astype(np.float32) / 255.0
            # bw = transform(Image.fromarray(bw))
            # bw = bw.unsqueeze(0)
            # # print(bw.shape)
            
            # start_time = cv2.getTickCount()
            # output = model(bw)
            # # predicted_class = classes[output.argmax().item()]
            # end_time = cv2.getTickCount()
            # time_taken = (end_time - start_time) / cv2.getTickFrequency()
            # fps = 1.0 / time_taken
            # # print(f"FPS: {fps:.2f}")

            
            # probs = torch.exp(output)
            # max_probs, preds = torch.max(probs, dim=1)
            # preds[max_probs < threshold] = -1

            # guess = preds.item()
            # max_prob = max_probs.item()

            # # print(guess)
            

            # if max_prob < threshold:
            #     # print(f"Low confidence ({max_prob:.2f}), skipping prediction")
            #     continue

            # print(max_probs.item(), classes[guess])
            # guesses[guess] += 1
            # queue.append(guess)
            
            # if len(queue) > 100:
            #     popped = queue.pop(0)
            #     guesses[popped] -= 1
            
            # print("Guesses:", guesses)
            

        # if len(cs):

if __name__ == "__main__":
    main()