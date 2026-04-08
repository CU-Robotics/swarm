import os

from PIL import Image, ImageOps

import numpy as np
import ultralytics

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


class YOLO(LabelStudioMLBase):
    def setup(self) -> None:

        # Model and labels
        self.model = self.load_model()
        self.labels = self.model.names
        self.from_name = "label"
        self.to_name = "image"

    def load_model(self) -> ultralytics.YOLO:
        model_dir = os.getenv("MODEL_DIR")
        model_filename = os.getenv("MODEL_FILENAME")
        model_filepath = os.path.join(model_dir, model_filename)
        model = ultralytics.YOLO(model_filepath)
        self.set("model_version", f"yolo-{model.task}")

        return model

    def load_image(self,
                   task: dict) -> Image.Image:
        # Get image path and task id
        image_path = task.get("data").get("image")
        task_id = task.get("id")

        # Extract local image path
        file_path = self.get_local_path(image_path,
                                        task_id=task_id)

        # Open image
        image = Image.open(file_path)
        image = ImageOps.exif_transpose(image)

        return image

    def predict(self, tasks: list[dict], **kwargs) -> ModelResponse:
        # Create blank list with results
        results = []

        # Create variable to calcualte scores
        score = 0
        counter = 0

        for task in tasks:
            # Load image
            image = self.load_image(task=task)

            # Height and width of image
            image_width, image_height = image.size

            # Getting prediction using model
            model_prediction = self.model.predict(image)

            # Getting boxes from model prediction
            for pred in model_prediction:
                for i, box in enumerate(pred.boxes):

                    # Points
                    xyxy = box.xyxy[0].tolist()
                    x = xyxy[0] / image_width * 100
                    y = xyxy[1] / image_height * 100
                    width = (xyxy[2] - xyxy[0]) / image_width * 100
                    height = (xyxy[3] - xyxy[1]) / image_height * 100

                    # Label
                    labels = [self.labels[int(box.cls.item())]]

                    result = {"from_name": self.from_name,
                              "to_name": self.to_name,
                              "id": str(i),
                              "type": "rectanglelabels",
                              "score": box.conf.item(),
                              "original_width": image_width,
                              "original_height": image_height,
                              "image_rotation": 0,
                              "value": {
                                  "rotation": 0,
                                  "x": x,
                                  "y": y,
                                  "width": width,
                                  "height":  height,
                                  "rectanglelabels": labels}}

                    # Append prediction to predictions
                    results.append(result)

                    # Add score
                    score += box.conf.item()
                    counter += 1

        predictions = [{"result": results,
                       "score": score / counter,
                        "model_version": self.model_version}]
        
        if counter == 0:
            print("No predictions, setting score to 0")
            score = 0
            counter = 1

        return ModelResponse(predictions=predictions)


    def fit(self, event, data, **kwargs):
        raise NotImplementedError("Training is not implemented yet")
