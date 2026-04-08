from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import os
import threading
from typing import List, Dict, Optional

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor

class SAM3Backend(LabelStudioMLBase):
    def setup(self):
        self.set("model_version", "sam3-plate-v1")
        self.target_labels = ["Blank Plate", "Blue Plate", "Red Plate"]
        self.exemplar_store = {label: [] for label in self.target_labels}
        
        self.predictor = None
        self.model_ready = False
        self.cached_image_url = None
        # Values used for coordinate scaling
        self.img_h = None
        self.img_w = None
        
        # takes to long to load SAM3 so we put it on its own thread to avoid being timed out by Label Studio
        threading.Thread(target=self._init_model).start()

    def _init_model(self):
        print("Initializing SAM 3 on GPU...")
        
        # model_dir = os.getenv("MODEL_DIR")
        # model_filename = os.getenv("MODEL_FILENAME")
        # model_filepath = os.path.join(model_dir, model_filename)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "sam3.pt")
        print("Model path: " + str(model_path))

        overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, device="cuda", verbose=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.predictor.setup_model()
        self.model_ready = True
        print("SAM 3 ready for Plate Detection.")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        if not self.model_ready:
            return ModelResponse(predictions=[])

        task = tasks[0]
        image_url = task['data'].get('image') or task['data'].get('image_url')
        
        # Have we seen this image? If not, load it and set it for the predictor
        if self.cached_image_url != image_url:
            image_path = self.get_local_path(image_url)
            self.predictor.set_image(image_path)
            img = cv2.imread(image_path)
            self.img_h, self.img_w = img.shape[:2]
            self.cached_image_url = image_url

        predictions = []

        if context and 'result' in context:
            for ctx in context['result']:
                if ctx['type'] == 'rectanglelabels':
                    v = ctx['value']
                    bx = [
                        v['x'] * self.img_w / 100, 
                        v['y'] * self.img_h / 100, 
                        (v['x'] + v['width']) * self.img_w / 100, 
                        (v['y'] + v['height']) * self.img_h / 100
                    ]
                    
                    _, boxes = self.predictor.inference_features(
                        self.predictor.features, bboxes=[bx], src_shape=(self.img_h, self.img_w)
                    )
                    if boxes is not None:
                        predictions.append(self._format_box(boxes[0], v['rectanglelabels'], ctx))
        else:
            for label_name in self.target_labels:
                exemplars = self.exemplar_store[label_name]
                prompt_kwargs = {"exemplars": exemplars} if exemplars else {"text": [label_name]}
                
                _, boxes = self.predictor.inference_features(
                    self.predictor.features, 
                    src_shape=(self.img_h, self.img_w),
                    **prompt_kwargs
                )
                
                if boxes is not None:
                    for b in boxes:
                        predictions.append(self._format_box(b, [label_name]))

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return

        image_url = data['task']['data'].get('image') or data['task']['data'].get('image_url')
        img = cv2.imread(self.get_local_path(image_url))
        h, w = img.shape[:2] # Define locally to avoid AttributeError

        for result in data['annotation']['result']:
            if result['type'] == 'rectanglelabels':
                v = result['value']
                # Safety check if user didn't select a label
                if not v.get('rectanglelabels'): continue
                
                label = v['rectanglelabels'][0]
                
                x1, y1 = int(v['x'] * w / 100), int(v['y'] * h / 100)
                x2, y2 = int((v['x'] + v['width']) * w / 100), int((v['y'] + v['height']) * h / 100)
                
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    crop = img[y1:y2, x1:x2]
                    feat = self.predictor.extract_exemplar_features(crop)
                    self.exemplar_store[label].append(feat)
                    self.exemplar_store[label] = self.exemplar_store[label][-15:]

    def _format_box(self, box, labels, ctx=None):
        b = box.cpu().numpy()
        return {
            "from_name": ctx['from_name'] if ctx else "label",
            "to_name": ctx['to_name'] if ctx else "image",
            "type": "rectanglelabels",
            "value": {
                "x": (b[0] / self.img_w) * 100, 
                "y": (b[1] / self.img_h) * 100,
                "width": ((b[2] - b[0]) / self.img_w) * 100, 
                "height": ((b[3] - b[1]) / self.img_h) * 100,
                "rectanglelabels": labels
            },
            "score": 0.9
        }

# class NewModel(LabelStudioMLBase):
#     """Custom ML Backend model
#     """
    
#     def setup(self):
#         """Configure any parameters of your model here
#         """
#         self.set("model_version", "0.0.1")

#     def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
#         """ Write your inference logic here
#             :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
#             :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
#             :return model_response
#                 ModelResponse(predictions=predictions) with
#                 predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
#         """
#         print(f'''\
#         Run prediction on {tasks}
#         Received context: {context}
#         Project ID: {self.project_id}
#         Label config: {self.label_config}
#         Parsed JSON Label config: {self.parsed_label_config}
#         Extra params: {self.extra_params}''')

#         # example for resource downloading from Label Studio instance,
#         # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
#         # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

#         # example for simple classification
#         # return [{
#         #     "model_version": self.get("model_version"),
#         #     "score": 0.12,
#         #     "result": [{
#         #         "id": "vgzE336-a8",
#         #         "from_name": "sentiment",
#         #         "to_name": "text",
#         #         "type": "choices",
#         #         "value": {
#         #             "choices": [ "Negative" ]
#         #         }
#         #     }]
#         # }]
        
#         return ModelResponse(predictions=[])
    
#     def fit(self, event, data, **kwargs):
#         """
#         This method is called each time an annotation is created or updated
#         You can run your logic here to update the model and persist it to the cache
#         It is not recommended to perform long-running operations here, as it will block the main thread
#         Instead, consider running a separate process or a thread (like RQ worker) to perform the training
#         :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
#         :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
#         """

#         # use cache to retrieve the data from the previous fit() runs
#         old_data = self.get('my_data')
#         old_model_version = self.get('model_version')
#         print(f'Old data: {old_data}')
#         print(f'Old model version: {old_model_version}')

#         # store new data to the cache
#         self.set('my_data', 'my_new_data_value')
#         self.set('model_version', 'my_new_model_version')
#         print(f'New data: {self.get("my_data")}')
#         print(f'New model version: {self.get("model_version")}')

#         print('fit() completed successfully.')

