from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import os
from typing import List, Dict, Optional

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor
import torch

class SAM3Backend(LabelStudioMLBase):
    _model_instance = None

    def __init__(self, project_id, **kwargs):
        super(SAM3Backend, self).__init__(project_id, **kwargs)
        
        if SAM3Backend._model_instance is None:
            print("--- INITIALIZING SAM 3 SINGLETON ---")
            model_path = os.path.join(os.path.dirname(__file__), "models", "sam3.pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")

            overrides = dict(
                model=model_path,
                device='cuda',
                task='segment',
                half=True 
            )

            predictor = SAM3SemanticPredictor(overrides=overrides)
            predictor.setup_model()
            SAM3Backend._model_instance = predictor
            print("SAM 3 model loaded successfully.")
        
        # Point to the shared singleton
        self.predictor = SAM3Backend._model_instance
        self.model_ready = True
        
        # State management
        self.target_labels = ["Blank Plate", "Blue Plate", "Red Plate"]
        self.exemplar_store = {label: [] for label in self.target_labels} # store exemplars for each label
        self.cached_image_url = None
        self.img_h, self.img_w = None, None

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        if not self.model_ready:
            print("Model not ready, returning empty predictions.")
            return ModelResponse(predictions=[])

        task = tasks[0]
        image_url = task['data'].get('image') or task['data'].get('image_url')
        image_path = self.get_local_path(image_url)
        
        # Load image and set features once per image
        if self.cached_image_url != image_url:
            self.predictor.set_image(image_path)
            img = cv2.imread(image_path)
            self.img_h, self.img_w = img.shape[:2]
            self.cached_image_url = image_url

        predictions = []

        # Use the high-level predictor call to handle exemplars/text correctly
        for label_name in self.target_labels:
            exemplars = self.exemplar_store[label_name]
            
            # If we have exemplars (few-shot), use the first one as the 'golden' reference
            if exemplars:
                # exemplar format: {"img": path, "bboxes": [[x1,y1,x2,y2]], "labels": [1]}
                results = self.predictor(exemplar=exemplars[0])
            else:
                # Fallback to text if no manual labels exist yet
                results = self.predictor(text=[label_name])
            
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    predictions.append(self._format_box(b.xyxy[0], [label_name]))

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """ Stores the user's manual annotations as exemplars for future predictions """
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return

        image_url = data['task']['data'].get('image') or data['task']['data'].get('image_url')
        image_path = self.get_local_path(image_url)
        
        for result in data['annotation']['result']:
            if result['type'] == 'rectanglelabels':
                v = result['value']
                if not v.get('rectanglelabels'): continue
                
                label = v['rectanglelabels'][0]
                
                # Convert percent coordinates to pixel coordinates
                x1 = v['x'] * self.img_w / 100
                y1 = v['y'] * self.img_h / 100
                x2 = (v['x'] + v['width']) * self.img_w / 100
                y2 = (v['y'] + v['height']) * self.img_h / 100
                
                # Store the data required for an 'exemplar' prompt
                new_exemplar = {
                    "img": image_path,
                    "bboxes": [[x1, y1, x2, y2]],
                    "labels": [1]
                }
                
                self.exemplar_store[label].insert(0, new_exemplar)
                self.exemplar_store[label] = self.exemplar_store[label][:5] # Keep last 5

    def _format_box(self, box, labels, ctx=None):
        # Ensure box is on CPU and numpy for scaling
        b = box.cpu().numpy()
        return {
            "from_name": "label",
            "to_name": "image",
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

