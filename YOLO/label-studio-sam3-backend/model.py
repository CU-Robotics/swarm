from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor
import torch


# Set up logging
load_dotenv() # This looks for the .env file and loads the variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class SAM3Backend(LabelStudioMLBase):
    _model_instance = None

    def __init__(self, project_id, **kwargs):
        super(SAM3Backend, self).__init__(project_id, **kwargs)
        
        if SAM3Backend._model_instance is None:
            logging.info("--- INITIALIZING SAM 3 SINGLETON ---")
            model_path = os.path.join(os.path.dirname(__file__), "models", "sam3.pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {device}")

            overrides = dict(
                model=model_path,
                device='cuda',
                task='segment',
                half=True 
            )

            predictor = SAM3SemanticPredictor(overrides=overrides)
            predictor.setup_model()
            SAM3Backend._model_instance = predictor
            logging.info("SAM 3 model loaded successfully.")
        
        # Point to the shared singleton
        self.predictor = SAM3Backend._model_instance
        self.model_ready = True
        
        # State management
        self.target_labels = os.getenv("LABELS")
        self.target_prompts = os.getenv("INITIAL_PROMPTS")

        print(f"Loaded LABELS: {self.target_labels}")
        print(f"Loaded INITIAL_PROMPTS: {self.target_prompts}")
        
        self.exemplar_store = {label: [] for label in self.target_labels} # store exemplars for each label
        self.cached_image_url = None
        self.img_h, self.img_w = 1200, 1920 # default values to avoid division by zero

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        if not self.model_ready:
            logging.info("Model not ready, returning empty predictions.")
            return ModelResponse(predictions=[])

        task = tasks[0]
        image_url = task['data'].get('image') or task['data'].get('image_url')
        image_path = self.get_local_path(image_url)

        logger.debug(f"Processing Image URL: {image_url}")
        
        
        # Load image and set features once per image
        if self.cached_image_url != image_url:
            self.predictor.set_image(image_path)
            img = cv2.imread(image_path)
            self.img_h, self.img_w = img.shape[:2]
            self.cached_image_url = image_url

    
        logger.debug(f"Active prompts for this prediction: {self.target_prompts}")
        predictions = []

        # Use the high-level predictor call to handle exemplars/text correctly
        for i, label_name in enumerate(self.target_labels):
            exemplars = self.exemplar_store[label_name]
            logger.debug(f"Exemplars for {label_name}: {exemplars}")
            
            # If we have exemplars (few-shot), use the first one as the 'golden' reference
            if exemplars:
                # exemplar format: {"img": path, "bboxes": [[x1,y1,x2,y2]], "labels": [1]}
                results = self.predictor(exemplar=exemplars, text=[self.target_prompts[i]])
            else:
                # Fallback to text if no manual labels exist yet
                results = self.predictor(text=[self.target_prompts[i]])
            
            if results and results[0].boxes is not None:
                logger.info(f"Detected {len(results[0].boxes)} potential plates for label {label_name}")
                for b in results[0].boxes:

                    # return the bounding box, label, and confidence to be translated to label-studio form
                    predictions.append(self._format_box(b.xyxy[0], [label_name], b.conf[0])) 

        # NEW (Valid format)
        return ModelResponse(predictions=[{'result': predictions}])

    def fit(self, event, data, **kwargs):
        """ Stores the user's manual annotations as exemplars for future predictions """
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return

        image_url = data['task']['data'].get('image') or data['task']['data'].get('image_url')
        image_path = self.get_local_path(image_url)
        full_img = cv2.imread(image_path)
        
        for result in data['annotation']['result']:
            if result['type'] == 'rectanglelabels':
                v = result['value']
                logger.debug(f"Processing user annotation: {v}")
                if not v.get('rectanglelabels'): continue
                
                label = v['rectanglelabels'][0]

                # Convert percent coordinates to pixel coordinates (integer division)
                x1 = int(v['x'] * self.img_w / 100)
                y1 = int(v['y'] * self.img_h / 100)
                x2 = int((v['x'] + v['width']) * self.img_w / 100)
                y2 = int((v['y'] + v['height']) * self.img_h / 100)
                
                # Store the data required for an 'exemplar' prompt
                new_exemplar = {
                    "img": image_path,
                    "bboxes": [[x1, y1, x2, y2]],
                    "labels": [label]
                }

                # 3. Handle Debug Image Saving
                if logger.isEnabledFor(logging.DEBUG):
                    # Use a subdirectory for each label to stay organized
                    debug_dir = os.path.join(os.path.dirname(__file__), "debug", label.replace(" ", "_"))
                    os.makedirs(debug_dir, exist_ok=True)

                    # Use a timestamp or unique ID to avoid filename collisions
                    unique_id = int(time.time() * 1000)
                    filename = f"crop_{unique_id}_{os.path.basename(image_path)}"
                    save_path = os.path.join(debug_dir, filename)

                    crop = full_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(save_path, crop)
                        logger.debug(f"Saved debug crop to: {save_path}")
                
                self.exemplar_store[label].insert(0, new_exemplar)
                self.exemplar_store[label] = self.exemplar_store[label][:5] # Keep last 5

    def _format_box(self, box, label, score=0.0):
        # .cpu().numpy() gets it off the GPU, 
        # but the array still contains float16 values.
        b = box.cpu().numpy()
        
        return {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                # Use float() to convert from numpy.float16 to python float
                "x": float((b[0] / self.img_w) * 100), 
                "y": float((b[1] / self.img_h) * 100),
                "width": float(((b[2] - b[0]) / self.img_w) * 100), 
                "height": float(((b[3] - b[1]) / self.img_h) * 100),
                "rectanglelabels": label
            },
            # Ensure the score isn't a float16 either
            "score": float(score) 
        }
