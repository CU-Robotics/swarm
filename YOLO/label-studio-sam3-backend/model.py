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
    _initalized = False
    _model = None
    _store = None
    _labels = None
    _prompts = None

    img_h = 1200
    img_w = 1920

    def setup(self):
        """ 
            Initializes the SAM3 Backend

            LabelStudioMLBase will create a new instance of SAM3Backend for each worker process, 
            but we only want to initialize the heavy SAM 3 model and exemplar storage once per process, not once per instance.
            To achieve this, we use class variables to store the model and related data, and check if it's already initialized before doing the setup work. 
        """
        if SAM3Backend._initalized:
            logger.info("--- SKIP: SAM 3 already initialized in this process ---")
            # Just ensure the instance variables are linked
            self.predictor = SAM3Backend._model
            self.exemplar_store = SAM3Backend._store
            self.target_labels = SAM3Backend._labels
            self.target_prompts = SAM3Backend._prompts
            return
        
        logging.info("--- INITIALIZING SAM 3 MODEL ---")
        model_path = os.path.join(os.path.dirname(__file__), "models", "sam3.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")

        overrides = dict(
            model=model_path,
            device='cuda',
            task='segment',
            half=True 
        )

        # Intilize all the stuff that needs to persist across instances in class variables, 
        SAM3Backend._model = SAM3SemanticPredictor(overrides=overrides)
        SAM3Backend._model.setup_model()
        SAM3Backend._labels = os.getenv("LABELS").split(",") 
        SAM3Backend._prompts = os.getenv("INITIAL_PROMPTS").split(",")
        SAM3Backend._store = {label: [] for label in SAM3Backend._labels} # store exemplars for each label 

        # and point instance variables to them
        self.predictor = SAM3Backend._model
        self.exemplar_store = SAM3Backend._store
        self.target_labels = SAM3Backend._labels
        self.target_prompts = SAM3Backend._prompts
                
        for label, prompt in zip(self.target_labels, self.target_prompts):
            logger.debug(f"Label: '{label}' will use initial prompt: '{prompt}'")

        SAM3Backend._initalized = True
        logging.info("--- SAM 3 MODEL INITIALIZATION COMPLETE ---")
        
        

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ 
        Uses the prompt and exeplars to generate predictions for the given task (image)
        
        returns a list of predictions in the label-studio format
        """
        if not SAM3Backend._initalized:
            logging.info("Model not ready, returning empty predictions.")
            return ModelResponse(predictions=[])

        # assume one task at a time and breaks out important info
        task = tasks[0]
        image_url = task['data'].get('image') or task['data'].get('image_url')
        image_path = self.get_local_path(image_url)

        logger.debug(f"Processing Image URL: {image_url}")
        
        # setup model
        self.predictor.set_image(image_path)
        
        # set image dimensions for the format_box
        img = cv2.imread(image_path)
        self.img_h, self.img_w = img.shape[:2]
        self.cached_image_url = image_url
    
        logger.debug(f"Active prompts for this prediction: {self.target_prompts}")
        predictions = []

        # For each of the possible labels, run model using the correct prompt and exemplars
        for i, label_name in enumerate(self.target_labels):
            exemplars = self.exemplar_store[label_name]
            logger.debug(f"Exemplars for {label_name}: {exemplars}")
            
            # If we have exemplars, use with model. Will be slower
            if exemplars:
                # exemplar format: {"img": path, "bboxes": [[x1,y1,x2,y2]], "labels": [1]}
                results = self.predictor(exemplar=exemplars, text=[self.target_prompts[i]])
            else:
                # Fallback to text if no manual labels exist yet
                results = self.predictor(text=[self.target_prompts[i]])
            
            # If we have some bounding boxes, format them for label-studio and add to our predictions list
            if results and results[0].boxes is not None:
                logger.info(f"Detected {len(results[0].boxes)} potential plates for label {label_name}")
                for b in results[0].boxes:

                    # return the bounding box, label, and confidence to be translated to label-studio form
                    predictions.append(self._format_box(b.xyxy[0], [label_name], b.conf[0])) 

        return ModelResponse(predictions=[{'result': predictions}])

    def fit(self, event, data, **kwargs):
        """ Stores the user's manual annotations as exemplars for future predictions """
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return

        image_url = data['task']['data'].get('image') or data['task']['data'].get('image_url')
        image_path = self.get_local_path(image_url)
                
        for result in data['annotation']['result']:
            if result['type'] == 'rectanglelabels':
                v = result['value']
                self.img_w = result.get('original_width')
                self.img_h = result.get('original_height')

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

                logger.debug(f"New exemplar for label '{label}': {new_exemplar}")

                # 3. Handle Debug Image Saving
                if logger.isEnabledFor(logging.DEBUG):
                    # Use a subdirectory for each label to stay organized
                    debug_dir = os.path.join(os.path.dirname(__file__), "debug", label.replace(" ", "_"))
                    os.makedirs(debug_dir, exist_ok=True)

                    # Use a timestamp or unique ID to avoid filename collisions
                    unique_id = int(time.time() * 1000)
                    filename = f"crop_{unique_id}_{os.path.basename(image_path)}"
                    save_path = os.path.join(debug_dir, filename)
                    full_img = cv2.imread(image_path)
                    crop = full_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(save_path, crop)
                        logger.debug(f"Saved debug crop to: {save_path}")
                
                self.exemplar_store[label].insert(0, new_exemplar)
                self.exemplar_store[label] = self.exemplar_store[label][:5] # Keep last 5

    def _format_box(self, box, label, score=0.0):
        """ converts model's box output to label-studio format """
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
