from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time

import cv2
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor
import torch


# Set up logging
load_dotenv() # This looks for the .env file and loads the variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
print(f"Log level set to: {LOG_LEVEL}")
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class SAM3Backend(LabelStudioMLBase):
    _initialized = False
    _model = None
    _labels = None
    _prompts = None

    img_h = 1200
    img_w = 1920

    def setup(self):
        """ 
            Initializes the SAM3 Backend

            LabelStudioMLBase will create a new instance of SAM3Backend for each worker process, 
            but we only want to initialize the heavy SAM 3 model once per process, not once per instance.
            To achieve this, we use class variables to store the model and related data, and check if it's already initialized before doing the setup work. 
        """
        if SAM3Backend._initialized:
            logger.info("--- SKIP: SAM 3 already initialized in this process ---")
            # Just ensure the instance variables are linked
            self.predictor = SAM3Backend._model
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
            half=True,
            imgsz=1024
        )

        # Intilize all the stuff that needs to persist across instances in class variables, 
        SAM3Backend._model = SAM3SemanticPredictor(overrides=overrides)
        SAM3Backend._model.setup_model()
        SAM3Backend._labels = os.getenv("LABELS").split(",") 
        SAM3Backend._prompts = os.getenv("INITIAL_PROMPTS").split(",")

        # and point instance variables to them
        self.predictor = SAM3Backend._model
        self.target_labels = SAM3Backend._labels
        self.target_prompts = SAM3Backend._prompts
                
        for label, prompt in zip(self.target_labels, self.target_prompts):
            logger.debug(f"Label: '{label}' will use initial prompt: '{prompt}'")

        SAM3Backend._initialized = True
        logging.info("--- SAM 3 MODEL INITIALIZATION COMPLETE ---")
        
        

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ 
        Uses the prompt and exeplars to generate predictions for the given task (image)
        
        returns a list of predictions in the label-studio format
        """
        if not SAM3Backend._initialized:
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
        with Image.open(image_path) as im:
            self.img_w, self.img_h = im.size
    
        logger.debug(f"Active prompts for this prediction: {self.target_prompts}")
        predictions = []

        # For each of the possible labels, run model using the correct prompt
        for i, label_name in enumerate(self.target_labels):
            results = self.predictor(text=[self.target_prompts[i]])
            
            # If we have some bounding boxes, format them for label-studio and add to our predictions list
            if results and results[0].boxes is not None:
                logger.info(f"Detected {len(results[0].boxes)} potential plates for label {label_name}")
                for b in results[0].boxes:

                    # return the bounding box, label, and confidence to be translated to label-studio form
                    predictions.append(self._format_box(b.xyxy[0], [label_name], b.conf[0])) 

        return ModelResponse(predictions=[{'result': predictions}])

    def fit(self, event, data, **kwargs):
        raise NotImplementedError("Training is not implemented yet")

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
