import ray
from ray import serve
from paddleocr import TextDetection, TextRecognition
import numpy as np
from typing import Dict, Any, List, Union
import base64
from io import BytesIO
from PIL import Image
import json


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}  # Adjust GPU if needed
)
class TextDetectionService:
    def __init__(self):
        """Initialize the PaddleOCR text detection model."""
        self.model = TextDetection(model_name="PP-OCRv5_server_det")
    
    async def __call__(self, request) -> Dict[str, Any]:
        """
        Handle incoming requests for text detection.
        
        Args:
            request: HTTP request containing image data
            
        Returns:
            Dict containing detection results
        """
        # Get request data
        request_data = await request.json()
        
        # Handle different input formats
        if "image_path" in request_data:
            # Direct file path
            image_input = request_data["image_path"]
        elif "image_base64" in request_data:
            # Base64 encoded image
            image_data = base64.b64decode(request_data["image_base64"])
            image = Image.open(BytesIO(image_data))
            # Save temporarily for PaddleOCR (it expects file path)
            temp_path = "/tmp/temp_detect_image.jpg"
            image.save(temp_path)
            image_input = temp_path
        else:
            return {"error": "No image provided. Use 'image_path' or 'image_base64'"}
        
        # Get batch size (default to 1)
        batch_size = request_data.get("batch_size", 1)
        
        # Perform text detection
        try:
            output = self.model.predict(input=image_input, batch_size=batch_size)
            
            # Process results
            results = []
            for res in output:
                # Extract detection boxes and scores
                detection_result = {
                    "boxes": res.boxes.tolist() if hasattr(res, 'boxes') else [],
                    "scores": res.scores.tolist() if hasattr(res, 'scores') else [],
                }
                results.append(detection_result)
                
                # Optionally save results if paths are provided
                if "save_img_path" in request_data:
                    res.save_to_img(save_path=request_data["save_img_path"])
                if "save_json_path" in request_data:
                    res.save_to_json(save_path=request_data["save_json_path"])
            
            return {
                "status": "success",
                "results": results,
                "num_detections": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}  # Adjust GPU if needed
)
class TextRecognitionService:
    def __init__(self):
        """Initialize the PaddleOCR text recognition model."""
        self.model = TextRecognition(model_name="PP-OCRv5_server_rec")
    
    async def __call__(self, request) -> Dict[str, Any]:
        """
        Handle incoming requests for text recognition.
        
        Args:
            request: HTTP request containing image data
            
        Returns:
            Dict containing recognition results
        """
        # Get request data
        request_data = await request.json()
        
        # Handle different input formats
        if "image_path" in request_data:
            # Direct file path
            image_input = request_data["image_path"]
        elif "image_base64" in request_data:
            # Base64 encoded image
            image_data = base64.b64decode(request_data["image_base64"])
            image = Image.open(BytesIO(image_data))
            # Save temporarily for PaddleOCR (it expects file path)
            temp_path = "/tmp/temp_recognize_image.jpg"
            image.save(temp_path)
            image_input = temp_path
        else:
            return {"error": "No image provided. Use 'image_path' or 'image_base64'"}
        
        # Get batch size (default to 1)
        batch_size = request_data.get("batch_size", 1)
        
        # Perform text recognition
        try:
            output = self.model.predict(input=image_input, batch_size=batch_size)
            
            # Process results
            results = []
            for res in output:
                # Extract recognized text and confidence
                recognition_result = {
                    "text": res.text if hasattr(res, 'text') else "",
                    "score": res.score if hasattr(res, 'score') else 0.0,
                }
                results.append(recognition_result)
                
                # Optionally save results if paths are provided
                if "save_img_path" in request_data:
                    res.save_to_img(save_path=request_data["save_img_path"])
                if "save_json_path" in request_data:
                    res.save_to_json(save_path=request_data["save_json_path"])
            
            return {
                "status": "success",
                "results": results,
                "num_recognitions": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Create the Ray Serve applications
detection_app = TextDetectionService.bind()
recognition_app = TextRecognitionService.bind()