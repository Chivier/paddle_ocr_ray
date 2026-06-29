import requests
import base64
import json

# Server URL
BASE_URL = "http://localhost:8000"

def detect_text(image_path=None, image_base64=None):
    """Call the text detection service."""
    payload = {"batch_size": 1}
    
    if image_path:
        payload["image_path"] = image_path
    elif image_base64:
        payload["image_base64"] = image_base64
    
    response = requests.post(f"{BASE_URL}/detect", json=payload)
    return response.json()

def recognize_text(image_path=None, image_base64=None):
    """Call the text recognition service."""
    payload = {"batch_size": 1}
    
    if image_path:
        payload["image_path"] = image_path
    elif image_base64:
        payload["image_base64"] = image_base64
    
    response = requests.post(f"{BASE_URL}/recognize", json=payload)
    return response.json()

def full_ocr_pipeline(image_path):
    """Run both detection and recognition on an image."""
    print(f"Processing image: {image_path}")
    
    # Step 1: Detect text regions
    print("\n1. Detecting text regions...")
    detection_result = detect_text(image_path=image_path)
    print(f"Found {detection_result.get('num_detections', 0)} text regions")
    
    # Step 2: Recognize text
    print("\n2. Recognizing text...")
    recognition_result = recognize_text(image_path=image_path)
    print(f"Recognized {recognition_result.get('num_recognitions', 0)} text items")
    
    # Combine results
    if detection_result["status"] == "success" and recognition_result["status"] == "success":
        print("\n3. Combined Results:")
        for i, (det, rec) in enumerate(zip(
            detection_result["results"], 
            recognition_result["results"]
        )):
            print(f"\nText Region {i+1}:")
            print(f"  Boxes: {det['boxes']}")
            print(f"  Detection Score: {det['scores']}")
            print(f"  Recognized Text: '{rec['text']}'")
            print(f"  Recognition Score: {rec['score']}")
    
    return detection_result, recognition_result

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    
    # Use individual services
    print("=== Testing Detection Service ===")
    det_result = detect_text(image_path=image_path)
    print(json.dumps(det_result, indent=2))
    
    print("\n=== Testing Recognition Service ===")
    rec_result = recognize_text(image_path=image_path)
    print(json.dumps(rec_result, indent=2))
    
    # Or use the full pipeline
    print("\n=== Full OCR Pipeline ===")
    full_ocr_pipeline(image_path)