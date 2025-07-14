import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Set paths
IMAGE_PATH = "ImagesVideos\img2.jpg"
MODEL_PATH = "models/best.pt"
NUMPLATE_FOLDER = "NumPlate"
TEXT_OUTPUT_FILE = os.path.join(NUMPLATE_FOLDER, "extracted_numbers.txt")

# Ensure output folder exists
os.makedirs(NUMPLATE_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)
CLASS_NAMES = ["license_plate"]


def generate_uniquefilename(extension=".jpg"):
    """Generates a unique filename based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"plate_{timestamp}{extension}"


def detect_license_plate(image):
    """Detects license plate and returns the cropped plate image (without saving)."""
    results = model.predict(image, conf=0.27)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            cls_name = CLASS_NAMES[int(box.cls[0])]

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f'{cls_name}: {conf}'
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Return the detected license plate region (in memory)
            return image[y1:y2, x1:x2]

    return None  # No plate detected


def extract_text_from_plate_paddleocr(plate_img):
    """Extracts text from the detected license plate image using PaddleOCR."""
    if plate_img is None:
        print("No license plate detected.")
        exit()

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Perform OCR on the plate image
    result = ocr.ocr(plate_img, cls=True)
    
    extracted_texts = []
    
    # Process the results
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]  # The recognized text
            confidence = line[1][1]  # Confidence score
            
            # Convert points to int (required for OpenCV)
            points = line[0]
            points = np.array(points).astype(int)
            
            # Draw box around text
            cv2.polylines(plate_img, [points], True, (0, 255, 0), 2)
            
            # Add text above the box
            min_x = min([p[0] for p in points])
            min_y = min([p[1] for p in points])
            cv2.putText(plate_img, text, (min_x, min_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"Detected text: {text} with confidence {confidence:.2f}")
            extracted_texts.append(text)
    
    # If no text is detected, add a message
    if not extracted_texts:
        print("No text detected in the license plate.")
        extracted_texts = ["No text detected"]

    # Append extracted texts to file
    with open(TEXT_OUTPUT_FILE, "a") as file:
        file.write("\n".join(extracted_texts) + "\n")
    
    return plate_img, extracted_texts


def main():
    # Load image
    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print(f"Error: Could not read image from {IMAGE_PATH}")
        return
    
    # Make a copy of the original image
    original_image = image.copy()
    
    # Detect license plate
    plate_img = detect_license_plate(image)
    
    if plate_img is None:
        print("No license plate detected.")
        return
    
    # Extract text from the plate
    annotated_plate, detected_texts = extract_text_from_plate_paddleocr(plate_img)
    
    # Generate unique filename
    plate_filename = generate_uniquefilename(".jpg")
    plate_path = os.path.join(NUMPLATE_FOLDER, plate_filename)
    
    # Save the annotated plate image
    cv2.imwrite(plate_path, annotated_plate)
    
    # Display the original image with detected plate
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Detected Plate")
    plt.axis('off')
    
    # Display the cropped and annotated plate
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated_plate, cv2.COLOR_BGR2RGB))
    plt.title("License Plate with Detected Text")
    plt.axis('off')
    
    print(f"\nFinal annotated image saved as: {plate_path}")
    print(f"Extracted text saved in: {TEXT_OUTPUT_FILE}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()