import cv2
import os
import math
import easyocr
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO

# Paths
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "ImagesVideos/DemoVideo.mp4"
OUTPUT_DIR = "NumPlate"
TEXT_FILE = os.path.join(OUTPUT_DIR, "final_result.txt")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize YOLO Model
model = YOLO(MODEL_PATH)
CLASS_NAMES = ["license_plate"]

# Initialize EasyOCR
reader = easyocr.Reader(['en'])  

# Set up Matplotlib figure
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()
img_display = None


def generate_unique_filename(extension=".jpg"):
    """Generates a unique filename using timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"plate_{timestamp}{extension}"


def process_frame(frame, frame_count, text_file):
    """Processes a single frame to detect and extract license plates."""
    results = model.predict(frame, conf=0.27)

    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            cls_name = CLASS_NAMES[int(box.cls[0])]

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f'{cls_name}: {conf}'
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Extract license plate region
            plate_region = frame[y1:y2, x1:x2]

            # Save plate image
            plate_filename = generate_unique_filename(".jpg")
            plate_path = os.path.join(OUTPUT_DIR, plate_filename)
            cv2.imwrite(plate_path, plate_region)

            # Perform OCR
            detected_texts = extract_text(plate_region)

            # Write detected text to file (Append)
            with open(text_file, "a") as f:
                for detected_text in detected_texts:
                    f.write(f"Frame {frame_count}, Plate {i + 1}: {detected_text}\n")

            print(f"Frame {frame_count}, Plate {i + 1}: {detected_texts}")


def extract_text(plate_img):
    """Extracts text from license plate using EasyOCR."""
    if plate_img is None or plate_img.size == 0:
        return []

    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    ocr_results = reader.readtext(gray_plate)

    extracted_texts = []
    for _, extracted_text, confidence in ocr_results:
        print(f"Detected text: {extracted_text} (Confidence: {confidence:.2f})")
        extracted_texts.append(extracted_text)

    return extracted_texts


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing the video.")
            break

        frame_count += 1
        print(f"Processing frame: {frame_count}")

        # Process current frame
        process_frame(frame, frame_count, TEXT_FILE)

        # Display frame with Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        global img_display
        if img_display is None:
            img_display = ax.imshow(frame_rgb)
        else:
            img_display.set_data(frame_rgb)

        plt.draw()
        plt.pause(0.001)

    # Release resources
    cap.release()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
