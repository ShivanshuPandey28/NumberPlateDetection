# 🚘 Number Plate Detection and Recognition using YOLO & OCR

## 🔍 Objective
This project aims to automate the detection and recognition of vehicle license plates from **images and videos** using deep learning. It combines a custom-trained **YOLOv8** model for plate detection with OCR engines like **PaddleOCR** (for images) and **EasyOCR** (for videos) for text extraction.

---

## 📷 Project Demo

- **Input:** Image or video of vehicles containing visible number plates.
- **Output:**
  - Annotated original image/video with bounding boxes and detected text.
  - Cropped license plate images.
  - Extracted license plate numbers saved in a `.txt` file.

---

## 🧠 Tech Stack

| Task                | Technology Used     |
|---------------------|---------------------|
| Object Detection    | [YOLOv8](https://github.com/ultralytics/ultralytics) |
| OCR (Image)         | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| OCR (Video)         | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Image Processing    | OpenCV              |
| Visualization       | Matplotlib          |
| Language            | Python              |

---

## 🏗️ Architecture / Workflow

       +------------------+
      | Image / Video    |
      +--------+---------+
               |
               ▼
    +------------------------+
    | YOLOv8 Model Inference |
    | Detects license plates |
    +-----------+------------+
                |
     +----------v------------+
     | Crop Detected Plate   |
     +----------+------------+
                |
    +-----------v-----------+
    | OCR (Paddle/EasyOCR) |
    | Extract Text          |
    +-----------+-----------+
                |
    +-----------v-----------+
    | Annotate & Save       |
    | - Image/Video output  |
    | - Text in .txt file   |
    +-----------------------+



---

## 🔧 Technology Overview

### 🧠 YOLOv8 (Ultralytics)
- Custom-trained model to detect license plates in real-time.
- Processes both images and video frames, returning bounding boxes with class and confidence.
- Chosen for its speed, accuracy, and support for custom object detection tasks.

### 🔤 OCR Engines
- **PaddleOCR (for images):** High-accuracy multilingual text recognition; handles angled and distorted plates.
- **EasyOCR (for videos):** Lightweight and fast, suitable for real-time frame-wise text detection.

### 🖼 OpenCV
- Performs image loading, cropping, drawing bounding boxes, and converting frames to grayscale for OCR.

### 📊 Matplotlib
- Visualizes annotated images and frames in real-time for debugging and demonstration.

### 🔣 NumPy
- Supports array and coordinate manipulation required for bounding boxes and OCR results.

### ⚙️ Python Standard Libraries
- `os`, `datetime`, `math`: For file handling, unique naming, and utility operations.

---


