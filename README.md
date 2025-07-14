# 🚘 Number Plate Detection and Recognition using YOLO & OCR

## 🔍 Objective
This project detects and reads vehicle license plates from **images and videos** using **YOLOv8** for object detection and **OCR (PaddleOCR / EasyOCR)** for text recognition.

---

## 📷 Project Demo

- **Input:** Image/Video of vehicles with visible number plates.
- **Output:** Cropped license plate image + Recognized text + Annotated output image/video.

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
