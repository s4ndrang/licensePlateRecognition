# licensePlateRecognition
LicensePlateRecognition is an Android mobile application that detects vehicle license plates in images using a custom-trained YOLOv8 model and extracts plate numbers via on-device OCR. The app runs entirely offline, providing fast and reliable license plate detection and recognition optimized for mobile devices.

## Features
- License plate detection using YOLOv8 (TFLite)
- Optimized for on-device Android inference
- OCR extraction of plate numbers using ML Kit
- Supports single-line and multi-line license plates
- Fast and offline processing
- Validation of detected license plates using regex and optional registry checks

## Technical Overview
### Detection
- Model: **YOLOv8n**
- Trained on ~10,000 public license plate images
- Single-class detection (`license_plate`)
- Exported to **TensorFlow Lite** for mobile deployment

### Recognition (OCR)
- Uses **Google ML Kit Text Recognition**
- Processes cropped license plate regions
- Sorts detected text lines by size to handle multi-line plates
- Normalizes and validates extracted text

## Tech Stack
- **Android (Java)**
- **TensorFlow Lite**
- **YOLOv8**
- **Google ML Kit (Text Recognition)**


## Workflow
1. User captures or selects an image
2. YOLOv8 model detects license plate bounding boxes
3. Detected plates are cropped and enhanced
4. OCR is applied to extract text
5. Results are validated and displayed to the user

## Installation
1. Clone the repository
2. Open in Android Studio
3. Place the `.tflite` model in the `assets/` folder
4. Build and run on an Android device


## Notes
- Performance depends on image quality and lighting
- Designed for offline, privacy-friendly usage


## License
This project uses open-source components and public datasets.  
Please verify dataset and model licenses before commercial use.

## Author
Developed as a custom license plate recognition solution optimized for mobile environments.
