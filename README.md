# FilterPix

FilterPix is an offline image culling and sorting application designed to help photographers quickly organize large photo collections. It provides a fast, accessible way to remove unwanted or low-quality images after events or when managing image repositories. FilterPix optionally sorts images using AI-based object detection, all while keeping your original files untouched.
---

## Key Features

- Sharpness-based filtering using Laplacian edge detection  
- Burst sequence analysis with automatic best-shot selection  
- Metadata-based rating filtering  
- Optional AI-powered image detection and sorting  
- Fully offline processing  
- Non-destructive workflow

---

## DISCLAIMER

- Ensure you have sufficient free space on the output drive, at least equal to the size of the input folder
- Filtering and sorting are not 100% accurate; false results may occur
- Processing may cause high hardware usage, especially on low-end systems
- Remember: your original files remain untouched throughout the process
- User is fully responsible for verifying results and managing files

---

## Quick Start

1. Launch **FilterPix**
2. Click **Input** and select the folder containing your images
3. (Optional) Adjust filter and sorting settings
4. Click **Start** to begin processing
5. Review results in the output folders:
   - **Sharp** – images that passed filters  
   - **Sorted** – images processed with object detection
6. (Optional) After reviewing the output, you can delete the input folder to save storage

Original image files are never modified.

---

## Use Cases

- Fast photo cleanup after events or shoots  
- Selecting best shots from burst photography  
- Removing out-of-focus or low-quality images  
- Offline image organization on low- to mid-range hardware  

---

## Documentation

- User Manual: `user manual.txt`  
- Detailed settings, workflows, and filter behavior are documented in the manual  

---

## Technical Overview

- Language: Python 3.11
- GUI: Tkinter  
- Image Processing: OpenCV, Pillow  
- AI Detection: YOLO (Ultralytics)  
- Designed for offline execution  

---


