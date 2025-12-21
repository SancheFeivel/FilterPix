# FilterPix

FilterPix is an offline image culling and sorting application designed to help photographers quickly clean large photo collections. It removes blurry images, selects the best shots from burst sequences, and optionally sorts images using AI-based content detection — all while keeping original files untouched.

---

## Key Features

- Sharpness-based filtering using Laplacian edge detection  
- Burst sequence analysis with automatic best-shot selection  
- Metadata-based rating filtering  
- Optional AI-powered image detection and sorting  
- Fully offline processing  
- Non-destructive workflow (original files are never modified)

---

## Quick Start

1. Launch **FilterPix**
2. Click **Input** and select the folder containing your images
3. (Optional) Adjust filter and sorting settings
4. Click **Start** to begin processing
5. Review results in the output folders:
   - **Sharp** – images that passed filters  
   - **Sorted** – rejected images  

Original image files are never modified.

---

## How It Works

1. Select a folder containing images  
2. Configure filtering and sorting options  
3. Start processing  
4. FilterPix copies images into organized output folders:
   - **Sharp** — images that passed filters  
   - **Sorted** — rejected images  

The original image folder remains unchanged.

---

## Supported Formats

- JPG  
- JPEG  
- PNG  
- Other image formats supported by OpenCV and Pillow  

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

- Language: Python  
- GUI: Tkinter  
- Image Processing: OpenCV, Pillow  
- AI Detection: YOLO (Ultralytics)  
- Designed for offline execution  

---

## Disclaimer

FilterPix only copies files for sorting purposes. It does not delete, modify, or overwrite original images. Ensure sufficient disk space is available before processing large image folders.
"""

