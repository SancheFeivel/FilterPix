import os
import time
import shutil
import multiprocessing
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS
import gc


class AISorter:
    def __init__(self, input_folder, solo, model_path="yolov8m.pt", target_classes=None, conf=0.4, imgsz=320, subject_threshold=0.009):
        
        # Fix: Set input_folder first, then modify it if needed
        self.input_folder = input_folder
        if not solo:
            self.input_folder = os.path.join(self.input_folder, "Sharp")
            
        self.output_base = os.path.join(self.input_folder, "Sorted")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.subject_threshold = subject_threshold
        self.cancel_flag = multiprocessing.Manager().Value("b", False)
        self.progress_callback = None
        
        # Target classes for detection
        self.target_classes = target_classes or [
            0, 1, 2, 3, 5, 7,           # person, bicycle, car, motorcycle, bus, truck
            16, 17, 18, 19, 20, 21,     # bird, cat, dog, horse, sheep, cow
            32, 36, 37, 39, 41          # sports ball, bat, glove, skateboard, racket
        ]

        # Categories for sophisticated sorting
        self.categories = {
            "people":   [0],                     # person
            "vehicles": [1, 2, 3, 5, 7],         # bicycle, car, motorcycle, bus, truck
            "sports":   [32, 36, 37, 39, 41],    # sports ball, bat, glove, skateboard, racket
            "animals":  [16, 17, 18, 19, 20, 21] # bird, cat, dog, horse, sheep, cow
        }

        print("Using device:", self.model.device)

    def cancel(self):
        self.cancel_flag.value = True
        print("Cancellation requested...")

    def get_exif_data(self, img_path):
        """Extract EXIF data as a readable dictionary."""
        try:
            image = Image.open(img_path)
            exif_data_raw = image._getexif()
            exif_data = {}
            if exif_data_raw:
                for tag_id, value in exif_data_raw.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            return exif_data
        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            return {}

    #Internal Helpers
    def _create_category_folders(self):
        """Create folders for all possible categories."""
        os.makedirs(self.output_base, exist_ok=True)
        categories = [
            "portraits", "group_photo", "large_group", "crowd", 
            "wideshot", "landscape", "vehicles", "sports", "animals", "other"
        ]
        for category in categories:
            os.makedirs(os.path.join(self.output_base, category), exist_ok=True)

    def _process_single_image(self, image_path):
        """Process a single image using sophisticated categorization logic."""
        if self.cancel_flag.value:
            return False

        # Run YOLO detection
        results = self.model(
            image_path,
            classes=self.target_classes,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )

        # Get EXIF data
        exif_info = self.get_exif_data(image_path)
        
        # Initialize tracking variables
        class_counts = {self.model.names[c]: 0 for c in self.target_classes}
        person_areas = []
        person_positions = []
        category_area_sums = {cat: 0 for cat in self.categories}
        category_counts = {cat: 0 for cat in self.categories}
        category_largest_areas = {cat: 0 for cat in self.categories}

        # Process detection results
        for result in results:
            img_height, img_width = result.orig_img.shape[:2]

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in self.target_classes:
                    class_name = self.model.names[cls_id]
                    class_counts[class_name] += 1

                    # Area calculation
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    area = width * height
                    area_norm = area / (img_width * img_height)

                    if cls_id == 0:  # person
                        person_areas.append(area_norm)
                        center_x = (x_min + x_max) / 2 / img_width
                        center_y = (y_min + y_max) / 2 / img_height
                        person_positions.append((center_x, center_y))

                    # Categorize detection
                    for cat, class_ids in self.categories.items():
                        if cls_id in class_ids:
                            category_area_sums[cat] += area_norm
                            category_counts[cat] += 1
                            if area_norm > category_largest_areas[cat]:
                                category_largest_areas[cat] = area_norm
                            break

        # Calculate average areas
        category_avg_areas = {}
        for cat in self.categories:
            if category_counts[cat] > 0:
                category_avg_areas[cat] = category_area_sums[cat] / category_counts[cat]
            else:
                category_avg_areas[cat] = 0

        total_area = sum(category_area_sums.values())
        total_count = sum(category_counts.values())
        overall_avg_area = total_area / total_count if total_count > 0 else 0

        # Log analysis results
        self.analyze_avg_detection_areas(category_avg_areas, overall_avg_area)

        # Determine category using sophisticated logic
        category = self.grouping(
            class_counts,
            person_areas,
            overall_avg_area,
            category_largest_areas,
            category_counts,
            category_area_sums,
        )

        # Move file to appropriate folder
        dest_folder = os.path.join(self.output_base, category)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, os.path.basename(image_path))
        shutil.copyfile(image_path, dest_path)
        print(f"✔ Moved {os.path.basename(image_path)} to {category}")
        return True

    def analyze_avg_detection_areas(self, category_avg_areas, overall_avg_area):
        """Log average detection sizes for analysis."""
        print("\n=== Average Detection Areas ===")
        for cat, avg_area in category_avg_areas.items():
            print(f"{cat.capitalize():<10}: {avg_area:.4f} ({avg_area*100:.2f}% of frame)")
        print(f"Overall Avg : {overall_avg_area:.4f} ({overall_avg_area*100:.2f}% of frame)")
        print("=" * 35)

    def grouping(self, class_counts, person_areas,
                 overall_avg_area, category_largest_areas, category_counts, category_area_sums):
        """Sophisticated categorization logic from PhotoSorter."""
        total_person_count = class_counts.get("person", 0)

        # STRICTER wide shot rule - much smaller threshold to catch more wide shots
        max_detection_size = max(category_largest_areas.values()) if category_largest_areas else 0

        # Multiple criteria for wideshot:
        total_detections = sum(category_counts.values())

        if (overall_avg_area < 0.003 or  # Much stricter threshold
            (max_detection_size < 0.008 and total_detections >= 2) or  # Multiple tiny detections
            (overall_avg_area < 0.005 and total_detections >= 4)):  # Many small detections
            return "wideshot"

        # If people exist → run portrait/group logic (people can have nuanced categories)
        if total_person_count > 0:
            if not person_areas:
                return "landscape"

            person_areas_sorted = sorted(person_areas, reverse=True)
            largest = person_areas_sorted[0]
            second = person_areas_sorted[1] if len(person_areas_sorted) > 1 else 0

            # Check if there's a clear dominant person (major vs minor subject)
            ratio = largest / (second + 1e-6)
            if ratio > 2.0 and largest > 0.02:  # Dominant person threshold
                return "portraits"

            # Much stricter main subject definition - must be substantial size AND similar to largest
            min_main_subject_size = max(0.015, largest * 0.6)  # At least 1.5% of image OR 60% of largest person
            main_subject_areas = [a for a in person_areas if a >= min_main_subject_size]
            main_subject_count = len(main_subject_areas)

            if main_subject_count == 1:
                return "portraits"
            elif 2 <= main_subject_count <= 5:
                # Additional check: ensure it's actually a group, not one big + tiny people
                avg_main_size = sum(main_subject_areas) / main_subject_count
                if avg_main_size > 0.012:  # All main subjects must be reasonably sized
                    return "group_photo"
                else:
                    return "portraits"  # Likely one main person with small background people
            elif main_subject_count > 5:
                avg_main_size = sum(main_subject_areas) / main_subject_count
                return "large_group" if avg_main_size > 0.015 else "crowd"

        # For non-people categories: SINGLE CATEGORY ONLY (no multiple assignments)
        non_people_categories = {k: v for k, v in category_largest_areas.items() 
                               if k != "people" and category_counts[k] > 0}

        if non_people_categories:
            # Primary method: largest single detection
            dominant_by_size = max(non_people_categories.items(), key=lambda x: x[1])

            # Secondary method: total area coverage
            non_people_total_areas = {k: category_area_sums[k] for k, v in category_largest_areas.items() 
                                    if k != "people" and category_counts[k] > 0}
            dominant_by_total = max(non_people_total_areas.items(), key=lambda x: x[1]) if non_people_total_areas else (None, 0)

            # Choose the category - prefer largest single detection unless total area is significantly different
            category_by_size, max_size = dominant_by_size
            category_by_total, total_area = dominant_by_total if dominant_by_total[0] else (category_by_size, 0)

            # If the largest detection is substantial, use that category
            if max_size > 0.015:  # Significant detection threshold
                return category_by_size

            # Otherwise, use total area method but only if it's substantial
            elif total_area > 0.02:  # Multiple smaller detections threshold
                return category_by_total

        return "other"

    def analyze_person_sizes(self, person_areas):
        """Log person size distribution."""
        if not person_areas:
            print("No people detected")
            return
        person_areas_sorted = sorted(person_areas, reverse=True)
        print("\n=== Person Size Analysis ===")
        for i, area in enumerate(person_areas_sorted, 1):
            print(f"Person {i}: {area:.3f} ({area*100:.1f}%)")
        if len(person_areas) > 1:
            largest, second = person_areas_sorted[:2]
            ratio = largest / second if second > 0 else float('inf')
            print(f"Largest to 2nd largest ratio: {ratio:.1f}x")
        print("=" * 30)

    def is_clustered(self, positions, threshold=0.2):
        """Check if people are spatially clustered."""
        if len(positions) <= 2:
            return True
        total_distance = 0
        pair_count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                total_distance += dist
                pair_count += 1
        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        return avg_distance < threshold

    #Main Logic
    def process_images_singlethreaded(self, progress_callback=None):
        """Process all images in the input folder using sophisticated categorization."""
        self.progress_callback = progress_callback
        start_time = time.time()
        self._create_category_folders()

        image_paths = [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_paths:
            print("No images found.")
            return

        print(f"Processing {len(image_paths)} images with sophisticated YOLO categorization...")

        # Track statistics
        category_stats = {}
        processed_count = 0
        
        for path in image_paths:
            if self.cancel_flag.value:
                print("Processing cancelled by user.")
                break
            
            success = self._process_single_image(path)
            if success:
                processed_count += 1
                if self.progress_callback:
                    self.progress_callback(processed_count, len(image_paths))

        gc.collect()
        
        if not self.cancel_flag.value:
            # Print final statistics
            print("\n=== Final Sorting Statistics ===")
            for category_folder in os.listdir(self.output_base):
                category_path = os.path.join(self.output_base, category_folder)
                if os.path.isdir(category_path):
                    count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    if count > 0:
                        print(f"{category_folder}: {count} images")
            print("=" * 35)
            print(f"\nFinished in {time.time() - start_time:.2f} seconds")
        
        return processed_count

    def process_folder(self, folder_path, output_dir):
        """Legacy method for backward compatibility."""
        # Update paths for compatibility
        self.input_folder = folder_path
        self.output_base = output_dir
        return self.process_images_singlethreaded()


#Entry Point
def main(folder, mode="fast", solo_process=None, cancel_flag=None, progress_callback=None):
    if mode == "fast":
        config = {
            "model_path": "yolov8s.pt",
            "conf": 0.6,
            "imgsz": 320
        }
    elif mode == "accurate":
        config = {
            "model_path": "yolov8m.pt",
            "conf": 0.4,
            "imgsz": 640
        }
    else:
        raise ValueError("Mode must be either 'fast' or 'accurate'")
    
    # Default target classes for comprehensive detection
    target_classes = [
        0, 1, 2, 3, 5, 7,           # person, bicycle, car, motorcycle, bus, truck
        16, 17, 18, 19, 20, 21,     # bird, cat, dog, horse, sheep, cow
        32, 36, 37, 39, 41          # sports ball, bat, glove, skateboard, racket
    ]

    sorter = AISorter(
        input_folder=folder,
        model_path=config["model_path"],
        solo=solo_process,
        target_classes=target_classes,
        conf=config["conf"],
        imgsz=config["imgsz"]
    )
    
    if cancel_flag:
        sorter.cancel_flag = cancel_flag
        
    return sorter.process_images_singlethreaded(progress_callback=progress_callback)

def main(folder, mode="fast", solo_process=None, cancel_flag=None, progress_callback=None):
    if mode == "fast":
        config = {
            "model_path": "yolov8n.pt",
            "conf": 0.6,
            "imgsz": 320
        }
    elif mode == "accurate":
        config = {
            "model_path": "yolov8m.pt",
            "conf": 0.4,
            "imgsz": 640
        }
    else:
        raise ValueError("Mode must be either 'fast' or 'accurate'")
    
    

    sorter = AISorter(
        input_folder=folder,
        model_path=config["model_path"],
        solo = solo_process,
        conf=config["conf"],
        imgsz=config["imgsz"]
    )
    
    if cancel_flag:
        sorter.cancel_flag = cancel_flag
        
    return sorter.process_images_singlethreaded(progress_callback=progress_callback)


