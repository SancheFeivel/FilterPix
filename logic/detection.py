import os
import time
import shutil
import multiprocessing
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS
import gc


class AISorter:
    def __init__(self, input_folder, solo, model_path="yolov8m.pt", target_classes=None, conf=0.4, imgsz=320, subject_threshold=0.009, output_dir=None):
        # Set input_folder first, then modify it if needed
        self.input_folder = input_folder
        if not solo:
            self.input_folder = os.path.join(self.input_folder, "Sharp")
            
        self.output_base = os.path.join(output_dir, "Sorted") if output_dir else os.path.join(self.input_folder, "Sorted")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.subject_threshold = subject_threshold
        self.cancel_flag = None
        self.progress_callback = None
        self.start_time = None
        self.total_time = None
        
        os.makedirs(self.output_base, exist_ok=True)
        
        # Target classes for detection
        self.target_classes = target_classes or [
            0, 1, 2, 3, 5, 7,           # person, bicycle, car, motorcycle, bus, truck
            16, 17, 18, 19, 20, 21,     # bird, cat, dog, horse, sheep, cow
            32, 36, 37, 39, 41          # sports ball, bat, glove, skateboard, racket
        ]

        # Categories for sorting
        self.categories = {
            "people":   [0],                     # person
            "vehicles": [1, 2, 3, 5, 7],         # bicycle, car, motorcycle, bus, truck
            "sports":   [32, 36, 37, 39, 41],    # sports ball, bat, glove, skateboard, racket
            "animals":  [16, 17, 18, 19, 20, 21] # bird, cat, dog, horse, sheep, cow
        }

        # Pre-compute category names for output folders
        self.output_categories = [
            "portraits", "group_photo", "large_group", "crowd", 
            "wideshot", "landscape", "vehicles", "sports", "animals", "other"
        ]

        # Supported image extensions (compiled once)
        self.supported_extensions = ('.jpg', '.jpeg', '.png')

        print("Using device:", self.model.device)

    def cancel(self):
        if self.cancel_flag:
            self.cancel_flag.set()
        print("Cancellation requested...")

    def get_exif_data(self, img_path):
        """Extract EXIF data as a readable dictionary."""
        try:
            with Image.open(img_path) as image:
                exif_data_raw = image._getexif()
                if not exif_data_raw:
                    return {}
                
                return {TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data_raw.items()}
        except Exception as e:
            print(f"Error reading EXIF data from {img_path}: {e}")
            return {}

    def _get_image_paths(self):
        """Get list of all supported image files in the input folder."""
        try:
            return [
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.lower().endswith(self.supported_extensions)
            ]
        except OSError as e:
            print(f"Error reading input folder {self.input_folder}: {e}")
            return []

    def _process_single_image(self, image_path):
        """Process single image using categorization logic."""
        if self.cancel_flag and self.cancel_flag.is_set():
            return False

        try:
            # Run YOLO detection
            results = self.model(
                image_path,
                classes=self.target_classes,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False
            )

            # Get EXIF data (currently not used in categorization, but preserved for future use)
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
                img_area = img_width * img_height

                for box in result.boxes:
                    cls_id = int(box.cls[0])

                    if cls_id in self.target_classes:
                        class_name = self.model.names[cls_id]
                        class_counts[class_name] += 1

                        # Area calculation
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        width = x_max - x_min
                        height = y_max - y_min
                        area = width * height
                        area_norm = area / img_area

                        if cls_id == 0:  # person
                            person_areas.append(area_norm)
                            center_x = (x_min + x_max) / (2 * img_width)
                            center_y = (y_min + y_max) / (2 * img_height)
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
            category_avg_areas = {
                cat: (category_area_sums[cat] / category_counts[cat] if category_counts[cat] > 0 else 0)
                for cat in self.categories
            }

            total_area = sum(category_area_sums.values())
            total_count = sum(category_counts.values())
            overall_avg_area = total_area / total_count if total_count > 0 else 0

            # Log analysis results
            # self.analyze_avg_detection_areas(category_avg_areas, overall_avg_area)

            # Determine category using logic
            category = self.grouping(
                class_counts,
                person_areas,
                overall_avg_area,
                category_largest_areas,
                category_counts,
                category_area_sums,
            )

            # Only create folder if we're actually moving something there
            dest_folder = os.path.join(self.output_base, category)
            os.makedirs(dest_folder, exist_ok=True)

            dest_path = os.path.join(dest_folder, os.path.basename(image_path))
            shutil.copyfile(image_path, dest_path)
            # print(f"Moved {os.path.basename(image_path)} to {category}")
            return True

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False

    # def analyze_avg_detection_areas(self, category_avg_areas, overall_avg_area):
    #     """Log average detection sizes for analysis."""
    #     print("\nAverage Detection Areas")
    #     for cat, avg_area in category_avg_areas.items():
    #         print(f"{cat.capitalize():<10}: {avg_area:.4f} ({avg_area*100:.2f}% of frame)")
    #     print(f"Overall Avg : {overall_avg_area:.4f} ({overall_avg_area*100:.2f}% of frame)")
    #     print("=" * 30)

    def grouping(self, class_counts, person_areas,
                 overall_avg_area, category_largest_areas, category_counts, category_area_sums):
        """Categorization logic from PhotoSorter."""
        total_person_count = class_counts.get("person", 0)

        # STRICTER wide shot rule
        max_detection_size = max(category_largest_areas.values()) if category_largest_areas else 0
        total_detections = sum(category_counts.values())

        if (overall_avg_area < 0.003 or  
            (max_detection_size < 0.008 and total_detections >= 2) or  
            (overall_avg_area < 0.005 and total_detections >= 4)):  
            return "wideshot"

        if total_person_count > 0:
            if not person_areas:
                return "landscape"

            person_areas_sorted = sorted(person_areas, reverse=True)
            largest = person_areas_sorted[0]
            second = person_areas_sorted[1] if len(person_areas_sorted) > 1 else 0

            ratio = largest / (second + 1e-6)
            if ratio > 2.0 and largest > 0.02:
                return "portraits"

            min_main_subject_size = max(0.015, largest * 0.6)
            main_subject_areas = [a for a in person_areas if a >= min_main_subject_size]
            main_subject_count = len(main_subject_areas)

            if main_subject_count == 1:
                return "portraits"
            elif 2 <= main_subject_count <= 5:
                avg_main_size = sum(main_subject_areas) / main_subject_count
                if avg_main_size > 0.012:
                    return "group_photo"
                else:
                    return "portraits"
            elif main_subject_count > 5:
                avg_main_size = sum(main_subject_areas) / main_subject_count
                return "large_group" if avg_main_size > 0.015 else "crowd"

        non_people_categories = {k: v for k, v in category_largest_areas.items() 
                               if k != "people" and category_counts[k] > 0}

        if non_people_categories:
            dominant_by_size = max(non_people_categories.items(), key=lambda x: x[1])
            non_people_total_areas = {k: category_area_sums[k] for k, v in category_largest_areas.items() 
                                    if k != "people" and category_counts[k] > 0}
            dominant_by_total = max(non_people_total_areas.items(), key=lambda x: x[1]) if non_people_total_areas else (None, 0)

            category_by_size, max_size = dominant_by_size
            category_by_total, total_area = dominant_by_total if dominant_by_total[0] else (category_by_size, 0)

            if max_size > 0.015:
                return category_by_size
            elif total_area > 0.02:
                return category_by_total

        return "other"

    def process_images_singlethreaded(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.start_time = time.time()

        image_paths = self._get_image_paths()
        total_images = len(image_paths)

        if not image_paths:
            return {
                "total_images": 0,
                "final_selection": 0,
                "elapsed_time": 0
            }

        processed_count = 0

        for path in image_paths:
            if self.cancel_flag and self.cancel_flag.is_set():
                break

            if self._process_single_image(path):
                processed_count += 1
                if self.progress_callback:
                    self.progress_callback(processed_count, total_images)

        self.total_time = time.time() - self.start_time

        return {
            "total_images": total_images,
            "final_selection": processed_count,
            "elapsed_time": self.total_time
        }


    def _print_final_statistics(self):
        """Print final sorting statistics."""
        print("\n=== Final Sorting Statistics ===")
        try:
            if os.path.exists(self.output_base):
                for category_folder in os.listdir(self.output_base):
                    category_path = os.path.join(self.output_base, category_folder)
                    if os.path.isdir(category_path):
                        count = len([f for f in os.listdir(category_path) 
                                   if f.lower().endswith(self.supported_extensions)])
                        if count > 0:
                            print(f"{category_folder}: {count} images")
        except OSError as e:
            print(f"Error reading output directory: {e}")
        print("=" * 30)

    def process_folder(self, folder_path, output_dir):
        """Legacy method for backward compatibility."""
        self.input_folder = folder_path
        self.output_base = output_dir
        return self.process_images_singlethreaded()


def main(folder, output=None, mode="fast", solo_process=None, cancel_flag=None, progress_callback=None):
    if mode == "fast":
        config = {"model_path": "yolov8n.pt", "conf": 0.6, "imgsz": 320}
    elif mode == "accurate":
        config = {"model_path": "yolov8m.pt", "conf": 0.4, "imgsz": 640}
    else:
        raise ValueError("Mode must be either 'fast' or 'accurate'")

    sorter = AISorter(
        input_folder=folder,
        model_path=config["model_path"],
        solo=solo_process,
        conf=config["conf"],
        imgsz=config["imgsz"],
        output_dir=output
    )

    if cancel_flag:
        sorter.cancel_flag = cancel_flag

    return sorter.process_images_singlethreaded(progress_callback)