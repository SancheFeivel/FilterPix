import os
import time
import shutil
import multiprocessing
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS

class AISorter:
    def __init__(self, input_folder, solo, model_path="yolov8m.pt", target_classes=None, conf=0.4, imgsz=320, subject_threshold=0.009, output_dir=None):
        # Normalize paths
        input_folder = os.path.normpath(input_folder)
        if output_dir:
            output_dir = os.path.normpath(output_dir)
        
        self.original_input_folder = input_folder
        
        if output_dir:
            self.input_folder = os.path.join(output_dir, "Sharp")
        elif solo:
            self.input_folder = input_folder
        else:
            self.input_folder = os.path.join(input_folder, "Sharp")
        
        self.output_base = os.path.join(self.input_folder, "Sorted")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.subject_threshold = subject_threshold
        self.cancel_flag = None
        self.progress_callback = None
        self.start_time = None
        self.total_time = None

        # Cache created folders to avoid repeated makedirs syscalls
        self._folder_cache = set()
        
        os.makedirs(self.output_base, exist_ok=True)
        self._folder_cache.add(self.output_base)
        
        self.target_classes = target_classes or [
            0, 1, 2, 3, 5, 7,
            16, 17, 18, 19, 20, 21,
            32, 36, 39, 41
        ]

        self.categories = {
            "people":   [0],
            "vehicles": [1, 2, 3, 5, 7],
            "sports":   [36, 39, 41],
            "animals":  [16, 17, 18, 19, 20, 21]
        }

        self.output_categories = [
            "solo", "group_photo", "large_group", "close-up",
            "wideshot", "vehicles", "sports", "animals", "other",
            "portrait", "landscape"
        ]

        self.supported_extensions = ('.jpg', '.jpeg', '.png')

        print("Using device:", self.model.device)

    def cancel(self):
        if self.cancel_flag:
            self.cancel_flag.set()
        print("Cancellation requested...")

    def _makedirs_cached(self, path):
        """Create directory only if not already known to exist."""
        if path not in self._folder_cache:
            os.makedirs(path, exist_ok=True)
            self._folder_cache.add(path)

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

    def get_image_orientation(self, img_path):
        """Determine if image is portrait or landscape (single open, no EXIF redundancy)."""
        try:
            with Image.open(img_path) as image:
                width, height = image.size
                exif_data = image._getexif()
                if exif_data and 274 in exif_data:
                    if exif_data[274] in (5, 6, 7, 8):
                        width, height = height, width
                return "landscape" if width > height else "portrait" if height > width else "square"
        except Exception:
            return "unknown"

    def _process_batch(self, batch_paths):
        """
        Run YOLO on a batch of images in one forward pass, then copy each
        image to its destination.  Returns the number of successfully handled
        files.
        """
        if self.cancel_flag and self.cancel_flag.is_set():
            return 0

        try:
            # Single batched inference call — much faster than N individual calls
            batch_results = self.model(
                batch_paths,
                classes=self.target_classes,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            print(f"Error running batch inference: {e}")
            return 0

        processed = 0
        for image_path, result in zip(batch_paths, batch_results):
            try:
                orientation = self.get_image_orientation(image_path)

                class_counts = {self.model.names[c]: 0 for c in self.target_classes}
                person_areas = []
                category_area_sums = {cat: 0 for cat in self.categories}
                category_counts = {cat: 0 for cat in self.categories}
                category_largest_areas = {cat: 0 for cat in self.categories}

                img_height, img_width = result.orig_img.shape[:2]
                img_area = img_width * img_height

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.target_classes:
                        continue

                    class_name = self.model.names[cls_id]
                    class_counts[class_name] += 1

                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    area_norm = ((x_max - x_min) * (y_max - y_min)) / img_area

                    if cls_id == 0:
                        person_areas.append(area_norm)

                    for cat, class_ids in self.categories.items():
                        if cls_id in class_ids:
                            category_area_sums[cat] += area_norm
                            category_counts[cat] += 1
                            if area_norm > category_largest_areas[cat]:
                                category_largest_areas[cat] = area_norm
                            break

                total_count = sum(category_counts.values())
                total_area = sum(category_area_sums.values())
                overall_avg_area = total_area / total_count if total_count > 0 else 0

                category = self.grouping(
                    class_counts,
                    person_areas,
                    overall_avg_area,
                    category_largest_areas,
                    category_counts,
                    category_area_sums,
                )

                dest_folder = os.path.join(self.output_base, category, orientation)
                self._makedirs_cached(dest_folder)

                dest_path = os.path.join(dest_folder, os.path.basename(image_path))
                shutil.copyfile(image_path, dest_path)
                processed += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return processed

    def grouping(
        self,
        class_counts,
        person_areas,
        overall_avg_area,
        category_largest_areas,
        category_counts,
        category_area_sums,
    ):
        total_person_count = class_counts.get("person", 0)

        if total_person_count > 0:
            person_areas_sorted = sorted(person_areas, reverse=True)
            largest_person = person_areas_sorted[0]

            # Close-up: one dominant subject filling the frame
            if largest_person > 0.40:
                if total_person_count == 1:
                    return "close-up"
                second_largest = person_areas_sorted[1] if total_person_count > 1 else 0
                if largest_person > second_largest * 4:
                    return "close-up"

            # Use 50% of largest as threshold — catches equal/similar sized pairs
            significant_threshold = max(0.005, largest_person * 0.50)
            significant_people = [a for a in person_areas if a >= significant_threshold]
            num_significant = len(significant_people)

            # 2+ significant people → always a group shot, no fallthrough
            if num_significant >= 2:
                avg_person_size = sum(significant_people) / num_significant
                if num_significant > 15 or avg_person_size < 0.01:
                    return "large_group"
                return "group_photo"

            # One significant person — only allow solo if any extras are truly negligible
            if total_person_count >= 2:
                background_people = person_areas_sorted[1:]
                if (
                    largest_person >= 0.15
                    and len(background_people) <= 1
                    and all(area < largest_person * 0.10 for area in background_people)
                    and all(area < 0.005 for area in background_people)
                ):
                    return "solo"
                return "group_photo"

            return "solo"

        # No people — check other categories
        non_people_categories = {
            cat: area
            for cat, area in category_largest_areas.items()
            if cat != "people" and category_counts.get(cat, 0) > 0
        }

        if non_people_categories:
            category_name, max_size = max(non_people_categories.items(), key=lambda x: x[1])
            total_coverage = category_area_sums[category_name]
            object_count = category_counts[category_name]

            if max_size > 0.025:
                return category_name
            if total_coverage > 0.05 and object_count >= 2:
                return category_name
            if max_size > 0.015 and total_coverage > 0.03:
                return category_name

        return "other"

    def process_images_singlethreaded(self, progress_callback=None, batch_size=8):
        """
        Process images in batches.  batch_size=8 is a good default; increase
        to 16 if VRAM allows, decrease to 4 if you hit OOM errors.
        """
        self.progress_callback = progress_callback
        self.start_time = time.time()

        image_paths = self._get_image_paths()
        total_images = len(image_paths)

        if not image_paths:
            return {"total_images": 0, "final_selection": 0, "elapsed_time": 0}

        processed_count = 0

        for i in range(0, total_images, batch_size):
            if self.cancel_flag and self.cancel_flag.is_set():
                break

            batch = image_paths[i:i + batch_size]
            processed_count += self._process_batch(batch)

            if self.progress_callback:
                self.progress_callback(
                    min(i + batch_size, total_images),
                    total_images,
                    "sorting"
                )

        self.total_time = time.time() - self.start_time

        return {
            "total_images": total_images,
            "final_selection": processed_count,
            "elapsed_time": self.total_time
        }

def main(folder, output=None, mode="fast", solo_process=None, cancel_flag=None, progress_callback=None):
    if mode == "fast":
        config = {"model_path": "yolov8n.pt", "conf": 0.4, "imgsz": 416}
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