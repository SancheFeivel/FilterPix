import os
import cv2
import time
import shutil
import multiprocessing
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict

class EXIFHelper:
    @staticmethod
    def get_exif_value(path, key, default=None):
        try:
            with Image.open(path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return default
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == key:
                        return value
        except Exception as e:
            print(f"Error reading {key} from {path}: {e}")
        return default

    @staticmethod
    def get_fstop(path):
        value = EXIFHelper.get_exif_value(path, 'FNumber', 8.0)
        return value[0] / value[1] if isinstance(value, tuple) else value

    @staticmethod
    def get_shutter_speed(path):
        return EXIFHelper.get_exif_value(path, 'ExposureTime', None)

    @staticmethod
    def get_iso(path):
        return EXIFHelper.get_exif_value(path, 'ISOSpeedRatings', 100)

    @staticmethod
    def get_rating(path):
        return EXIFHelper.get_exif_value(path, 'Rating', "0")

    @staticmethod
    def get_datetime_original(path):
        return EXIFHelper.get_exif_value(path, 'DateTimeOriginal', None)

    @staticmethod
    def get_subsec_time(path):
        return EXIFHelper.get_exif_value(path, 'SubSecTimeOriginal', '00')


class ImageAnalyzer:
    @staticmethod
    def crop_center(image, fraction=0.75):
        h, w = image.shape[:2]
        ch, cw = int(h * fraction), int(w * fraction)
        y, x = (h - ch) // 2, (w - cw) // 2
        return image[y:y+ch, x:x+cw]

    @staticmethod
    def resize_short_side(image, target_short_side=683):
        h, w = image.shape[:2]

        # Determine scale factor based on shorter side
        scale = target_short_side / min(h, w)

        # Compute new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def resize_then_crop(image, target_short_side=683, fraction=0.75):
        resized = ImageAnalyzer.resize_short_side(image, target_short_side)
        cropped = ImageAnalyzer.crop_center(resized, fraction)
        return cropped



    @staticmethod
    def is_sharp(image, path, base_blur, tolerance):
        cropped = ImageAnalyzer.crop_center(image)
        laplacian = cv2.Laplacian(cropped, cv2.CV_64F).var()

        fstop = EXIFHelper.get_fstop(path)
        iso = EXIFHelper.get_iso(path)
        shutter = EXIFHelper.get_shutter_speed(path)

        if fstop < 4 and iso < 2000:
            threshold = 30
        elif iso > 5000:
            threshold = 410
        elif iso > 2000 or (shutter and shutter <= 0.05):
            threshold = 200
        else:
            threshold = 75

        threshold += base_blur + tolerance
        
        # Debug output for first few images
        filename = os.path.basename(path)
        if filename.endswith('.jpg'):  # Only print for first few to avoid spam
            print(f"DEBUG {filename}: laplacian={laplacian:.1f}, threshold={threshold:.1f}, fstop={fstop}, iso={iso}, sharp={laplacian > threshold}")
        
        return laplacian > threshold, laplacian


def compute_laplacian_variance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    cropped = ImageAnalyzer.resize_then_crop(image, target_short_side=683, fraction=0.75)
    return cv2.Laplacian(cropped, cv2.CV_64F).var()


def process_image_sharpness(folder, filename, base_blur, tolerance):
    """Process single image for sharpness check"""
    if not filename.lower().endswith(".jpg"):
        return None

    path = os.path.join(folder, filename)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    try:
        if image is None:
            print(f"Failed to read {filename}")
            return None

        is_sharp, laplacian = ImageAnalyzer.is_sharp(image, path, base_blur, tolerance)
        return filename, is_sharp, laplacian

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    finally:
        if 'image' in locals():
            del image
        cv2.destroyAllWindows()


class ImageSharpnessProcessor:
    def __init__(self, folder, base_blur=0, tolerance=0, burst_count=2):
        self.folder = folder
        self.base_blur = base_blur
        self.tolerance = tolerance
        self.burst_count = burst_count  # Configurable burst selection count
        self.cancel_flag = multiprocessing.Manager().Value("b", False)
        self.progress_callback = None

    def cancel(self):
        self.cancel_flag.value = True
        print("Cancellation requested...")

    def stage1_star_check(self, all_images):
        """Stage 1: Filter out rated images"""
        print("STAGE 1: Star Check")
        unrated_images = []
        rated_count = 0
        
        for filename in all_images:
            if self.cancel_flag.value:
                print("Cancelled during star check.")
                return []
                
            path = os.path.join(self.folder, filename)
            rating = EXIFHelper.get_rating(path)
            
            if rating == "0":  # Unrated images only
                unrated_images.append(filename)
            else:
                rated_count += 1
                print(f"Filtered out rated image: {filename} (rating: {rating})")
        
        print(f"Star check complete: {len(unrated_images)} unrated, {rated_count} rated (filtered)")
        return unrated_images

    def stage2_sharpness_check(self, unrated_images):
        """Stage 2: Filter out blurry images using multiprocessing"""
        print("\nSTAGE 2: Laplacian Sharpness Check")
        
        if not unrated_images:
            print("No unrated images to process.")
            return []

        args = [
            (self.folder, filename, self.base_blur, self.tolerance)
            for filename in unrated_images
        ]
        pool_size = max(1, multiprocessing.cpu_count() - 2)

        sharp_images = []
        
        with multiprocessing.Pool(pool_size) as pool:
            result_async = pool.starmap_async(process_image_sharpness, args)

            while not result_async.ready():
                if self.cancel_flag.value:
                    pool.terminate()
                    pool.join()
                    print("Cancelled during sharpness check.")
                    return []
                if self.progress_callback:
                    self.progress_callback("Processing sharpness...")
                time.sleep(0.1)

            try:
                results = result_async.get()
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                return []

        # Process results
        sharp_count = 0
        blurry_count = 0
        
        for result in results:
            if result is None:
                continue
            filename, is_sharp, laplacian = result
            if is_sharp:
                sharp_images.append(filename)
                sharp_count += 1
            else:
                blurry_count += 1

        print(f"Sharpness check complete: {sharp_count} sharp, {blurry_count} blurry (filtered)")
        return sharp_images

    def stage3_burst_grouping(self, sharp_images):
        """Stage 3: Group bursts and select best from each group"""
        print("\nSTAGE 3: Burst Grouping")
        
        if not sharp_images:
            print("No sharp images to process.")
            return []

        # Create path list for sharp images only
        sharp_paths = [os.path.join(self.folder, filename) for filename in sharp_images]
        
        # Group by DateTime
        burst_groups = defaultdict(list)
        non_burst_images = []
        
        for path in sharp_paths:
            if self.cancel_flag.value:
                print("Cancelled during burst grouping.")
                return []
                
            dt = EXIFHelper.get_datetime_original(path)
            if dt:
                burst_groups[dt].append(path)
            else:
                non_burst_images.append(path)

        # Filter to only actual burst groups (more than 1 image)
        actual_bursts = {k: v for k, v in burst_groups.items() if len(v) > 1}
        
        # Add non-burst images from single-image "groups"
        for k, v in burst_groups.items():
            if len(v) == 1:
                non_burst_images.extend(v)

        final_selection = []
        
        # Process burst groups
        burst_groups_processed = 0
        images_from_bursts = 0
        
        for dt, group in actual_bursts.items():
            if self.cancel_flag.value:
                print("Cancelled during burst processing.")
                return []
                
            # Score by laplacian variance
            scored = [(compute_laplacian_variance(path), path) for path in group]
            scored.sort(reverse=True)  # Highest laplacian first
            
            # Select top N from burst
            selected_from_burst = scored[:self.burst_count]
            for score, path in selected_from_burst:
                final_selection.append(path)
                images_from_bursts += 1
                print(f"Selected from burst {dt}: {os.path.basename(path)} (score: {score:.1f})")
            
            burst_groups_processed += 1

        # Add all non-burst images
        final_selection.extend(non_burst_images)
        
        print(f"Burst grouping complete:")
        print(f"  - Burst groups found: {len(actual_bursts)}")
        print(f"  - Images selected from bursts: {images_from_bursts}")
        print(f"  - Non-burst images: {len(non_burst_images)}")
        print(f"  - Total final selection: {len(final_selection)}")
        
        return final_selection

    def copy_final_images(self, final_paths, output_folder):
        """Copy final selected images to output folder"""
        print(f"\nCOPYING {len(final_paths)} IMAGES TO OUTPUT")
        
        os.makedirs(output_folder, exist_ok=True)
        copied_count = 0
        
        for path in final_paths:
            if self.cancel_flag.value:
                print("Cancelled during copying.")
                return copied_count
                
            try:
                filename = os.path.basename(path)
                dest_path = os.path.join(output_folder, filename)
                shutil.copy(path, dest_path)
                copied_count += 1
                print(f"Copied: {filename}")
            except Exception as e:
                print(f"Error copying {path}: {e}")
        
        return copied_count

    def run(self, use_starcheck=True, use_laplaciancheck=True, group_bursts=True, progress_callback=None):
        """Run the three-stage filtering process"""
        self.progress_callback = progress_callback
        output_folder = os.path.join(self.folder, "Sharp")

        if self.cancel_flag.value:
            print("Cancelled before any processing.")
            return

        # Get all JPG files
        all_images = [f for f in os.listdir(self.folder) if f.lower().endswith(".jpg")]
        print(f"Found {len(all_images)} JPG files to process")
        
        if not all_images:
            print("No JPG files found.")
            return

        # Stage 1: Star Check (if enabled)
        if use_starcheck:
            remaining_images = self.stage1_star_check(all_images)
        else:
            print("STAGE 1: Star Check (SKIPPED)")
            remaining_images = all_images

        if self.cancel_flag.value or not remaining_images:
            return

        # Stage 2: Sharpness Check (if enabled)
        if use_laplaciancheck:
            sharp_images = self.stage2_sharpness_check(remaining_images)
        else:
            print("\nSTAGE 2: Sharpness Check (SKIPPED)")
            sharp_images = remaining_images

        if self.cancel_flag.value or not sharp_images:
            return

        # Stage 3: Burst Grouping (if enabled)
        if group_bursts:
            final_paths = self.stage3_burst_grouping(sharp_images)
        else:
            print("\nSTAGE 3: Burst Grouping (SKIPPED)")
            final_paths = [os.path.join(self.folder, f) for f in sharp_images]

        if self.cancel_flag.value or not final_paths:
            return

        # Copy final selection
        copied = self.copy_final_images(final_paths, output_folder)
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Started with: {len(all_images)} images")
        print(f"Final selection: {len(final_paths)} images")
        print(f"Successfully copied: {copied} images")
        print(f"Output folder: {output_folder}")


def main(folder, base_blur=0, tolerance=0, burst_count=2,
         use_starcheck=False, use_laplaciancheck=True, group_bursts=True,
         cancel_flag=None, progress_callback=None):

    processor = ImageSharpnessProcessor(folder, base_blur, tolerance, burst_count)

    if cancel_flag:
        processor.cancel_flag = cancel_flag

    processor.run(
        use_starcheck=use_starcheck,
        use_laplaciancheck=use_laplaciancheck,
        group_bursts=group_bursts,
        progress_callback=progress_callback
    )