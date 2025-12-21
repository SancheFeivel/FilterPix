import os
import cv2
import time
import shutil
import multiprocessing
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
import numpy as np

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
        rating = EXIFHelper.get_exif_value(path, 'Rating', None)
        if rating is None:
            return None
        try:
            return int(rating)
        except (ValueError, TypeError):
            return None

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
        scale = target_short_side / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def resize_then_crop(image, target_short_side=683, fraction=0.75):
        resized = ImageAnalyzer.resize_short_side(image, target_short_side)
        cropped = ImageAnalyzer.crop_center(resized, fraction)
        return cropped

    @staticmethod
    def detect_sharp_regions(image):
        """
        Divide image into grid and analyze sharpness distribution.
        Returns multiple metrics to better discriminate sharp vs blurry images.
        """
        h, w = image.shape[:2]
        grid_size = 4  # 4x4 grid = 16 regions
        cell_h, cell_w = h // grid_size, w // grid_size
        
        sharpness_values = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                region = image[y1:y2, x1:x2]
                
                # Calculate Laplacian variance for this region
                laplacian = cv2.Laplacian(region, cv2.CV_64F).var()
                sharpness_values.append(laplacian)
        
        sharpness_values.sort(reverse=True)
        
        # Multiple metrics for better detection
        max_sharpness = sharpness_values[0]
        top_3_avg = np.mean(sharpness_values[:3])  # Top 3 sharpest regions
        top_quarter = sharpness_values[:len(sharpness_values)//4]
        avg_top_quarter = np.mean(top_quarter) if top_quarter else 0
        median_sharpness = np.median(sharpness_values)
        
        return {
            'max': max_sharpness,
            'top3_avg': top_3_avg,
            'top_quarter_avg': avg_top_quarter,
            'median': median_sharpness,
            'all_values': sharpness_values
        }

    @staticmethod
    def is_sharp(image, path, base_blur, tolerance, exif_data=None):
        """
        Enhanced sharpness detection with stricter criteria.
        Requires BOTH sharp regions AND acceptable overall sharpness.
        """
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        cropped = ImageAnalyzer.crop_center(image)
        
        # Get regional analysis
        regional_metrics = ImageAnalyzer.detect_sharp_regions(cropped)
        
        # Calculate traditional full-image Laplacian
        full_laplacian = cv2.Laplacian(cropped, cv2.CV_64F).var()

        if exif_data is None:
            # Get EXIF data if not provided
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
                except Exception:
                    pass
                return default
            
            fstop_value = get_exif_value(path, 'FNumber', 8.0)
            fstop = fstop_value[0] / fstop_value[1] if isinstance(fstop_value, tuple) else fstop_value
            iso = get_exif_value(path, 'ISOSpeedRatings', 100)
            shutter = get_exif_value(path, 'ExposureTime', None)
        else:
            fstop = exif_data['fstop']
            iso = exif_data['iso']
            shutter = exif_data['shutter']

        # STRICTER thresholds based on shooting conditions
        if fstop < 4 and iso < 2000:
            # Low light, wide aperture - expect sharp subject
            regional_threshold = 80
            global_threshold = 40
        elif iso > 5000:
            # Very high ISO - more lenient but still require minimum
            regional_threshold = 500
            global_threshold = 200
        elif iso > 2000 or (shutter and shutter <= 0.05):
            # High ISO or fast shutter
            regional_threshold = 250
            global_threshold = 100
        else:
            # Normal conditions - expect good sharpness
            regional_threshold = 120
            global_threshold = 60

        # Apply user adjustments
        regional_threshold += base_blur + tolerance
        global_threshold += base_blur + tolerance
        
        # CRITICAL: Use weighted combination of top regions
        # This gives credit for sharp subjects while requiring overall quality
        regional_score = regional_metrics['top3_avg'] * 0.7 + regional_metrics['max'] * 0.3
        
        # STRICTER logic: Must pass BOTH tests
        # 1. At least some sharp regions (subject in focus)
        # 2. Minimum overall sharpness (not completely blurry background)
        passes_regional = regional_score > regional_threshold
        passes_global = full_laplacian > global_threshold
        
        # For very sharp images, allow slightly lower global threshold
        if regional_score > regional_threshold * 2:
            passes_global = full_laplacian > global_threshold * 0.7
        
        is_sharp_final = passes_regional and passes_global
        
        import os
        filename = os.path.basename(path)
        if filename.endswith('.jpg'):
            print(f"DEBUG {filename}: "
                  f"regional={regional_score:.1f} (threshold={regional_threshold:.1f}, pass={passes_regional}), "
                  f"global={full_laplacian:.1f} (threshold={global_threshold:.1f}, pass={passes_global}), "
                  f"fstop={fstop}, iso={iso}, "
                  f"FINAL={'SHARP' if is_sharp_final else 'BLURRY'}")
        
        # Return the regional score for burst comparison (higher is better)
        return is_sharp_final, regional_score


def process_image_sharpness(folder, filename, base_blur, tolerance, exif_cache):
    """Process single image for sharpness check with cached EXIF"""
    if not filename.lower().endswith(".jpg"):
        return None

    path = os.path.join(folder, filename)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    try:
        if image is None:
            print(f"Failed to read {filename}")
            return None

        # Use cached EXIF
        exif_data = exif_cache.get(path)
        is_sharp, laplacian = ImageAnalyzer.is_sharp(image, path, base_blur, tolerance, exif_data)
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
        self.burst_count = burst_count
        self.cancel_flag = None
        self.progress_callback = None

        # Caches
        self.exif_cache = {}       # Stores EXIF per image
        self.laplacian_map = {}    # Stores Laplacian per image
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'rated_images': 0,
            'sharp_images': 0,
            'final_selection': 0,
            'copied_images': 0,
            'start_time': None,
            'end_time': None,
            'elapsed_time': 0
        }

    def cache_exif(self, path):
        """Cache EXIF data for one image"""
        if path in self.exif_cache:
            return self.exif_cache[path]
        fstop = EXIFHelper.get_fstop(path)
        iso = EXIFHelper.get_iso(path)
        shutter = EXIFHelper.get_shutter_speed(path)
        rating = EXIFHelper.get_rating(path)
        dt = EXIFHelper.get_datetime_original(path)
        subsec = EXIFHelper.get_subsec_time(path)
        self.exif_cache[path] = {
            'fstop': fstop,
            'iso': iso,
            'shutter': shutter,
            'rating': rating,
            'datetime': dt,
            'subsec': subsec
        }
        return self.exif_cache[path]

    def cancel(self):
        if self.cancel_flag:
            self.cancel_flag.set()
        print("Cancellation requested...")

    def stage1_star_check(self, all_images):
        """Stage 1: Separate rated images but keep unrated ones for later filters."""
        print("STAGE 1: Star Check")
        rated, unrated = [], []

        for filename in all_images:
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during star check.")
                return []

            path = os.path.join(self.folder, filename)
            exif = self.cache_exif(path)
            rating = exif['rating']

            if rating is not None and rating > 0:
                rated.append(filename)
                print(f"Keeping rated image: {filename} (rating: {rating})")
            else:
                unrated.append(filename)
                print(f"Unrated image: {filename} (rating: {rating}) → will process in later stages")

        self.stats['rated_images'] = len(rated)

        # If no later filters are enabled, only return rated images
        if not self.use_laplaciancheck and not self.group_bursts:
            print("No later filters enabled → Only keeping rated images")
            return rated

        # If laplacian check is enabled, combine rated and unrated
        # If only burst grouping is enabled, also combine all images
        combined = rated + unrated
        print(f"Star check complete: {len(rated)} rated, {len(unrated)} unrated (kept for later stages)")
        return combined

    def stage2_sharpness_check(self, unrated_images):
        print("\nSTAGE 2: Laplacian Sharpness Check")
        if not unrated_images:
            print("No unrated images to process.")
            return []

        args = [
            (self.folder, filename, self.base_blur, self.tolerance, self.exif_cache)
            for filename in unrated_images
        ]
        pool_size = max(1, multiprocessing.cpu_count() - 2)
        sharp_images = []

        with multiprocessing.Pool(pool_size) as pool:
            result_async = pool.starmap_async(process_image_sharpness, args)

            while not result_async.ready():
                if self.cancel_flag and self.cancel_flag.is_set():
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

        sharp_count, blurry_count = 0, 0
        for result in results:
            if result is None:
                continue
            filename, is_sharp, laplacian = result
            path = os.path.join(self.folder, filename)
            self.laplacian_map[path] = laplacian  # cache Laplacian
            if is_sharp:
                sharp_images.append(filename)
                sharp_count += 1
            else:
                blurry_count += 1

        self.stats['sharp_images'] = sharp_count
        print(f"Sharpness check complete: {sharp_count} sharp, {blurry_count} blurry (filtered)")
        return sharp_images
    
    def calculate_laplacian_scores(self, images):
        """Calculate Laplacian scores for images without filtering (used for burst grouping)"""
        print("\nCALCULATING LAPLACIAN SCORES (for burst grouping)")
        
        args = [
            (self.folder, filename, self.base_blur, self.tolerance, self.exif_cache)
            for filename in images
        ]
        pool_size = max(1, multiprocessing.cpu_count() - 2)

        with multiprocessing.Pool(pool_size) as pool:
            result_async = pool.starmap_async(process_image_sharpness, args)

            while not result_async.ready():
                if self.cancel_flag and self.cancel_flag.is_set():
                    pool.terminate()
                    pool.join()
                    print("Cancelled during score calculation.")
                    return images
                if self.progress_callback:
                    self.progress_callback("Calculating sharpness scores...")
                time.sleep(0.1)

            try:
                results = result_async.get()
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                return images

        for result in results:
            if result is None:
                continue
            filename, is_sharp, laplacian = result
            path = os.path.join(self.folder, filename)
            self.laplacian_map[path] = laplacian

        print(f"Laplacian scores calculated for {len(self.laplacian_map)} images")
        return images

    def stage3_burst_grouping(self, sharp_images):
        print("\nSTAGE 3: Burst Grouping")
        if not sharp_images:
            print("No sharp images to process.")
            return []

        sharp_paths = [os.path.join(self.folder, f) for f in sharp_images]
        
        # CRITICAL FIX: Cache EXIF for all images before grouping
        print("Caching EXIF data for burst grouping...")
        for path in sharp_paths:
            if path not in self.exif_cache:
                self.cache_exif(path)
        
        burst_groups, non_burst_images = defaultdict(list), []

        # Build groups by datetime
        for path in sharp_paths:
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during burst grouping.")
                return []

            dt = self.exif_cache.get(path, {}).get('datetime')
            if dt:
                burst_groups[dt].append(path)
            else:
                non_burst_images.append(path)
                print(f"No datetime for: {os.path.basename(path)}")

        # DEBUG: Show what we found
        print(f"\n=== BURST GROUP ANALYSIS ===")
        print(f"Total images to process: {len(sharp_paths)}")
        print(f"Images with datetime: {sum(len(v) for v in burst_groups.values())}")
        print(f"Images without datetime: {len(non_burst_images)}")
        print(f"\nDatetime groups found: {len(burst_groups)}")
        
        # Show breakdown of group sizes
        group_size_counts = defaultdict(int)
        for dt, group in burst_groups.items():
            group_size_counts[len(group)] += 1
        
        print(f"\nGroup size distribution:")
        for size in sorted(group_size_counts.keys()):
            print(f"  {group_size_counts[size]} groups with {size} image(s)")

        # Filter to actual bursts (more than 1 image)
        actual_bursts = {k: v for k, v in burst_groups.items() if len(v) > 1}
        
        # Single-image groups go to non_burst_images
        for k, v in burst_groups.items():
            if len(v) == 1:
                non_burst_images.extend(v)

        print(f"\n=== PROCESSING BURSTS ===")
        print(f"Actual burst groups (>1 image): {len(actual_bursts)}")
        print(f"Non-burst images: {len(non_burst_images)}")

        final_selection, images_from_bursts = [], 0
        
        # Process each burst
        for burst_num, (dt, group) in enumerate(actual_bursts.items(), 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during burst processing.")
                return []

            print(f"\nBurst {burst_num} at {dt}: {len(group)} images")
            
            # Score images by sharpness
            scored = [(self.laplacian_map.get(path, 0.0), path) for path in group]
            scored.sort(reverse=True)
            
            # Select top N
            selected_from_burst = scored[:self.burst_count]
            
            # Show all images in burst with selection status
            for i, (score, path) in enumerate(scored):
                selected = i < self.burst_count
                marker = "✓ KEEP" if selected else "✗ DROP"
                print(f"  {marker} {os.path.basename(path)} (sharpness: {score:.1f})")

            # Add selected images to final selection
            for score, path in selected_from_burst:
                final_selection.append(path)
                images_from_bursts += 1

        # Add non-burst images
        final_selection.extend(non_burst_images)
        
        self.stats['final_selection'] = len(final_selection)
        
        print(f"\n=== BURST GROUPING COMPLETE ===")
        print(f"Burst groups processed: {len(actual_bursts)}")
        print(f"Images from bursts: {images_from_bursts}")
        print(f"Non-burst images: {len(non_burst_images)}")
        print(f"Total final selection: {len(final_selection)}")
        
        return final_selection

    def copy_final_images(self, final_paths, output_folder):
        print(f"\nCOPYING {len(final_paths)} IMAGES TO OUTPUT")
        os.makedirs(output_folder, exist_ok=True)
        copied_count = 0

        for path in final_paths:
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during copying.")
                self.stats['copied_images'] = copied_count
                return copied_count
            try:
                dest_path = os.path.join(output_folder, os.path.basename(path))
                shutil.copy(path, dest_path)
                copied_count += 1
                print(f"Copied: {os.path.basename(path)}")
            except Exception as e:
                print(f"Error copying {path}: {e}")

        self.stats['copied_images'] = copied_count
        return copied_count

    def run(self, use_starcheck=True, use_laplaciancheck=True, group_bursts=True, output_folder=None, progress_callback=None):
            self.progress_callback = progress_callback
            self.stats['start_time'] = time.time()
            self.use_starcheck, self.use_laplaciancheck, self.group_bursts = use_starcheck, use_laplaciancheck, group_bursts

            if output_folder is None:
                output_folder = os.path.join(self.folder, "Sharp")

            all_images = [f for f in os.listdir(self.folder) if f.lower().endswith(".jpg")]
            self.stats['total_images'] = len(all_images)
            print(f"Found {len(all_images)} JPG files to process")

            if not all_images:
                self.stats['end_time'] = time.time()
                self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
                return self.stats

            # Stage 1: Star check
            remaining_images = self.stage1_star_check(all_images) if use_starcheck else all_images
            if not remaining_images or (self.cancel_flag and self.cancel_flag.is_set()):
                self.stats['end_time'] = time.time()
                self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
                return self.stats

            # Stage 2: Sharpness check
            if use_laplaciancheck:
                sharp_images = self.stage2_sharpness_check(remaining_images)
            else:
                sharp_images = remaining_images
                # When laplacian check is disabled, all remaining images are "sharp"
                self.stats['sharp_images'] = len(sharp_images)
                if group_bursts:
                    self.calculate_laplacian_scores(sharp_images)

            if not sharp_images or (self.cancel_flag and self.cancel_flag.is_set()):
                self.stats['end_time'] = time.time()
                self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
                return self.stats

            # Stage 3: Burst grouping
            if group_bursts:
                final_paths = self.stage3_burst_grouping(sharp_images)
            else:
                final_paths = [os.path.join(self.folder, f) for f in sharp_images]
                self.stats['final_selection'] = len(final_paths)

            # Copy final images
            copied = self.copy_final_images(final_paths, output_folder)

            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            
            # Calculate rejection stats
            self.stats['filtered_by_sharpness'] = self.stats['total_images'] - self.stats.get('sharp_images', self.stats['total_images'])
            self.stats['filtered_by_bursts'] = self.stats.get('sharp_images', len(sharp_images)) - self.stats['final_selection']
            self.stats['total_rejected'] = self.stats['total_images'] - self.stats['final_selection']
            
            print(
                f"\n=== PROCESSING COMPLETE ===\n"
                f"Total images: {self.stats['total_images']}\n"
                f"Filtered by sharpness: {self.stats['filtered_by_sharpness']}\n"
                f"Filtered by burst grouping: {self.stats['filtered_by_bursts']}\n"
                f"Final selection: {self.stats['final_selection']} images\n"
                f"Copied: {copied}\n"
                f"Time elapsed: {self.stats['elapsed_time']:.2f}s\n"
                f"Output: {output_folder}"
            )
            return self.stats



def main(folder, base_blur=0, tolerance=0, burst_count=2, use_starcheck=False, use_laplaciancheck=True, group_bursts=True, output=None, cancel_flag=None, progress_callback=None):
    processor = ImageSharpnessProcessor(folder, base_blur, tolerance, burst_count)
    if cancel_flag:
        processor.cancel_flag = cancel_flag
    stats = processor.run(use_starcheck, use_laplaciancheck, group_bursts, output, progress_callback)
    return stats