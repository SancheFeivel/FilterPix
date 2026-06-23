import os
import io
import sys
import cv2
import time
import shutil
import multiprocessing
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
import numpy as np

# ── Multiprocessing safety (required for PyInstaller + Windows/macOS) ─────────
# Must be called once at app startup, before any Pool is created.
def init_multiprocessing():
    multiprocessing.freeze_support()
    method = multiprocessing.get_start_method(allow_none=True)
    print(f"DEBUG INIT: current start method = {method!r}")
    if method is None:
        multiprocessing.set_start_method('spawn')
        print("DEBUG INIT: start method set to 'spawn'")
    else:
        print(f"DEBUG INIT: start method already set, leaving as {method!r}")

def _pool_initializer():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    FaceDetector._load()  # load cascades once per worker
    print(f"DEBUG WORKER: pool worker initialised (pid={os.getpid()})")

# ─────────────────────────────────────────────────────────────────────────────

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
        value = EXIFHelper.get_exif_value(path, 'FNumber', None)
        if value is not None:
            try:
                return float(value[0]) / float(value[1]) if isinstance(value, tuple) else float(value)
            except Exception:
                pass
        apex = EXIFHelper.get_exif_value(path, 'ApertureValue', None)
        if apex is not None:
            try:
                apex_val = float(apex[0]) / float(apex[1]) if isinstance(apex, tuple) else float(apex)
                return 2 ** (apex_val / 2)
            except Exception:
                pass
        return 8.0

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

    
class FaceDetector:
    """Lazy-loaded Haar cascade face detector."""
    _frontal = None
    _profile = None

    @classmethod
    def _load(cls):
        if cls._frontal is None:
            if getattr(sys, 'frozen', False):
                base = os.path.join(sys._MEIPASS, 'cv2', 'data', '')
            else:
                base = cv2.data.haarcascades

            frontal_path = base + "haarcascade_frontalface_default.xml"
            profile_path = base + "haarcascade_profileface.xml"

            cls._frontal = cv2.CascadeClassifier(frontal_path)
            cls._profile = cv2.CascadeClassifier(profile_path)

            if cls._frontal.empty():
                print(f"WARNING: Failed to load frontal cascade from: {frontal_path}")
            if cls._profile.empty():
                print(f"WARNING: Failed to load profile cascade from: {profile_path}")

    @classmethod
    def detect(cls, gray_image):
        cls._load()

        def _run(cascade, img, scale, neighbors):
            if cascade.empty():
                return []
            faces = cascade.detectMultiScale(
                img,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return list(faces) if len(faces) > 0 else []

        faces = _run(cls._frontal, gray_image, 1.1, 5)
        if not faces:
            faces = _run(cls._frontal, gray_image, 1.05, 3)
        if not faces:
            faces = _run(cls._profile, gray_image, 1.1, 4)

        return faces


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
        h, w = image.shape[:2]
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        sharpness_values = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                region = image[y1:y2, x1:x2]
                laplacian = cv2.Laplacian(region, cv2.CV_64F).var()
                sharpness_values.append(laplacian)
        sharpness_values.sort(reverse=True)
        return {
            'max': sharpness_values[0],
            'top3_avg': np.mean(sharpness_values[:3]),
            'top_quarter_avg': np.mean(sharpness_values[:len(sharpness_values)//4]),
            'median': np.median(sharpness_values),
            'all_values': sharpness_values
        }

    @staticmethod
    def is_sharp(image, path, base_blur, tolerance, exif_data=None):
        img = ImageAnalyzer.resize_then_crop(image, 720, 0.8)
        
        iso = exif_data.get('iso', 100) if exif_data else 100
        if iso >= 1200:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        h, w = img.shape
        name = os.path.basename(path)

        def lap(region):
            return cv2.Laplacian(region, cv2.CV_64F).var()

        def metrics(region):
            l = lap(region)
            sx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            ten = np.mean(sx**2 + sy**2)
            hist = cv2.calcHist([region], [0], None, [256], [0, 256])
            hist /= hist.sum() + 1e-7
            ent = -np.sum(hist * np.log2(hist + 1e-7))
            return l, ten, ent

        raw_level = base_blur + tolerance
        strictness = raw_level / 20.0

        grid = 4
        cell_h, cell_w = h // grid, w // grid
        all_cell_laps = sorted([
            lap(img[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w])
            for gi in range(grid) for gj in range(grid)
        ], reverse=True)
        top3_avg = np.mean(all_cell_laps[:3])

        effective_floor = max(1.0, 80.0 + strictness * 40.0)

        if top3_avg < effective_floor:
            print(f"DEBUG {name} REJECTED global floor: top3={top3_avg:.1f}")
            return False, 0.0

        fstop = exif_data.get('fstop', 99) if exif_data else 99
        face_action = "not checked"

        band = int(min(h, w) * 0.18)
        bg_regions = [img[0:band, :], img[h-band:h, :], img[:, 0:band], img[:, w-band:w]]

        def get_bg_lap():
            return np.mean([lap(r) for r in bg_regions])

        def get_eye_laps(faces, cx_lo=0.20, cx_hi=0.80, cy_lo=0.20, cy_hi=0.80):
            results = []
            for (fx, fy, fw, fh) in faces:
                face_cx = fx + fw / 2
                face_cy = fy + fh / 2
                if not (w * cx_lo < face_cx < w * cx_hi and h * cy_lo < face_cy < h * cy_hi):
                    continue
                eye_zone = img[fy + int(fh*0.15):fy + int(fh*0.55),
                               fx + int(fw*0.05):fx + fw - int(fw*0.05)]
                if eye_zone.size > 0:
                    results.append(lap(eye_zone))
            return results

        if fstop <= 5.6:
            faces = FaceDetector.detect(img)
            if faces:
                eye_laps = get_eye_laps(faces)
                if eye_laps:
                    face_lap = np.mean(eye_laps)
                    bg = get_bg_lap()
                    eye_vs_bg = face_lap / (bg + 1e-6)

                    veto_ratio = 0.45 + strictness * 0.10
                    sharp_floor = 250.0 + strictness * 100.0

                    if eye_vs_bg < veto_ratio:
                        print(
                            f"DEBUG {name} FACE VETO OOF: "
                            f"eye_lap={face_lap:.1f} bg={bg:.1f} ratio={eye_vs_bg:.2f} "
                            f"veto={veto_ratio:.2f} f/{fstop} X BLUR"
                        )
                        return False, 0.0

                    elif face_lap >= sharp_floor:
                        print(
                            f"DEBUG {name} FACE PASS sharp eyes: "
                            f"eye_lap={face_lap:.1f} bg={bg:.1f} ratio={eye_vs_bg:.2f} "
                            f"floor={sharp_floor:.0f} f/{fstop} top3={top3_avg:.1f} V SHARP"
                        )
                        return True, face_lap

                    else:
                        face_action = f"soft eye_lap={face_lap:.1f} ratio={eye_vs_bg:.2f}"
                else:
                    face_action = "face out of bounds"
            else:
                face_action = "no face"

        ch, cw = int(h * 0.35), int(w * 0.35)
        cy, cx = (h - ch) // 2, (w - cw) // 2
        center = img[cy:cy+ch, cx:cx+cw]

        outer_regions = bg_regions
        cl, ct, ce = metrics(center)
        outer_metrics_list = [metrics(r) for r in outer_regions]
        ol = np.mean([m[0] for m in outer_metrics_list])
        ot = np.mean([m[1] for m in outer_metrics_list])
        oe = np.mean([m[2] for m in outer_metrics_list])

        def safe_ratio(c, o):
            if c >= o:
                o = max(o, c * 0.18)
            return c / (o + 1e-6)

        lap_ratio = safe_ratio(cl, ol)
        ten_ratio = safe_ratio(ct, ot)
        ent_ratio = safe_ratio(ce, oe)
        shallow_dof = lap_ratio > 2.6 or ten_ratio > 2.6

        score = (lap_ratio * 0.50 + ten_ratio * 0.45 + ent_ratio * 0.05)
        if shallow_dof:
            score *= 1.10

        threshold = 1.12 + strictness * 0.20
        if lap_ratio > 3.0:
            threshold -= 0.12
        if top3_avg > 800:
            threshold -= 0.15
        if top3_avg > 2000:
            threshold -= 0.10
        if fstop <= 5.6 and top3_avg > 800:
            threshold -= 0.45
        if 'soft' in face_action:
            threshold += 0.55

        sharp = score > threshold

        print(
            f"DEBUG {name} "
            f"LAP {lap_ratio:.2f} TEN {ten_ratio:.2f} ENT {ent_ratio:.2f} "
            f"SCORE {score:.2f} THR {threshold:.2f} "
            f"f/{fstop} shallow_dof={shallow_dof} face=({face_action}) "
            f"top3={top3_avg:.1f} "
            f"{'V SHARP' if sharp else 'X BLUR'}"
        )
        return sharp, score


def process_image_sharpness(folder, filename, base_blur, tolerance, exif_cache):
    if not filename.lower().endswith(".jpg"):
        return None

    path = os.path.join(folder, filename)
    image = None
    pid = os.getpid()

    print(f"DEBUG WORKER [{pid}]: starting {filename}")

    try:
        print(f"DEBUG WORKER [{pid}]: reading image {filename}")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"DEBUG WORKER [{pid}]: ERROR cv2.imread returned None for {filename}")
            return None

        image = ImageAnalyzer.resize_short_side(image, 720)
        print(f"DEBUG WORKER [{pid}]: image loaded {filename} shape={image.shape} dtype={image.dtype}")

        exif_data = exif_cache.get(path)
        if exif_data is None:
            print(f"DEBUG WORKER [{pid}]: WARNING no EXIF cache for {filename}")
            return None

        print(f"DEBUG WORKER [{pid}]: running is_sharp for {filename} fstop={exif_data.get('fstop')}")
        is_sharp, laplacian = ImageAnalyzer.is_sharp(
            image, path, base_blur, tolerance, exif_data
        )
        
        print(f"DEBUG WORKER [{pid}]: done {filename} is_sharp={is_sharp} laplacian={laplacian:.2f}")
        return filename, is_sharp, laplacian

    except Exception as e:
        import traceback
        print(f"DEBUG WORKER [{pid}]: EXCEPTION processing {filename}: {e}")
        print(traceback.format_exc())
        return None
    
    finally:
        if image is not None:
            del image
        # NOTE: cv2.destroyAllWindows() intentionally removed — can hang in
        # subprocess workers on platforms with no display (macOS/Windows)


def _run_pool(args, cancel_flag, progress_callback, stage_name, pool_timeout=300):
    """
    Shared helper that creates a spawn-safe Pool, runs starmap_async for
    process_image_sharpness, polls for cancellation, and returns results.
    Returns None if cancelled or on error.
    """
    pool_size = max(1, min(6, multiprocessing.cpu_count() // 2))
    ctx = multiprocessing.get_context('spawn')

    print(f"DEBUG POOL [{stage_name}]: starting pool size={pool_size} jobs={len(args)} start_method={ctx.get_start_method()}")

    with ctx.Pool(pool_size, initializer=_pool_initializer) as pool:
        print(f"DEBUG POOL [{stage_name}]: pool created, dispatching starmap_async")
        result_async = pool.starmap_async(process_image_sharpness, args)
        total = len(args)
        elapsed = 0

        while not result_async.ready():
            if cancel_flag and cancel_flag.is_set():
                print(f"DEBUG POOL [{stage_name}]: cancel flag detected, terminating pool")
                pool.terminate()
                pool.join()
                print(f"Cancelled during {stage_name}.")
                return None

            if progress_callback:
                progress_callback(-1, total, stage_name)

            time.sleep(0.5)
            elapsed += 0.5
            if elapsed % 10 == 0:
                print(f"DEBUG POOL [{stage_name}]: still waiting... {elapsed:.0f}s elapsed")

        print(f"DEBUG POOL [{stage_name}]: result_async ready, calling .get(timeout={pool_timeout})")
        try:
            results = result_async.get(timeout=pool_timeout)
            print(f"DEBUG POOL [{stage_name}]: got {len(results)} results")
            return results
        except multiprocessing.TimeoutError:
            print(f"DEBUG POOL [{stage_name}]: ERROR timed out after {pool_timeout}s")
            pool.terminate()
            return None
        except Exception as e:
            import traceback
            print(f"DEBUG POOL [{stage_name}]: ERROR in .get(): {e}")
            print(traceback.format_exc())
            pool.terminate()
            return None


class ImageSharpnessProcessor:
    def __init__(self, folder, base_blur=0, tolerance=0, burst_count=2):
        self.folder = folder
        self.base_blur = base_blur
        self.tolerance = tolerance
        self.burst_count = burst_count
        self.cancel_flag = None
        self.progress_callback = None

        self.exif_cache = {}
        self.laplacian_map = {}
        
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
        print("STAGE 1: Star Check")
        rated, unrated = [], []
        total = len(all_images)

        for idx, filename in enumerate(all_images, 1):
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
            
            if self.progress_callback:
                self.progress_callback(idx, total, "star_check")

        self.stats['rated_images'] = len(rated)

        if not self.use_laplaciancheck and not self.group_bursts:
            print("No later filters enabled → Only keeping rated images")
            return rated

        combined = rated + unrated
        print(f"Star check complete: {len(rated)} rated, {len(unrated)} unrated (kept for later stages)")
        return combined

    def stage2_sharpness_check(self, unrated_images):
        print("\nSTAGE 2: Laplacian Sharpness Check")
        if not unrated_images:
            print("No unrated images to process.")
            return []

        print(f"DEBUG STAGE2: {len(unrated_images)} images to process")
        print("Caching EXIF data for sharpness check...")
        total_exif = len(unrated_images)
        for idx, filename in enumerate(unrated_images, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during EXIF caching.")
                return []
            path = os.path.join(self.folder, filename)
            if path not in self.exif_cache:
                self.cache_exif(path)
            if self.progress_callback:
                self.progress_callback(idx, total_exif, "exif_caching")

        print(f"DEBUG STAGE2: EXIF cached for {len(self.exif_cache)} images, dispatching pool")
        args = [
            (self.folder, filename, self.base_blur, self.tolerance, self.exif_cache)
            for filename in unrated_images
        ]

        results = _run_pool(args, self.cancel_flag, self.progress_callback, "sharpness_check")
        if results is None:
            print("DEBUG STAGE2: pool returned None (cancelled or error)")
            return []

        print(f"DEBUG STAGE2: pool finished, processing {len(results)} results")
        sharp_images, sharp_count, blurry_count = [], 0, 0
        for idx, result in enumerate(results, 1):
            if result is None:
                print(f"DEBUG STAGE2: result {idx} is None (worker failed)")
                continue
            filename, is_sharp, laplacian = result
            path = os.path.join(self.folder, filename)
            self.laplacian_map[path] = laplacian
            if is_sharp:
                sharp_images.append(filename)
                sharp_count += 1
            else:
                blurry_count += 1
            if self.progress_callback:
                self.progress_callback(idx, len(results), "sharpness_results")

        self.stats['sharp_images'] = sharp_count
        print(f"Sharpness check complete: {sharp_count} sharp, {blurry_count} blurry (filtered)")
        return sharp_images

    def calculate_laplacian_scores(self, images):
        print("\nCALCULATING LAPLACIAN SCORES (for burst grouping)")
        print(f"DEBUG SCORES: {len(images)} images to score")

        print("Caching EXIF data for score calculation...")
        total_exif = len(images)
        for idx, filename in enumerate(images, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during EXIF caching.")
                return images
            path = os.path.join(self.folder, filename)
            if path not in self.exif_cache:
                self.cache_exif(path)
            if self.progress_callback:
                self.progress_callback(idx, total_exif, "exif_caching_scores")

        print(f"DEBUG SCORES: EXIF cached, dispatching pool")
        args = [
            (self.folder, filename, self.base_blur, self.tolerance, self.exif_cache)
            for filename in images
        ]

        results = _run_pool(args, self.cancel_flag, self.progress_callback, "calculating_scores")
        if results is None:
            print("DEBUG SCORES: pool returned None (cancelled or error)")
            return images

        print(f"DEBUG SCORES: pool finished, processing {len(results)} results")
        for idx, result in enumerate(results, 1):
            if result is None:
                print(f"DEBUG SCORES: result {idx} is None (worker failed)")
                continue
            filename, is_sharp, laplacian = result
            path = os.path.join(self.folder, filename)
            self.laplacian_map[path] = laplacian
            if self.progress_callback:
                self.progress_callback(idx, len(results), "score_results")

        print(f"Laplacian scores calculated for {len(self.laplacian_map)} images")
        return images

    def stage3_burst_grouping(self, sharp_images):
        print("\nSTAGE 3: Burst Grouping")
        if not sharp_images:
            print("No sharp images to process.")
            return []

        sharp_paths = [os.path.join(self.folder, f) for f in sharp_images]
        
        print("Caching EXIF data for burst grouping...")
        total_exif = len(sharp_paths)
        for idx, path in enumerate(sharp_paths, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during EXIF caching.")
                return []
            if path not in self.exif_cache:
                self.cache_exif(path)
            if self.progress_callback:
                self.progress_callback(idx, total_exif, "exif_caching")
        
        burst_groups, non_burst_images = defaultdict(list), []

        for idx, path in enumerate(sharp_paths, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during burst grouping.")
                return []

            dt = self.exif_cache.get(path, {}).get('datetime')
            if dt:
                burst_groups[dt].append(path)
            else:
                non_burst_images.append(path)
                print(f"No datetime for: {os.path.basename(path)}")
            
            if self.progress_callback:
                self.progress_callback(idx, len(sharp_paths), "grouping_bursts")

        group_size_counts = defaultdict(int)
        for dt, group in burst_groups.items():
            group_size_counts[len(group)] += 1
        
        print(f"\nGroup size distribution:")
        for size in sorted(group_size_counts.keys()):
            print(f"  {group_size_counts[size]} groups with {size} image(s)")

        actual_bursts = {k: v for k, v in burst_groups.items() if len(v) > 1}
        
        for k, v in burst_groups.items():
            if len(v) == 1:
                non_burst_images.extend(v)

        print(f"\n=== PROCESSING BURSTS ===")
        print(f"Actual burst groups (>1 image): {len(actual_bursts)}")
        print(f"Non-burst images: {len(non_burst_images)}")

        final_selection, images_from_bursts = [], 0
        total_bursts = len(actual_bursts)
        
        for burst_num, (dt, group) in enumerate(actual_bursts.items(), 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during burst processing.")
                return []

            print(f"\nBurst {burst_num} at {dt}: {len(group)} images")
            
            scored = [(self.laplacian_map.get(path, 0.0), path) for path in group]
            scored.sort(reverse=True)
            selected_from_burst = scored[:self.burst_count]
            
            for i, (score, path) in enumerate(scored):
                marker = "V KEEP" if i < self.burst_count else "X DROP"
                print(f"  {marker} {os.path.basename(path)} (sharpness: {score:.1f})")

            for score, path in selected_from_burst:
                final_selection.append(path)
                images_from_bursts += 1
            
            if self.progress_callback:
                self.progress_callback(burst_num, total_bursts, "processing_bursts")

        final_selection.extend(non_burst_images)
        self.stats['final_selection'] = len(final_selection)
        return final_selection

    def copy_final_images(self, final_paths, output_folder, all_images):
        print(f"\nCOPYING {len(final_paths)} IMAGES TO OUTPUT")
        
        if os.path.basename(output_folder) == "Sharp":
            base_output = os.path.dirname(output_folder)
            sharp_folder = output_folder
        else:
            base_output = output_folder
            sharp_folder = os.path.join(output_folder, "Sharp")
        
        os.makedirs(sharp_folder, exist_ok=True)
        rejected_folder = os.path.join(base_output, "Rejected")
        os.makedirs(rejected_folder, exist_ok=True)
        
        copied_count = 0
        rejected_count = 0
        final_basenames = {os.path.basename(path) for path in final_paths}
        
        for idx, path in enumerate(final_paths, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during copying.")
                self.stats['copied_images'] = copied_count
                self.stats['rejected_images'] = rejected_count
                return copied_count
            try:
                dest_path = os.path.join(sharp_folder, os.path.basename(path))
                shutil.copy(path, dest_path)
                copied_count += 1
                print(f"Copied to Sharp: {os.path.basename(path)}")
                if self.progress_callback:
                    self.progress_callback(idx, len(final_paths), "copying_sharp")
            except Exception as e:
                print(f"Error copying {path}: {e}")
        
        rejected_images = [f for f in all_images if f not in final_basenames]
        
        for idx, filename in enumerate(rejected_images, 1):
            if self.cancel_flag and self.cancel_flag.is_set():
                print("Cancelled during copying rejected images.")
                self.stats['copied_images'] = copied_count
                self.stats['rejected_images'] = rejected_count
                return copied_count
            try:
                source_path = os.path.join(self.folder, filename)
                dest_path = os.path.join(rejected_folder, filename)
                shutil.copy(source_path, dest_path)
                rejected_count += 1
                print(f"Copied to Rejected: {filename}")
                if self.progress_callback:
                    self.progress_callback(idx, len(rejected_images), "copying_rejected")
            except Exception as e:
                print(f"Error copying rejected image {filename}: {e}")

        self.stats['copied_images'] = copied_count
        self.stats['rejected_images'] = rejected_count
        print(f"\nCopied {rejected_count} rejected images to: {rejected_folder}")
        return copied_count

    def run(self, use_starcheck=True, use_laplaciancheck=True, group_bursts=True, output_folder=None, progress_callback=None):
        self.progress_callback = progress_callback
        self.stats['start_time'] = time.time()
        self.use_starcheck, self.use_laplaciancheck, self.group_bursts = use_starcheck, use_laplaciancheck, group_bursts

        import platform
        print(f"DEBUG RUN: platform={platform.system()} {platform.release()} python={sys.version}")
        print(f"DEBUG RUN: cpu_count={multiprocessing.cpu_count()} start_method={multiprocessing.get_start_method(allow_none=True)}")
        print(f"DEBUG RUN: frozen={getattr(sys, 'frozen', False)} pid={os.getpid()}")
        print(f"DEBUG RUN: flags use_starcheck={use_starcheck} use_laplaciancheck={use_laplaciancheck} group_bursts={group_bursts}")

        if output_folder is None:
            output_folder = self.folder

        all_images = [f for f in os.listdir(self.folder) if f.lower().endswith(".jpg")]
        self.stats['total_images'] = len(all_images)
        print(f"Found {len(all_images)} JPG files to process")

        if not all_images:
            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            return self.stats

        remaining_images = self.stage1_star_check(all_images) if use_starcheck else all_images
        if not remaining_images or (self.cancel_flag and self.cancel_flag.is_set()):
            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            return self.stats

        print(f"DEBUG RUN: {len(remaining_images)} images remaining after stage 1")

        if use_laplaciancheck:
            sharp_images = self.stage2_sharpness_check(remaining_images)
        else:
            sharp_images = remaining_images
            self.stats['sharp_images'] = len(sharp_images)
            if group_bursts:
                self.calculate_laplacian_scores(sharp_images)

        print(f"DEBUG RUN: {len(sharp_images) if sharp_images else 0} images after stage 2")

        if not sharp_images or (self.cancel_flag and self.cancel_flag.is_set()):
            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            return self.stats

        if group_bursts:
            final_paths = self.stage3_burst_grouping(sharp_images)
        else:
            final_paths = [os.path.join(self.folder, f) for f in sharp_images]
            self.stats['final_selection'] = len(final_paths)

        print(f"DEBUG RUN: {len(final_paths) if final_paths else 0} images after stage 3, copying now")
        copied = self.copy_final_images(final_paths, output_folder, all_images)

        self.stats['end_time'] = time.time()
        self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
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