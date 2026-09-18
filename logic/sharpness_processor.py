import os
import time
import shutil
from collections import defaultdict

from exif_helper import EXIFHelper
from pool_worker import run_pool

SUPPORTED_EXTS = ('.jpg', '.jpeg')


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

        self.stats = {}

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

        results = run_pool(args, self.cancel_flag, self.progress_callback, "sharpness_check")
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
                self.progress_callback(idx, len(results), "sharpness_results", extra={"rejected": blurry_count})

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

        results = run_pool(args, self.cancel_flag, self.progress_callback, "calculating_scores")
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
        burst_dropped = 0
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

        final_selection = []
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
                if i >= self.burst_count:
                    burst_dropped += 1
                print(f"  {marker} {os.path.basename(path)} (sharpness: {score:.1f})")

            for score, path in selected_from_burst:
                final_selection.append(path)

            if self.progress_callback:
                self.progress_callback(burst_num, total_bursts, "processing_bursts", extra={"rejected": burst_dropped})

        final_selection.extend(non_burst_images)
        self.stats['final_selection'] = len(final_selection)
        return final_selection

    def copy_final_images(self, final_paths, output_folder, all_images, keep_rejected=True):
        print(f"\nCOPYING {len(final_paths)} IMAGES TO OUTPUT")

        if os.path.basename(output_folder) == "Sharp":
            base_output = os.path.dirname(output_folder)
            sharp_folder = output_folder
        else:
            base_output = output_folder
            sharp_folder = os.path.join(output_folder, "Sharp")

        os.makedirs(sharp_folder, exist_ok=True)

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

        if self.progress_callback:
            self.progress_callback(len(rejected_images), len(rejected_images), "rejected_known")

        if keep_rejected:
            rejected_folder = os.path.join(base_output, "Rejected")
            os.makedirs(rejected_folder, exist_ok=True)

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

            print(f"\nCopied {rejected_count} rejected images to: {rejected_folder}")
        else:
            # keep_rejected is off — don't touch the source files, but the
            # stat still needs to reflect how many were rejected.
            rejected_count = len(rejected_images)
            print(f"\nkeep_rejected=False — {rejected_count} rejected images left untouched in source folder")

        self.stats['copied_images'] = copied_count
        self.stats['rejected_images'] = rejected_count
        return copied_count

    def run(self, use_starcheck=True, use_laplaciancheck=True, group_bursts=True, output_folder=None, progress_callback=None, keep_rejected=False):
        self.progress_callback = progress_callback
        self.stats['start_time'] = time.time()
        self.use_starcheck, self.use_laplaciancheck, self.group_bursts = use_starcheck, use_laplaciancheck, group_bursts

        import sys
        import platform
        import multiprocessing
        print(f"DEBUG RUN: platform={platform.system()} {platform.release()} python={sys.version}")
        print(f"DEBUG RUN: cpu_count={multiprocessing.cpu_count()} start_method={multiprocessing.get_start_method(allow_none=True)}")
        print(f"DEBUG RUN: frozen={getattr(sys, 'frozen', False)} pid={os.getpid()}")
        print(f"DEBUG RUN: flags use_starcheck={use_starcheck} use_laplaciancheck={use_laplaciancheck} group_bursts={group_bursts}")

        if output_folder is None:
            output_folder = self.folder

        all_images = [f for f in os.listdir(self.folder) if f.lower().endswith(SUPPORTED_EXTS)]
        self.stats['total_images'] = len(all_images)
        print(f"Found {len(all_images)} JPG files to process")

        if not all_images:
            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            self.stats.setdefault('rejected_images', self.stats['total_images'] - self.stats.get('final_selection', 0))
            return self.stats

        remaining_images = self.stage1_star_check(all_images) if use_starcheck else all_images
        if not remaining_images or (self.cancel_flag and self.cancel_flag.is_set()):
            self.stats['end_time'] = time.time()
            self.stats['elapsed_time'] = self.stats['end_time'] - self.stats['start_time']
            self.stats.setdefault('rejected_images', self.stats['total_images'] - self.stats.get('final_selection', 0))
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
            self.stats.setdefault('rejected_images', self.stats['total_images'] - self.stats.get('final_selection', 0))
            return self.stats

        if group_bursts:
            final_paths = self.stage3_burst_grouping(sharp_images)
        else:
            final_paths = [os.path.join(self.folder, f) for f in sharp_images]
            self.stats['final_selection'] = len(final_paths)

        print(f"DEBUG RUN: {len(final_paths) if final_paths else 0} images after stage 3, copying now")
        copied = self.copy_final_images(final_paths, output_folder, all_images, keep_rejected=keep_rejected)

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
