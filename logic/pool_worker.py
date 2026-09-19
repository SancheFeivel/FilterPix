import os
import io
import sys
import time
import multiprocessing

import cv2

from face_detector import FaceDetector
from image_analyzer import ImageAnalyzer

SUPPORTED_EXTS = ('.jpg', '.jpeg')


def _pool_initializer():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    FaceDetector._load()  # load cascades once per worker
    print(f"DEBUG WORKER: pool worker initialised (pid={os.getpid()})")


def process_image_sharpness(folder, filename, base_blur, tolerance, exif_cache):
    if not filename.lower().endswith(SUPPORTED_EXTS):
        return None

    path = os.path.join(folder, filename)
    image = None
    pid = os.getpid()
    t_start = time.perf_counter()

    print(f"DEBUG WORKER [{pid}]: starting {filename}")

    try:
        print(f"DEBUG WORKER [{pid}]: reading image {filename}")
        t = time.perf_counter()
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        t_read = time.perf_counter() - t

        if image is None:
            print(f"DEBUG WORKER [{pid}]: ERROR cv2.imread returned None for {filename}")
            return None

        t = time.perf_counter()
        image = ImageAnalyzer.resize_short_side(image, 720)
        t_resize = time.perf_counter() - t
        print(f"DEBUG WORKER [{pid}]: image loaded {filename} shape={image.shape} dtype={image.dtype}")

        exif_data = exif_cache.get(path)
        if exif_data is None:
            print(f"DEBUG WORKER [{pid}]: WARNING no EXIF cache for {filename}")
            return None

        print(f"DEBUG WORKER [{pid}]: running is_sharp for {filename} fstop={exif_data.get('fstop')}")
        t = time.perf_counter()
        is_sharp, laplacian = ImageAnalyzer.is_sharp(
            image, path, base_blur, tolerance, exif_data
        )
        t_sharp = time.perf_counter() - t

        t_total = time.perf_counter() - t_start
        print(f"DEBUG WORKER [{pid}]: done {filename} is_sharp={is_sharp} laplacian={laplacian:.2f}")
        print(
            f"TIMING WORKER [{pid}] {filename}: "
            f"read={t_read*1000:.1f}ms resize={t_resize*1000:.1f}ms "
            f"is_sharp={t_sharp*1000:.1f}ms total={t_total*1000:.1f}ms"
        )
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


def run_pool(args, cancel_flag, progress_callback, stage_name, pool_timeout=300):
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
        ticks = 0

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
            ticks += 1
            if ticks % 20 == 0:
                print(f"DEBUG POOL [{stage_name}]: still waiting... {ticks * 0.5:.0f}s elapsed")

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