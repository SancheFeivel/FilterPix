from sharpness_processor import ImageSharpnessProcessor

# Re-exported for backwards compatibility with any code importing these
# names from blur_sorter directly.
from exif_helper import EXIFHelper  # noqa: F401
from face_detector import FaceDetector  # noqa: F401
from image_analyzer import ImageAnalyzer  # noqa: F401
from pool_worker import process_image_sharpness, run_pool as _run_pool  # noqa: F401


def main(folder, base_blur=0, tolerance=0, burst_count=2, use_starcheck=False, use_laplaciancheck=True, group_bursts=True, output=None, cancel_flag=None, progress_callback=None, keep_rejected=True):
    processor = ImageSharpnessProcessor(folder, base_blur, tolerance, burst_count)
    if cancel_flag:
        processor.cancel_flag = cancel_flag
    stats = processor.run(use_starcheck, use_laplaciancheck, group_bursts, output, progress_callback, keep_rejected=keep_rejected)
    return stats
