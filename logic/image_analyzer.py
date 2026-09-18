import os
import cv2
import numpy as np

from face_detector import FaceDetector


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
