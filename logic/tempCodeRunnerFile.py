    @staticmethod
    def is_sharp(image, path, base_blur, tolerance, exif_data=None):
        """
        Improved dual-mode sharpness detection with stabilization.
        Structure preserved, thresholds stabilized.
        """

        # --- NORMALIZE RESOLUTION + CROP ---
        cropped = ImageAnalyzer.resize_then_crop(image)

        # --- NOISE STABILIZATION ---
        cropped = cv2.GaussianBlur(cropped, (3, 3), 0)

        regional_metrics = ImageAnalyzer.detect_sharp_regions(cropped)

        # --- CONTRAST NORMALIZATION ---
        contrast = np.std(cropped) + 1e-5
        norm_factor = contrast / 64.0  # 64 ≈ mid contrast baseline

        # --- EXIF ---
        if exif_data is None:
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

        filename = os.path.basename(path)
        shallow_dof = fstop <= 3.0

        # --- USER FACTOR (PROPORTIONAL INSTEAD OF ADDITIVE) ---
        user_factor = 1 + (base_blur + tolerance) / 100.0

        # --- ISO SMOOTH FACTOR ---
        iso_factor = np.clip((iso - 100) / 2900, 0, 1)

        if shallow_dof:
            # ================= SHALLOW DOF =================
            full_laplacian = cv2.Laplacian(cropped, cv2.CV_64F).var() * norm_factor

            regional_score = (
                regional_metrics['top3_avg'] * 0.7 +
                regional_metrics['max'] * 0.3
            ) * norm_factor

            # Base thresholds by aperture
            if fstop <= 1.8:
                base_regional = 70
                base_global = 30
            elif fstop <= 2.0:
                base_regional = 85
                base_global = 40
            else:
                base_regional = 105
                base_global = 50

            # Smooth ISO scaling
            regional_threshold = base_regional * (1 + iso_factor * 0.6) * user_factor
            global_threshold = base_global * (1 + iso_factor * 0.4) * user_factor

            passes_regional = regional_score > regional_threshold
            passes_global = full_laplacian > global_threshold

            # Strong subject boost
            if regional_score > regional_threshold * 2:
                passes_global = full_laplacian > global_threshold * 0.7

            is_sharp_final = passes_regional and passes_global
            regional_score_return = regional_score

            print(
                f"DEBUG {filename} [SHALLOW]: "
                f"regional={regional_score:.1f}/{regional_threshold:.1f}, "
                f"global={full_laplacian:.1f}/{global_threshold:.1f}, "
                f"fstop={fstop}, iso={iso}, "
                f"FINAL={'SHARP' if is_sharp_final else 'BLURRY'}"
            )

        else:
            # ================= DEEP DOF =================
            h, w = cropped.shape
            border_width = int(min(h, w) * 0.15)

            top_edge = cropped[0:border_width, :]
            bottom_edge = cropped[h-border_width:h, :]
            left_edge = cropped[:, 0:border_width]
            right_edge = cropped[:, w-border_width:w]

            top_lap = cv2.Laplacian(top_edge, cv2.CV_64F).var()
            bottom_lap = cv2.Laplacian(bottom_edge, cv2.CV_64F).var()
            left_lap = cv2.Laplacian(left_edge, cv2.CV_64F).var()
            right_lap = cv2.Laplacian(right_edge, cv2.CV_64F).var()

            edge_sharpness = np.mean([top_lap, bottom_lap, left_lap, right_lap]) * norm_factor

            center_region = cropped[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            center_sharpness = cv2.Laplacian(center_region, cv2.CV_64F).var() * norm_factor

            global_sharpness = cv2.Laplacian(cropped, cv2.CV_64F).var() * norm_factor

            regional_score = (
                regional_metrics['top3_avg'] * 0.6 +
                regional_metrics['max'] * 0.3 +
                regional_metrics['top_quarter_avg'] * 0.1
            ) * norm_factor

            # Base threshold by aperture
            if fstop < 4:
                base_threshold = 60
            elif fstop < 8:
                base_threshold = 80
            else:
                base_threshold = 110

            base_threshold *= (1 + iso_factor * 0.5)

            center_threshold = base_threshold * user_factor
            min_edge = max(30, base_threshold * 0.75) * user_factor
            min_global = max(35, base_threshold * 0.85) * user_factor

            passes_center = center_sharpness >= center_threshold
            edges_acceptable = edge_sharpness >= min_edge
            global_acceptable = global_sharpness >= min_global

            # Weighted secondary instead of binary
            secondary_score = (0.4 if edges_acceptable else 0) + (0.6 if global_acceptable else 0)
            passes_secondary = secondary_score >= 0.6

            # Softer obvious blur veto
            obviously_blurry = (
                center_sharpness < center_threshold * 0.6 or
                global_sharpness < min_global * 0.55 or
                edge_sharpness < min_edge * 0.5
            )

            is_sharp_final = passes_center and passes_secondary and not obviously_blurry
            regional_score_return = regional_score

            print(
                f"DEBUG {filename} [DEEP]: "
                f"CENTER={center_sharpness:.1f}/{center_threshold:.1f}, "
                f"EDGE={edge_sharpness:.1f}/{min_edge:.1f}, "
                f"GLOBAL={global_sharpness:.1f}/{min_global:.1f}, "
                f"FINAL={'SHARP' if is_sharp_final else 'BLURRY'}"
            )

        return is_sharp_final, regional_score_return