import os
import sys
import cv2


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
