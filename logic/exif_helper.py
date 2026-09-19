from PIL import Image
from PIL.ExifTags import TAGS


def _ratio(v):
    return float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)


class EXIFHelper:
    @staticmethod
    def read_all(path):
        """One open, one parse. Returns {tag_name: value}, {} on failure."""
        try:
            with Image.open(path) as img:
                raw = img._getexif() or {}
        except Exception as e:
            print(f"Error reading EXIF from {path}: {e}")
            return {}
        return {TAGS.get(k, k): v for k, v in raw.items()}

    @staticmethod
    def parse(exif):
        """Turn read_all() dict into the fields the sorter uses."""
        fstop = 8.0
        try:
            fstop = _ratio(exif['FNumber'])
        except Exception:
            try:
                fstop = 2 ** (_ratio(exif['ApertureValue']) / 2)
            except Exception:
                pass

        try:
            rating = int(exif['Rating'])
        except (KeyError, ValueError, TypeError):
            rating = None

        return {
            'fstop': fstop,
            'iso': exif.get('ISOSpeedRatings', 100),
            'shutter': exif.get('ExposureTime'),
            'rating': rating,
            'datetime': exif.get('DateTimeOriginal'),
            'subsec': exif.get('SubSecTimeOriginal', '00'),
        }

    # --- legacy single-value getters (each still opens the file once) ---

    @staticmethod
    def get_exif_value(path, key, default=None):
        return EXIFHelper.read_all(path).get(key, default)

    @staticmethod
    def get_fstop(path):
        return EXIFHelper.parse(EXIFHelper.read_all(path))['fstop']

    @staticmethod
    def get_shutter_speed(path):
        return EXIFHelper.get_exif_value(path, 'ExposureTime', None)

    @staticmethod
    def get_iso(path):
        return EXIFHelper.get_exif_value(path, 'ISOSpeedRatings', 100)

    @staticmethod
    def get_rating(path):
        return EXIFHelper.parse(EXIFHelper.read_all(path))['rating']

    @staticmethod
    def get_datetime_original(path):
        return EXIFHelper.get_exif_value(path, 'DateTimeOriginal', None)

    @staticmethod
    def get_subsec_time(path):
        return EXIFHelper.get_exif_value(path, 'SubSecTimeOriginal', '00')