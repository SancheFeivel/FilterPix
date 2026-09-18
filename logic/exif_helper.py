from PIL import Image
from PIL.ExifTags import TAGS


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
