from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time

import webview

def get_base_path():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def get_project_root():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


BASE_PATH = get_base_path()
HTML_PATH = BASE_PATH / "index.html"
SPLASH_PATH = BASE_PATH / "splash.html"


class SettingsManager:
    def __init__(self, app_name="FilterPix"):
        if sys.platform == "win32":
            app_data = os.getenv("APPDATA")
            self.config_dir = Path(app_data) / app_name if app_data else Path.home() / f".{app_name.lower()}"
        else:
            self.config_dir = Path.home() / f".{app_name.lower()}"
        self.config_file = self.config_dir / "settings.json"

    def save(self, settings):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def load(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return {}


def import_logic_modules():
    try:
        project_root = get_project_root()
        logic_path = project_root / "logic"

        for p in (str(logic_path), str(project_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

        import blur_sorter as blur
        import detection as detect

        print("Modules imported successfully")
        return blur, detect
    except Exception as e:
        print(f"Error importing logic modules: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _first(d: dict, *keys, default=0):
    """Return the first non-None value found under *keys* in *d*, else default."""
    return next((d[k] for k in keys if d.get(k) is not None), default)


class Api:
    def __init__(self, window_getter):
        self._get_window = window_getter
        self.settings_mgr = SettingsManager()

        self.folder_path = None
        self.output_directory = None

        self.sorter_cancel_event = threading.Event()
        self.detection_cancel_event = threading.Event()
        self.is_processing = False
        self.cancelled = False

        self.sorter_stats = None
        self._kept_count = 0
        self._rejected_count = 0
        self.detection_stats = None

        self._scanned_count = 0
        self._kept_count = 0
        self._rejected_count = 0

        self.blur = None
        self.detect = None

        self._start_time = None
        self._save_lock = threading.Lock()

        saved = self.settings_mgr.load()
        self.settings = {
            "star_enabled": saved.get("star_enabled", True),
            "laplacian_enabled": saved.get("laplacian_enabled", True),
            "burst_enabled": saved.get("burst_enabled", True),
            "img_detect_enabled": saved.get("img_detect_enabled", False),
            "sharpness_level": saved.get("sharpness_level", 0),
            "tolerance": saved.get("tolerance", 0),
            "detection_mode": saved.get("detection_mode", "fast"),
            "keep_rejected": saved.get("keep_rejected", False),
            "last_folder": saved.get("last_folder", None),
            "output_directory": saved.get("output_directory", None),
        }
        if self.settings["last_folder"]:
            self.folder_path = self.settings["last_folder"]
        if self.settings["output_directory"]:
            self.output_directory = self.settings["output_directory"]

    def attach_logic(self, blur, detect):
        self.blur = blur
        self.detect = detect

    def get_settings(self):
        return self.settings

    def update_settings(self, payload):
        with self._save_lock:
            self.settings.update(payload)
            if payload.get("last_folder"):
                self.folder_path = payload["last_folder"]
            if "output_directory" in payload:
                self.output_directory = payload["output_directory"]
            ok = self.settings_mgr.save(self.settings)
            if not ok:
                print("WARNING: settings_mgr.save() failed in update_settings")
            return ok

    def select_folder(self):
        result = self._get_window().create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            folder = result[0]
            self.folder_path = folder
            self.settings["last_folder"] = folder
            self.settings_mgr.save(self.settings)
            return folder
        return None

    def select_output_folder(self):
        result = self._get_window().create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            self.output_directory = result[0]
            self.settings["output_directory"] = self.output_directory
            self.settings_mgr.save(self.settings)
            return self.output_directory
        return None

    def open_folder(self, path):
        target = path or self.folder_path
        if not target or not os.path.isdir(target):
            return {"error": "folder does not exist"}
        try:
            if sys.platform == "win32":
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return {"opened": True}
        except Exception as e:
            print(f"open_folder failed: {e}")
            return {"error": str(e)}

    def start_cull(self, options):
        if self.is_processing:
            return {"error": "already running"}
        if not options.get("folder") or not os.path.isdir(options["folder"]):
            self._push_progress(tag="sys", msg="invalid or missing source folder.")
            return {"error": "invalid folder"}

        self.folder_path = options["folder"]
        self.output_directory = options.get("output") or None
        self.cancelled = False
        self.sorter_stats = None
        self.detection_stats = None
        self._scanned_count = 0
        self._kept_count = 0
        self._rejected_count = 0
        self.is_processing = True
        self._start_time = time.time()

        star_enabled = options.get("use_starcheck", True)
        laplacian_enabled = options.get("use_laplaciancheck", True)
        burst_enabled = options.get("group_bursts", True)
        img_detect_enabled = options.get("img_detect", False)
        keep_rejected = options.get("keep_rejected", False)

        # True when the caller wants detection but no sorter passes at all.
        solo = not (star_enabled or laplacian_enabled or burst_enabled)

        sorter_options = {
            "folder": self.folder_path,
            "output": self.output_directory,
            "base_blur": options.get("base_blur", 0),
            "tolerance": options.get("tolerance", 0),
            "use_starcheck": star_enabled,
            "use_laplaciancheck": laplacian_enabled,
            "group_bursts": burst_enabled,
            "keep_rejected": keep_rejected,
            "cancel_flag": self.sorter_cancel_event,
            "progress_callback": self._make_progress_callback(),
        }

        self.sorter_cancel_event.clear()
        self.detection_cancel_event.clear()

        threading.Thread(
            target=self._run_worker,
            args=(sorter_options, options, img_detect_enabled, solo),
            daemon=True,
        ).start()

        return {"started": True}

    def cancel_cull(self):
        if not self.is_processing:
            return False
        self.cancelled = True
        self.sorter_cancel_event.set()
        self.detection_cancel_event.set()
        return True

    # ---------- single unified worker ----------

    def _run_worker(self, sorter_options, original_options, img_detect_enabled, solo):
        if not solo:
            self.sorter_stats = self._run_sorter(sorter_options)
            if self.cancelled:
                self._finish(cancelled=True)
                return

        if solo or img_detect_enabled:
            self._run_detection(original_options, solo=solo)
        else:
            self._finish()

    def _run_sorter(self, sorter_options) -> dict:
        try:
            return self.blur.main(**sorter_options) or {}
        except Exception as e:
            print(f"Error in sorter: {e}")
            return {}

    def _run_detection(self, original_options, solo):
        detection_options = {
            "folder": self.folder_path,
            "output": self.output_directory,
            "mode": original_options.get("detection_mode", "fast"),
            "solo_process": solo,
            "cancel_flag": self.detection_cancel_event,
            "progress_callback": self._make_progress_callback(),
        }
        try:
            self.detection_stats = self.detect.main(**detection_options) or {}
        except Exception as e:
            print(f"Error in detection: {e}")
            self.detection_stats = {}

        self._finish(cancelled=self.cancelled)

    # ---------- progress plumbing ----------

    def _make_progress_callback(self):
        """Adapts the existing (current, total, stage_name) callback contract
        from blur_sorter / detection into log lines + stat updates in the UI.
        Counters are cumulative across stages, not reset per-stage."""

        # which stage_names represent a KEEP / REJECT event per-item
        KEEP_STAGES = {"copying_sharp"}
        REJECT_STAGES = {"copying_rejected"}
        # stages where current==total items have been freshly *scanned*
        # (i.e. count toward the SCANNED stat as new items seen)
        SCAN_STAGES = {"star_check", "sharpness_results", "score_results"}

        def callback(current, total, stage_name="processing"):
            if current == -1:
                self._push_progress(tag="sys", msg=f"{stage_name.replace('_', ' ')}...")
                return

            tag = "scan"
            payload = {}

            if stage_name in KEEP_STAGES:
                tag = "keep"
                self._kept_count = current  # current IS the running copied count
                payload["kept"] = self._kept_count
            elif stage_name in REJECT_STAGES:
                tag = "drop"
                self._rejected_count = current
                payload["rejected"] = self._rejected_count
            elif stage_name in SCAN_STAGES:
                # these stages fire once per image in order -> safe to treat
                # current as "items scanned so far in this stage", but we only
                # want it monotonic overall, so just mirror total_images scope
                self._scanned_count = current
                payload["scanned"] = self._scanned_count

            percent = int((current / total) * 100) if total else None
            if percent is not None:
                payload["percent"] = percent
            payload["tag"] = tag
            payload["msg"] = f"{stage_name.replace('_', ' ')}: {current}/{total}"

            self._push_progress(**payload)

        return callback
    
    def _push_progress(self, **payload):
        window = self._get_window()
        if not window:
            return
        try:
            window.evaluate_js(f"window.onCullProgress({json.dumps(payload)})")
        except Exception as e:
            print(f"progress push failed: {e}")

    def _finish(self, cancelled=False):
        self.is_processing = False
        stats = self._compute_final_stats()
        elapsed = time.time() - (self._start_time or time.time())
        rejected = max(stats["total_images"] - stats["final_selection"], 0)

        result = {
            "kept": stats["final_selection"],
            "total": stats["total_images"],
            "rejected": rejected,
            "elapsed": elapsed,
            "cancelled": cancelled,
        }

        self._push_progress(kept=stats["final_selection"], rejected=rejected)

        window = self._get_window()
        if window:
            try:
                window.evaluate_js(f"window.onCullComplete({json.dumps(result)})")
            except Exception as e:
                print(f"completion push failed: {e}")

        self.cancelled = False

    def _compute_final_stats(self):
        sorter = self.sorter_stats or {}
        detection = self.detection_stats or {}

        # solo_detection path: sorter_stats is None, so read from detection
        if not sorter:
            return {
                "total_images": _first(detection, "total_images", "images_processed", "total"),
                "final_selection": _first(detection, "selected_images", "rated_images", "final_selection"),
            }

        return {
            "total_images": _first(sorter, "total_images", "images_processed"),
            "final_selection": _first(sorter, "final_selection", "sharp_images"),
        }


def main():
    window_ref = {"win": None, "splash": None}

    api = Api(window_getter=lambda: window_ref["win"])

    splash = webview.create_window(
        "FilterPix",
        str(SPLASH_PATH),
        width=640,
        height=480,
        resizable=False,
        frameless=False,
        background_color="#08090a",
    )
    window_ref["splash"] = splash

    window = webview.create_window(
        "FilterPix",
        str(HTML_PATH),
        js_api=api,
        width=1040,
        height=680,
        resizable=True,
        min_size=(900, 600),
        background_color="#0d0a06",
        hidden=True,
    )
    window_ref["win"] = window

    def push_boot(tag, msg, cls=""):
        try:
            splash.evaluate_js(f"window.onBootMessage({json.dumps({'tag': tag, 'msg': msg, 'cls': cls})})")
        except Exception as e:
            print(f"splash push failed: {e}")

    def boot_sequence():
        push_boot("SYS", "loading saved settings ...")
        time.sleep(0.15)

        push_boot("SYS", "importing blur_sorter.py ...")
        blur, detect = import_logic_modules()

        if blur and detect:
            api.attach_logic(blur, detect)
            push_boot("OK", "backend modules ready", "ok")
            time.sleep(0.25)
            try:
                splash.evaluate_js("window.onBootComplete()")
            except Exception as e:
                print(f"splash complete push failed: {e}")
            time.sleep(0.3)
            window.show()
            splash.destroy()
        else:
            push_boot("ERR", "backend modules failed to import", "err")
            try:
                splash.evaluate_js(
                    "window.onBootError(%s)"
                    % json.dumps({
                        "msg": "backend modules failed to import.",
                        "detail": "check that blur_sorter.py and detection.py are present under /logic "
                                  "and importable, then restart the application. see console output for "
                                  "the full traceback.",
                    })
                )
            except Exception as e:
                print(f"splash error push failed: {e}")
            print("CRITICAL: backend logic modules failed to import. main window will not be shown.")

    def on_splash_loaded():
        threading.Thread(target=boot_sequence, daemon=True).start()

    splash.events.loaded += on_splash_loaded

    webview.start(debug=False)


if __name__ == "__main__":
    main()