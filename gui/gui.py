from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, filedialog, END
from tkinter.font import Font
import os
import threading
import sys
import json
from multiprocessing import freeze_support


def get_base_path():
    """Get the correct base path for both development and frozen (EXE) modes"""
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE - use _MEIPASS for temporary extracted files
        return Path(sys._MEIPASS)
    else:
        # Running as script - gui.py is in gui folder
        return Path(__file__).parent

BASE_PATH = get_base_path()
ASSETS_PATH = BASE_PATH / "assets" / "frame0"

def get_project_root():
    """Get project root (prod folder) for logic imports"""
    if getattr(sys, 'frozen', False):
        # When frozen, logic is in sys._MEIPASS
        return Path(sys._MEIPASS)
    else:
        # When running as script, go up one level from gui folder
        return Path(__file__).parent.parent

def relative_to_assets(path: str) -> Path:
    """Convert relative path to absolute asset path"""
    return ASSETS_PATH / Path(path)

class SettingsManager:
    """Manages saving and loading user settings to/from disk"""
    
    def __init__(self, app_name='FilterPix'):
        # Store settings in user's AppData or home directory
        if sys.platform == 'win32':
            app_data = os.getenv('APPDATA')
            if app_data:
                self.config_dir = Path(app_data) / app_name
            else:
                self.config_dir = Path.home() / f'.{app_name.lower()}'
        else:
            self.config_dir = Path.home() / f'.{app_name.lower()}'
        
        self.config_file = self.config_dir / 'settings.json'
        
    def save(self, settings):
        """Save settings to JSON file"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Settings saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def load(self):
        """Load settings from JSON file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                print(f"Settings loaded from {self.config_file}")
                return settings
            else:
                print("No saved settings found, using defaults")
                return {}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return {}
    
    def get_defaults(self):
        """Return default settings"""
        return {
            'star_enabled': True,
            'laplacian_enabled': True,
            'burst_enabled': True,
            'img_detect_enabled': False,
            'sharpness_level': 0,
            'detection_mode': 'fast',
            'tolerance': 0,
            'last_folder': None
        }


class ImageCache:
    """Cache for PhotoImage objects to prevent garbage collection"""
    def __init__(self):
        self._cache = {}
    
    def load(self, name):
        """Load and cache an image"""
        if name not in self._cache:
            try:
                img_path = relative_to_assets(name)
                if not img_path.exists():
                    print(f"WARNING: Image not found: {img_path}")
                    # Create a placeholder
                    self._cache[name] = PhotoImage(width=1, height=1)
                else:
                    self._cache[name] = PhotoImage(file=str(img_path))
            except Exception as e:
                print(f"Error loading image {name}: {e}")
                self._cache[name] = PhotoImage(width=1, height=1)
        return self._cache[name]
    
    def get(self, name):
        """Get cached image"""
        return self._cache.get(name)

class SplashScreen(tk.Toplevel):
    def __init__(self, root, image_path, min_duration=2000, on_close=None):
        super().__init__(root)

        self.on_close = on_close
        self.overrideredirect(True)
        self.configure(bg="#271B28")

        root.update_idletasks()

        # Convert Path to string and ensure it exists
        image_path = str(image_path)
        loading_path = str(relative_to_assets("loading.png"))

        # Load images with error handling
        try:
            if not Path(image_path).exists():
                print(f"Splash image not found: {image_path}")
            self.image = tk.PhotoImage(file=image_path)
        except Exception as e:
            print(f"Failed to load splash image: {e}")
            # Create minimal splash
            self.image = tk.PhotoImage(width=400, height=300)

        try:
            if not Path(loading_path).exists():
                print(f"Loading image not found: {loading_path}")
            self.loading_img = tk.PhotoImage(file=loading_path)
        except Exception as e:
            print(f"Failed to load loading image: {e}")
            self.loading_img = tk.PhotoImage(width=50, height=50)

        # Set geometry
        width = self.image.width()
        height = self.image.height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Place images
        self.splash_label = tk.Label(self, image=self.image, bg="#271B28", borderwidth=0)
        self.splash_label.pack()

        self.loading_label = tk.Label(self, image=self.loading_img, bg="#271B28", borderwidth=0)
        self.loading_label.place(relx=0.5, rely=0.85, anchor="n")

        self.update_idletasks()

        self.min_time_passed = False
        self.app_ready = False
        self.after(min_duration, self._min_time_reached)

    def _min_time_reached(self):
        self.min_time_passed = True
        self._try_close()

    def mark_app_ready(self):
        self.app_ready = True
        self._try_close()

    def _try_close(self):
        if self.min_time_passed and self.app_ready:
            self.destroy()
            if self.on_close:
                self.on_close()


class MainApp:
    def __init__(self, root):
        self.root = root
        self.folder_path = None
        self.output_directory = None
        self._setup_window()
        self._init_state()
        self._init_frames()
        self._setup_ui()
        self._load_saved_settings()
        self.show_home()

    def _setup_window(self):
        """Configure main window properties"""
        self.root.geometry("720x480")
        self.root.configure(bg="#271B28")
        self.root.resizable(False, False)
        self.root.title("FilterPix")
        
        # Icon handling with better error checking
        try:
            icon_path = relative_to_assets("filterpix.ico")
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception as e:
            print(f"Could not set icon: {e}")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _init_state(self):
        """Initialize application state variables"""
        self.settings_mgr = SettingsManager()
        
        # Threading
        self.sorter_thread = None
        self.detection_thread = None
        self.sorter_cancel_event = threading.Event()
        self.detection_cancel_event = threading.Event()
        
        # Processing state
        self.is_processing = False
        self.cancelled = False
        self.solo_detection = False
        
        # Settings
        self.default_output_enabled = True
        self.star_enabled = True
        self.laplacian_enabled = True
        self.burst_enabled = True
        self.img_detect_enabled = False
        self.sharpness_level = 0
        self.tolerance_comp_val = 0
        self.detection_mode = "fast"
        
        # Animation
        self.dot_count = 0
        self.animation_id = None
        
        # Statistics
        self.sorter_stats = None
        self.detection_stats = None
        
        # Image cache
        self.images = ImageCache()
        
        self._settings_initialized = False

    def _init_frames(self):
        """Create main frames"""
        self.home_frame = tk.Frame(self.root, bg="#271B28")
        self.settings_frame = tk.Frame(self.root, bg="#271B28")

    def _setup_ui(self):
        """Setup all UI components"""
        self.setup_homescreen()

    def _load_saved_settings(self):
        """Load previously saved settings from disk"""
        saved_settings = self.settings_mgr.load()
        
        self.star_enabled = saved_settings.get('star_enabled', True)
        self.laplacian_enabled = saved_settings.get('laplacian_enabled', True)
        self.burst_enabled = saved_settings.get('burst_enabled', True)
        self.img_detect_enabled = saved_settings.get('img_detect_enabled', False)
        self.sharpness_level = saved_settings.get('sharpness_level', 0)
        self.tolerance_comp_val = saved_settings.get('tolerance', 0)
        self.detection_mode = saved_settings.get('detection_mode', 'fast')
        
    def _save_current_settings(self):
        """Save current settings to disk"""
        settings = {
            'star_enabled': self.star_enabled,
            'laplacian_enabled': self.laplacian_enabled,
            'burst_enabled': self.burst_enabled,
            'img_detect_enabled': self.img_detect_enabled,
            'sharpness_level': self.sharpness_level,
            'detection_mode': self.detection_mode,
            'tolerance': self._get_tolerance_value(),
            'last_folder': self.folder_path
        }
        self.settings_mgr.save(settings)
    
    def _on_closing(self):
        """Called when window is closing - save settings"""
        self._save_current_settings()
        self.root.destroy()

    # ===============================
    # Animation System
    # ===============================
    def animate_processing_dots(self):
        """Animates the dots after 'Processing' text"""
        if not self.is_processing:
            self.animation_id = None
            return

        self.dot_count = (self.dot_count % 3) + 1
        dots = "." * self.dot_count
        progress_text = f"Processing{dots}"
        self.canvas.itemconfig(self.processing_text, text=progress_text)
        self.canvas.lift(self.processing_text)

        self.animation_id = self.root.after(500, self.animate_processing_dots)

    def start_animation(self):
        """Start the animated dots"""
        if self.animation_id:
            try:
                self.root.after_cancel(self.animation_id)
            except Exception:
                pass
            self.animation_id = None

        self.dot_count = 0
        self.canvas.lift(self.processing_text)
        self.animation_id = self.root.after(0, self.animate_processing_dots)

    def stop_animation(self):
        """Stop the animated dots"""
        if self.animation_id:
            try:
                self.root.after_cancel(self.animation_id)
            except Exception:
                pass
            self.animation_id = None

    # ===============================
    # Frame Navigation
    # ===============================
    def show_home(self):
        self.settings_frame.place_forget()
        self.home_frame.place(x=0, y=0, width=720, height=480)

    def show_settings(self):
        if not self._settings_initialized:
            print("Initializing settings screen (lazy load)...")
            self.setup_settings()
            self._apply_current_settings_to_ui()
            self._settings_initialized = True
        
        self.home_frame.place_forget()
        self.settings_frame.place(x=0, y=0, width=720, height=480)
    
    def _apply_current_settings_to_ui(self):
        """Apply current settings values to UI widgets"""
        self._toggle_button_state(self.star_button, self.star_enabled)
        self._toggle_button_state(self.laplacian_button, self.laplacian_enabled)
        self._toggle_button_state(self.burst_button, self.burst_enabled)
        self._toggle_button_state(self.img_detection_button, self.img_detect_enabled)
        
        self.set_blur_level(self.sharpness_level)
        
        self.tolerance_comp.delete(0, END)
        if self.tolerance_comp_val != 0:
            self.tolerance_comp.insert(0, str(self.tolerance_comp_val))
        
        if self.detection_mode == "fast":
            self.fast_button.config(image=self.images.get("fast_on.png"))
            self.accurate_button.config(image=self.images.get("acc_off.png"))
        else:
            self.fast_button.config(image=self.images.get("fast_off.png"))
            self.accurate_button.config(image=self.images.get("acc_on.png"))
        
        self._update_minor_buttons_ui()

    def back_clicked(self):
        self._save_current_settings()
        self.show_home()

    # ===============================
    # UI Component Builders
    # ===============================
    def _create_canvas(self, parent):
        """Create a standard canvas for the app"""
        canvas = Canvas(
            parent,
            bg="#271B28",
            height=480,
            width=720,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        canvas.place(x=0, y=0)
        return canvas

    def _create_sidebar(self, canvas):
        """Create left sidebar with logo and branding"""
        canvas.create_image(62.0, 240.0, image=self.images.load("rect.png"))
        canvas.create_image(59.67, 51.78, image=self.images.load("logo.png"))
        canvas.create_image(62.0, 102.0, image=self.images.load("filterpix.png"))

    def _create_button(self, parent, image_name, x, y, width, height, command, bg="#352237"):
        """Create a standard button with image"""
        img = self.images.load(image_name)
        button = Button(
            parent,
            image=img,
            borderwidth=0,
            highlightthickness=0,
            command=command,
            relief="flat",
            bg=bg,
            activebackground=bg
        )
        button.place(x=x, y=y, width=width, height=height)
        return button

    def _create_toggle_button(self, parent, x, y, command, initial_state=True):
        """Create a toggle button (on/off)"""
        img = self.images.get("on.png" if initial_state else "off.png")
        button = Button(
            parent,
            image=img,
            borderwidth=0,
            highlightthickness=0,
            command=command,
            relief="flat",
            bg="#332632",
            activebackground="#332632"
        )
        button.place(x=x, y=y, width=44.0, height=24.0)
        return button

    # ===============================
    # UI Setup: Home Screen
    # ===============================
    def setup_homescreen(self):
        self.canvas = self._create_canvas(self.home_frame)
        self._create_sidebar(self.canvas)
        
        self.canvas.create_image(60.0, 162.0, image=self.images.load("sort_on.png"))
        
        self._create_button(self.home_frame, "settings_off.png", 32.0, 193.0, 59.0, 17.0,
                        self.settings_clicked, bg="#241822")

        BOX_1_Y = 100.0
        BOX_2_Y = 245.0
        BOX_3_Y = 380.0
        
        self.canvas.create_image(420.0, BOX_1_Y, image=self.images.load("box_1.png"))

        self.canvas.create_text(156.0, 25.0, anchor="nw", text="Folder: ",
                            fill="#D9D9D9", font=("Segoe UI", 15, "bold"))
        
        self.folder_display = self.canvas.create_text(223.0, 25.0, anchor="nw", text="None",
                                                    fill="#D9D9D9", font=("Segoe UI", 15, "bold"))

        self.input_button = self._create_button(self.home_frame, "input_button.png",
                                                156.0, 65.0, 261.0, 45.0, self.input_button_clicked)
        self.folder_button = self._create_button(self.home_frame, "folder_button.png",
                                                422.0, 65.0, 261.0, 45.0, self.folder_clicked)
        self.cancel_button = self._create_button(self.home_frame, "cancel.png",
                                                156.0, 118.0, 528.0, 50.0, self.cancel_clicked)
        self.start_button = self._create_button(self.home_frame, "start.png",
                                            156.0, 118.0, 528.0, 50.0, self.start_clicked)

        self.processing_text = self.canvas.create_text(156.0, 236, anchor="nw", text="",
                                                    fill="#FFFFFF", font=("Segoe UI", 27, "bold"))
        self.label_1 = self.canvas.create_text(156.0, 198.0, anchor="nw", text="",
                                            fill="#D9D9D9", font=("Segoe UI", 12, "bold italic"))
        
        self.process_report()
        self.toggle_process_report(False)

    # ===============================
    # UI Setup: Settings Screen
    # ===============================
    def setup_settings(self):
        self.settings_canvas = self._create_canvas(self.settings_frame)
        self._create_sidebar(self.settings_canvas)
        
        self._setup_custom_folder()

        self._create_button(self.settings_frame, "sort_off.png", 43.0, 154.0, 39.0, 17.0,
                           self.back_clicked, bg="#241822")
        self.settings_canvas.create_image(62.0, 200.0, image=self.images.load("settings_on.png"))

        self.settings_canvas.create_image(229.0, 28.0, image=self.images.load("output.png"))
        self.settings_canvas.create_image(229.0, 114.0, image=self.images.load("filter_options.png"))
        self.settings_canvas.create_image(228.0, 318.0, image=self.images.load("sort_settings.png"))

        self.settings_canvas.create_image(418.0, 71.0, image=self.images.load("settings_1.png"))
        self.settings_canvas.create_image(420.0, 216.0, image=self.images.load("settings_2.png"))
        self.settings_canvas.create_image(420.0, 372.0, image=self.images.load("settings_3.png"))

        self.images.load("on.png")
        self.images.load("off.png")

        self.star_button = self._create_toggle_button(self.settings_frame, 645.0, 140.0,
                                                      self.star_clicked, True)
        self.laplacian_button = self._create_toggle_button(self.settings_frame, 645.0, 172.0,
                                                           self.laplacian_clicked, True)
        self.burst_button = self._create_toggle_button(self.settings_frame, 645.0, 204.0,
                                                       self.burst_clicked, True)

        self._setup_sharpness_buttons()
        self._setup_entry_fields()

        self.img_detection_button = self._create_toggle_button(self.settings_frame, 645.0, 344.0,
                                                               self.img_detection_clicked, False)
        
        self._setup_detection_mode_buttons()

    def _setup_sharpness_buttons(self):
        """Setup high/med/low sharpness buttons"""
        self.images.load("high_off.png")
        self.images.load("high_on.png")
        self.images.load("med_off.png")
        self.images.load("med_on.png")
        self.images.load("low_off.png")
        self.images.load("low_on.png")

        self.high_button = self._create_button(self.settings_frame, "high_off.png",
                                               564.0, 237.0, 40.0, 24.0, self.high_clicked, bg="#332632")
        self.med_button = self._create_button(self.settings_frame, "med_on.png",
                                              607.0, 237.0, 40.0, 24.0, self.med_clicked, bg="#332632")
        self.low_button = self._create_button(self.settings_frame, "low_off.png",
                                              649.0, 237.0, 40.0, 24.0, self.low_clicked, bg="#332632")

    def _setup_detection_mode_buttons(self):
        """Setup fast/accurate detection mode buttons"""
        self.images.load("fast_off.png")
        self.images.load("fast_on.png")
        self.images.load("acc_off.png")
        self.images.load("acc_on.png")

        self.fast_button = self._create_button(self.settings_frame, "fast_on.png",
                                               565.0, 376.0, 44.0, 24.0, self.fast_clicked, bg="#332632")
        self.accurate_button = self._create_button(self.settings_frame, "acc_off.png",
                                                   612.0, 376.0, 77.0, 24.0, self.accurate_clicked, bg="#332632")
    
    def _setup_custom_folder(self):
        self.images.load("def_off.png")
        self.images.load("def_on.png")
        self.images.load("cus_off.png")
        self.images.load("cus_on.png")
        
        self.custom_folder_button = self._create_button(self.settings_frame, "cus_off.png",
                                               625.0, 59.0, 63.0, 24.0, self.output_clicked, bg="#332632")
        self.default_folder_button = self._create_button(self.settings_frame, "def_on.png",
                                               563.0, 59.0, 59.0, 24.0, self.default_clicked, bg="#332632")

    def _setup_entry_fields(self):
        """Setup input entry fields"""
        self.settings_canvas.create_image(648.0, 279.5, image=self.images.load("entry.png"))
        self.tolerance_comp = Entry(
            self.settings_frame, bd=0, bg="#493C48", fg="#D9D9D9",
            highlightthickness=0,
            justify="center", insertbackground="#D9D9D9",
            font=("Segoe UI", 15, "bold")
        )
        self.tolerance_comp.place(x=617.0, y=268.0, width=66.0, height=21.0)

    # ===============================
    # Toggle Settings Handlers
    # ===============================
    def _update_minor_buttons_ui(self):
        """Sync Detection + Laplacian dependent UI states"""
        if hasattr(self, "fast_button"):
            if self.img_detect_enabled:
                self.fast_button.config(state="normal", relief="flat")
                self.accurate_button.config(state="normal", relief="flat")

                if self.detection_mode == "fast":
                    self.fast_button.config(image=self.images.get("fast_on.png"))
                    self.accurate_button.config(image=self.images.get("acc_off.png"))
                else:
                    self.fast_button.config(image=self.images.get("fast_off.png"))
                    self.accurate_button.config(image=self.images.get("acc_on.png"))
            else:
                self.fast_button.config(state="disabled", relief="sunken")
                self.accurate_button.config(state="disabled", relief="sunken")

        if hasattr(self, "high_button"):
            if self.laplacian_enabled:
                self.high_button.config(state="normal", relief="flat")
                self.med_button.config(state="normal", relief="flat")
                self.low_button.config(state="normal", relief="flat")
                self.set_blur_level(self.sharpness_level)
            else:
                self.high_button.config(state="disabled", relief="sunken")
                self.med_button.config(state="disabled", relief="sunken")
                self.low_button.config(state="disabled", relief="sunken")

            if hasattr(self, "tolerance_comp"):
                self.tolerance_comp.delete(0, END)
                if self.laplacian_enabled:
                    self.tolerance_comp.insert(0, str(self.tolerance_comp_val))
                
    def _toggle_button_state(self, button, state):
        """Update button image based on state"""
        img = self.images.get("on.png" if state else "off.png")
        button.config(image=img)
        
    def laplacian_clicked(self):
        self.laplacian_enabled = not self.laplacian_enabled
        self._toggle_button_state(self.laplacian_button, self.laplacian_enabled)
        self._update_minor_buttons_ui()
        self._save_current_settings()
        
    def burst_clicked(self):
        self.burst_enabled = not self.burst_enabled
        self._toggle_button_state(self.burst_button, self.burst_enabled)
        self._save_current_settings()
        
    def star_clicked(self):
        self.star_enabled = not self.star_enabled
        self._toggle_button_state(self.star_button, self.star_enabled)
        self._save_current_settings()

    def img_detection_clicked(self):
        self.img_detect_enabled = not self.img_detect_enabled
        self._toggle_button_state(self.img_detection_button, self.img_detect_enabled)
        self._update_minor_buttons_ui()
        self._save_current_settings()
            
    def set_blur_level(self, level):
        """Update sharpness level and button states"""
        self.sharpness_level = int(level)
        self.high_button.config(image=self.images.get("high_on.png" if level == 30 else "high_off.png"))
        self.med_button.config(image=self.images.get("med_on.png" if level == 0 else "med_off.png"))
        self.low_button.config(image=self.images.get("low_on.png" if level == -20 else "low_off.png"))

    def low_clicked(self):
        self.set_blur_level(-20)
        self._save_current_settings()

    def med_clicked(self):
        self.set_blur_level(0)
        self._save_current_settings()

    def high_clicked(self):
        self.set_blur_level(30)
        self._save_current_settings()
        
    def output_clicked(self):
        self.output_directory = filedialog.askdirectory(title="Select a Folder")
        if self.output_directory:
            self.custom_folder_button.config(image=self.images.get("cus_on.png"))
            self.default_folder_button.config(image=self.images.get("def_off.png"))
            self._save_current_settings()
        
    def default_clicked(self):
        self.output_directory = self.folder_path
        self.default_folder_button.config(image=self.images.get("def_on.png"))
        self.custom_folder_button.config(image=self.images.get("cus_off.png"))
        self._save_current_settings()
        
    def fast_clicked(self):
        self.detection_mode = "fast"
        self.fast_button.config(image=self.images.get("fast_on.png"))
        self.accurate_button.config(image=self.images.get("acc_off.png"))
        self._save_current_settings()
        
    def accurate_clicked(self):
        self.detection_mode = "accurate"
        self.accurate_button.config(image=self.images.get("acc_on.png"))
        self.fast_button.config(image=self.images.get("fast_off.png"))
        self._save_current_settings()

    # ===============================
    # Main Button Event Handlers
    # ===============================
    def input_button_clicked(self):
        """Show input dialog for folder path"""
        folder = filedialog.askdirectory(title="Select a Folder")
        if not folder:
            print("No folder selected.")
            return
        self.folder_path = folder
        
        display_text = (
            self.folder_path if len(self.folder_path) < 50
            else self.folder_path[:45] + "..."
        )
        self.canvas.itemconfig(self.folder_display, text=display_text)
        print("Selected folder:", self.folder_path)
        
        self._save_current_settings()

    def folder_clicked(self):
        """Opens the selected folder."""
        try:
            if self.folder_path and os.path.exists(self.folder_path):
                if sys.platform == 'win32':
                    os.startfile(self.folder_path)
                elif sys.platform == 'darwin':
                    os.system(f'open "{self.folder_path}"')
                else:
                    os.system(f'xdg-open "{self.folder_path}"')
            else:
                print(f"Input folder does not exist: {self.folder_path}")
        except Exception as e:
            print(f"Error opening folder: {e}")
            
    def settings_clicked(self):
        self.show_settings()

    def cancel_clicked(self):
        """Cancel ongoing processing"""
        if not self.is_processing:
            return
            
        self.is_processing = False
        self.cancelled = True
        
        self.stop_animation()
        self._update_status("Cancelling...")
        
        self.sorter_cancel_event.set()
        self.detection_cancel_event.set()
            
        self._update_status("Processing cancelled.")
        self.root.after(100, self._check_thread_finished)
        self.start_button.lift()
        
    def _check_thread_finished(self):
        """Poll until processing threads are finished"""
        if ((self.sorter_thread and self.sorter_thread.is_alive()) or 
            (self.detection_thread and self.detection_thread.is_alive())):
            self.root.after(100, self._check_thread_finished)
        else:
            print("Processing thread finished\n")

    # ===============================
    # Processing Logic
    # ===============================
    def start_clicked(self):
        """Start the processing workflow"""
        if self.is_processing:
            return
        
        self.toggle_process_report(False)
            
        if not self.folder_path:
            self._show_error("Please select a folder first.")
            return
        
        tolerance = self._get_tolerance_value()
        self.sorter_cancel_event.clear()
        
        options = {
            "folder": self.folder_path,
            "output": self.output_directory,
            "base_blur": self.sharpness_level,
            "tolerance": tolerance,
            "use_starcheck": self.star_enabled,
            "use_laplaciancheck": self.laplacian_enabled,
            "group_bursts": self.burst_enabled,
            "cancel_flag": self.sorter_cancel_event,
        }

        self.canvas.create_image(420.0, 245.0, image=self.images.load("box_2.png"))
        self.canvas.itemconfig(self.label_1, text="Processing with:")
        self.canvas.lift(self.label_1)
        
        self.is_processing = True
        self.cancel_button.lift()
        
        if not self.laplacian_enabled and not self.star_enabled and not self.burst_enabled:
            self._update_status("Starting detection...")
            self.solo_detection = True
            self.root.after(100, self.start_detection)
        else:
            self.start_animation()
            self.sorter_thread = threading.Thread(target=self.run_sorter, args=(options,))
            self.sorter_thread.daemon = True
            self.sorter_thread.start()
            self.root.after(100, self._check_sorter_done)

    def _get_tolerance_value(self):
        """Get tolerance value from entry field and update stored value"""
        if not hasattr(self, 'tolerance_comp'):
            return self.tolerance_comp_val
        
        entry_text = self.tolerance_comp.get().strip()
        if entry_text:
            try:
                self.tolerance_comp_val = int(entry_text)
            except ValueError:
                pass
        return self.tolerance_comp_val

    def _show_error(self, message):
        """Display error message"""
        self.canvas.create_image(420.0, 245.0, image=self.images.load("box_2.png"))
        self.canvas.itemconfig(self.label_1, text="Error")
        self.canvas.lift(self.label_1)
        self._update_status(message)

    def _update_status(self, text):
        """Update processing status text"""
        self.canvas.itemconfig(self.processing_text, text=text)
        self.canvas.lift(self.processing_text)

    def _check_sorter_done(self):
        """Check if sorting is complete"""
        if self.sorter_thread and self.sorter_thread.is_alive():
            self.root.after(100, self._check_sorter_done)
        else:
            self.stop_animation()
            if self.img_detect_enabled and self.is_processing:
                self._update_status("Filtering Complete")
                self.root.after(500, self.start_detection)
            else:
                self.finish_processing()

    def start_detection(self):
        """Start image detection process"""
        if not self.is_processing:
            return
            
        self._update_status("Starting detection...")
        self.detection_cancel_event.clear()
        
        detection_options = {
            "folder": self.folder_path,
            "output": self.output_directory,
            "mode": self.detection_mode,
            "solo_process": self.solo_detection,
            "cancel_flag": self.detection_cancel_event,
            "progress_callback": self.detection_progress_callback,
        }
        
        self.detection_thread = threading.Thread(target=self.run_detection, args=(detection_options,))
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.root.after(100, self._check_detection_done)

    def detection_progress_callback(self, current, total):
        """Callback for detection progress updates"""
        if self.is_processing:
            progress_text = f"Detection: {current}/{total}"
            self.root.after(0, lambda: self._update_status(progress_text))

    def _check_detection_done(self):
        """Check if detection is complete"""
        if self.detection_thread and self.detection_thread.is_alive():
            self.root.after(100, self._check_detection_done)
        else:
            self.finish_processing()

    def finish_processing(self):
        """Finalize processing and update UI with real statistics"""
        self.stop_animation()
        self.is_processing = False
        self.start_button.lift()
        
        final_stats = self.get_final_stats()
        
        if self.cancelled:
            self._update_status("Process cancelled")
            self.canvas.itemconfig(self.label_1, text="Error")
            self.cancelled = False
        else:
            self._update_status("Process complete")
            self.canvas.itemconfig(self.label_1, text="Done")
            if final_stats:
                self.update_process_report_with_stats(final_stats)
            else:
                self.update_process_report_with_stats({})
            self.toggle_process_report(True)


    def get_final_stats(self):
        sorter = self.sorter_stats or {}
        detection = self.detection_stats or {}

        # Check if this was a detection-only run
        if self.solo_detection:
            total_images = (detection.get("total_images")
                or detection.get("images_processed")
                or detection.get("total")
                or 0
            )
            selected_images = (detection.get("selected_images")
                or detection.get("rated_images")
                or detection.get("final_selection")
                or 0
            )
            detection_time = float(detection.get("elapsed_time", 0) or 0)
            
            return {
                "total_images": total_images,
                "final_selection": selected_images,
                "elapsed_time": detection_time,
                "detection_only": True
            }
        
        # Regular filtering or combined run
        total_images = (sorter.get("total_images")
            or sorter.get("images_processed")
            or 0
        )

        selected_images = (sorter.get("final_selection")
            or sorter.get("sharp_images")
            or 0
        )

        sorter_time = float(sorter.get("elapsed_time", 0) or 0)
        detection_time = float(detection.get("elapsed_time", 0) or 0)

        return {
            "total_images": total_images,
            "final_selection": selected_images,
            "elapsed_time": sorter_time + detection_time,
            "detection_only": False
        }

    def run_sorter(self, options):
        """Run blur sorter in thread and capture statistics"""
        try:
            self.sorter_stats = self.blur.main(**options) or {}
        except Exception as e:
            print(f"Error in sorter: {e}")
            self.sorter_stats = {}

    def run_detection(self, detection_options):
        """Run detection in thread and capture statistics"""
        try:
            self.detection_stats = self.detect.main(**detection_options) or {}
        except Exception as e:
            print(f"Error in detection: {e}")
            self.detection_stats = {}

    # ===============================
    # Process Report
    # ===============================
    def process_report(self):
        """Display process report with stats"""
        self.box_3 = self.canvas.create_image(420.0, 380.0, image=self.images.load("box_3.png"))
        
        self.label_2 = self.canvas.create_text(156.0, 316.0, anchor="nw", text="Process report:",
                                            fill="#D9D9D9", font=("Segoe UI", 12, "bold italic"))

        self.stats_images_processed = self.canvas.create_text(156.0, 344.0, anchor="nw",
                                                            text="Images Processed: 0",
                                                            fill="#FFFFFF", font=("Segoe UI", 15, "bold italic"))
        self.stats_sharp_images = self.canvas.create_text(156.0, 369.0, anchor="nw",
                                                        text="Selected Images: 0",
                                                        fill="#FFFFFF", font=("Segoe UI", 15, "bold italic"))
        self.stats_sorted_images = self.canvas.create_text(156.0, 394.0, anchor="nw",
                                                        text="Discarded Images: 0",
                                                        fill="#FFFFFF", font=("Segoe UI", 15, "bold italic"))
        self.stats_time_elapsed = self.canvas.create_text(156.0, 419.0, anchor="nw",
                                                        text="Time Elapsed: 0 mins",
                                                        fill="#FFFFFF", font=("Segoe UI", 15, "bold italic"))

    def toggle_process_report(self, visible: bool):
        """Show or hide the process report section"""
        state = "normal" if visible else "hidden"
        
        report_items = [
            "box_3", "label_2", "stats_images_processed",
            "stats_sharp_images", "stats_sorted_images", "stats_time_elapsed"
        ]
        
        for item_name in report_items:
            item = getattr(self, item_name, None)
            if item is not None:
                try:
                    self.canvas.itemconfigure(item, state=state)
                except Exception as e:
                    print(f"Could not toggle item {item_name}: {e}")
                    
    def update_process_report_with_stats(self, stats):
        """Update the canvas text items with the latest processing stats"""
        total_images = stats.get('total_images') or stats.get('total') or stats.get('images_processed') or 0
        selected_images = stats.get('final_selection') or stats.get('sharp_images') or stats.get('rated_images') or 0
        elapsed_seconds = stats.get('elapsed_time') or stats.get('elapsed') or 0

        self.canvas.itemconfig(
            self.stats_images_processed,
            text=f"Images Processed: {total_images}"
        )
        
        self.canvas.itemconfig(
            self.stats_sharp_images,
            text=f"Selected Images: {selected_images}"
        )
        
        self.canvas.itemconfig(
            self.stats_sorted_images,
            text=f"Rejected Images: {total_images - selected_images}"
        )
        
        if elapsed_seconds is None:
            elapsed_seconds = 0
        try:
            elapsed_seconds = float(elapsed_seconds)
        except Exception:
            elapsed_seconds = 0

        if elapsed_seconds < 60:
            elapsed_str = f"{elapsed_seconds:.2f} seconds"
        else:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            elapsed_str = f"{minutes} mins {seconds} secs"
        
        self.canvas.itemconfig(
            self.stats_time_elapsed,
            text=f"Time Elapsed: {elapsed_str}"
        )


def show_error_and_exit(root):
    """Show error dialog and exit"""
    from tkinter import messagebox
    root.withdraw()
    messagebox.showerror(
        "Startup Error",
        "Failed to load required modules.\n\n"
        "Please ensure all files are properly installed."
    )
    root.destroy()
    sys.exit(1)


def import_logic_modules():
    """Safely import logic modules with proper path handling"""
    try:

        project_root = get_project_root()
        logic_path = project_root / "logic"
        

        logic_str = str(logic_path)
        if logic_str not in sys.path:
            sys.path.insert(0, logic_str)
        

        root_str = str(project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        
        print(f"Project root: {project_root}")
        print(f"Logic path: {logic_path}")
        print(f"Logic path exists: {logic_path.exists()}")
        
        # Import modules
        import blur_sorter as blur
        import detection as detect
        
        print("✓ Modules imported successfully")
        return blur, detect
        
    except Exception as e:
        print(f"✗ Error importing logic modules: {e}")
        print(f"sys.path: {sys.path}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    freeze_support()

    root = Tk()
    root.withdraw()

    # Show splash first
    splash = SplashScreen(
        root,
        image_path=relative_to_assets("splashscreen.png"),
        min_duration=2000
    )

    def init_main_app():
        """Initialize app in background thread"""
        # Import modules
        blur, detect = import_logic_modules()
        
        if blur is None or detect is None:
            print("CRITICAL ERROR: Could not import required modules!")
            # Show error and exit
            root.after(0, lambda: show_error_and_exit(root))
            return

        def show_main():
            """Show main app window"""
            try:
                root.deiconify()
                root.attributes('-topmost', True)
                app = MainApp(root)
                app.blur = blur
                app.detect = detect
                root.after(1, lambda: root.attributes('-topmost', False))
            except Exception as e:
                print(f"Error initializing main app: {e}")
                import traceback
                traceback.print_exc()

        splash.on_close = show_main
        splash.mark_app_ready()

    # Start initialization
    threading.Thread(target=init_main_app, daemon=True).start()

    root.mainloop()