from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, END
from tkinter.scrolledtext import ScrolledText
import types
import os
import threading
import sys
import multiprocessing
from multiprocessing import freeze_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "logic"))
import blur_sorter as blur
import detection as detect

if getattr(sys, 'frozen', False):
    BASE_PATH = Path(sys._MEIPASS)
else:
    BASE_PATH = Path(__file__).parent

ASSETS_PATH = BASE_PATH / "assets" / "frame0"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def redirect_stdout(widget):
    def write(s):
        widget.insert(END, s)
        widget.see(END)
        widget.update_idletasks()
    def flush():
        pass
    return types.SimpleNamespace(write=write, flush=flush)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("720x480")
        self.root.configure(bg="#000000")
        self.root.resizable(False, False)
        self.root.title("Image Culler")
        self.root.iconbitmap(str(relative_to_assets("Rei.ico")))
        self.sorter_thread = None
        self.detection_thread = None
        self.is_processing = False
        
        # CREATE MANAGER ONCE during init to avoid freezing on button click
        self.mp_manager = multiprocessing.Manager()
        self.sorter_cancel_flag = None
        self.detection_cancel_flag = None
        
        self.star_enabled = True
        self.laplacian_enabled = True
        self.burst_enabled = True
        self.img_detect_enabled = False
        self.solo_detection = False
        self.sharpness_level = 0
        self.detection_mode = "fast"
        
        # Animation variables
        self.dot_count = 0
        self.animation_id = None

        self.home_frame = tk.Frame(self.root, bg="white")
        self.settings_frame = tk.Frame(self.root, bg="black")        

        self.setup_homescreen()
        self.setup_settings()
        self.show_home()

    # ===============================
    # Animated Dots Helper
    # ===============================
    def animate_processing_dots(self):
        """Animates the dots after 'Processing Images' text"""
        if self.is_processing and self.animation_id is not None:
            self.dot_count = (self.dot_count % 3) + 1
            dots = "." * self.dot_count
            self.canvas.itemconfig(self.processing_text, text=f"Processing Images{dots}")
            self.animation_id = self.root.after(500, self.animate_processing_dots)
    
    def start_animation(self, text="Processing Images"):
        """Start the animated dots"""
        self.dot_count = 0
        self.canvas.itemconfig(self.processing_text, text=text)
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        self.animation_id = self.root.after(500, self.animate_processing_dots)
    
    def stop_animation(self):
        """Stop the animated dots"""
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

    # ===============================
    # Frame Navigation
    # ===============================
    def show_home(self):
        self.settings_frame.place_forget()
        self.home_frame.place(x=0, y=0, width=720, height=480)

    def show_settings(self):
        self.home_frame.place_forget()
        self.settings_frame.place(x=0, y=0, width=720, height=480)

    def back_clicked(self):
        self.canvas.itemconfig(self.processing_text, text="")    
        self.show_home()

    # ===============================
    # UI Setup: Home and Settings
    # ===============================
    def setup_homescreen(self):
        self.canvas = Canvas(self.home_frame, bg="#252827", height=480, width=720, bd=0, highlightthickness=0, relief="ridge")
        self.canvas.place(x=0, y=0)

        self.canvas.create_text(8.0, 455.0, anchor="nw", text="Version 1.2.0", fill="#D9D9D9", font=("Inter ExtraLightItalic", 16))
        self.canvas.create_text(22.0, 20.0, anchor="nw", text="Folder Directory:", fill="#FFFFFF", font=("Inter", 30))

        entry_image_1 = PhotoImage(file=str(relative_to_assets("entry_1.png")))
        self.canvas.create_image(186.0, 85.0, image=entry_image_1)
        self.entry_image_1 = entry_image_1  
        self.entry_1 = Entry(self.home_frame, font=("Helvetica", 20), bd=0, bg="#1E1E1E", fg="#D9D9D9", highlightthickness=0)
        self.entry_1.place(x=25.0, y=68.0, width=322.0, height=34.0)

        cancel_button_image = PhotoImage(file=str(relative_to_assets("button_3.png")))
        self.cancel_button = Button(self.home_frame, image=cancel_button_image, borderwidth=0, highlightthickness=0,
                                    command=self.cancel_clicked, relief="flat")
        self.cancel_button.image = cancel_button_image
        self.cancel_button.place(x=22.0, y=115.0, width=100.0, height=40.0)

        button_image_1 = PhotoImage(file=str(relative_to_assets("button_1.png")))
        self.button_1 = Button(self.home_frame, image=button_image_1, borderwidth=0, highlightthickness=0,
                               command=self.start_clicked, relief="flat")
        self.button_1.image = button_image_1  
        self.button_1.place(x=22.0, y=115.0, width=100.0, height=40.0)

        button_image_2 = PhotoImage(file=str(relative_to_assets("button_2.png")))
        self.button_2 = Button(self.home_frame, image=button_image_2, borderwidth=0, highlightthickness=0,
                               command=self.folder_clicked, relief="flat")
        self.button_2.image = button_image_2  
        self.button_2.place(x=127.0, y=115.0, width=100.0, height=40.0)

        settings_image = PhotoImage(file=str(relative_to_assets("settings.png")))
        self.settings_button = Button(self.home_frame, image=settings_image, borderwidth=0, highlightthickness=0,
                                      command=self.settings_clicked, relief="flat")
        self.settings_button.image = settings_image
        self.settings_button.place(x=232.0, y=115.0, width=118.0, height=40.0)

        self.processing_text = self.canvas.create_text(22.0, 165.0, anchor="nw", text="", fill="#D9D9D9", font=("Inter", 26))
        self.error_text_1 = self.canvas.create_text(22.0, 200.0, anchor="nw", text="", fill="#D9D9D9", font=("Inter", 22))
        self.error_text_2 = self.canvas.create_text(22.0, 222.0, anchor="nw", text="", fill="#D9D9D9", font=("Inter", 22))

        self.output_box = ScrolledText(self.home_frame, font=("tkfont", 12), bg="#252827", fg="#D9D9D9",
                                       insertbackground="#D9D9D9", wrap="word", borderwidth=0)

    def setup_settings(self):
        self.settings_canvas = Canvas(self.settings_frame, bg="#252827", height=480, width=720, bd=0, highlightthickness=0, relief="ridge")
        self.settings_canvas.place(x=0, y=0)

        back_image = PhotoImage(file=relative_to_assets("Back.png"))
        self.back_button = Button(self.settings_frame, image=back_image, borderwidth=0, highlightthickness=0,
                                command=self.back_clicked, relief="flat")
        self.back_button.image = back_image
        self.back_button.place(x=8, y=406.0, width=100.0, height=40.0)

        # Grid layout parameters
        row_height = 38
        start_y = 25
        label_x = 19
        control_x = 310

        # Create table structure with vertical and horizontal lines
        table_left = 10
        table_right = 710
        table_top = start_y - 10
        table_bottom = start_y + (row_height * 10) -10
        separator_x = 300  # Vertical line between labels and controls

        # Outer table border
        self.settings_canvas.create_rectangle(table_left, table_top, table_right, table_bottom, fill="", outline="#555555", width=2)
        
        # Vertical separator between labels and controls
        self.settings_canvas.create_line(separator_x, table_top, separator_x, table_bottom, fill="#555555", width=1)
        
        # Horizontal lines for each row
        for i in range(11):  # 0 to 10 for 11 horizontal lines
            y_pos = table_top + (i * row_height)
            self.settings_canvas.create_line(table_left, y_pos, table_right, y_pos, fill="#555555", width=1)

        # Row labels - CORRECTED ORDER
        self.settings_canvas.create_text(label_x, start_y, anchor="nw", text="Rating Filter:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height, anchor="nw", text="Sharpness Filter:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height * 2, anchor="nw", text="Burst Selection:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height * 3, anchor="nw", text="Sharpness Threshold:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height * 4, anchor="nw", text="Threshold Compensation:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height * 5, anchor="nw", text="Image Detection:", fill="#D9D9D9", font=("Inter", 18))
        self.settings_canvas.create_text(label_x, start_y + row_height * 6, anchor="nw", text="Detection Mode:", fill="#D9D9D9", font=("Inter", 18))

        # Load images
        self.low_image = PhotoImage(file=str(relative_to_assets("Low.png")))
        self.low_image_active = PhotoImage(file=str(relative_to_assets("Low1.png")))
        self.med_image = PhotoImage(file=str(relative_to_assets("Med.png")))
        self.med_image_active = PhotoImage(file=str(relative_to_assets("Med1.png")))
        self.high_image = PhotoImage(file=str(relative_to_assets("High.png")))
        self.high_image_active = PhotoImage(file=str(relative_to_assets("High1.png")))
        self.on_image = PhotoImage(file=str(relative_to_assets("on.png")))
        self.off_image = PhotoImage(file=str(relative_to_assets("off.png")))
        
        # Detection mode images
        self.accurate_image = PhotoImage(file=str(relative_to_assets("Accurate.png")))
        self.accurate_image_active = PhotoImage(file=str(relative_to_assets("Accurate1.png")))
        self.fast_image = PhotoImage(file=str(relative_to_assets("Fast.png")))
        self.fast_image_active = PhotoImage(file=str(relative_to_assets("Fast1.png")))

        # Row 0 Controls: Rating Filter (Star rating toggle)
        self.star_on = Button(self.settings_frame, image=self.on_image, borderwidth=0, highlightthickness=0,
                            command=self.star_clicked, bg="#262827", relief="flat")
        self.star_on.place(x=control_x, y=start_y - 1, width=60, height=21)

        # Row 1 Controls: Sharpness Filter (Laplacian toggle)
        self.laplacian_on = Button(self.settings_frame, image=self.on_image, borderwidth=0, highlightthickness=0,
                                command=self.laplacian_clicked, bg="#262827", relief="flat")
        self.laplacian_on.place(x=control_x, y=start_y + row_height - 1, width=60, height=21)

        # Row 2 Controls: Burst Selection (Burst grouping toggle)
        self.burst_on = Button(self.settings_frame, image=self.on_image, borderwidth=0, highlightthickness=0,
                            command=self.burst_clicked, bg="#262827", relief="flat")
        self.burst_on.place(x=control_x, y=start_y + row_height * 2 - 1, width=60, height=21)

        # Row 3 Controls: Sharpness Threshold (Low/Med/High buttons)
        self.low_button = Button(self.settings_frame, image=self.low_image, borderwidth=0, highlightthickness=0,
                                command=self.low_clicked, bg="#262827", relief="flat")
        self.low_button.place(x=control_x, y=start_y + row_height * 3 - 1, width=60, height=21)

        self.med_button = Button(self.settings_frame, image=self.med_image_active, borderwidth=0, highlightthickness=0,
                                command=self.med_clicked, bg="#262827", relief="flat")
        self.med_button.place(x=control_x + 65, y=start_y + row_height * 3 - 1, width=60, height=21)

        self.high_button = Button(self.settings_frame, image=self.high_image, borderwidth=0, highlightthickness=0,
                                command=self.high_clicked, bg="#262827", relief="flat")
        self.high_button.place(x=control_x + 130, y=start_y + row_height * 3 - 1, width=60, height=21)

        # Row 4 Controls: Threshold Compensation (Entry field)
        settings_entry_image_1 = PhotoImage(file=relative_to_assets("Threshold.png"))
        self.settings_canvas.create_image(control_x + 30, start_y + row_height * 4 + 9, image=settings_entry_image_1)

        self.tolerance_comp = Entry(self.settings_frame, font=("Helvetica", 16), bd=0, bg="#1E1E1E", 
                                    fg="#D9D9D9", highlightthickness=0)
        self.tolerance_comp.image = settings_entry_image_1
        self.tolerance_comp.place(x=control_x+2, y=start_y + row_height * 4 -1, width=54.0, height=21.0)
        
        # Row 5 Controls: Image Detection toggle
        self.img_detection_on = Button(self.settings_frame, image=self.off_image, borderwidth=0, highlightthickness=0,
                                    command=self.img_detection_clicked, bg="#262827", relief="flat")
        self.img_detection_on.place(x=control_x, y=start_y + row_height * 5 - 1, width=60, height=21)

        # Row 6 Controls: Detection Mode (Accurate/Fast buttons)
        self.accurate_button = Button(self.settings_frame, image=self.accurate_image, borderwidth=0, highlightthickness=0,
                                    command=self.accurate_clicked, bg="#262827", relief="flat")
        self.accurate_button.place(x=control_x, y=start_y + row_height * 6 - 1, width=60, height=21)

        self.fast_button = Button(self.settings_frame, image=self.fast_image_active, borderwidth=0, highlightthickness=0,
                                command=self.fast_clicked, bg="#262827", relief="flat")
        self.fast_button.place(x=control_x + 65, y=start_y + row_height * 6 - 1, width=60, height=21)

        # Version text
        self.settings_canvas.create_text(8.0, 455.0, anchor="nw", text="Version 1.2.0", fill="#D9D9D9", font=("Inter ExtraLightItalic", 16))

    # ===============================
    # Toggle Settings
    # ===============================
    def laplacian_clicked(self):
        self.laplacian_enabled = not self.laplacian_enabled
        self.laplacian_on.config(image=self.on_image if self.laplacian_enabled else self.off_image)
        
    def burst_clicked(self):
        self.burst_enabled = not self.burst_enabled
        self.burst_on.config(image=self.on_image if self.burst_enabled else self.off_image)
        
    def star_clicked(self):
        self.star_enabled = not self.star_enabled
        self.star_on.config(image=self.on_image if self.star_enabled else self.off_image)

    def img_detection_clicked(self):
        self.img_detect_enabled = not self.img_detect_enabled
        self.img_detection_on.config(image=self.on_image if self.img_detect_enabled else self.off_image)

    def set_blur_level(self, level):
        self.sharpness_level = int(level)
        buttons = {
            -20: (self.low_button, self.low_image, self.low_image_active),
            0: (self.med_button, self.med_image, self.med_image_active),
            30: (self.high_button, self.high_image, self.high_image_active)
        }
        for val, (button, off, on) in buttons.items():
            button.config(image=on if val == level else off)

    def low_clicked(self):
        self.set_blur_level(-20)

    def med_clicked(self):
        self.set_blur_level(0)

    def high_clicked(self):
        self.set_blur_level(30)
        
    def fast_clicked(self):
        self.detection_mode = "fast"
        self.fast_button.config(image=self.fast_image_active)
        self.accurate_button.config(image=self.accurate_image)
        
    def accurate_clicked(self):
        self.detection_mode = "accurate"
        self.accurate_button.config(image=self.accurate_image_active)
        self.fast_button.config(image=self.fast_image)
        
    # ===============================
    # Main Button Event Handlers
    # ===============================
    def folder_clicked(self):  
        try:
            if self.entry_1.get() != "":
                folder_path = self.entry_1.get()
                os.startfile(os.path.join(folder_path, 'sharp'))
        except Exception as e:
            print(f"Error: {e}")
            self.canvas.itemconfig(self.processing_text, text="Invalid directory.") 

    def settings_clicked(self):
        self.show_settings()

    def cancel_clicked(self):
        if not self.is_processing:
            return
        self.stop_animation()
        self.canvas.itemconfig(self.processing_text, text="Cancelling...")
        self.is_processing = False
        
        # Cancel both processes
        if self.sorter_cancel_flag:
            self.sorter_cancel_flag.value = True
        if self.detection_cancel_flag:
            self.detection_cancel_flag.value = True
            
        self.canvas.itemconfig(self.processing_text, text="Processing cancelled.")
        self.button_1.lift()
        self.root.after(100, self.check_thread_finished)

    def check_thread_finished(self):
        if ((self.sorter_thread and self.sorter_thread.is_alive()) or 
            (self.detection_thread and self.detection_thread.is_alive())):
            self.root.after(100, self.check_thread_finished)
        else:
            print("Processing thread finished\n")

    # ===============================
    # Progress Callback for Detection
    # ===============================
    def detection_progress_callback(self, current, total):
        if self.is_processing:  # Only update if still processing
            progress_text = f"Detection: {current}/{total} images"
            self.root.after(0, lambda: self.canvas.itemconfig(self.processing_text, text=progress_text))

    # ===============================
    # Sorter + Detection Logic
    # ===============================
    def start_clicked(self):
        if self.is_processing:
            return
            
        folder = self.entry_1.get().strip()
        if not folder:
            self.canvas.itemconfig(self.processing_text, text="Invalid directory.")
            return
            
        # Clear error messages
        self.canvas.itemconfig(self.error_text_1, text="")
        self.canvas.itemconfig(self.error_text_2, text="")
        
        tolerance = 0
        if self.tolerance_comp.get():
            try:
                tolerance = int(self.tolerance_comp.get())
            except ValueError:
                tolerance = 0
                self.canvas.itemconfig(self.error_text_1, text="Invalid compensation value;")
                self.canvas.itemconfig(self.error_text_2, text="using default of 0.")
        
        # Use pre-created manager - much faster than creating new Manager()
        self.sorter_cancel_flag = self.mp_manager.Value("b", False)
        
        options = {
            "folder": folder,
            "base_blur": self.sharpness_level,
            "tolerance": tolerance,
            "use_starcheck": self.star_enabled,
            "use_laplaciancheck": self.laplacian_enabled,
            "group_bursts": self.burst_enabled,
            "cancel_flag": self.sorter_cancel_flag,  # CRITICAL: Pass the cancel flag!
        }

        # Set up output box and redirect stdout
        self.output_box.place(x=410, y=20, width=290, height=440)
        sys.stdout = redirect_stdout(self.output_box)
        self.is_processing = True
        self.cancel_button.lift()
        
        # Check if we should skip sorting and go straight to detection
        if not self.laplacian_enabled and not self.star_enabled and not self.burst_enabled:
            self.canvas.itemconfig(self.processing_text, text="Skipping Filtering - starting detection...")
            self.solo_detection = True
            self.root.after(100, self.start_detection)
            
        else:
            # Start the normal sorting process with animated dots
            self.start_animation("Processing Images")
            self.sorter_thread = threading.Thread(target=self.run_sorter, args=(options,))
            self.sorter_thread.daemon = True
            self.sorter_thread.start()
            self.root.after(100, self.check_sorter_done)
            
    def check_sorter_done(self):
        if self.sorter_thread and self.sorter_thread.is_alive():
            self.root.after(100, self.check_sorter_done)
        else:
            # Stop the animation
            self.stop_animation()
            # Sorter is done, check if we need to run detection
            if self.img_detect_enabled and self.is_processing:  # Check is_processing in case cancelled
                self.canvas.itemconfig(self.processing_text, text="Main Sorting Complete.")
                self.root.after(500, self.start_detection)  # Small delay for UI update
            else:
                self.finish_processing()

    def start_detection(self):
        if not self.is_processing:  # Check if cancelled during delay
            return
            
        self.canvas.itemconfig(self.processing_text, text="Starting image detection...")
        
        # Use pre-created manager
        self.detection_cancel_flag = self.mp_manager.Value("b", False)
        
        detection_options = {
            "folder": self.entry_1.get().strip(),
            "mode": self.detection_mode,
            "solo_process": self.solo_detection,
            "cancel_flag": self.detection_cancel_flag,
            "progress_callback": self.detection_progress_callback,
        }
        
        self.detection_thread = threading.Thread(target=self.run_detection, args=(detection_options,))
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start checking for detection completion
        self.root.after(100, self.check_detection_done)

    def check_detection_done(self):
        if self.detection_thread and self.detection_thread.is_alive():
            self.root.after(100, self.check_detection_done)
        else:
            self.finish_processing()

    def finish_processing(self):
        self.stop_animation()
        self.is_processing = False
        self.canvas.itemconfig(self.processing_text, text="Process Complete.")
        self.button_1.lift()  # Show the start button again

    def run_sorter(self, options):
        try:
            blur.main(**options)
        except Exception as e:
            print(f"Error in sorter: {e}")

    def run_detection(self, detection_options):
        try:
            detect.main(**detection_options)
        except Exception as e:
            print(f"Error in detection: {e}")
        
if __name__ == "__main__":
    freeze_support()
    window = Tk()
    app = MainApp(window)
    window.mainloop()