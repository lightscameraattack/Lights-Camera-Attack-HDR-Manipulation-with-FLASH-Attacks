import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime
from typing import List, Tuple
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from hdr_pipeline_gui import ModularHDR
    from hdr_pipeline_organized import compare_results
except ImportError:

    class ModularHDR:

        def __init__(self, *args):
            pass

        def process(self, *args):
            return (np.zeros((300, 300, 3), dtype=np.uint8), {'base': '.'})

        def process_generated(self, *args):
            return (np.zeros((300, 300, 3), dtype=np.uint8), {'base': '.'})

    def compare_results(*args):
        pass

class RoundedButton(tk.Canvas):

    def __init__(self, parent, text, command=None, width=120, height=35, corner_radius=10, bg_color='#007AFF', fg_color='white', hover_color='#0056b3', parent_bg='#FFFFFF'):
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, relief='flat')
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.radius = corner_radius
        self.bg_normal = bg_color
        self.bg_hover = hover_color
        self.fg_color = fg_color
        self.normal_shape = self._draw_shape(self.bg_normal)
        self.hover_shape = None
        self.text_item = self.create_text(width / 2, height / 2, text=text, fill=fg_color, font=('Segoe UI', 10, 'bold'))
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _draw_shape(self, color):
        self.delete('shape')
        r = self.radius
        w, h = (self.width, self.height)
        shapes = []
        shapes.append(self.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90, fill=color, outline=color, tags='shape'))
        shapes.append(self.create_arc(w - 2 * r, 0, w, 2 * r, start=0, extent=90, fill=color, outline=color, tags='shape'))
        shapes.append(self.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=color, outline=color, tags='shape'))
        shapes.append(self.create_arc(0, h - 2 * r, 2 * r, h, start=180, extent=90, fill=color, outline=color, tags='shape'))
        shapes.append(self.create_rectangle(r, 0, w - r, h, fill=color, outline=color, tags='shape'))
        shapes.append(self.create_rectangle(0, r, w, h - r, fill=color, outline=color, tags='shape'))
        self.tag_lower('shape')
        return shapes

    def _on_enter(self, event):
        self._draw_shape(self.bg_hover)

    def _on_leave(self, event):
        self._draw_shape(self.bg_normal)

    def _on_click(self, event):
        self._draw_shape(self.bg_hover)
        self.move(self.text_item, 1, 1)

    def _on_release(self, event):
        self.move(self.text_item, -1, -1)
        if self.command:
            self.command()

class HDRLabApp:

    def __init__(self, root):
        self.root = root
        self.root.title('HDRLab')
        self.root.geometry('1400x950')
        self.colors = {'bg_main': '#F3F4F6', 'bg_panel': '#FFFFFF', 'fg_text': '#374151', 'fg_header': '#111827', 'accent': '#2563EB', 'accent_hover': '#1D4ED8', 'input_bg': '#F9FAFB', 'border': '#E5E7EB', 'canvas_bg': '#E5E7EB', 'secondary': '#6B7280'}
        self.fonts = {'main': ('Segoe UI', 10), 'bold': ('Segoe UI', 10, 'bold'), 'header': ('Segoe UI', 11, 'bold'), 'title': ('Segoe UI', 22, 'bold')}
        self.root.configure(bg=self.colors['bg_main'])
        self.setup_styles()
        self.image_paths = []
        self.exposures = []
        self.current_result = None
        self.current_dirs = None
        self.processing = False
        self.save_directory = None
        self.initializing = True
        self.alignment_var = tk.StringVar(value='mtb')
        self.merging_var = tk.StringVar(value='debevec')
        self.tonemapping_var = tk.StringVar(value='reinhard')
        self.single_image_mode_var = tk.BooleanVar(value=False)
        self.exposure_stops_var = tk.StringVar(value='2')
        self.generated_images = []
        self.setup_gui()
        self.alignment_var.trace('w', self.on_method_change)
        self.merging_var.trace('w', self.on_method_change)
        self.tonemapping_var.trace('w', self.on_method_change)
        self.single_image_mode_var.trace('w', self.on_single_image_mode_change)
        self.root.after(100, self.load_default_image)

    def setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            style.theme_use('default')
        style.configure('Main.TFrame', background=self.colors['bg_main'])
        style.configure('Panel.TFrame', background=self.colors['bg_panel'])
        style.configure('TLabel', background=self.colors['bg_panel'], foreground=self.colors['fg_text'], font=self.fonts['main'])
        style.configure('Title.TLabel', background=self.colors['bg_main'], foreground=self.colors['fg_header'], font=self.fonts['title'])
        style.configure('Header.TLabel', background=self.colors['bg_panel'], foreground=self.colors['fg_header'], font=self.fonts['header'])
        style.configure('Status.TLabel', background=self.colors['bg_panel'], foreground=self.colors['fg_text'], font=('Segoe UI', 9), padding=5)
        style.configure('TLabelframe', background=self.colors['bg_panel'], bordercolor=self.colors['border'], relief='flat')
        style.configure('TLabelframe.Label', background=self.colors['bg_panel'], foreground=self.colors['fg_header'], font=self.fonts['header'])
        style.configure('TEntry', fieldbackground=self.colors['input_bg'], foreground=self.colors['fg_header'], bordercolor=self.colors['border'], relief='flat', padding=5)
        style.configure('TCheckbutton', background=self.colors['bg_panel'], foreground=self.colors['fg_text'], font=self.fonts['main'])
        style.map('TCheckbutton', background=[('active', self.colors['bg_panel'])], indicatorcolor=[('selected', self.colors['accent'])])
        style.configure('TRadiobutton', background=self.colors['bg_panel'], foreground=self.colors['fg_text'], font=self.fonts['main'])
        style.map('TRadiobutton', background=[('active', self.colors['bg_panel'])], indicatorcolor=[('selected', self.colors['accent'])])
        style.configure('TButton', font=self.fonts['main'])

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(1, weight=1)
        header_frame = ttk.Frame(main_frame, style='Main.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 25))
        title_label = ttk.Label(header_frame, text='HDRLab', style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        self.setup_control_panel(main_frame)
        self.setup_preview_panel(main_frame)
        self.setup_status_bar(self.root)

    def setup_control_panel(self, parent):
        control_container = ttk.Frame(parent, style='Main.TFrame')
        control_container.grid(row=1, column=0, sticky='nsew', padx=(0, 20))
        img_grp = self.create_card(control_container, 'Source Images')
        img_grp.pack(fill=tk.X, pady=(0, 20))
        btn_import = RoundedButton(img_grp, text='Import Images', command=self.select_images, bg_color=self.colors['accent'], hover_color=self.colors['accent_hover'], parent_bg=self.colors['bg_panel'], width=280)
        btn_import.pack(pady=(5, 10))
        list_frame = tk.Frame(img_grp, bg=self.colors['input_bg'], bd=1, relief='solid')
        list_frame.config(highlightbackground=self.colors['border'], highlightthickness=1)
        list_frame.pack(fill=tk.X, pady=5, padx=5)
        self.image_listbox = tk.Listbox(list_frame, height=5, bg=self.colors['input_bg'], fg=self.colors['fg_text'], selectbackground=self.colors['accent'], selectforeground='white', relief='flat', borderwidth=0, font=self.fonts['main'], highlightthickness=0)
        self.image_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        mode_grp = self.create_card(control_container, 'Exposure Settings')
        mode_grp.pack(fill=tk.X, pady=(0, 20))
        self.single_image_check = ttk.Checkbutton(mode_grp, text='Auto Bracket (Single Image)', variable=self.single_image_mode_var)
        self.single_image_check.pack(anchor=tk.W, pady=(5, 10), padx=5)
        param_frame = ttk.Frame(mode_grp, style='Panel.TFrame')
        param_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(param_frame, text='Stops (plus minus):').grid(row=0, column=0, padx=(0, 5), sticky='w')
        self.exposure_stops_entry = ttk.Entry(param_frame, textvariable=self.exposure_stops_var, width=8)
        self.exposure_stops_entry.grid(row=0, column=1, sticky='w')
        ttk.Label(param_frame, text='Manual EV:').grid(row=1, column=0, padx=(0, 5), pady=(10, 0), sticky='w')
        self.exposure_entry = ttk.Entry(param_frame)
        self.exposure_entry.grid(row=1, column=1, sticky='ew', pady=(10, 0))
        param_frame.columnconfigure(1, weight=1)
        self.exposure_entry.insert(0, '0.1, 1.0, 10.0')
        self.setup_method_selection(control_container)
        self.setup_processing_options(control_container)
        self.setup_action_buttons(control_container)

    def create_card(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=15)
        return frame

    def setup_method_selection(self, parent):
        method_frame = self.create_card(parent, 'Pipeline Configuration')
        method_frame.pack(fill=tk.X, pady=(0, 20))
        method_frame.columnconfigure(0, weight=1)
        method_frame.columnconfigure(1, weight=1)
        ttk.Label(method_frame, text='Alignment', style='Header.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
        ttk.Radiobutton(method_frame, text='MTB', variable=self.alignment_var, value='mtb').grid(row=1, column=0, sticky='w', padx=5)
        ttk.Radiobutton(method_frame, text='Homography', variable=self.alignment_var, value='homography').grid(row=1, column=1, sticky='w')
        ttk.Separator(method_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)
        ttk.Label(method_frame, text='Merging', style='Header.TLabel').grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 8))
        ttk.Radiobutton(method_frame, text='Robertson', variable=self.merging_var, value='robertson').grid(row=4, column=0, sticky='w', padx=5)
        ttk.Radiobutton(method_frame, text='Debevec', variable=self.merging_var, value='debevec').grid(row=4, column=1, sticky='w')
        ttk.Radiobutton(method_frame, text='Mertens', variable=self.merging_var, value='mertens').grid(row=5, column=0, sticky='w', padx=5)
        ttk.Separator(method_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky='ew', pady=15)
        ttk.Label(method_frame, text='Tone Mapping', style='Header.TLabel').grid(row=7, column=0, columnspan=2, sticky='w', pady=(0, 8))
        ttk.Radiobutton(method_frame, text='Mantiuk', variable=self.tonemapping_var, value='mantiuk').grid(row=8, column=0, sticky='w', padx=5)
        ttk.Radiobutton(method_frame, text='Reinhard', variable=self.tonemapping_var, value='reinhard').grid(row=8, column=1, sticky='w')
        ttk.Radiobutton(method_frame, text='Drago', variable=self.tonemapping_var, value='drago').grid(row=9, column=0, sticky='w', padx=5)
        ttk.Radiobutton(method_frame, text='Local', variable=self.tonemapping_var, value='local').grid(row=9, column=1, sticky='w')

    def setup_processing_options(self, parent):
        options_frame = self.create_card(parent, 'Options')
        options_frame.pack(fill=tk.X, pady=(0, 20))
        self.save_intermediate_var = tk.BooleanVar(value=False)
        self.save_intermediate_check = ttk.Checkbutton(options_frame, text='Save Intermediate Steps', variable=self.save_intermediate_var, command=self.on_save_intermediate_toggle)
        self.save_intermediate_check.pack(anchor=tk.W, pady=2, padx=5)
        self.auto_process_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text='Auto Process Preview', variable=self.auto_process_var).pack(anchor=tk.W, pady=2, padx=5)

    def on_save_intermediate_toggle(self):
        if self.save_intermediate_var.get():
            save_dir = filedialog.askdirectory()
            if save_dir:
                self.save_directory = save_dir
                self.update_status(f'Saving steps to: {save_dir}')
            else:
                self.save_intermediate_var.set(False)

    def on_single_image_mode_change(self, *args):
        if self.single_image_mode_var.get():
            self.update_status('Mode: Auto Bracketing (Single Image)')
            self.exposure_entry.state(['disabled'])
        else:
            self.update_status('Mode: Standard (Multi Image)')
            self.exposure_entry.state(['!disabled'])
            self.generated_images = []

    def setup_action_buttons(self, parent):
        button_frame = ttk.Frame(parent, style='Main.TFrame')
        button_frame.pack(fill=tk.X, pady=5)
        btn_save = RoundedButton(button_frame, text='Save Result', command=self.save_result, bg_color='#10B981', hover_color='#059669', parent_bg=self.colors['bg_main'], width=300)
        btn_save.pack(fill=tk.X, pady=(0, 15))
        grid_frame = ttk.Frame(button_frame, style='Main.TFrame')
        grid_frame.pack(fill=tk.X)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        btn_compare = RoundedButton(grid_frame, text='Compare All', command=self.compare_methods, bg_color=self.colors['secondary'], hover_color='#4B5563', parent_bg=self.colors['bg_main'], width=140)
        btn_compare.grid(row=0, column=0, padx=(0, 10), sticky='ew')
        btn_clear = RoundedButton(grid_frame, text='Clear All', command=self.clear_all, bg_color=self.colors['secondary'], hover_color='#4B5563', parent_bg=self.colors['bg_main'], width=140)
        btn_clear.grid(row=0, column=1, padx=(10, 0), sticky='ew')

    def setup_preview_panel(self, parent):
        preview_frame = ttk.Frame(parent, style='Main.TFrame')
        preview_frame.grid(row=1, column=1, sticky='nsew')
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        out_container = self.create_card(preview_frame, 'Result Preview')
        out_container.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=(0, 20))
        self.output_canvas = tk.Canvas(out_container, bg=self.colors['canvas_bg'], highlightthickness=0, width=600, height=400)
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        in_container = self.create_card(preview_frame, 'Source Input')
        in_container.grid(row=1, column=0, sticky='nsew', padx=(0, 20))
        self.input_canvas = tk.Canvas(in_container, bg=self.colors['canvas_bg'], highlightthickness=0, width=250, height=200)
        self.input_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_container = self.create_card(preview_frame, 'Process Details')
        info_container.grid(row=1, column=1, sticky='nsew')
        self.info_text = tk.Text(info_container, height=8, width=30, bg=self.colors['input_bg'], fg=self.colors['fg_text'], font=('Segoe UI', 9), relief='flat', borderwidth=0, padx=10, pady=10)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.show_message('Ready for Processing')

    def setup_status_bar(self, parent):
        self.status_var = tk.StringVar(value=' Ready')
        status_frame = tk.Frame(parent, bg=self.colors['border'], height=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar = ttk.Label(parent, textvariable=self.status_var, style='Status.TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_images(self):
        filetypes = [('Images', '*.jpg *.jpeg *.png *.tiff *.tif'), ('All', '*.*')]
        files = filedialog.askopenfilenames(title='Import Images', filetypes=filetypes)
        if files:
            self.image_paths = list(files)
            self.update_image_list()
            self.update_status(f'Imported {len(files)} images')
            self.load_and_display_input_image()
            self.auto_adjust_methods()
            if self.auto_process_var.get():
                self.process_images()

    def auto_adjust_methods(self):
        num = len(self.image_paths)
        method = self.merging_var.get()
        if num == 1:
            if method in ['robertson', 'debevec']:
                if not self.single_image_mode_var.get():
                    self.single_image_mode_var.set(True)
                    self.update_status('Switched to Auto Bracketing (Required for single image)')
            elif method == 'mertens' and self.single_image_mode_var.get():
                self.single_image_mode_var.set(False)
        elif num >= 3:
            if self.single_image_mode_var.get():
                self.single_image_mode_var.set(False)
                self.update_status('Switched to Standard Mode (Multiple images detected)')
        if method == 'mertens' and self.single_image_mode_var.get():
            self.single_image_mode_var.set(False)

    def load_and_display_input_image(self):
        if not self.image_paths:
            return
        try:
            img = cv2.imread(self.image_paths[0])
            if img is None:
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            self._display_on_canvas(pil_image, self.input_canvas)
        except Exception:
            pass

    def _display_on_canvas(self, pil_image, canvas):
        canvas.update()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1:
            w = 300
        if h <= 1:
            h = 200
        iw, ih = pil_image.size
        scale = min(w / iw, h / ih, 1.0)
        nw, nh = (int(iw * scale), int(ih * scale))
        resized = pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete('all')
        canvas.create_image(w // 2, h // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo

    def update_image_list(self):
        self.image_listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            self.image_listbox.insert(tk.END, f' {i + 1}.  {filename}')

    def generate_bracketed_exposures(self, image_path: str) -> Tuple[List[np.ndarray], List[float]]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError('Load failed')
            img_float = img.astype(np.float32) / 255.0
            stops = float(self.exposure_stops_var.get())
            exposures = [1.0 / 2 ** stops, 1.0, 2 ** stops]
            generated = []
            for exp in exposures:
                adj = np.clip(img_float * exp, 0, 1)
                generated.append((adj * 255).astype(np.uint8))
            self.update_status(f'Generated brackets plus minus {stops} EV')
            return (generated, exposures)
        except Exception as e:
            raise ValueError(str(e))

    def parse_exposures(self):
        if self.single_image_mode_var.get():
            return True
        txt = self.exposure_entry.get().strip()
        if txt:
            try:
                self.exposures = [float(x.strip()) for x in txt.split(',')]
                return True
            except ValueError:
                return False
        return True

    def on_method_change(self, *args):
        if self.initializing:
            return
        self.auto_adjust_methods()
        if self.auto_process_var.get() and self.image_paths and (not self.processing):
            if self.parse_exposures():
                self.process_images()

    def process_images(self):
        if not self.image_paths:
            return
        if self.processing:
            return
        if not self.parse_exposures():
            return
        num = len(self.image_paths)
        meth = self.merging_var.get()
        if self.single_image_mode_var.get():
            if num != 1:
                return
            if meth not in ['robertson', 'debevec']:
                return
        self.processing = True
        self.update_status('Processing...')
        self.root.config(cursor='watch')
        thread = threading.Thread(target=self._process_images_thread)
        thread.daemon = True
        thread.start()

    def _process_images_thread(self):
        try:
            out = self.save_directory if self.save_intermediate_var.get() and self.save_directory else 'temp_output'
            pipe = ModularHDR(self.alignment_var.get(), self.merging_var.get(), self.tonemapping_var.get())
            if self.single_image_mode_var.get():
                imgs, exps = self.generate_bracketed_exposures(self.image_paths[0])
                res, dirs = pipe.process_generated(imgs, exps, out, self.save_intermediate_var.get())
            else:
                exps = None if self.merging_var.get() == 'mertens' else self.exposures
                res, dirs = pipe.process(self.image_paths, exps, out, self.save_intermediate_var.get())
            self.root.after(0, self._process_complete, res, dirs)
        except Exception as e:
            self.root.after(0, self._process_error, str(e))

    def _process_complete(self, result, dirs):
        self.processing = False
        self.root.config(cursor='')
        self.current_result = result
        self.current_dirs = dirs
        self.update_preview(result)
        self.update_info(result, dirs)
        self.update_status('Processing Complete')

    def _process_error(self, msg):
        self.processing = False
        self.root.config(cursor='')
        self.update_status('Error Occurred')
        messagebox.showerror('Error', msg)

    def update_preview(self, result):
        if result is None:
            return
        disp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
        self._display_on_canvas(Image.fromarray(disp), self.output_canvas)

    def update_info(self, result, dirs):
        if result is None:
            return
        mode = 'Auto Bracket' if self.single_image_mode_var.get() else 'Standard'
        txt = f'PIPELINE CONFIG\n'
        txt += f'──────────────────\n'
        txt += f'Align : {self.alignment_var.get()}\n'
        txt += f'Merge : {self.merging_var.get()}\n'
        txt += f'Tone  : {self.tonemapping_var.get()}\n'
        txt += f'Mode  : {mode}\n\n'
        txt += f'IMAGE STATISTICS\n'
        txt += f'──────────────────\n'
        txt += f'Size  : {result.shape[1]}x{result.shape[0]}\n'
        txt += f'Range : {result.min()}-{result.max()}\n'
        txt += f'Mean  : {result.mean():.1f}'
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, txt)

    def show_message(self, msg):
        self.input_canvas.delete('all')
        self.output_canvas.delete('all')
        self.output_canvas.create_text(300, 200, text=msg, font=self.fonts['title'], fill='#9CA3AF')

    def save_result(self):
        if self.current_result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')])
        if path:
            cv2.imwrite(path, self.current_result)
            self.update_status(f'Saved to {os.path.basename(path)}')

    def compare_methods(self):
        if not self.image_paths:
            return
        if not self.parse_exposures():
            return
        save_dir = filedialog.askdirectory(title='Select Output Folder')
        if not save_dir:
            return
        self.update_status('Running Comparison...')
        methods = [('mtb', 'debevec', 'reinhard'), ('mtb', 'debevec', 'drago'), ('mtb', 'mertens', 'reinhard')]
        results = {}
        for a, m, t in methods:
            try:
                p = ModularHDR(a, m, t)
                exps = None if m == 'mertens' else self.exposures
                res, _ = p.process(self.image_paths, exps, 'comp_temp')
                results[f'{a}_{m}_{t}'] = res
            except Exception:
                pass
        if results:
            compare_results(results, os.path.join(save_dir, 'comparison'), 'report')
            self.update_status('Comparison Saved')

    def load_default_image(self):
        try:
            default = 'tests/test_exposure_2.jpg'
            if os.path.exists(default):
                self.image_paths = [default]
                self.update_image_list()
                self.load_and_display_input_image()
                self.auto_adjust_methods()
                self.update_status('Loaded Default')
                self._process_images_thread()
        except Exception:
            pass
        self.initializing = False

    def clear_all(self):
        self.image_paths = []
        self.exposures = []
        self.current_result = None
        self.image_listbox.delete(0, tk.END)
        self.show_message('Ready')
        self.info_text.delete(1.0, tk.END)
        self.update_status('Reset')

    def update_status(self, msg):
        self.status_var.set(f' {msg}')

def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = HDRLabApp(root)
    root.mainloop()
if __name__ == '__main__':
    main()