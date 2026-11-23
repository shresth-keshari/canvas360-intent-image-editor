import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import re
import sys

# --- SAFETY CHECK ---
print(f"Running on Python {sys.version.split()[0]}")
try:
    import rembg
    from rembg import remove
    print("✅ AI Engine (Rembg/ONNX) loaded successfully.")
    REMBG_AVAILABLE = True
except ImportError as e:
    print(f"❌ AI Engine Error: {e}")
    print("Background removal will be disabled.")
    REMBG_AVAILABLE = False

class SafeEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Editor (Safe Mode)")
        self.root.geometry("1100x850")
        self.root.configure(bg="#202020")

        self.cv_image = None
        self.display_image = None
        self.history = []

        self.setup_ui()

    def setup_ui(self):
        # Tools
        btn_frame = tk.Frame(self.root, bg="#333")
        btn_frame.pack(side=tk.TOP, fill=tk.X, ipady=5)
        
        btns = [("Open Image", self.load_image), ("Undo", self.undo), ("Save", self.save), ("Reset", self.reset)]
        for text, cmd in btns:
            tk.Button(btn_frame, text=text, command=cmd, bg="#555", fg="white", relief="flat").pack(side=tk.LEFT, padx=5)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#151515", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Prompt Bar
        prompt_frame = tk.Frame(self.root, bg="#252525", height=60)
        prompt_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(prompt_frame, text="Command >", fg="#00ff00", bg="#252525", font=("Consolas", 12)).pack(side=tk.LEFT, padx=10)
        
        self.entry = tk.Entry(prompt_frame, bg="#333", fg="white", font=("Consolas", 12), insertbackground="white")
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10, pady=15)
        self.entry.bind("<Return>", self.process_prompt)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.cv_image = cv2.imread(path)
        self.history = [self.cv_image.copy()]
        self.refresh_canvas()

    def refresh_canvas(self):
        if self.cv_image is None: return
        
        # Resize for display (keep aspect ratio)
        h, w = self.cv_image.shape[:2]
        display_h = 700
        scale = display_h / h
        display_w = int(w * scale)
        
        # Convert BGR to RGB
        resized = cv2.resize(self.cv_image, (display_w, display_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.display_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        self.canvas.delete("all")
        # Center image
        cw, ch = self.root.winfo_width(), self.root.winfo_height()
        self.canvas.create_image(cw//2, 350, image=self.display_image)

    def process_prompt(self, event=None):
        if self.cv_image is None: return
        prompt = self.entry.get().lower().strip()
        self.entry.delete(0, tk.END)
        print(f"Processing: {prompt}")

        self.history.append(self.cv_image.copy()) # Save for undo

        # --- LOGIC ENGINE ---
        try:
            img = self.cv_image.astype(np.float32)

            if "warm" in prompt:
                img[:, :, 2] += 20 # Red up
                img[:, :, 0] -= 20 # Blue down
            elif "cool" in prompt:
                img[:, :, 2] -= 20
                img[:, :, 0] += 20
            elif "bright" in prompt:
                img *= 1.2
            elif "dark" in prompt:
                img *= 0.8
            elif "gray" in prompt or "bw" in prompt:
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
            elif "blur" in prompt:
                k = 15 if "heavy" in prompt else 5
                img = cv2.GaussianBlur(img, (k, k), 0)
            elif "crop" in prompt:
                # Syntax: crop 100 100 400 400
                coords = [int(n) for n in re.findall(r'\d+', prompt)]
                if len(coords) == 4:
                    x, y, w, h = coords
                    self.cv_image = self.cv_image[y:y+h, x:x+w]
                    img = self.cv_image.astype(np.float32) # Reset img var
            elif "bg" in prompt and "remove" in prompt:
                if REMBG_AVAILABLE:
                    self.thread_rembg()
                    return # Exit to let thread finish
                else:
                    messagebox.showerror("Error", "AI Module missing.")
            
            # Finalize
            self.cv_image = np.clip(img, 0, 255).astype(np.uint8)
            self.refresh_canvas()

        except Exception as e:
            print(e)
            messagebox.showerror("Error", str(e))

    def thread_rembg(self):
        def task():
            print("Running AI Background Removal...")
            rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            out = remove(rgb) # Returns RGBA
            
            # Simple composite over white
            alpha = out[:, :, 3] / 255.0
            fg = out[:, :, :3]
            bg = np.ones_like(fg) * 255 # White
            
            # Blend
            for i in range(3):
                bg[:,:,i] = (1. - alpha) * bg[:,:,i] + alpha * fg[:,:,i]
            
            self.cv_image = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2BGR)
            self.root.after(0, self.refresh_canvas)
            print("Done.")
        
        threading.Thread(target=task).start()

    def undo(self):
        if self.history:
            self.cv_image = self.history.pop()
            self.refresh_canvas()

    def reset(self):
        if self.history:
            self.cv_image = self.history[0].copy()
            self.history = []
            self.refresh_canvas()

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path and self.cv_image is not None:
            cv2.imwrite(path, self.cv_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = SafeEditorApp(root)
    root.mainloop()