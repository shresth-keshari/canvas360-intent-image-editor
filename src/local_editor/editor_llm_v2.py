import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import json
import re
import sys

# --- 1. SETUP LLM ---
print("⏳ Loading Local LLM...")
LLM_AVAILABLE = False
try:
    from llama_cpp import Llama
    llm = Llama(
        model_path="Llama-3.2-1B-Instruct-Q4_K_M.gguf", 
        n_ctx=2048, 
        verbose=False 
    )
    LLM_AVAILABLE = True
    print("✅ Local LLM Loaded!")
except Exception as e:
    print(f"❌ LLM Load Error: {e}")
    print("Ensure 'model.gguf' is in this folder.")

# --- 2. SETUP VISION ---
REMBG_AVAILABLE = False
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠️ Rembg not installed. 'Remove background' command will be disabled.")

class SmartEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Editor Pro (Final Version)")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1e1e1e")

        # --- State ---
        self.cv_image = None       # The base pixel image (BGR format)
        self.display_image = None  # The Tkinter-friendly image
        
        # History Stacks for Undo/Redo
        self.history = []          # List of snapshots
        self.redo_stack = []
        
        # Virtual Text Layer
        # Format: [{'content': "Hello", 'x': 100, 'y': 100, 'color': (255,255,255), 'size': 1.5}]
        self.text_objects = []     

        self.setup_ui()

    def setup_ui(self):
        # --- Toolbar ---
        btn_frame = tk.Frame(self.root, bg="#2d2d2d", pady=5)
        btn_frame.pack(fill=tk.X)
        
        style = {"bg": "#3d3d3d", "fg": "white", "relief": "flat", "padx": 10, "pady": 5}
        
        tk.Button(btn_frame, text="📂 Open", command=self.load_image, **style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="💾 Save", command=self.save_image, **style).pack(side=tk.LEFT, padx=5)
        tk.Label(btn_frame, text="|", bg="#2d2d2d", fg="#555").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="↩️ Undo", command=self.undo, **style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="↪️ Redo", command=self.redo, **style).pack(side=tk.LEFT, padx=5)
        
        # --- Main Canvas ---
        self.canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#101010", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # --- Prompt Area ---
        prompt_frame = tk.Frame(self.root, bg="#2d2d2d", height=80, pady=10)
        prompt_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(prompt_frame, text="AI Prompt >", fg="#00ff99", bg="#2d2d2d", font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=15)
        
        self.prompt_var = tk.StringVar()
        # FIX: Explicitly creating self.entry so it can be accessed later
        self.entry = tk.Entry(prompt_frame, textvariable=self.prompt_var, font=("Segoe UI", 12), bg="#404040", fg="white", insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=5)
        self.entry.bind("<Return>", self.process_prompt_with_llm)
        
        self.status = tk.Label(self.root, text="Ready.", bg="#1e1e1e", fg="#888", anchor="e")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

    # --- HISTORY SYSTEM ---
    def save_state(self):
        """Snapshots the current image and text objects for Undo"""
        if self.cv_image is not None:
            # Deep copy the text list so modifying future text doesn't change history
            snapshot = {
                "image": self.cv_image.copy(),
                "text": [t.copy() for t in self.text_objects]
            }
            self.history.append(snapshot)
            self.redo_stack.clear() # Clear redo stack on new action
            
            # Limit memory usage (keep last 15 steps)
            if len(self.history) > 15: self.history.pop(0)

    def undo(self):
        if not self.history: return
        
        # Save current state to redo stack before reverting
        current_state = {
            "image": self.cv_image.copy(),
            "text": [t.copy() for t in self.text_objects]
        }
        self.redo_stack.append(current_state)
        
        # Pop previous state
        prev_state = self.history.pop()
        self.cv_image = prev_state["image"]
        self.text_objects = prev_state["text"]
        
        self.refresh_canvas()
        self.status.config(text="Undo performed")

    def redo(self):
        if not self.redo_stack: return
        
        # Save current state to history before redoing
        self.history.append({
            "image": self.cv_image.copy(),
            "text": [t.copy() for t in self.text_objects]
        })
        
        # Pop redo state
        next_state = self.redo_stack.pop()
        self.cv_image = next_state["image"]
        self.text_objects = next_state["text"]
        
        self.refresh_canvas()
        self.status.config(text="Redo performed")

    # --- IMAGE IO ---
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_image = cv2.imread(path)
            self.history.clear()
            self.redo_stack.clear()
            self.text_objects.clear()
            self.refresh_canvas()
            self.status.config(text=f"Loaded: {path}")

    def save_image(self):
        if self.cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            # Burn text into pixels for saving
            final_img = self.get_composited_image()
            cv2.imwrite(path, final_img)
            messagebox.showinfo("Saved", "Image saved successfully!")

    def get_composited_image(self):
        """Merges virtual text onto the pixel image"""
        if self.cv_image is None: return None
        img_copy = self.cv_image.copy()
        
        for txt in self.text_objects:
            # cv2.putText uses BGR colors
            cv2.putText(img_copy, txt['content'], (txt['x'], txt['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, txt['size'], txt['color'], 3, cv2.LINE_AA)
        return img_copy

    def refresh_canvas(self):
        if self.cv_image is None: return
        
        # 1. Get image with text overlays
        display_img = self.get_composited_image()
        
        # 2. Resize to fit window
        h, w = display_img.shape[:2]
        canvas_h = self.canvas.winfo_height()
        if canvas_h < 100: canvas_h = 600
        
        ratio = canvas_h / h
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized = cv2.resize(display_img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        self.display_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, image=self.display_image, anchor=tk.CENTER)

    # --- AI LOGIC ---
    def process_prompt_with_llm(self, event=None):
        # FIX: Safely access self.entry
        user_input = self.prompt_var.get().strip()
        if not user_input: return
        if self.cv_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return
        if not LLM_AVAILABLE:
            messagebox.showerror("Error", "LLM model not loaded.")
            return
        
        self.entry.delete(0, tk.END)
        self.status.config(text="🤖 Analyzing...", fg="#00ff99")
        self.root.update()
        
        threading.Thread(target=self.run_llm_inference, args=(user_input,)).start()

    def run_llm_inference(self, user_input):
        system_prompt = """
        You are an image editing API. Return ONLY a JSON object. No Markdown.
        
        AVAILABLE TOOLS:
        1. {"tool": "auto_enhance"}  (For: "fix lighting", "auto fix", "make it better")
        2. {"tool": "sharpen"}       (For: "sharpen", "clarity")
        3. {"tool": "undo"}
        4. {"tool": "redo"}
        5. {"tool": "text_add", "content": "string", "color": "red/blue/green/white/black"} 
        6. {"tool": "text_color", "target_content": "string", "new_color": "string"}
           (Use this if user wants to change color of existing text. 'target_content' is the word they mentioned)
        7. {"tool": "remove_bg"}
        8. {"tool": "crop", "x": int, "y": int, "w": int, "h": int}
        9. {"tool": "warmth", "value": "up" or "down"}
        
        Example: "Change Hello to blue" -> {"tool": "text_color", "target_content": "Hello", "new_color": "blue"}
        Example: "Add text Sale in red" -> {"tool": "text_add", "content": "Sale", "color": "red"}
        """
        
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"
        
        try:
            output = llm(prompt, max_tokens=150, stop=["<|end|>", "}"], echo=False)
            raw = output['choices'][0]['text'].strip()
            
            # Robust JSON extraction
            if not raw.endswith("}"): raw += "}"
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            
            if json_match:
                cmd = json.loads(json_match.group(0))
                # Use after() to safely update GUI from thread
                self.root.after(0, lambda: self.execute_command(cmd))
            else:
                print(f"Failed to parse: {raw}")
                self.root.after(0, lambda: self.status.config(text="⚠️ AI confused.", fg="orange"))
        except Exception as e:
            print(e)
            self.root.after(0, lambda: self.status.config(text="❌ Inference Error", fg="red"))

    def execute_command(self, cmd):
        tool = cmd.get("tool")
        print(f"Executing: {tool}")
        
        # Don't save state for Undo/Redo commands themselves
        if tool not in ["undo", "redo"]:
            self.save_state()

        try:
            # --- GLOBAL ---
            if tool == "undo":
                self.undo()
                return
            elif tool == "redo":
                self.redo()
                return

            # --- PIXEL MANIPULATION ---
            if tool == "auto_enhance":
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                merged = cv2.merge((cl,a,b))
                self.cv_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            elif tool == "sharpen":
                # Sharpening Kernel
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)

            elif tool == "remove_bg":
                if REMBG_AVAILABLE:
                    rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                    out = remove(rgb)
                    # Composite on white
                    alpha = out[:, :, 3] / 255.0
                    bg = np.ones_like(out[:,:,:3]) * 255
                    for i in range(3):
                        bg[:,:,i] = (1. - alpha) * bg[:,:,i] + alpha * out[:,:,i]
                    self.cv_image = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    messagebox.showerror("Error", "Rembg not installed.")

            elif tool == "warmth":
                val = cmd.get("value")
                img = self.cv_image.astype(np.float32)
                if val == "up":
                    img[:, :, 2] += 15 # Red
                    img[:, :, 0] -= 15 # Blue
                else:
                    img[:, :, 2] -= 15
                    img[:, :, 0] += 15
                self.cv_image = np.clip(img, 0, 255).astype(np.uint8)

            elif tool == "crop":
                x, y = int(cmd.get("x", 0)), int(cmd.get("y", 0))
                w, h = int(cmd.get("w", 200)), int(cmd.get("h", 200))
                # Validate bounds
                H, W = self.cv_image.shape[:2]
                x = max(0, min(x, W-1))
                y = max(0, min(y, H-1))
                self.cv_image = self.cv_image[y:y+h, x:x+w]

            # --- TEXT MANIPULATION ---
            elif tool == "text_add":
                c_str = cmd.get("color", "white").lower()
                colors = {
                    "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
                    "black": (0, 0, 0), "white": (255, 255, 255), "yellow": (0, 255, 255)
                }
                bgr = colors.get(c_str, (255, 255, 255))
                
                new_text = {
                    "content": cmd.get("content", "Text"),
                    "x": 50, 
                    "y": 100 + (len(self.text_objects) * 60), 
                    "color": bgr,
                    "size": 2.0
                }
                self.text_objects.append(new_text)

            elif tool == "text_color":
                target = cmd.get("target_content", "").lower()
                c_str = cmd.get("new_color", "white").lower()
                colors = {
                    "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
                    "black": (0, 0, 0), "white": (255, 255, 255)
                }
                new_bgr = colors.get(c_str, (255, 255, 255))
                
                found = False
                for txt in self.text_objects:
                    if target in txt["content"].lower():
                        txt["color"] = new_bgr
                        found = True
                        break
                
                if not found:
                    self.status.config(text=f"Text '{target}' not found.", fg="orange")
                    return

            self.refresh_canvas()
            self.status.config(text=f"Done: {tool}", fg="#00ff99")

        except Exception as e:
            print(e)
            self.status.config(text="Error applying edit", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartEditorApp(root)
    root.mainloop()