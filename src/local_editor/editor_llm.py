import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import json
import re
import sys

# --- 1. SETUP LLM (The Brains) ---
print("⏳ Loading Local LLM (this happens once)...")
LLM_AVAILABLE = False
try:
    from llama_cpp import Llama
    # Lower n_ctx if you run out of RAM. 2048 is standard for Llama 3.
    # Check if model.gguf exists before running!
    llm = Llama(
        model_path="Llama-3.2-1B-Instruct-Q4_K_M.gguf", 
        n_ctx=2048, 
        verbose=False 
    )
    LLM_AVAILABLE = True
    print("✅ Local LLM Loaded!")
except Exception as e:
    print(f"❌ LLM Load Error: {e}")
    print("Make sure you downloaded a GGUF model and renamed it to 'model.gguf'")

# --- 2. SETUP VISION (The Eyes) ---
REMBG_AVAILABLE = False
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠️ Rembg not installed. 'Remove background' command will be disabled.")

class SmartEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local LLM Image Editor")
        self.root.geometry("1024x800")
        self.root.configure(bg="#1e1e1e")

        self.cv_image = None
        self.display_image = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- Toolbar ---
        btn_frame = tk.Frame(self.root, bg="#2d2d2d", pady=5)
        btn_frame.pack(fill=tk.X)
        
        btn_style = {"bg": "#3d3d3d", "fg": "white", "relief": "flat", "padx": 15, "pady": 5}
        
        tk.Button(btn_frame, text="📂 Open Image", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="💾 Save", command=self.save_image, **btn_style).pack(side=tk.LEFT, padx=10)
        
        # --- Main Canvas ---
        self.canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#101010", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # --- Prompt Area ---
        prompt_frame = tk.Frame(self.root, bg="#2d2d2d", height=80, pady=10)
        prompt_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        lbl = tk.Label(prompt_frame, text="AI Prompt >", fg="#00ff99", bg="#2d2d2d", font=("Consolas", 12, "bold"))
        lbl.pack(side=tk.LEFT, padx=15)
        
        self.prompt_var = tk.StringVar()
        self.entry = tk.Entry(prompt_frame, textvariable=self.prompt_var, font=("Segoe UI", 12), bg="#404040", fg="white", insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=5)
        self.entry.bind("<Return>", self.process_prompt_with_llm)
        
        self.status = tk.Label(self.root, text="Ready. Load an image to start.", bg="#1e1e1e", fg="#888", anchor="e")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg;*.bmp")])
        if path:
            # Read image using OpenCV
            self.cv_image = cv2.imread(path)
            self.refresh_canvas()
            self.status.config(text=f"Loaded: {path}")

    def save_image(self):
        if self.cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imwrite(path, self.cv_image)
            messagebox.showinfo("Saved", "Image saved successfully!")

    def refresh_canvas(self):
        if self.cv_image is None: return
        
        # 1. Get dimensions
        h, w = self.cv_image.shape[:2]
        
        # 2. Calculate scaling to fit window
        canvas_h = self.canvas.winfo_height()
        if canvas_h < 100: canvas_h = 600 # Fallback if window not fully drawn
        
        ratio = canvas_h / h
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 3. Resize for display (fast)
        resized = cv2.resize(self.cv_image, (new_w, new_h))
        
        # 4. Convert BGR (OpenCV) to RGB (Tkinter)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.display_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        # 5. Draw
        self.canvas.delete("all")
        # Center the image
        x_center = self.canvas.winfo_width() // 2
        y_center = self.canvas.winfo_height() // 2
        self.canvas.create_image(x_center, y_center, image=self.display_image, anchor=tk.CENTER)

    # --- THE CORE AI LOGIC ---

    def process_prompt_with_llm(self, event=None):
        user_input = self.prompt_var.get().strip()
        if not user_input: return
        if self.cv_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return
        if not LLM_AVAILABLE:
            messagebox.showerror("Error", "LLM model not loaded. Check console for errors.")
            return
        
        # Clear entry and show loading state
        self.entry.delete(0, tk.END)
        self.status.config(text="🤖 AI is thinking...", fg="#00ff99")
        self.root.update()

        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self.run_llm_inference, args=(user_input,)).start()

    def run_llm_inference(self, user_input):
        try:
            # 1. Construct Prompt
            # We teach the LLM to speak JSON only.
            system_prompt = """
            You are an image editing assistant. Convert the user request into a strict JSON command.
            
            AVAILABLE TOOLS:
            - {"tool": "warmth", "value": "up" | "down"}
            - {"tool": "brightness", "value": "up" | "down"}
            - {"tool": "contrast", "value": "up" | "down"}
            - {"tool": "blur", "value": "low" | "high"}
            - {"tool": "grayscale"}
            - {"tool": "flip", "direction": "horizontal" | "vertical"}
            - {"tool": "remove_bg"} 
            - {"tool": "crop", "x": int, "y": int, "w": int, "h": int} 
              (Note: If user says "crop", guess coordinates like x=100, y=100, w=400, h=400 if not specified)

            EXAMPLE 1: "Make it warmer please" -> {"tool": "warmth", "value": "up"}
            EXAMPLE 2: "Remove the background" -> {"tool": "remove_bg"}
            EXAMPLE 3: "Crop the face" -> {"tool": "crop", "x": 200, "y": 100, "w": 300, "h": 300}

            OUTPUT JSON ONLY. NO MARKDOWN.
            """

            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

            # 2. Run Inference
            output = llm(
                full_prompt, 
                max_tokens=100, 
                stop=["<|end|>", "}"], # Stop early to save time
                echo=False
            )
            
            raw_text = output['choices'][0]['text']
            # Re-add the closing bracket if LLM stopped early
            if not raw_text.strip().endswith("}"): 
                raw_text += "}"

            print(f"🤖 Raw LLM Output: {raw_text}")

            # 3. Extract JSON
            # Use regex to find the first { ... } block
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                command_data = json.loads(json_str)
                # Execute on main thread (TKinter isn't thread safe)
                self.root.after(0, lambda: self.execute_command(command_data))
            else:
                self.root.after(0, lambda: self.status.config(text="⚠️ AI didn't understand.", fg="orange"))

        except Exception as e:
            print(f"Inference Error: {e}")
            self.root.after(0, lambda: self.status.config(text="❌ AI Error", fg="red"))

    def execute_command(self, cmd):
        tool = cmd.get("tool")
        val = cmd.get("value")
        
        self.status.config(text=f"Applying: {tool}...", fg="#ffff00")
        self.root.update()
        
        img = self.cv_image.astype(np.float32)

        try:
            # --- COLOR COMMANDS ---
            if tool == "warmth":
                if val == "up":
                    img[:, :, 2] += 20  # Boost Red
                    img[:, :, 0] -= 20  # Reduce Blue
                else:
                    img[:, :, 2] -= 20
                    img[:, :, 0] += 20
                self.cv_image = np.clip(img, 0, 255).astype(np.uint8)

            elif tool == "brightness":
                factor = 1.2 if val == "up" else 0.8
                img *= factor
                self.cv_image = np.clip(img, 0, 255).astype(np.uint8)

            elif tool == "contrast":
                factor = 1.2 if val == "up" else 0.8
                mean = np.mean(img)
                img = (img - mean) * factor + mean
                self.cv_image = np.clip(img, 0, 255).astype(np.uint8)

            # --- FILTER COMMANDS ---
            elif tool == "grayscale":
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                self.cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif tool == "blur":
                k = 15 if val == "high" else 5
                self.cv_image = cv2.GaussianBlur(self.cv_image, (k, k), 0)

            elif tool == "flip":
                code = 1 if cmd.get("direction") == "horizontal" else 0
                self.cv_image = cv2.flip(self.cv_image, code)

            # --- GEOMETRY COMMANDS ---
            elif tool == "crop":
                x = int(cmd.get("x", 0))
                y = int(cmd.get("y", 0))
                w = int(cmd.get("w", 100))
                h = int(cmd.get("h", 100))
                
                # Bounds checking
                H, W = self.cv_image.shape[:2]
                x = max(0, min(x, W-1))
                y = max(0, min(y, H-1))
                w = min(w, W-x)
                h = min(h, H-y)
                
                self.cv_image = self.cv_image[y:y+h, x:x+w]

            # --- AI COMMANDS ---
            elif tool == "remove_bg":
                if REMBG_AVAILABLE:
                    # Rembg expects RGB, OpenCV is BGR
                    rgb_in = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                    rgba_out = remove(rgb_in)
                    # Convert back to BGR (and handling alpha is complex, 
                    # so we just composite over white for this demo)
                    alpha = rgba_out[:, :, 3] / 255.0
                    foreground = rgba_out[:, :, :3]
                    background = np.ones_like(foreground) * 255 # White bg
                    
                    # Blend
                    for i in range(3):
                        background[:,:,i] = (1.0 - alpha) * background[:,:,i] + alpha * foreground[:,:,i]
                    
                    self.cv_image = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    messagebox.showerror("Error", "Rembg library missing.")

            else:
                print(f"Unknown tool: {tool}")

            # Finalize
            self.refresh_canvas()
            self.status.config(text=f"Done: {tool}", fg="#00ff99")

        except Exception as e:
            print(e)
            self.status.config(text=f"Error applying {tool}", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartEditorApp(root)
    root.mainloop()