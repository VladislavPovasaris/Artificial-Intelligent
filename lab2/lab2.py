import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import requests
import io

token = "5o63l43cp0i8dq43it38gvfa2s"
project_id = "60984"
model = "lab2"

headers = {"X-Auth-token": token, "Content-Type": "application/octet-stream"}

# Define colors for each label
label_colors = {
    "Lobster": "aqua",
    "Shrimp": "blue",
    "Crab": "GreenYellow"
}

# Define font size for labels
font_size = 30  # Adjust as needed

def classify_image():
    global image_filename
    if image_filename:
        with open(image_filename, 'rb') as handle:
            r = requests.post('https://platform.sentisight.ai/api/predict/{}/{}/'.format(project_id,model), headers=headers, data=handle)
        
        if r.status_code == 200:
            json_response = r.json()
            img = Image.open(image_filename)
            draw = ImageDraw.Draw(img)
            
            for prediction in json_response:
                label = prediction["label"]
                score = prediction["score"]
                x0, y0, x1, y1 = prediction["x0"], prediction["y0"], prediction["x1"], prediction["y1"]
                color = label_colors.get(label, "red")  # Default to red if label not found
                
                # Load font
                font = ImageFont.truetype("arial.ttf", font_size)
                
                draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)
                draw.text((x0 + 10, y0 + 3), f"{label}", fill=color, font=font)
                draw.text((x1 - 185, y1 - 35), f"Score: {score:.1f}%", fill=color, font=font)
                
            # Resize image proportionally if it's larger than the screen size
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            if img.width > screen_width or img.height > screen_height:
                img.thumbnail((screen_width-400, screen_height-400))
            
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img)
            panel.image = img  # Keep a reference to the image to prevent garbage collection
        else:
            result_label.config(text='Error occurred with REST API.\nStatus code: {}\nError message: {}'.format(r.status_code, r.text))
    else:
        result_label.config(text='Please select an image first.')

def browse_image():
    global image_filename
    image_filename = filedialog.askopenfilename()
    if image_filename:
        image = Image.open(image_filename)
        # Resize image proportionally if it's larger than the screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        if image.width > screen_width or image.height > screen_height:
            image.thumbnail((screen_width-400, screen_height-400))
        photo = ImageTk.PhotoImage(image)
        panel.configure(image=photo)
        panel.image = photo  # Keep a reference to the image to prevent garbage collection
        result_label.config(text='Image selected: {}'.format(image_filename))
        root.geometry(f"{image.width}x{image.height}")
    else:
        result_label.config(text='No image selected.')

# Create main window
root = tk.Tk()
root.title("Image Classifier")

# Set initial window size
initial_width = 300
initial_height = 100
root.geometry(f"{initial_width}x{initial_height}")

# Create widgets
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
classify_button = tk.Button(root, text="Classify Image", command=classify_image)
panel = tk.Label(root)
result_label = tk.Label(root, text="")

# Layout widgets
browse_button.pack(pady=10)
classify_button.pack(pady=5)
panel.pack(padx=10, pady=10)
result_label.pack()

# Run the application
root.mainloop()
