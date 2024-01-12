import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from greenScreen import removeGreenBackground, removeGreenEdge, mask_alpha

class ImageThresholdApp:
  def __init__(self, master):
    self.master = master
    self.master.title("GreenScreen removal")

    self.original_image = None
    self.removed_green = None
    self.preview_image = None
    self.object_mask = None
    self.border_mask = None
    self.final_mask = None
    self.threshold_value = tk.DoubleVar()
    self.threshold_value.set(0)
    self.target_preview_height = 300

    self.create_widgets()


  def create_widgets(self):
    # Import button
    self.import_button = tk.Button(self.master, text="Import Image", command=self.load_image)
    self.import_button.pack(pady=10)

    # Image preview
    self.preview = tk.Label(self.master)
    #self.preview.grid(row=0, column=0, padx=5, pady=5)
    self.preview.pack(pady=10)
    self.border_preview = tk.Label(self.master)
    #self.preview.grid(row=0, column=0, padx=5, pady=5)
    self.border_preview.pack(pady=10)

    # Threshold slider
    self.threshold_slider = tk.Scale(
      self.master, from_=0, to=255, orient=tk.HORIZONTAL,
      label="Threshold", variable=self.threshold_value, command=self.update_threshold
    )
    self.threshold_slider.pack(pady=10)

    # Export button
    self.export_button = tk.Button(self.master, text="Export Image", command=self.export_image)
    self.export_button.pack(pady=10)

  def load_image(self):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
      self.original_image = cv2.imread(file_path)
      self.remove_green()
      self.update_preview()

  def export_image(self):
    filename = filedialog.asksaveasfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if filename:
      out = mask_alpha(self.original_image, self.final_mask)
      cv2.imwrite(filename + ".png", out)
      messagebox.showinfo("info", "export successful")

  def resize_image(self, image, target_height):
    if len(image.shape) == 2:
      im_h, im_w = image.shape
    elif len(image.shape) == 3:
      im_h, im_w, _ = image.shape
    else:
      raise Exception('Shape of image not supported')
    height_ratio = target_height / im_h
    new_height = int(im_h * height_ratio)
    new_width = int(new_height * (im_w / im_h))
    img = Image.fromarray(image)
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    return ImageTk.PhotoImage(resized)
  
  def remove_green(self):
    removed_bg, mask = removeGreenBackground(self.original_image)
    self.removed_green = removed_bg
    self.preview_image = removed_bg
    self.object_mask = mask
    self.update_threshold()
  
  def update_threshold(self, *args):
    threshold_value = int(self.threshold_value.get())
    removed_edge, border_mask, final_mask = removeGreenEdge(self.removed_green, threshold_value, self.object_mask)
    self.preview_image = removed_edge
    self.border_mask = border_mask
    self.final_mask = final_mask
    self.update_preview()

  def update_preview(self, *args):
    removed_edge_preview = self.resize_image(self.preview_image, self.target_preview_height)
    border_mask_preview = self.resize_image(self.border_mask, self.target_preview_height)

    # Display the thresholded image
    self.preview.configure(image=removed_edge_preview)
    self.preview.image = removed_edge_preview

    # Display the border image
    self.border_preview.configure(image=border_mask_preview)
    self.border_preview.image = border_mask_preview


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageThresholdApp(root)
    root.mainloop()