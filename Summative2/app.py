# Importing libraries
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from joblib import load
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

knn = load('knn_model.joblib')

# image uploader function
def findImage():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)
    return path

def imageShow(path):
    # if file is selected
    if len(path):
        img = Image.open(path)
        img = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img)

        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("560x300")
        label.config(image=pic)
        label.image = pic

# Main method
if __name__ == "__main__":

    # defining tkinter object
    app = tk.Tk()

    # setting title and basic size to our App
    app.title("Genetic detection")
    app.geometry("560x270")

    label = tk.Label(app)
    label.pack(pady=10)

    # defining our upload buttom
    uploadButton = tk.Button(app, text="Upload image", command=findImage)
    uploadButton.pack(side=tk.BOTTOM, pady=20)

    app.mainloop()
