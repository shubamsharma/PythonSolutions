# Import required packages 
import cv2 
import pytesseract 
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import HTML, display
import imutils

def show_image(image) :
    #Function shows inage in Jupyter Notebook.
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
  
pages = convert_from_path("sample.pdf", dpi = 500)

img = np.array(pages[0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  

cnts = imutils.grab_contours(contours)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

im2 = img.copy() 

for cnt in cnts : 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    cropped = im2[y:y + h, x:x + w] 

    text = pytesseract.image_to_string(cropped)
      
