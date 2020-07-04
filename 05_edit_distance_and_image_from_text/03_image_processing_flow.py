#Language - Python
#Imp Linrary - OpenCV, pytesseract

############################
#Libraries
############################

from os import listdir
from os.path import isfile,join
import os
import time
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import cv2
from matplotlib import pyplot as plt
from IPython.display import HTML, display

############################
#Read all the PDF's from path
############################

current_dir = os.path.abspath(os.path.join(os.getcwd()))
directory = "files/"
directory = os.path.abspath(os.path.join(current_dir,directory))
pdf_files_list = [f for f in listdir(directory) if isfile(join(directory,f))]
for file in pdf_files_list:
    path = os.path.abspath(os.path.join(directory,file))
    print(path)
    
	############################
	Convert PDF to image using convert_from_path
	############################
	
	############################
	Convert imaage to numpy array
	############################

	############################
	Convert imaage to Grayscale using cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	############################
	
	############################
	Apply Adaptive Thresholding 
	############################
	
	+++++++++++++Optional+++++++++++++
	Specify structure shape using cv2.getStructuringElement() and apply dilation
	but this will create problem in our case as we have dottted lines which will additionally be considered as contour objects
	+++++++++++++Optional+++++++++++++
	
	############################
	Find Contours
	############################
	
	############################
	Grab contours
	############################
	
	############################
	Sort contours and select the largest five based on contourArea
	############################
	
	############################
	Crop the orignal image to get the relevant area
	############################
	
	############################
	Apply pytesseract.image_to_string to exract text from image part and store in variable
	############################
	
	############################
	Create list with above data appended
	############################
	
############################
#Finally write to csv file
############################
