#!/usr/bin/env python
# coding: utf-8

# # Day 2 Programming Lab

# The following exercises will not count towards your course mark, but they provide you with an opportunity to receive feedback on your programming skills in advance of you completing your summative assignments.
# 
# The goal of the following exercises is to make you apply the concepts and general methods seen in Day 2 of the Bootcamp and develop Python scripts to perform Compter Vision and image processing tasks.

# ## OpenCV

# In the following exercises you will use OpenCV library to read images and conduct image processing

# In[5]:


get_ipython().system('pip install opencv-python')


# In[6]:


# The following libraries will let you use opencv
import cv2 #opencv itself
import numpy as np # matrix manipulations

#the following are to do with this interactive notebook code
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook


# Open the image "Day 2_Programming Lab_People.jpg". To complete this task you need to download the Day 2_Programming Lab_Exercises People image.jpg from Canvas, then use cv2.imread() method.

# In[ ]:


# Please provide your solution here


# Print the image size, shape and type

# In[2]:


# Please provide your solution here


# Display the image using plt.imshow() method

# In[3]:


# Please provide your solution here


# Do you  notice anything wrong with the image?

# In[4]:


# Please provide your solution here
#
#


# use the method cv2.split to split the color channels of the image

# In[6]:


# Please provide your solution here
# split channels


# Convert and merge between color spacees to correct the the image from RGB to BGR

# In[7]:


# Please provide you solution here

# Merge takes an array of single channel matrices


# Use the cv2.cvtColor method to correct the input image without splitting the color channels

# In[8]:


# Please provide your solution here


# Display part of the above image corresponding to [60:250, 70:350]

# In[9]:


# Please provide your solution here


# Apply a vertical flip on the input image

# In[10]:


# Please provide your solution here


# Apply a horizontal flip on the input image

# In[11]:


# Please provide you solution here


# Rotate the inout image 90 degrees counterclockwise

# In[12]:


# Please provide you solution here


# Rotate input image 90 degrees clockwise

# In[13]:


# Please provide you solution here


# Create a blank image

# In[14]:


# Please provide you solution here


# Create a white image

# In[15]:


# Please provide you solution here


# Add a green square to the blank image at x = 240:310 and y = 60:150

# In[16]:


# Please provide you solution here


# Add the blank image with the green square to the input image, then correct the channels of the new image from BGR to RGB

# In[17]:


# Please provide you solution here


# Use Gaussian Kernel Blur with a Threshold = 3 to blur the input image, then correct the channels of the new image from BGR to RGB 

# In[18]:


# Please provide you solution here


# Try different value for the threshold and analyse the differnce

# In[19]:


# Please provide you solution here
#
#


# Create a 3X3 numpy array ([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) then use it as a kernel "Mask" to sharpen the opencv_merged image using the cv2.filter2D method.

# In[20]:


# Please provide your solution here


# Create a 3X3 numpy array ([[0, -1, 0],[-1, 5,-1], [0, -1, 0]]) then use it as a kernel "Mask" to sharpen the opencv_merged image using the cv2.filter2D method.

# In[21]:


# Please provide your solution here


# In the following exercise you will apply Haar Cascade algorithm to detect faces, eyes and smiles in an image. To complete this exercise you need to download the Day 2_Programmong Lab_Person.jpg. Make a copy of the person_image and convert it to gray.

# In[22]:


# Please provide your solution here


# Use the pre-trained model haarcascade_frontalface_default.xml to detect faces in the photo

# In[23]:


# Please provide your solution here


# Use pre-trained model haarcascade_eye.xml to detect the eyes in the photo

# In[24]:


# Please provide your solution here


# Have you noticed few false positives in eye detection. How can you improve the accuracy of your model?

# Have you noticed few false positives in eye detection. How can you improve the accuracy of your model

# In[26]:


#Please provide your solution here


# ## PIL

# In the following exercises you will use PIL library to read images and conduct image processing. To complete the following exercises you need to download the Day 2_Programming Lab_zero.jpg, Day 2_Programming Lab_one.jpg and Day 2_Programming Lab 3_puppy.jpg

# In[7]:


from PIL import Image
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageDraw


# In[8]:


#Please uncomment the following lines to import the required library and open the images
#from PIL import Image
img_01 = Image.open("Day 2_Programming Lab_zero.jpg.jpg")
#img_02 = Image.open("Day 2_Programming Lab_one.jpg.jpg")
#img_03 = Image.open("Day 2_Programming Lab_one.jpg.jpg")
#img_04 = Image.open("Day 2_Programming Lab_zero.jpg.jpg")
 


# Find the size of img_01, img_02, img_03 and img_04

# In[36]:


#Please provide your solution here


# In[9]:


# Run the following line to create an image 4 times the size of img_01
new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))


# In[10]:


# Run the following code to blend all four images in one image "new_im"
new_im.paste(img_01, (0,0))
new_im.paste(img_02, (img_01_size[0],0))
new_im.paste(img_03, (0,img_01_size[1]))
new_im.paste(img_04, (img_01_size[0],img_01_size[1]))
display(new_im)


# Use the .save command to save the new image as "merged_images.png"

# In[11]:


# Please provide your solution here


# Diplay the Day 2_Programming Lab_puppy.jpg, the use image filter.GaussianBlur(4) to blur the Day 2_Programming Lab_puppy.jpg. Use different values for the GaussianBlur and analyse the results

# In[12]:


# Please provide your solution here


# Add "Python is cool" to the Day 2_Programming Lab_puppy.jpg image

# In[13]:


# Please run and analyse the following commands


image = Image.open("Day 2_Programming Lab_puppy.jpg")
 
# creating a copy of original image
watermark_image = image.copy()
 
# Image is converted into editable form using
# Draw function and assigned to draw
draw = ImageDraw.Draw(watermark_image)
 
# ("font type",font size)
font = ImageFont.truetype("Day 2_Prograaming Lab_Calibri Regular.ttf", 50)
 
# Decide the text location, color and font
# (255,255,255)-White color text
draw.text((0, 0), "Python is Cool!", (255, 255, 255), font=font)
 
watermark_image.show()
display(watermark_image)


# Write the following 3 lines on the Day 2_Programming Lab_puppy.jpg image: <br>
# I <br>
# Love <br> 
# Python

# In[35]:


text = u"""\
I \n
LOVE \n 
Python"""

# Please provide your solution here


# In[ ]:




