import napari
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Import packages for czi
from utils import pylibczirw_tools, misc


# Import packages for svs
import os
os.add_dll_directory(r"C:\Users\ZSPANIYA\Documents\Projects\Sun Yet Sen University\Dependencies\openslide\bin")
import openslide
import numpy as np

# Import packages for tiff
import tifffile

# Enter image path
# filename = r"C:\\Users\\zcxilin\\Desktop\\ALS-ImgConversion\\Images\\imgTest.tif"

# open s simple dialog to select an image file
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

imgArray = []

# Check file suffix
if filename.endswith(".czi"):
    # Open czi file
    # imgArray6d, _ = czird.read(filename)
    # imgArray_bgr = np.squeeze(imgArray6d)   
    # imgArray = imgArray_bgr[:,:,::-1]
    imgArray6d = pylibczirw_tools.read_7darray(filename)
    imgArrayBGR = np.squeeze(imgArray6d)
    imgArray = imgArrayBGR[:,:,::-1]
    print(imgArray.shape)
    
elif filename.endswith(".SVS"):
    # Open svs file
    slide = openslide.OpenSlide(filename)
    imgArray = np.array(slide.read_region((0,0), 0, slide.level_dimensions[0]))
    print(imgArray.shape)
    #imgArray.tiffsave(r'C:\Temp\testSVStoTIFF.tiff')
    
elif filename.endswith(".tiff") or filename.endswith(".tif") or filename.endswith(".TIFF"):
    # Open tif file
    imgArray = tifffile.imread(filename)
    print(imgArray.shape)
    
elif filename == '':
    print("Please choose an image")

else:
    print("Image format is not supported")

if np.any(imgArray):
   viewer = napari.view_image(imgArray)
   napari.run()


