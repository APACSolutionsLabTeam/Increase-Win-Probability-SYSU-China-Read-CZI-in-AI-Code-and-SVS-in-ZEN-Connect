# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:48:47 2021

@author: ZSPANIYA
"""
import os
import time
import argparse  
#from czi_utils import SVSToCZI

import os
from tifffile import TiffFile
from pylibCZIrw import czi as pyczi
from imagecodecs import jpeg2k_decode



class CZIConverter:
    def __init__(self, bfconvert_path: str):
        self.bfconvert_path = bfconvert_path

    def __call__(self, input_file_path: str, output_file_path: str) -> None:
        print("converting {} to {}".format(input_file_path, output_file_path))
        exitCode = os.system('{} -overwrite "{}" "{}"'.format(self.bfconvert_path, input_file_path, output_file_path))
        if exitCode != 0:
            raise Exception("exception occurred, please check logs above")


def SVSToCZI(filename):
    scale = None
    with TiffFile(filename) as tif:
        for series in tif.pages:
            for tag in series.tags:
                if str(tag.name) in ["ImageDescription"] and "MPP" in str(tag.value):
                    tagValue = str(tag.value).replace(" ", "")
                    #print(tagValue)
                    scale_start_idx = tagValue.find("MPP=") + 4
                    scale = float(tagValue[scale_start_idx:scale_start_idx+4])
                    break
            if scale:
                break
    
    with TiffFile(filename) as tif:
       
        for idx, page in enumerate(tif.pages):
            
            with pyczi.create_czi(r"C:\temp\sample.czi") as czidoc:
                czidoc.write(page.asarray(), plane={"T": 0,
                                                    "Z": 0,
                                                    "C": 0},
                             scene=0)
                czidoc.write_metadata(scale_x=scale, scale_y=scale, scale_z=1)
            return



parser = argparse.ArgumentParser(description='Read Filename')
parser.add_argument('-f', action="store", dest='filename')


#get the arguments
args = parser.parse_args()

imageFile = args.filename

#imageFile = r"C:\temp\SYSU\IPM_SYSU_Demo\SVS_Images\7255.SVS"
print("Running SVS to CZI conversion")
SVSToCZI(imageFile)

