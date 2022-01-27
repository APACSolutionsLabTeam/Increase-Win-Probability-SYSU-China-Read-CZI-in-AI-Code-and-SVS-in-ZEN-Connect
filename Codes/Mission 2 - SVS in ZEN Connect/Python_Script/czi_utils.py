import os
from tifffile import TiffFile
from pylibCZIrw import czi as pyczi


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
            
            with pyczi.create_czi(r"C:\temp\SVS_to_CZI_Python\sample.czi") as czidoc:
                czidoc.write(page.asarray(), plane={"T": 0,
                                                    "Z": 0,
                                                    "C": 0},
                             scene=0)
                czidoc.write_metadata(scale_x=scale, scale_y=scale, scale_z=1)
            return
