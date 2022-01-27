#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import glob
import shutil
import os
import openslide
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import math
import sys
import collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading
import cv2
import json
import matplotlib.pyplot as plt 
import math
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


os.listdir('../../../../../home/wzp/jupyterlab')


# In[ ]:


os.listdir('../../../../../group_homes/Pediatrics/home/share/PLN')


# In[ ]:


# os.listdir('../../../../../group_homes/Pediatrics/home/share/PLN/PAS')


# In[3]:


Project = 'Pediatrics'
data_set = 'HE'

mount_path = f'/home/wzp/jupyterlab/{Project}/{Project}'
ALL_WSI_PATH = f'/group_homes/Pediatrics/home/share/PLN/{data_set}'
RAW_DATA_PATH = f'{mount_path}/raw_data'
PATCHES_ABS_PATH = f'/group_homes/ALL_DATA/{Project}/{data_set}'
PATCHES_PATH = f'{mount_path}/patches/'


def make_all_dir():
    if os.path.exists(RAW_DATA_PATH):
        pass
    else:
        os.makedirs(RAW_DATA_PATH)
        print('Done making RAW_DATA_PATH')

    if os.path.exists(PATCHES_PATH):
        pass
    else:
        os.makedirs(PATCHES_PATH)
        print('Done making PATCHES_PATH')
        
    if os.path.exists('./labels/'):
        pass
    else:
        os.makedirs('./labels/')
        os.makedirs('./labels/patch/')
        os.makedirs('./labels/slide/')
        print('Done making PATCHES_LABELS_PATH')
    
    if os.path.exists('./models/'):
        pass
    else:
        os.makedirs('./models/')
        os.makedirs('./models/patch/')
        os.makedirs('./models/slide/')
        print('Done making PATCHES_MODELS_PATH')
    
    
make_all_dir()


# In[ ]:


def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # 灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red']=color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    #青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
    return dict


def get_color(frame):
    hsv = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum+=cv2.contourArea(c)
        if sum > maxsum :
            maxsum = sum
            color = d
    return color


# In[ ]:


def get_cut_point(slide):
    thumbnail = slide.get_thumbnail((1024, 1024))
    width, height = thumbnail.width, thumbnail.height
    thumbnail = ImageEnhance.Contrast(thumbnail).enhance(2.5)
    line = thumbnail.resize((width, 1), Image.ANTIALIAS).convert('L')
    line = np.array(line)
    index = width//4
    cut_point = (np.argmax(line[0][index:width-index])+index)/width
    return cut_point


def fn(path, ptid, slide, suffix):
    try:
        file = open_slide(f'{path}/{ptid}/{slide}.{suffix}')
        cp = get_cut_point(file)
    except Exception as e:
        cp = -1
    return ptid, slide, cp


def tile(pg, begin, row, col, slide_name):
    tile = pg.dz.get_tile(pg.level, (row, col))
    w, h = tile.size
    if w < 450 or h < 450:
        return
    color = get_color(tile)
    if color not in ['red', 'purple']:
        return
    temp = ImageEnhance.Brightness(tile).enhance(1.2)
    temp = ImageEnhance.Contrast(temp).enhance(1.2)
    temp = temp.convert('L')
    bw = temp.point(lambda x: 0 if x < 225 else 1, '1')
    avgBkg = np.average(bw)
    if avgBkg >= 0.70:
        return
    tile_name = '{}_{}.jpeg'.format(row-begin, col)
    lnk_path = os.path.join(pg.dest_path, pg.ptid, slide_name, str(pg.mag), tile_name)
    tile.save(lnk_path, quality=95)

    
    
class PatchGenerator:
    def __init__(self, slide_path, dest_path, ptid, tile_size, mag=5, avgBkg=0.4):
        self.slide_path = slide_path
        self.dest_path = dest_path
        self.ptid = ptid
        self.tile_size = tile_size
        self.mag = mag
        self.avgBkg=avgBkg
        self.slide = open_slide(self.slide_path)
        self.dz = DeepZoomGenerator(self.slide, tile_size=self.tile_size, overlap=1.)
        self.level = self.dz.level_count - int(math.log(40/self.mag, 2)) - 1
        self.cols, self.rows = self.dz.level_tiles[self.level]
        print('\tslide size', self.cols, self.rows)
        self.queue = []
        self.lock = threading.Lock()
        
    def generate(self, begin, end, slide_name):
        patch_store_path = os.path.join(
            self.dest_path, 
            self.ptid, 
            slide_name, 
            str(self.mag)
        )
        
        if os.path.exists(patch_store_path):
            print('\t Already Done!')
            return
        
        os.makedirs(patch_store_path)
        cols, rows = self.dz.level_tiles[self.level]
        for row in range(begin, end):
            for col in range(self.rows):
                self.queue.append(
                t.submit(tile, self, begin, row, col, slide_name)
                )
        print('\tDone submitting %d patches' % ((end-begin)*self.rows)) 
        
    def start(self):
        if not os.path.exists(os.path.join(self.dest_path, self.ptid)):
            os.mkdir(os.path.join(self.dest_path, self.ptid))
        
        basename = os.path.basename(self.slide_path).split('.')[0]
        print('\tPatient[{}] cases {}'.format(self.ptid, basename))
        
        if len(basename.split('+')) == 2930039123:
#         if len(basename.split('+')) == 2:
            name1, name2 = basename.split('+')
            
            cut_point = int(memo[(self.ptid, name1+'+'+name2)] * self.cols)
            
            print('\tSave patch from', name1)
            self.generate(0, cut_point, name1)

            print('\tSave patch from', name2)
            self.generate(cut_point, self.cols, name2)

        else:
            print('\tSave patch from', basename)
            self.generate(0, self.dz.level_tiles[self.level][0], basename)


# In[ ]:


def joint_patch(path, patch_size=32):
    patches = os.listdir(path)
    width = -1
    height = -1
    for i in patches:
        if '.ipy' in i:
            continue
        y, x = list(map(int, i.split('.')[0].split('_')))
        width = max(width, x)
        height = max(height, y)
        
    print('width', width, 'height', height)
    
    arr = np.ones([(width+1)*patch_size, (height+1)*patch_size, 3])
    arr.fill(255)
    for i in patches:
        if '.ipy' in i:
            continue
        y, x = list(map(int, i.split('.')[0].split('_')))
        arr[x*patch_size:x*patch_size+patch_size,y*patch_size:y*patch_size+patch_size] =             np.array(Image.open(os.path.join(path, i)).resize((patch_size,patch_size), Image.ANTIALIAS))
    return width, height, arr


def reading_json(mount_path, Project, json_file, ratio, label):
    basename = os.path.basename(json_file).split('.')[0]
    ptid = basename.split('-')[0]
    slides = [basename.split('-')[1]]
    print('PTID:', ptid, 'SLIDE:', slides)

    raw_path = glob.glob('{}/{}/raw_data/{}/{}*'.format(mount_path, Project, ptid, '+'.join(slides)))
    raw_file = open_slide(raw_path[0])
        
    width, height = raw_file.dimensions
    mask = np.zeros([height//ratio,width//ratio,1], np.uint8)
    
    try:
        json.load(open(json_file))
    except Exception:
        print(json_file,'not exist')
        return
    for poly in json.load(open(json_file)):
        try:
            a = poly['properties']['classification']['name']
        except Exception:
            print('ERROR')
            return
        if poly['properties']['classification']['name'] != label:
            continue
        _poly = poly['geometry']['coordinates']
        if len(_poly) != 1:
            _poly = _poly[1] if len(_poly[1]) > len(_poly[0]) else _poly[0]
        try:
            cv2.fillPoly(mask, np.asarray(_poly).astype('int')//ratio, 255)
        except Exception:
            pass
        
    print('\tDone making mask!')
    dz = DeepZoomGenerator(raw_file, tile_size=512, overlap=1.)
    level = dz.level_count - int(math.log(ratio, 2)) - 1
    cols, rows = dz.level_tiles[level]

    for idx, slide in enumerate(slides):
        print('\tselect '+label+' patch from slide', slide)
        if idx != 0:
            cut_point = cut_point_record[(ptid, '+'.join(slides))]
            print('\tCut point', cut_point)
        
        patches_path = f'{mount_path}/{Project}/patches_sysucc_20/{ptid}/{slide}/{mag}/'
        tumor_patches_path = f'{mount_path}/{Project}/patches_sysucc_20/{ptid}/{slide}/{mag}_'+label+'/'
        if os.path.exists(tumor_patches_path):
            shutil.rmtree(tumor_patches_path)
    
        os.mkdir(tumor_patches_path)
        for patch in os.listdir(patches_path):
            if '.ipy' in patch:
                continue
            x, y = list(map(int, patch.split('.')[0].split('_')))
            x, y = x*512, y*512
            if idx != 0:
                x += int(cut_point * cols)*512
                
            im_patch = mask[y:y+512, x:x+512,:]
            percent = im_patch.sum()/(255*512*512)
            
            if percent >= 0.5:
                os.symlink(
                    patches_path+patch, 
                    tumor_patches_path+patch
                )
                
                
def show_joint_image(Project, Patient_id, Slide_id, Mag='5'):
    path = f'../{Project}/patches_sysucc_20/{Patient_id}/{Slide_id}/{Mag}/*'
    width, height, im = joint_patch(path[:-1])
    plt.figure(figsize=(8, 8))
    plt.imshow(im.astype(np.uint8))
    plt.xticks(range(0, height*32 + 32, 32))
    plt.yticks(range(0, width*32 + 32, 32))
    plt.grid()
    plt.axis('off')
    plt.show()
    plt.close()
    return im


def get_txt(Project, Pat_id, labels, Mag='5',set_id='train'):
    f = open(f'./labels/patch/{set_id}.txt','a')
    path = f'{PATCHES_PATH}{Pat_id}/'
    slide_list = os.listdir(path)
    for i in range(len(labels)):
        for sl in slide_list:
            patch_list = glob.glob(f'{PATCHES_PATH}{Pat_id}/{sl}/{Mag}_{labels[i]}/*.jpeg')
            for pl in patch_list:
                f.write(pl+' '+str(i)+'\n')
    f.close()
    
    
def get_test(Project, Pat_id, slide_id, Mag='5',set_id='test'):
    f = open(f'./labels/patch/{set_id}.txt','w')
    path = f'{PATCHES_PATH}{Pat_id}/'
    patch_list = glob.glob(f'{PATCHES_PATH}{Pat_id}/{slide_id}/{Mag}/*.jpeg')
    for pl in patch_list:
        f.write(pl+' '+str(0)+'\n')
    f.close()
    
    
def get_all(Project, Pat_id, Mag='5',set_id='trainval'):
    f = open(f'./labels/patch/{set_id}.txt','a')
    path = f'{PATCHES_PATH}{Pat_id}/'
    slide_list = os.listdir(path)
    for sl in slide_list:
        patch_list = glob.glob(f'{PATCHES_PATH}{Pat_id}/{sl}/{Mag}/*.jpeg')
        for pl in patch_list:
            f.write(pl+' '+str(0)+'\n')
    f.close()


# In[ ]:


try:
    memo = np.load('./cp.npy', allow_pickle=True).item()
except Exception:
    memo = {}
t =  ThreadPoolExecutor(max_workers=8, thread_name_prefix='generate')


# In[ ]:


all_wsi = os.listdir(ALL_WSI_PATH)
all_ptid = list(set(['-'.join(i.split('-')[:-1]) for i in all_wsi]))
                
for ptid in all_ptid:
    print(ptid)
    if os.path.exists(f'{RAW_DATA_PATH}/{ptid}'):
        shutil.rmtree(f'{RAW_DATA_PATH}/{ptid}')
    os.mkdir(f'{RAW_DATA_PATH}/{ptid}')
    for slide in all_wsi:
        if ptid in slide:
            suffix = slide.split('.')[-1]
            cases = slide.split('.')[0]
            cases = [i for i in cases.split('-')[1:] if i]
            new_name = '+'.join(cases) + '.' + suffix
            print(ptid, slide, new_name)
            os.symlink(os.path.join(ALL_WSI_PATH, slide),
                       os.path.join(f'{RAW_DATA_PATH}/{ptid}/{new_name}'))


# In[ ]:


for ptid in os.listdir(RAW_DATA_PATH):
    print(ptid)
    tasks = []
    for slide in os.listdir(f'{RAW_DATA_PATH}/{ptid}/'):
        ends = slide.split('.')[-1]
        if ends not in ['svs', 'tif', 'SVS', 'TIF']:
            continue
        slide_name = slide.split('.')[0]
        suffix = slide.split('.')[1]
        if (ptid, slide_name) not in memo or memo[(ptid, slide_name)] == -1:
            tasks.append(t.submit(fn, RAW_DATA_PATH, ptid, slide_name, suffix))

    wait(tasks)
    for future in as_completed(tasks):
        ptid, slide, cp = future.result()
        memo[(ptid, slide)] = cp
        print(ptid, slide, cp)
    np.save('./cp.npy', memo)


# In[ ]:


ptids = os.listdir(RAW_DATA_PATH)
mag = 40
tile_size = 512
avgBkg = 0.6

for ptid in ptids:
    print('Start tile slides from patient ', ptid)
    for cases in os.listdir(os.path.join(RAW_DATA_PATH, ptid)):
        ends = cases.split('.')[-1]
        if ends in ['svs', 'tif', 'SVS', 'TIF']:
            try:
                pg = PatchGenerator(
                    slide_path=os.path.join(RAW_DATA_PATH, ptid, cases), 
                    ptid=ptid, 
                    dest_path=PATCHES_PATH, 
                    tile_size=tile_size, 
                    mag=mag,
                    avgBkg=avgBkg)
                pg.start()
                wait(pg.queue) 
                print('\tdone')
            except Exception:
                print(ptid, 'ERROR!')
                pass


# In[15]:


print(f'All the segmentation of {data_set}-{mag}X done!!!')


# In[23]:


all_data = os.listdir(f'../{Project}/patches/')
# all_data.extend(os.listdir(f'../{Project}/patches_180new/'))
# all_data.extend(os.listdir(f'../{Project}/patches_sysucc_20/'))

label_df = pd.read_excel('LN_label_0928.xlsx')
label_df['pat_id'] = label_df['pat_id'].astype(str)
label_df = label_df[label_df['pat_id'].isin(all_data)]
all_data = list(set(all_data)&set(label_df['pat_id'].tolist()))

pat_dict = {}
for pat in all_data:
    pat_dict[pat] = {}
    pat_dict[pat]['patient-label'] = label_df[label_df['pat_id']==pat]['label'].tolist()[0]
        
with open('./pat_labels.json3','w') as f:
    data_map = json.dump(pat_dict, f)


# In[25]:


# pat_dict


# In[27]:


# for pat in label_df['pat_id'].tolist():
#     if pat in list(pat_dict.keys()):
#         if label_df[label_df['pat_id']==pat]['label'].tolist()[0] == pat_dict[pat]['patient-label']:
#             print('TRUE')
#         else:
#             print('FALSE')


# In[33]:


os.listdir('/home/wzp/.cache/torch/hub/checkpoints')


# In[ ]:





# In[ ]:





# In[ ]:




