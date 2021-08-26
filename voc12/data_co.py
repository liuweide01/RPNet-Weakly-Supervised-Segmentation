import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
import random

# IMG_FOLDER_NAME = "JPEGImages"
# ANNOT_FOLDER_NAME = "Annotations"

base_img_ref_path = '/home/wdliu/VOCCLASS/train/'

IMG_FOLDER_NAME = "images"
ANNOT_FOLDER_NAME = "segmentations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy').item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def load_image_label_from_name(image_name):

    cls_labels_dict = np.load('voc12/cls_labels.npy').item()

    return cls_labels_dict[image_name]

def get_sudo_mask_path(img_name):
    return os.path.join('/home/wdliu/VOC/sudo_mask/', img_name + '.png')


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.png')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img

def get_ref_img(class_index):
    list_path = os.path.join(base_img_ref_path,str(class_index)+'.txt')
    list_ref = open(list_path).read().splitlines()
    ref_name = random.choice(list_ref)
    return ref_name

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None,from_val=False):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.from_val = from_val

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        ref_list = [i for i, value in enumerate(label) if value == 1]
        idx_ref = random.choice(ref_list)
        idx_ref_class = idx_ref +1

        ref_name = get_ref_img(idx_ref_class)

        ref_img = PIL.Image.open(get_img_path(ref_name, self.voc12_root)).convert("RGB")

        # if self.transform:
        # if self.from_val is False:
        ref_img = self.transform(ref_img)

        ref_label = load_image_label_from_name(ref_name)
        ref_label = torch.from_numpy(ref_label)

        # ref_zeros = np.zeros(20)
        # ref_zeros[idx_ref] = 1
        # ref_label = ref_zeros
        # ref_label = torch.from_numpy(ref_label)
        ref_label_common = torch.mul(ref_label, label)
        label_common = (torch.where(ref_label_common == 1)[0][0]).item()

        return name, img, label,ref_name,ref_img,ref_label,label_common


class VOC12PartsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])


        mask_x_path = get_sudo_mask_path(name)

        mask_x = PIL.Image.open(mask_x_path).convert("RGB")

        mask_x = self.transform(mask_x)


        return name, img, label ,mask_x


class VOC12ClsValDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        ref_list = [i for i, value in enumerate(label) if value == 1]
        idx_ref = random.choice(ref_list)
        idx_ref_class = idx_ref +1

        ref_name = get_ref_img(idx_ref_class)

        ref_img = PIL.Image.open(get_img_path(ref_name, self.voc12_root)).convert("RGB")

        # if self.transform:
        ref_img = self.transform(ref_img)

        ref_label = load_image_label_from_name(ref_name)
        ref_label = torch.from_numpy(ref_label)

        ref_label_common = torch.mul(ref_label, label).float()

        # seg_lable = PIL.Image.open(os.path.join('VOC2012EX/VOC2012_SEG_AUG/segmentations/',name))

        return name, img, label,ref_name,ref_img,ref_label,ref_label_common

class VOC12SegmentDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, crop,transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.crop = crop

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        ref_list = [i for i, value in enumerate(label) if value == 1]
        idx_ref = random.choice(ref_list)
        idx_ref_class = idx_ref +1

        ref_name = get_ref_img(idx_ref_class)

        ref_img = PIL.Image.open(get_img_path(ref_name, self.voc12_root)).convert("RGB")

        # if self.transform:
        ref_img = self.transform(ref_img)

        ref_label = load_image_label_from_name(ref_name)
        ref_label = torch.from_numpy(ref_label)

        ref_label_common = torch.mul(ref_label,label).float()
        # ref_zeros = np.zeros(20)
        # ref_zeros[idx_ref] = 1
        # ref_label = ref_zeros
        # ref_label = torch.from_numpy(ref_label)

        mask_x_path = get_sudo_mask_path(name)
        mask_ref_path = get_sudo_mask_path(ref_name)
        mask_x = PIL.Image.open(mask_x_path).convert("RGB")
        mask_ref = PIL.Image.open(mask_ref_path).convert("RGB")
        mask_x = self.transform(mask_x)
        mask_ref = self.transform(mask_ref)

        h, w, c = mask_x.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        return name, img, label,ref_name,ref_img,ref_label ,ref_label_common, mask_x,mask_ref

def Normallize(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] / 255. - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] / 255. - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] / 255. - mean[2]) / std[2]

    return proc_img

import torch.nn.functional as F
import numpy as np
import cv2

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None,transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=transform,from_val=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label,ref_name,ref_img,ref_label,label_common = super().__getitem__(idx)

        rounded_size = (int(round(img.shape[1]/self.unit)*self.unit), int(round(img.shape[2]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[1]*s),
                           round(rounded_size[0]*s))
            # target_size = (round(rounded_size[0]*s),
            #                round(rounded_size[1]*s))

            # s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            h,w = target_size
            img = img.transpose(1,2,0)
            s_img =cv2.resize(img,(h,w))
            s_img = s_img.transpose(2,0,1)
            img = img.transpose(2, 0, 1)
            ms_img_list.append(s_img)

        # if self.inter_transform:
        #     for i in range(len(ms_img_list)):
        #         ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label,ref_name,ref_img,ref_label ,label_common

# class VOC12ClsDatasetMSF(VOC12ImageDataset):
#
#     def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
#         super().__init__(img_name_list_path, voc12_root, transform=None)
#         self.scales = scales
#         self.unit = unit
#         self.inter_transform = inter_transform
#
#         self.label_list = load_image_label_list_from_npy(self.img_name_list)
#
#     def __getitem__(self, idx):
#
#         name, img = super().__getitem__(idx)
#
#         label = torch.from_numpy(self.label_list[idx])
#
#         ref_list = [i for i, value in enumerate(label) if value == 1]
#
#         ref_label_list = []
#         ref_name_list = []
#         ref_img_list = []
#
#         # for j in range(5):
#         #
#         #     idx_ref = random.choice(ref_list)
#         #     ref_index_list.append(idx_ref)
#         #
#         #     idx_ref_class = idx_ref +1
#         #
#         #     ref_name = get_ref_img(idx_ref_class)
#         #
#         #     ref_img = PIL.Image.open(get_img_path(ref_name, self.voc12_root)).convert("RGB")
#         #
#         #     # if self.transform:
#         #     ref_img = self.inter_transform(ref_img)
#         #
#         #     # ref_zeros = np.zeros(20)
#         #     # ref_zeros[idx_ref] = 1
#         #     # ref_label = ref_zeros
#         #     # ref_label = torch.from_numpy(ref_label)
#         #     ref_label = load_image_label_from_name(ref_name)
#         #     ref_label = torch.from_numpy(ref_label)
#         #
#         #     ref_label_common = torch.eq(ref_label,label).float()
#         #     ref_label_list.append(ref_label)
#         #     ref_name_list.append(ref_name)
#         #     ref_img_list.append(ref_img)
#
#         idx_ref = random.choice(ref_list)
#
#         idx_ref_class = idx_ref +1
#
#         ref_name = get_ref_img(idx_ref_class)
#
#         ref_img = PIL.Image.open(get_img_path(ref_name, self.voc12_root)).convert("RGB")
#
#         # if self.transform:
#         ref_img = self.inter_transform(ref_img)
#
#
#         # ref_label = load_image_label_from_name(ref_name)
#         # ref_label = torch.from_numpy(ref_label)
#         #
#         # ref_label_common = torch.eq(ref_label,label).float()
#         # ref_label_list.append(ref_label)
#         ref_name_list.append(ref_name)
#         ref_img_list.append(ref_img)
#
#         rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))
#
#         ms_img_list = []
#         for s in self.scales:
#             target_size = (round(rounded_size[0]*s),
#                            round(rounded_size[1]*s))
#             s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
#             ms_img_list.append(s_img)
#
#         if self.inter_transform:
#             for i in range(len(ms_img_list)):
#                 ms_img_list[i] = self.inter_transform(ms_img_list[i])
#
#         msf_img_list = []
#         for i in range(len(ms_img_list)):
#             msf_img_list.append(ms_img_list[i])
#             msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
#
#         # for item in ms_img_list:
#         #     item_nor = Normallize(item)
#         #     # msf_img_list
#         #     loc = list.index(item)
#         #     list.remove(item)
#         #     list.insert(loc,item_nor)
#         # ref_img = Normallize(ref_img)
#
#         return name, msf_img_list, label,ref_name_list,ref_img_list,ref_label_list,idx_ref