import os.path
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2 as cv

from scipy.ndimage import zoom
from torch.utils.data import DataLoader

import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def load_clinical_map_from_txt(txt_path: str):
    """
    txt 每行格式（逗号分隔）：
      Folder,Age,Sex,Atrial,Alcohol,NIHSS,Monocyte_Percentage,Lymphocyte_Percentage
    其中 Folder 就是 patient_name。
    """
    df = pd.read_csv(txt_path)

    patient_col = df.columns[0]  # e.g., "Folder"

    need = ["Age", "Sex", "Atrial", "Alcohol", "NIHSS", "Monocyte_Percentage", "Lymphocyte_Percentage"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{txt_path} 缺少列：{missing}，当前列：{list(df.columns)}")

    clinical_map = {}
    for _, r in df.iterrows():
        patient = str(r[patient_col]).strip()
        vec = np.array([
            pd.to_numeric(r["Age"], errors="coerce"),
            pd.to_numeric(r["Sex"], errors="coerce"),
            pd.to_numeric(r["Atrial"], errors="coerce"),
            pd.to_numeric(r["Alcohol"], errors="coerce"),
            pd.to_numeric(r["NIHSS"], errors="coerce"),
            pd.to_numeric(r["Monocyte_Percentage"], errors="coerce"),
            pd.to_numeric(r["Lymphocyte_Percentage"], errors="coerce"),
        ], dtype=np.float32)

        clinical_map[patient] = vec

    return clinical_map


def window_transform(ct_array, windowWidth=400, windowCenter=40, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newing = (ct_array - minWindow) / float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    if not normal:
        newing = (newing * 255).astype('uint8')
    return newing


def read_nii(path, resize_width=None, preprocess=True, crop_indexes=None):
    dst_z, dst_x, dst_y = 64, 128, 128

    data_nii = nib.load(path)

    if preprocess == True:
        temp = window_transform(data_nii.get_fdata(), normal=True)
    else:
        temp = data_nii.get_fdata()

    temp = temp.astype(np.float32)

    t_max, t_min = temp.max(), temp.min()
    temp = (temp - t_min) / (t_max - t_min)

    temp = zoom(temp, (dst_z / temp.shape[0], dst_x / temp.shape[1], dst_y / temp.shape[2]))

    if resize_width != None:
        temp = cv.resize(temp, (resize_width, resize_width))

    if crop_indexes != None:
        temp = temp[crop_indexes[0][0]:crop_indexes[0][1], crop_indexes[1][0]:crop_indexes[1][1]]

    temp = np.expand_dims(temp, axis=0)

    return temp


def read_nii6(path, resize_width=None, preprocess=True, crop_indexes=None):
    dst_z, dst_x, dst_y = 64, 128, 128

    data_nii = nib.load(path)

    if preprocess == True:
        temp = window_transform(data_nii.get_fdata(), normal=True)
    else:
        temp = data_nii.get_fdata()

    temp = temp.astype(np.float32)

    temp = zoom(temp, (dst_z / temp.shape[0], dst_x / temp.shape[1], dst_y / temp.shape[2]))

    if resize_width != None:
        temp = cv.resize(temp, (resize_width, resize_width))

    if crop_indexes != None:
        temp = temp[crop_indexes[0][0]:crop_indexes[0][1], crop_indexes[1][0]:crop_indexes[1][1]]

    temp = np.expand_dims(temp, axis=0)

    return temp


# ============================================================================
# 合并数据集支持：单个 train_file.txt（行首带 CENTER@ 前缀）+ 单个 train_text.txt
# ============================================================================

def read_and_process_data_string_combined(data_string, data_src_path):  # wmh高信号
    data_string = data_string.strip()

    # 行格式: CENTER@patient_name,mrs_label,DWI_b0,DWI_b1000,T1,T2
    if '@' in data_string.split(',')[0]:
        center_name, rest = data_string.split('@', 1)
        center_name = center_name.strip()
    else:
        # 兼容没有前缀的情况：退回旧逻辑用第一个下划线段
        center_name = data_string.split('_')[0]
        rest = data_string

    split_list = rest.split(',')

    patient_name, mrs_label, DWI_b0_name, DWI_b1000_name, T1_name, T2_name = \
        split_list[0].strip(), split_list[1].strip(), split_list[2].strip(), \
            split_list[3].strip(), split_list[4].strip(), split_list[5].strip()

    mrs_label = int(mrs_label)
    if mrs_label in [0, 1, 2]:
        mrs_label = 0
    elif mrs_label in [3, 4, 5, 6]:
        mrs_label = 1

    data_path = os.path.join(data_src_path, center_name)

    T2_nii_wmh = os.path.join(data_path, 'MR_NII_mni_wmh', patient_name, T2_name)
    t2_img_wmh = read_nii6(T2_nii_wmh, resize_width=None, preprocess=False)
    t2_mask = t2_img_wmh

    DWI_b1000_nii_raw = os.path.join(data_path, 'MR_NII_mni', patient_name, DWI_b1000_name)
    T1_nii_raw = os.path.join(data_path, 'MR_NII_mni', patient_name, T1_name)
    T2_nii_raw = os.path.join(data_path, 'MR_NII_mni', patient_name, T2_name)

    dwi_b1_img_raw = read_nii(DWI_b1000_nii_raw, resize_width=None, preprocess=False)
    t1_img_raw = read_nii(T1_nii_raw, resize_width=None, preprocess=False)
    t2_img_raw = read_nii(T2_nii_raw, resize_width=None, preprocess=False)

    dwi_b1_img_wmh = dwi_b1_img_raw * t2_mask
    t1_img_wmh = t1_img_raw * t2_mask
    t2_img_wmh = t2_img_raw * t2_mask

    key_name = f'{center_name}_{patient_name}'
    return key_name, {
        'label': mrs_label,
        'patient': patient_name,
        'dwi_b1_img': dwi_b1_img_raw,
        't1_img': t1_img_raw,
        't2_img': t2_img_raw,
        'dwi_b1_img_WMH': dwi_b1_img_wmh,
        't1_img_WMH': t1_img_wmh,
        't2_img_WMH': t2_img_wmh,
        't2_mask': t2_mask,
        'dwi_b1_path': DWI_b1000_nii_raw,
        't1_path': T1_nii_raw,
        't2_path': T2_nii_wmh,
    }


class MRViewDataset_text_combined(Dataset):
    """
    单个合并 label 文件 + 单个合并 clinical txt 的 Dataset。
    接口与 MRViewDataset_text01 完全一致 (data_files, data_src_path, text_file)，
    返回 (dwi_t1_t2_data, dwi_t1_t2_data_WMH, clinical_vec, target)。

    data_files: list[str]，通常就是 ['train_file.txt']（单个合并文件）。
    text_file : str，合并后的 train_text.txt。
    """

    def __init__(self, data_files, data_src_path, text_file):
        super().__init__()

        self.clinical_map = load_clinical_map_from_txt(text_file)
        self.default_vec = np.zeros((7,), dtype=np.float32)

        self.data_dict = {}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for data_file in data_files:
                with open(data_file, 'r') as f:
                    data_strings = f.readlines()
                for data_string in data_strings:
                    if not data_string.strip():
                        continue
                    futures.append(
                        executor.submit(read_and_process_data_string_combined,
                                        data_string, data_src_path))

            for future in as_completed(futures):
                key_name, data_entry = future.result()
                self.data_dict[key_name] = data_entry

        print(len(self.data_dict.keys()))
        self.keys = list(self.data_dict.keys())

    def __getitem__(self, index):
        key_name = self.keys[index]
        data_ins_dict = self.data_dict[key_name]

        target = int(data_ins_dict['label'])
        patient = str(data_ins_dict['patient']).strip()

        dwi_b1_data = self._ImgtoTensor(data_ins_dict['dwi_b1_img'])
        t1_data = self._ImgtoTensor(data_ins_dict['t1_img'])
        t2_data = self._ImgtoTensor(data_ins_dict['t2_img'])

        dwi_b1_data_WMH = self._ImgtoTensor(data_ins_dict['dwi_b1_img_WMH'])
        t1_data_WMH = self._ImgtoTensor(data_ins_dict['t1_img_WMH'])
        t2_data_WMH = self._ImgtoTensor(data_ins_dict['t2_img_WMH'])

        dwi_t1_t2_data = torch.cat([dwi_b1_data, t1_data, t2_data], dim=0)
        dwi_t1_t2_data_WMH = torch.cat([dwi_b1_data_WMH, t1_data_WMH, t2_data_WMH], dim=0)

        vec_np = self.clinical_map.get(patient, self.default_vec)
        clinical_vec = torch.from_numpy(vec_np).float()

        return dwi_t1_t2_data, dwi_t1_t2_data_WMH, clinical_vec, target

    def __len__(self):
        return len(self.keys)

    def _ImgtoTensor(self, numpy_img):
        return torch.tensor(numpy_img, dtype=torch.float32)
