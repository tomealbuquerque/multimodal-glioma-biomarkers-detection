"""
Loop over images, extract patches (ps*ps*ps from center_of_mass), save a new npy file of ROI
"""
# import openslide
# import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, center_of_mass



folder = '/dss/dssfs04/pn25ke/pn25ke-dss-0001/Glioma_EPIC'
sequence_order = ["t1c", "flair"]
ps = 96

df = pd.read_csv(os.path.join(folder, "TUM_dataset_glioma.csv"))
df = df.dropna(subset=['Pseudonym', 'IDH', 'LOH_1p19q'])

subfolder_ids = []
for root, dirs, files in os.walk(folder):
    if 'preop' in root:
        subfolder_ids.append(root.split(os.sep)[-2])

df = df[df['Pseudonym'].isin(subfolder_ids)]

global_patch_lst = []

def pad_to_shape(patch_size, img_arr):
    # Zero-pads arrays to shape [patch_size, patch_size, patch_size]
    diff_x = patch_size - img_arr.shape[0]
    diff_x_l = diff_x // 2
    diff_x_r = (diff_x // 2) + (diff_x % 2)

    diff_y = patch_size - img_arr.shape[1]
    diff_y_l = diff_y // 2
    diff_y_r = (diff_y // 2) + (diff_y % 2)

    diff_z = patch_size - img_arr.shape[2]
    diff_z_l = diff_z // 2
    diff_z_r = (diff_z // 2) + (diff_z % 2)

    return np.pad(img_arr, ((diff_x_l, diff_x_r),
                            (diff_y_l, diff_y_r),
                            (diff_z_l, diff_z_r)),
                  mode="constant",
                  constant_values=0.)


for _, row in df.iterrows():
    # Load data
    images_dict = {}
    for sequence in sequence_order:
        image_path = os.path.join(folder, row.Pseudonym, 'preop',
                                  f'sub-{row.Pseudonym}_ses-preop_space-sri_{sequence}.nii.gz')
        images_dict[sequence] = nib.load(image_path).get_fdata()

        seg_path = os.path.join(folder, row.Pseudonym, 'preop',
                                f'sub-{row.Pseudonym}_ses-preop_space-sri_seg.nii.gz')
        seg = nib.load(seg_path).get_fdata()
        seg[seg > 0] = 1  # For our purpose here, we binarize the segmentation

    temp_bm = np.zeros(seg.shape)
    temp_bm[images_dict["t1c"] != 0] = 1

    # Normalize entire volume (inside brainmask to) [0;1]
    for sequence in sequence_order:
        images_dict[sequence] = np.clip(images_dict[sequence], np.percentile(images_dict[sequence][temp_bm == 1], 1.),
                                        np.percentile(images_dict[sequence][temp_bm == 1], 99.))
        images_dict[sequence] -= images_dict[sequence][temp_bm == 1].min()
        images_dict[sequence] = images_dict[sequence] / images_dict[sequence][temp_bm == 1].max()
        images_dict[sequence] *= temp_bm
        images_dict[sequence] = images_dict[sequence].astype(np.float32)

    # Label the segmentation and find the largest tumor (to take care of multifocality)
    seg_label, num_features = label(seg, structure=np.ones((3, 3, 3)))  # To ensure full connectivity
    largest_label, largest_label_volume = 0, 0
    for ctr in range(1, num_features + 1):
        if len(seg[seg_label == ctr]) > largest_label_volume:
            largest_label = ctr
            largest_label_volume = len(seg[seg_label == ctr])

    # Find CoM
    seg_target = np.zeros(seg.shape)
    seg_target[seg_label == largest_label] = 1

    c_o_m = center_of_mass(seg_target)
    com_x = int(c_o_m[0])
    com_y = int(c_o_m[1])
    com_z = int(c_o_m[2])

    # Crop (Take care of "out-of-bounds")
    # tmp_patch_lst = []
    # images_dict_new = {}
    for sequence in sequence_order:
        tmp_patch = images_dict[sequence][
                    max(0, com_x - (ps // 2)):min(images_dict[sequence].shape[0], com_x + (ps // 2)),
                    max(0, com_y - (ps // 2)):min(images_dict[sequence].shape[1], com_y + (ps // 2)),
                    max(0, com_z - (ps // 2)):min(images_dict[sequence].shape[2], com_z + (ps // 2))]
        # tmp_patch_lst.append(pad_to_shape(patch_size=ps, img_arr=tmp_patch))
        block_path = image_path = os.path.join(folder, row.Pseudonym, 'preop',
                                  f'sub-{row.Pseudonym}_ses-preop_space-sri_{sequence}.npy')
        np.save(block_path, pad_to_shape(patch_size=ps, img_arr=tmp_patch))
        print(f'{row.Pseudonym} - {sequence} npy saved!')

    print(f'{row.Pseudonym} done!')

# img_path = '/dss/dsshome1/09/ge86jin2/artifact_detection/samples/sample_111920.svs'
# slides = openslide.OpenSlide(img_path)
# dims = slides.level_dimensions[1]
# img = slides.read_region((0,0), 1, dims)
# plt.imshow(img)
# print(slides)
# print(slides.level_dimensions)
# plt.imwrite(img, 'img.jpg')

# data_path = '/dss/dssfs04/pn25ke/pn25ke-dss-0001/albuquerque/Glioma_EPIC/Gliome_EPIC_PhenoData.csv'

# pdf = pd.read_csv(data_path)

# print(pdf.shape)
