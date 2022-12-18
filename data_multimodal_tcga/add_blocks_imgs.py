"""
Loop over images, extract patches (ps*ps*ps from center_of_mass)
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import pickle


K_folds_dataset = pd.read_pickle(r'multimodal_glioma_data.pickle')

seg_folder = "tcga"
sequence_order = ["t1", "t1c", "t2", "flair"]
ps = 96
df = pd.read_csv("phenoData.csv")


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


def modify_dictionaries(dictionaries):
    for original_dict in dictionaries:
        Patient_ID = original_dict[sequence_order[0]].split('\\')[1].lower()
        row = df[df['Patient'] == Patient_ID]
        if Patient_ID in blocks_dirs:
            images_dict = {}
            for sequence in sequence_order:
                image_path = os.path.join(row.Dataset.values[0], row.Patient.values[0], 'preop',
                                          f'sub-{row.Patient.values[0]}_ses-preop_space-sri_{sequence}.nii.gz')
                images_dict[sequence] = nib.load(image_path).get_fdata()
                seg_path = os.path.join(row.Dataset.values[0], row.Patient.values[0], 'preop',
                                        f'sub-{row.Patient.values[0]}_ses-preop_space-sri_seg.nii.gz')
                seg = nib.load(seg_path).get_fdata()
                seg[seg > 0] = 1  # For our purpose here, we binarize the segmentation

            temp_bm = np.zeros(seg.shape)
            temp_bm[images_dict["t1c"] != 0] = 1

            # Normalize entire volume (inside brainmask to) [0;1]
            for sequence in sequence_order:
                images_dict[sequence] = np.clip(images_dict[sequence],
                                                np.percentile(images_dict[sequence][temp_bm == 1], 1.),
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
            tmp_patch_lst = []

            for sequence in sequence_order:
                tmp_patch = images_dict[sequence][
                            max(0, com_x - (ps // 2)):min(images_dict[sequence].shape[0], com_x + (ps // 2)),
                            max(0, com_y - (ps // 2)):min(images_dict[sequence].shape[1], com_y + (ps // 2)),
                            max(0, com_z - (ps // 2)):min(images_dict[sequence].shape[2], com_z + (ps // 2))]
                tmp_patch_lst.append(pad_to_shape(patch_size=ps, img_arr=tmp_patch))
                if sequence == 't1c':
                    sequence = 't1ce'
                path_arr = r'{}'.format(original_dict[sequence]).split('\\')
                block_base_path = os.path.join(path_arr[0], path_arr[1])
                tmp_file_arr = path_arr[2].split('.')
                tmp_file_arr[0] += '_block'
                block_file_name = '.'.join(tmp_file_arr)
                block_path = os.path.join(block_base_path, block_file_name + '.npy')
                np.save(block_path, pad_to_shape(patch_size=ps, img_arr=tmp_patch))
                original_dict[sequence + '_block'] = block_path
            # print(row.Dataset.values[0] + "/" + row.Patient.values[0] + " done!")


for current_fold in K_folds_dataset:
    train_set, test_set = current_fold['train'], current_fold['test']
    X_train, X_test = train_set[0].tolist(), test_set[0].tolist()

    original_ds_dirs = [train_sample[sequence_order[0]].split('\\')[1] for train_sample in X_train]
    blocks_dirs = [x for x in os.listdir(seg_folder) if x.upper() in original_ds_dirs]

    modify_dictionaries(X_train)
    modify_dictionaries(X_test)

pickle.dump(K_folds_dataset, open('modified_multimodal_glioma_data.pickle', 'wb'))
# pickle.dump(K_folds_dataset, open(args.name_file, 'wb'))

# # # np.save(ordner+'patches.npy', final_patches_arr)
