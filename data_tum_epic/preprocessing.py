# =============================================================================
# Create Kfolds for TUM epic cross-validation
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


def _read_files(root_dir='/dss/dssfs04/pn25ke/pn25ke-dss-0001/Glioma_EPIC'):

    df = pd.read_csv(os.path.join(root_dir, 'TUM_dataset_glioma.csv'))

    df = df.dropna(subset=['IDH', '1p19q'])
    X, y = [], []

    for root, dirs, files in os.walk(root_dir):
        if 'preop' in root:
            subfolder_id = root.split(os.sep)[-2]
            if subfolder_id in df.Pseudonym.values:
                X.append({
                 't1ce_block': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1c.npy'),
                 'flair_block': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_flair.npy'),
                 'flair': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_flair.nii.gz'),
                 't1ce':  os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1c.nii.gz'),
                 't1': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1.nii.gz'),
                 't2': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t2.nii.gz'),
                })

                idx = df[df['Pseudonym'] == subfolder_id]
                y.append({
                    'idh1': int(idx['IDH'].item()),
                    '1p19q': int(idx['1p19q'].item())
                })
    return np.array(X), np.array(y)


def _create_cv_fold(X, y, output_file, seed=1234, fold=5):
    state = np.random.RandomState(seed)
    kfold = KFold(fold, shuffle=True, random_state=state)
    folds = [{'train': (X[tr], y[tr]), 'test': (X[ts], y[ts])} for tr, ts in kfold.split(X, y)]
    pickle.dump(folds, open(output_file, 'wb'))


if __name__ == '__main__':
    X, y = _read_files()
    output_file = os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "data_tum_epic", "multimodal_glioma_data.pickle",)
    _create_cv_fold(X, y, output_file=output_file)
