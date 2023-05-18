import os
import random
import pandas as pd
import numpy as np


def get_images_labels(modality, fold=0, dataset_type='train', data_type='epic'):
    """
    data_type is either tum epic or tcga dataset
    """
    
    root_dir = 'deep-multimodal-glioma-prognosis'
    data_path = 'data_tum_epic' if data_type == 'epic' else 'data_multimodal_tcga'
    pickle_path = os.path.join(root_dir, data_path, 'multimodal_glioma_data.pickle')
    # codeletion column name is different in csv. TUM: 1p19q; TCGA: ioh1p15q
    codeletion = '1p19q' if data_type == 'epic' else 'ioh1p15q'

    images, labels = pd.read_pickle(pickle_path)[fold][dataset_type]

    ims, lbs = [], []

    for mod in modality:
        if data_type == 'epic':
            ims += [x[mod] for x in images]
            
        else:
            if '_block' in mod:
                ims += [os.path.join(data_path, f[mod]) for f in images]
            else:
                ims += [os.path.join(data_path, f[mod].replace('\\', os.sep)) for f in images]
        
        lbs += [lb['idh1']+lb[codeletion] for lb in labels]

    return ims, np.array(lbs, dtype=np.int64), ims


def get_label_distribution(data_type='epic'):
    
    root_dir = 'deep-multimodal-glioma-prognosis'
    data_path = 'data_tum_epic' if data_type == 'epic' else 'data_multimodal_tcga'
    pickle_path = os.path.join(root_dir, data_path, 'multimodal_glioma_data.pickle')
    # codeletion column name is different in csv. TUM: 1p19q; TCGA: ioh1p15q
    codeletion = '1p19q' if data_type == 'epic' else 'ioh1p15q'
    
    full_data = pd.read_pickle(pickle_path)
    # images, labels = pd.read_pickle(pickle_path)[fold][dataset_type]
    
    train_folds, test_folds = [], []

    for i in range(5):
        _, train_labels = full_data[i]['train']
        _, test_labels = full_data[i]['test']
        print(train_labels)
        break
        # tr_lb_list = [lb['idh1']+lb[codeletion] for lb in train_labels]
        # te_lb_list = [lb['idh1']+lb[codeletion] for lb in test_labels]
        # train_folds += {x: tr_lb_list.count(x) for x in tr_lb_list}
        # test_folds += {x: te_lb_list.count(x) for x in te_lb_list}

    return train_folds, test_folds


if __name__ == "__main__":
    train_folds, test_folds = get_label_distribution(data_type='epic')
    # print(f"""
    # train_folds: {train_folds},

    # test_folds: {test_folds}
    # """)