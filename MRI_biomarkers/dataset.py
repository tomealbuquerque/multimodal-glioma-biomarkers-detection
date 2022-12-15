import os
from pathlib import Path
import torchio
import pandas as pd


class MRIDatasets:
    def __init__(self, dataset_path, metadata_path):
        self.dataset_path = dataset_path
        self.metadata = metadata_path

    def tcga(self):
        metadata_df = pd.read_csv(self.metadata)

        imgs = []
        for path, currentDir, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('t1ce.nii.gz'):
                    subject_id = os.path.basename(path)
                    img_path = Path(path + os.sep + 't1ce.nii.gz')
                    row_df = metadata_df[metadata_df['subject_id'] == subject_id]
                    # there might be more than one row found in csv file
                    label = int(row_df['IDH1_mut'].unique()[0] + row_df['loh1p/19q_cnv'].unique()[0])
                    imgs.append(torchio.Subject(t1=torchio.ScalarImage(img_path), label=label,))

        return imgs

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch

    imgs = MRIDatasets(dataset_path='../data_multimodal_tcga/Radiology',
                       metadata_path='../data_multimodal_tcga/patient-info-tcga.csv').tcga()

    subjects_dataset = torchio.SubjectsDataset(imgs)

    test_set_samples = int(len(imgs) * 0.3)
    train_set_samples = len(imgs) - test_set_samples

    trainset, _ = torch.utils.data.random_split(subjects_dataset, [train_set_samples, test_set_samples],
                                                        generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True)
