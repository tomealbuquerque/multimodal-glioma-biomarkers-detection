#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition lrz-dgx-a100-80x8
#SBATCH --mem=180G 
#SBATCH -t 72:00:00
#SBATCH -o outputs_512_bin_fold_0.out
#SBATCH -e enroot_test_512_bin_fold_0.err
##SBATCH --mail-user=tome.albuquerque@tum.de
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END

srun --container-image='/dss/dsshome1/0E/ge92tis2/enroot/my_container.sqsh'\
	--container-workdir=/dss/dsshome1/0E/ge92tis2/deep-multimodal-glioma-prognosis/MIL_big \
	--container-mounts=/dss/dssfs04/pn25ke/pn25ke-dss-0001 \
  	python mil_train_bin.py --fold 0 --model 'checkpoint_best_512_bin_multi_level_fold_0.pth'