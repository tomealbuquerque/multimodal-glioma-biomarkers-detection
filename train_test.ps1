$TILES = @(20)
$LOSSES = @("ordinal")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python RNN_train_ordinal_bin_MIL.py --fold 0 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_0.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 1 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_1.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 2 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_2.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 3 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_3.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 4 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_4.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
        }
    }
}

$TILES = @(20)
$LOSSES = @("ordinal")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python RNN_test_ordinal_bin_MIL.py --fold 0 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_0.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_test_ordinal_bin_MIL.py --fold 1 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_1.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_test_ordinal_bin_MIL.py --fold 2 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_2.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_test_ordinal_bin_MIL.py --fold 3 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_3.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_test_ordinal_bin_MIL.py --fold 4 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_4.pth' --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold'
        }
    }
}

$TILES = @(20)
$LOSSES = @("ordinal")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python print_MIL_results_tables.py --fold 0 --s $TILE --mix $MIX  --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold_Pathology_only'
		python print_MIL_results_tables.py --fold 1 --s $TILE --mix $MIX  --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold_Pathology_only'
		python print_MIL_results_tables.py --fold 2 --s $TILE --mix $MIX  --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold_Pathology_only'
		python print_MIL_results_tables.py --fold 3 --s $TILE --mix $MIX  --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold_Pathology_only'
		python print_MIL_results_tables.py --fold 4 --s $TILE --mix $MIX  --weights $LOSS --results_folder '6-Partial_dataset_MIL_bin_multiclass_kfold_Pathology_only'
        }
    }
}

$TILES = @(20)
$LOSSES = @("CE")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python RNN_train_ordinal_bin_MIL.py --fold 0 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_0.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_train_ordinal_bin_MIL.py --fold 1 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_1.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_train_ordinal_bin_MIL.py --fold 2 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_2.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_train_ordinal_bin_MIL.py --fold 3 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_3.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_train_ordinal_bin_MIL.py --fold 4 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_4.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
        }
    }
}


$TILES = @(20)
$LOSSES = @("CE")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python RNN_test_ordinal_bin_MIL.py --fold 0 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_0.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_test_ordinal_bin_MIL.py --fold 1 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_1.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_test_ordinal_bin_MIL.py --fold 2 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_2.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_test_ordinal_bin_MIL.py --fold 3 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_3.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python RNN_test_ordinal_bin_MIL.py --fold 4 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_4.pth' --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
        }
    }
}


$TILES = @(20)
$LOSSES = @("CE")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python print_MIL_results_tables.py --fold 0 --s $TILE --mix $MIX --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python print_MIL_results_tables.py --fold 1 --s $TILE --mix $MIX --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python print_MIL_results_tables.py --fold 2 --s $TILE --mix $MIX --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python print_MIL_results_tables.py --fold 3 --s $TILE --mix $MIX --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
		python print_MIL_results_tables.py --fold 4 --s $TILE --mix $MIX --weights $LOSS --results_folder '15-Partial_dataset_MIL_bin_multiclass_kfold_CE'
        }
    }
}


$TILES = @(4,8,10,20)
$LOSSES = @("CE","ordinal")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	        python RNN_train_ordinal_bin_MIL.py --fold 0 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_0.pth' --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 1 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_1.pth' --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 2 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_2.pth' --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 3 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_3.pth' --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
		python RNN_train_ordinal_bin_MIL.py --fold 4 --s $TILE --mix $MIX --model 'checkpoint_best_512_bin_fold_4.pth' --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
	        python print_MIL_results_tables.py --fold 'all' --s $TILE --mix $MIX  --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
	}
    }
}


$TILES = @(4,8,10,20)
$LOSSES = @("CE")
$MIXTURE = @("mix","global","expected")

foreach($LOSS in $LOSSES) {
    foreach($MIX in $MIXTURE) {
        foreach($TILE in $TILES) {
	 python print_MIL_results_tables.py --fold 'all' --s $TILE --mix $MIX  --weights $LOSS --results_folder '28-Partial_dataset_MIL_bin_multiclass_kfold'
	}
    }
}
