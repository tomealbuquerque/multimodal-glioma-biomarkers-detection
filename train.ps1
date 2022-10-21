$MODELS = @("UniMRI")
$MRIS = @("flair", "t1", "t1ce", "t2")
foreach($MODEL in $MODELS) {
	foreach($MRI in $MRIS) {
   		for($FOLD=0; $FOLD -le 5; $FOLD++) {
			python train.py --method $MODEL --MRI_type $MRI --fold $FOLD 
	}
    }
}

