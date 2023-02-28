# =============================================================================
# Create dataset from top tiles in each class for MLP train
# =============================================================================

import pickle
import numpy as np
import pandas as pd
import os

pd.options.mode.chained_assignment = None

# =============================================================================
# sort by top  value per class (mix)
# ============================================================================= 

def get_grids(typee, fold, K,dataset):
        
    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}{dataset}.csv'), header=None) 
    
     
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f

    
    start_index=0
    
    for i in range(len(lib)):

        len_grid=len(lib[i]['tiles_coords'])

        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
     
        
        grids_prob_tmp1=grids_prob_tmp
        grids_prob_tmp2=grids_prob_tmp
        grids_prob_tmp3=grids_prob_tmp
        
        grids_prob_tmp1["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp2["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp3["tiles_coords"] = lib[i]['tiles_coords']
        
        df_final1 = grids_prob_tmp1.sort_values(by=0, ascending=[False]).reset_index()
        df_final2 = grids_prob_tmp2.sort_values(by=1, ascending=[False]).reset_index()
        df_final3 = grids_prob_tmp3.sort_values(by=2, ascending=[False]).reset_index()
        
        dfs = [df_final1.loc[0:K-1,'tiles_coords'],df_final2.loc[0:K-1,'tiles_coords'],df_final3.loc[0:K-1,'tiles_coords']]
    
        result = pd.concat(dfs)
    
        lib[i].update({"sorted_coords": result.tolist()})
        
        start_index += len_grid

    return np.array(lib), np.array(targets)


def create_max_mix_tiles_dataset(name_file,fold,K,dataset):

    X_train, Y_train = get_grids('train',fold,K,dataset)
    X_test, Y_test = get_grids('test',fold,K,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]

    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    
    
# =============================================================================
# sort by global top value 
# =============================================================================    
    
def get_grids_top(typee,fold,dataset):
    
    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}{dataset}.csv'), header=None) 
    
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
    
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        # print(len_grid)
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        
        df = grids_prob_tmp.max(axis=1)
        
        df = pd.DataFrame(grids_prob_tmp.max(axis=1))
        
        df["tiles_coords"] = lib[i]['tiles_coords']
        
        df_final= df.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        
        start_index += len_grid
        # print(start_index)
    
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset(name_file,fold,dataset):

    X_train, Y_train = get_grids_top('train',fold,dataset)
    X_test, Y_test = get_grids_top('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    

# =============================================================================
# sort by global top value - binary 
# ============================================================================= 

def get_grids_bin(typee, fold, K,dataset):
        
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
        
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
     
        
        grids_prob_tmp1=grids_prob_tmp
        grids_prob_tmp2=grids_prob_tmp
        grids_prob_tmp3=grids_prob_tmp
        
        grids_prob_tmp1["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp2["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp3["tiles_coords"] = lib[i]['tiles_coords']
        
        df_final2 = grids_prob_tmp3.sort_values(by=1, ascending=[False]).reset_index()

        dfs = [df_final2.loc[0:K-1,'tiles_coords']]
    
        result = pd.concat(dfs)
    
        
        lib[i].update({"sorted_coords": result.tolist()})
        
        start_index += len_grid

    return np.array(lib), np.array(targets)


def create_max_tiles_dataset_bin(name_file,fold,K,dataset):

    X_train, Y_train = get_grids_bin('train',fold,K,dataset)
    X_test, Y_test = get_grids_bin('test',fold,K,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))

# =============================================================================
# sort by top  value per class (mix) binary 
# =============================================================================   


def get_grids_mix_bin(typee, fold, K,dataset):
        

    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]

    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
        
        lib=lib_f
        targets=target_f
    
    start_index=0
    
    for i in range(len(lib)):

        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        
        grids_prob_tmp1=grids_prob_tmp
        grids_prob_tmp2=grids_prob_tmp
        grids_prob_tmp3=grids_prob_tmp
        
        grids_prob_tmp1["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp2["tiles_coords"] = lib[i]['tiles_coords']
        grids_prob_tmp3["tiles_coords"] = lib[i]['tiles_coords']
        
        df_final1 = grids_prob_tmp1.sort_values(by=0, ascending=[False]).reset_index()
        df_final2 = grids_prob_tmp2.sort_values(by=1, ascending=[False]).reset_index()
        
        dfs = [df_final1.loc[0:K-1,'tiles_coords'],df_final2.loc[0:K-1,'tiles_coords']]
    
        result = pd.concat(dfs)
    
        
        lib[i].update({"sorted_coords": result.tolist()})
        
        start_index += len_grid


    return np.array(lib), np.array(targets)


def create_max_mix_tiles_dataset_bin(name_file,fold,K,dataset):

    X_train, Y_train = get_grids_mix_bin('train',fold,K,dataset)
    X_test, Y_test = get_grids_mix_bin('test',fold,K,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    

# =============================================================================
# sort by top expected value tiles (p0*1+p1*2*p2*3)
# =============================================================================

def get_grids_top_expected_value(typee,fold,dataset):
    
    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}{dataset}.csv'), header=None) 
    
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
    
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2+grids_prob_tmp.iloc[i,2]*3 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords"] = lib[i]['tiles_coords']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset_expected_value(name_file,fold,dataset):

    X_train, Y_train = get_grids_top_expected_value('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    
# =============================================================================
# sort by top expected value tiles (p0*1+p1*2) binary
# =============================================================================

def get_grids_top_expected_value_bin(typee,fold,dataset):
    
 
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords"] = lib[i]['tiles_coords']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset_expected_value_bin(name_file,fold,dataset):

    X_train, Y_train = get_grids_top_expected_value_bin('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value_bin('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    
    
# =============================================================================
# sort by top expected value tiles (p0*1+p1*2) binary - MULTILEVEL
# =============================================================================


def get_grids_top_expected_value_bin_same(typee,fold,dataset):
    
 
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords"] = lib[i]['tiles_coords']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)




def get_grids_top_expected_value_multi_level_bin_same(typee,fold,dataset):
    
 
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data_multi_level.pickle', 'rb'))[fold][typee]
    
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords"] = lib[i]['tiles_coords']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset_expected_value_multilevel_bin_same(name_file,fold,dataset):

    X_train, Y_train = get_grids_top_expected_value_bin_same('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value_bin_same('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    
    X_train, Y_train = get_grids_top_expected_value_multi_level_bin_same('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value_multi_level_bin_same('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted_multilevel.pickle', 'wb'))
    
    
# =============================================================================
#     NEW TILES
# =============================================================================
def get_grids_top_expected_value_bin(typee,fold,dataset):
    
 
    grids_prob = pd.read_csv(os.path.join('1_prediction_grids', f'predictions_grid_{typee}_fold_{fold}_bin{dataset}.csv'), header=None)

    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data{dataset}.pickle', 'rb'))[fold][typee]
    
    if dataset=='_full':
        lib_f=[]
        target_f=[]
        for i in range(len(lib)):
            for z in range(len(lib[i])):
                lib_f.append(lib[i][z])
                target_f.append(targets[i][z])
    
        lib=lib_f
        targets=target_f
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords"] = lib[i]['tiles_coords']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)

def get_grids_top_expected_value_multi_level_bin(typee,fold,dataset):
    
 
    grids_prob = pd.read_csv(f'predictions_grid_{typee}_fold_{fold}_multi_level_bin{dataset}.csv', header=None)
    lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data_multi_level.pickle', 'rb'))[fold][typee]
    
        
    start_index=0
    
    for i in range(len(lib)):
        len_grid=len(lib[i]['tiles_coords_level'])
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        df_exp = pd.DataFrame([grids_prob_tmp.iloc[i,0]*1 +grids_prob_tmp.iloc[i,1]*2 for i in range(len(grids_prob_tmp))])
        df_exp["tiles_coords_level"] = lib[i]['tiles_coords_level']
        df_final= df_exp.sort_values(by=0, ascending=False).reset_index()
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords_level'].tolist()})
        start_index += len_grid
    
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset_expected_value_multilevel_bin(name_file,fold,dataset):

    X_train, Y_train = get_grids_top_expected_value_bin('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value_bin('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'wb'))
    
    X_train, Y_train = get_grids_top_expected_value_multi_level_bin('train',fold,dataset)
    X_test, Y_test = get_grids_top_expected_value_multi_level_bin('test',fold,dataset)
    
    folds = [{'train': (X_train, Y_train), 'test': (X_test, Y_test)}]
    pickle.dump(folds, open(f'multimodal_glioma_data_sorted_multilevel.pickle', 'wb'))
    

