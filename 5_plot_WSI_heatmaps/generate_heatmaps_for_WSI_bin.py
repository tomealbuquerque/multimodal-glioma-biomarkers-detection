import pickle
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def get_grids_top(typee,fold):
    
    lib, targets = pickle.load(open(f'../data_multimodal_tcga/multimodal_glioma_data.pickle', 'rb'))[fold][typee]
    
    
    slide_names=[lib[i]['slide'] for i in range(len(lib))]
    target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]
    

    slide_names=[lib[i]['slide'] for i in range(len(lib))]
    target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]
    
    #Encoding targets 
    for i in range(len(target)):
        if target[i]==[0,0]:
            target[i]=0
        else:
            target[i]=1
    
    
    grids_prob = pd.read_csv(f'predictions_grid_{typee}_fold_{fold}_bin.csv', header=None)  
    
    #for debug
    # print(len(grids_prob))
    
    # all_grids=sum([len(lib[i]['tiles_coords']) for i in range(len(lib))])
    # print(all_grids)
    
    start_index=0
    
    for i in range(len(lib)):
        pred_target=[]
        len_grid=len(lib[i]['tiles_coords'])
        # print(len_grid)
        grids_prob_tmp=grids_prob.iloc[start_index:len_grid+start_index]
        
        df = grids_prob_tmp.max(axis=1)
        
        df = pd.DataFrame(grids_prob_tmp.max(axis=1))
        
        df["tiles_coords"] = lib[i]['tiles_coords']
        
        df_final= df.sort_values(by=0, ascending=False).reset_index()
        
        for t in range(len_grid):
            pred_target.append(np.array(grids_prob_tmp.iloc[t,:]).argmax(0))
        
        df_pred_target = pd.DataFrame(pred_target, columns = ['Pred'])
        
        lib[i].update({"sorted_coords": df_final.loc[:,'tiles_coords'].tolist()})
        
        lib[i].update({"pred_target": df_pred_target.loc[:,'Pred'].tolist()})
        
        start_index += len_grid
        # print(start_index)
        
        #print(np.array(grids_prob_tmp.iloc[0,:]).argmax(0))
    
        #print(df_pred_target)
    return np.array(lib), np.array(targets)


def create_max_tiles_dataset(name_file,fold):

   # X_train, Y_train = get_grids_top('train',fold)
    X_test, Y_test = get_grids_top('test',fold)
    
    folds = [{'test': (X_test, Y_test)}]
    pickle.dump(folds, open(name_file, 'wb'))


create_max_tiles_dataset('multimodal_glioma_data_sorted.pickle', 0)
#%%
################### Create Heatmaps ############################
import os
from openslide import *
import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
import pickle
import numpy as np
import random


lib, targets = pickle.load(open(f'multimodal_glioma_data_sorted.pickle', 'rb'))[0]['test']


slide_names=[lib[i]['slide'] for i in range(len(lib))]
target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]


slide_names=[lib[i]['slide'] for i in range(len(lib))]
target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]

#Encoding targets 
for i in range(len(target)):
    if target[i]==[0,0]:
        target[i]=0
    else:
        target[i]=1


for sd in range(len(target)):
    print('slide: ',sd)
    slide_number=sd
    
    slide = OpenSlide(os.path.join(f'E:/Pathology',slide_names[slide_number]))
    
    grids=np.array([lib[i]['tiles_coords'] for i in range(len(lib))],dtype=object)
    Pred=np.array([lib[i]['pred_target'] for i in range(len(lib))],dtype=object)
    
    from collections import Counter
    
    #get info from counts
    # for i in range(len(Pred)):
    #     print('number_slide: ',i ,'name',slide_names[slide_number],'target',targets[i],Counter(Pred[i]).values())
    
    
    a,b=slide.level_dimensions[0]
    
    #Get the thumbnail of the original image (206X400)
    simg = slide.get_thumbnail((math.ceil(a/16+16),math.ceil(b/16+16)))
    
    
    array=np.ones((math.ceil(a/16+16), math.ceil(b/16+16)), dtype=int)*4
    
    new_matrix_cords=np.zeros((len(grids[slide_number]),3))
    
    for i in range(len(grids[slide_number])):
        cord=grids[slide_number][i]
        #cord_list = ast.literal_eval(cord)
        new_matrix_cords[i,0]=math.ceil(cord[0]/16)
        new_matrix_cords[i,1]=math.ceil(cord[1]/16)
        new_matrix_cords[i,2]=Pred[slide_number][i]
    
    for z in range(len(new_matrix_cords)):
        x_axis=new_matrix_cords[z,0]
        y_axis=new_matrix_cords[z,1]
        value=new_matrix_cords[z,2]
        for i in range(math.ceil(512/16)):
            x=int(x_axis+i)
            for j in range(math.ceil(512/16)):
                y=int(y_axis+j)
                #print(x,y)
                try:
                    array[x,y]=value
                except Exception:
                    pass
                
    
    os.makedirs("plots", exist_ok=True)
    
    os.makedirs(f"plots/{slide_number}", exist_ok=True)
    
    #Save both side by side
    title_fig='Biomarkers map: '+os.path.basename(slide_names[slide_number])
    fig= plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_axis_off()
    im=ax1.imshow(np.fliplr(array), cmap="Pastel1")
    #plt.savefig('iqa_map_'+os.path.basename(slide_names[slide_number])+'heatmap.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
    # fig.colorbar(im)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(np.rot90(simg, 3))
    ax2.set_axis_off()
    fig.suptitle(title_fig, fontsize=16)
    plt.savefig(os.path.join(f"plots/{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'.png'), dpi = 1500)
    
    
    #Save individuals
    fig= plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_axis_off()
    im=ax1.imshow(np.fliplr(array), cmap="Pastel1")
    plt.savefig(os.path.join(f"plots/{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'_heatmap.png'), dpi = 1500, bbox_inches='tight', pad_inches = 0)
    
    # fig.colorbar(im)
    fig2= plt.figure()
    ax2 = fig2.add_subplot()
    ax2.imshow(np.rot90(simg, 3))
    ax2.set_axis_off()
    plt.savefig(os.path.join(f"plots/{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'_slide.png'), dpi = 1500, bbox_inches='tight', pad_inches = 0)
    
    
    try:
        from PIL import Image
    except ImportError:
        import Image
    
    background = Image.open(os.path.join(f"plots\\{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'_slide.png'))
    # background = Image.fromarray(np.rot90(simg,3))
    # overlay = Image.fromarray(np.fliplr(array[:,:-2])*255)
    overlay = Image.open(os.path.join(f"plots\\{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'_heatmap.png'))
    width, height = background.size
    overlay=overlay.resize((width, height))
    
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(os.path.join(f"plots/{slide_number}",'bio_map_'+os.path.basename(slide_names[slide_number])+'_fused.png'),"PNG")


    #plot some images of results per class
    
    number_of_images=40
    number_of_classes =2
    tile_size=512
    
    for nc in range(0,number_of_classes):
        # print(nc)
        data_0=new_matrix_cords[new_matrix_cords[:,2]==nc]
        if len(data_0)>100:
            examples=number_of_images    
        else:
            exmples=len(data_0)
        for cc in range(examples):
            # print(cc)
            try:
                img_0 = slide.read_region((int(data_0[cc,0])*16,int(data_0[cc,1])*16), 0, (tile_size, tile_size)).convert('RGB')
                os.makedirs(os.path.join(f"plots/{slide_number}",f'class_{nc}'), exist_ok=True)
                plt.imshow(img_0)
                plt.axis('off')
                plt.savefig(os.path.join(f"plots/{slide_number}",f'class_{nc}',f'figure{cc}.png'), dpi = 250, bbox_inches='tight', pad_inches = 0)
            except Exception:
                    pass
                        
            


