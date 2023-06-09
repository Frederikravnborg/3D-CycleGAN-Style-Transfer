import torch
from utils import save_checkpoint, load_checkpoint
import config
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import torch.optim as optim
#from torcheval.metrics import BinaryConfussionMatrix
from PlotSpecifikkePointclouds import visualize_pc
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader_dataset import PointCloudDataset
import numpy as np



def validation(disc_FM, disc_M, gen_FM, gen_M, POINTNET_classifier, val_loader, opt_disc, opt_gen, vis_list_female, vis_list_male):
    val_loop = tqdm(val_loader, leave=True)
    #TF, TM, FF, FM = []
    cf_mat = dict()
    for type in ['TF','FF','TM','FM']:
        cf_mat[type] = [0 for i in range(3)]
    vis_female = dict()
    vis_male = dict()

    for idx, data in enumerate(val_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        # Generating fakes
        fake_female = gen_FM(male)[0]
        fake_male = gen_M(female)[0]
        

        # Generate cycles
        cycle_female = gen_FM(fake_male)[0]
        cycle_male = gen_M(fake_female)[0]

        # Classify fakes and cycles - female
        True_female = POINTNET_classifier(female)[0]
        False_female = POINTNET_classifier(fake_female)[0]
        c_female = POINTNET_classifier(cycle_female)[0]

        # Classify fakes and cycles - male
        True_male = POINTNET_classifier(male)[0]
        False_male = POINTNET_classifier(fake_male)[0]
        c_male = POINTNET_classifier(cycle_male)[0]

        
        #Calculate predictions
        pred_choice_female = True_female.data.max(1)[1]
        pred_choice_ffemale = False_female.data.max(1)[1]
        pred_choice_cfemale = c_female.data.max(1)[1]

        pred_choice_male = True_male.data.max(1)[1]
        pred_choice_fmale = False_male.data.max(1)[1]
        pred_choice_cmale = c_male.data.max(1)[1]

        #True male and female matrix:
        for i, pred_male, pred_female in zip(range(3),[pred_choice_male,pred_choice_fmale,pred_choice_cmale],[pred_choice_female,pred_choice_ffemale,pred_choice_cfemale]):
            cf_mat['TF'][i] += pred_female.sum()
            cf_mat['FM'][i] += pred_male.sum()
            cf_mat['TM'][i] += (len(pred_male)-pred_male.sum())
            cf_mat['FF'][i] += (len(pred_female)-pred_female.sum())

        

        #Cycle male and female visualizations:
        for i in range(len(vis_list_female)):
            if vis_list_female[i] in fem_ids:
                vis_female[str(vis_list_female[i])] = cycle_female[fem_ids.index(vis_list_female[i])]

        for i in range(len(vis_list_male)):
            if vis_list_male[i] in male_ids:
                vis_male[str(vis_list_male[i])] = cycle_male[male_ids.index(vis_list_male[i])]

    return cf_mat, (vis_female, vis_male)
    # Visualize confusion matrix
    
    


def main():
    ### Initialize model and optimizer ###
    args_gen = config.get_parser_gen()
    disc_M = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    disc_FM = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)

    #Pointnet classifier
    POINTNET_classifier = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_FM.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_FM.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_class = optim.Adam(
        list(POINTNET_classifier.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    ### Load model ### 
    
    

    

    load_checkpoint(
        "CLASSIFIER_MODEL46.pth.tar",
        models=[POINTNET_classifier],
        optimizers=[opt_class],
        lr=config.LEARNING_RATE
    )
    
    val_dataset = PointCloudDataset(
        root_female=config.VAL_DIR + "/female_test",
        root_male=config.VAL_DIR + "/male_test",
        transform=False
    )

    val_loader = DataLoader(val_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=config.collate_fn
            )

    # list of pointclouds we wish to visualize:
    vis_list_female = ['SPRING0380.obj','SPRING0400.obj','SPRING0469.obj','SPRING0600.obj','SPRING1050.obj']
    vis_list_male = ['SPRING0223.obj','SPRING0300.obj','SPRING0320.obj','SPRING0420.obj','SPRING0450.obj']

    for shape in ['plane', 'sphere', 'gaussian','feature']:
        args_gen.shape = shape
        gen_M = Generator_Fold(args_gen).to(config.DEVICE)
        gen_FM = Generator_Fold(args_gen).to(config.DEVICE)
        load_checkpoint(
            f"MODEL_OPTS_LOSSES_{shape}_1201.pth.tar",
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
            lr=config.LEARNING_RATE,
        )

        cf_mat, visualizations = validation(disc_FM, disc_M, gen_FM, gen_M, POINTNET_classifier, val_loader, opt_disc, opt_gen, vis_list_female, vis_list_male)

        for i in range(3):
            df_cm = pd.DataFrame(np.array([[cf_mat['TF'][i],cf_mat['FM'][i]],[cf_mat['FF'][i],cf_mat['TM'][i]]]), index=['Female','Male'], columns=['Female_True','Male_true'])
            plt.figure(figsize=(10,7))
            sn.heatmap(df_cm,annot=True)
            plt.show()

        for i in range(len(vis_list_female)):
            visualize_pc(visualizations[0][vis_list_female[i]].transpose(-2,1))
            visualize_pc(visualizations[1][vis_list_male[i]].transpose(-2,1))
        

if __name__ == "__main__":
    main()


# for params in gen_M.parameters():
#     g = params

# with open('readme1.txt', 'w') as f:
#     f.write(str(g))
