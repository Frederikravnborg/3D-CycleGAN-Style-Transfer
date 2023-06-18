from load_data import ObjDataset
import config
from torch.utils.data import DataLoader
import trimesh

# define train dataset
dataset = ObjDataset(
    root_male=config.TRAIN_DIR + "/male", 
    root_female=config.TRAIN_DIR + "/female",
    transform=None,
    n_points=config.N_POINTS
)

# define dataloader for train and validation dataset
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

for i, (female, male) in enumerate(loader):
    female_vertices = female.transpose(2,1)[0].detach().cpu().numpy().transpose(1,0)
    female = trimesh.Trimesh(vertices=female_vertices)
    female.export(f"results_pcd/OG/female_{i}.obj")

    male_vertices = male.transpose(2,1)[0].detach().cpu().numpy().transpose(1,0)
    male = trimesh.Trimesh(vertices=male_vertices)
    male.export(f"results_pcd/OG/male_{i}.obj")    
    