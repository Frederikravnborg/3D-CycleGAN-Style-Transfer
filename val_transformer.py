# Transform val through loaded checkpoint Generator
from utilities.utilities import load_checkpoint
from foldingnet_model import Generator
import torch.optim as optim
from load_data import ObjDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import config
n_epoch = 1200

M_path = f"saved_models/genM{n_epoch}.pth.tar"
F_path = f"saved_models/genF{n_epoch}.pth.tar"

gen_F = Generator().to(config.DEVICE)
gen_M = Generator().to(config.DEVICE)
opt_gen = optim.Adam(
        list(gen_F.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

load_checkpoint(M_path, gen_M, opt_gen, config.LEARNING_RATE,)
load_checkpoint(F_path, gen_F, opt_gen, config.LEARNING_RATE,)

# from data/val/: import pcds through dataloader and transform them through generator
# define validation dataset
val_dataset = ObjDataset(
    root_male=config.VAL_DIR + "/male",
    root_female=config.VAL_DIR + "/female",
    transform=None,
    n_points=config.N_POINTS
)
val_loader = DataLoader(val_dataset, batch_size=10, pin_memory=True)

loop = tqdm(val_loader, leave=True) #Progress bar

# loop through the data loader
for idx, (female, male) in enumerate(loop):
    female = female.to(config.DEVICE).float()
    male = male.to(config.DEVICE).float()

    fake_female, _, _ = gen_F(male)
    fake_male, _, _ = gen_M(female)

    # save the generated point clouds through Trimesh
    female_vertices = fake_female[0].detach().cpu().numpy().transpose(1,0)
    fake_female = trimesh.Trimesh(vertices=female_vertices)
    fake_female.export(f"data/val/generated_female/female_{idx}.obj")

    male_vertices = fake_male[0].detach().cpu().numpy().transpose(1,0)
    fake_male = trimesh.Trimesh(vertices=male_vertices)
    fake_male.export(f"data/val/generated_male/male_{idx}.obj")

