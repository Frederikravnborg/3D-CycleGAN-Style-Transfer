import wandb
import trimesh

api = wandb.Api()
run = api.run("fagprojekt-/Fagprojekt/efcnpv6y")
pcd = run.history()["OG_male"][460]

# Convert the point cloud to a mesh
pcd = pcd.convex_hull

# Save the mesh as an .obj file
pcd.export("results_pcd/OG/OG_male_460.obj")