# Import open3d and glob packages
import open3d as o3d
import glob
import numpy as np

n_males = 1518
n_females = 1532


def load_data(path):
    # Specify the path of the folder containing obj files
    path = "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git_Privat/SPRING_FEMALE"
    # path = f"{path}SPRING_FEMALE"
    print(path)
    # List all the files with .obj extension in the path
    files = glob.glob(path + "*.obj")
    # Create an empty list to store meshes
    female_meshes = []
    # Loop over each file name
    for file in files[:4]:
        # Read a cloud from an obj file
        mesh = o3d.io.read_triangle_mesh(file)
        # Append the cloud to the list of point clouds
        female_meshes.append(mesh)

    path = "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git_Privat/SPRING_MALE"

    # path = f"{path}SPRING_FEMALE"    
    # List all the files with .obj extension in the path
    files = glob.glob(path + "*.obj")
    # Create an empty list to store meshes
    male_meshes = []
    # Loop over each file name
    for file in files[:4]:
        # Read a cloud from an obj file
        mesh = o3d.io.read_triangle_mesh(file)
        # Append the cloud to the list of clouds
        male_meshes.append(mesh)
    
    return sample_points(female_meshes, male_meshes)

def sample_points(female_meshes, male_meshes):
    female_pcds = []
    male_pcds = []
    for mesh in female_meshes:
        # Create a point cloud object from the mesh
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        # Save it as a pcd file
        pdc = o3d.io.write_point_cloud("example.pcd", pcd)
        female_pcds.append(pcd)
    
    for mesh in male_meshes:
        # Create a point cloud object from the mesh
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        # Save it as a pcd file
        pdc = o3d.io.write_point_cloud("example.pcd", pcd)
        male_pcds.append(pcd)

    return female_pcds, male_pcds

def normalize_point_clouds(pcds):
    # Convert point clouds to numpy arrays
    points_list = [np.asarray(pcd.points) for pcd in pcds]
    
    # Compute the global centroid and the maximum distance from the centroid
    global_centroid = np.mean(np.concatenate(points_list, axis=0), axis=0)
    max_dist = np.max([np.max(np.linalg.norm(points - global_centroid, axis=1)) for points in points_list])
    
    # Translate and scale the points to fit within a unit circle
    normalized_points_list = [(points - global_centroid) / max_dist for points in points_list]
    
    # Create new point clouds with the normalized points
    normalized_pcds = [o3d.geometry.PointCloud() for _ in range(len(pcds))]
    for i in range(len(pcds)):
        normalized_pcds[i].points = o3d.utility.Vector3dVector(normalized_points_list[i])
    
    # Return the normalized point clouds
    return normalized_pcds


female_pcds, male_pcds = load_data("/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git_Privat/x")
normalized_females = normalize_point_clouds(female_pcds)


# o3d.visualization.draw_geometries([female_pcds[0]])
o3d.visualization.draw_geometries([normalized_females[0]])


