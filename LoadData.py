# Import open3d and glob packages
import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt


# Define a function that loads data from obj files
def load_data():
    # Specify the path of the folder containing obj files
    path = "data/mesh_female/"
    # List all the files with .obj extension in the path
    files = glob.glob(path + "*.obj")
    # Create an empty list to store meshes
    female_meshes = []
    # Loop over each file name
    for file in files:
        # Read a cloud from an obj file
        mesh = o3d.io.read_triangle_mesh(file)
        # Append the cloud to the list of point clouds
        female_meshes.append(mesh)

    path = "data/mesh_male/"
    # List all the files with .obj extension in the path
    files = glob.glob(path + "*.obj")
    # Create an empty list to store meshes
    male_meshes = []
    # Loop over each file name
    for file in files:
        # Read a cloud from an obj file
        mesh = o3d.io.read_triangle_mesh(file)
        # Append the cloud to the list of clouds
        male_meshes.append(mesh)
    
    return sample_points(female_meshes, male_meshes)

# Define a function that samples points from a list of meshes
def sample_points(female_meshes, male_meshes):
    female_pcds = []
    male_pcds = []
    for mesh in female_meshes:
        # Create a point cloud object from the mesh
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        # Save it as a pcd file
        # pdc = o3d.io.write_point_cloud("example.pcd", pcd)
        female_pcds.append(pcd)
    
    for mesh in male_meshes:
        # Create a point cloud object from the mesh
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        # Save it as a pcd file
        # pdc = o3d.io.write_point_cloud("example.pcd", pcd)
        male_pcds.append(pcd)

    return female_pcds, male_pcds

# Define a function that normalizes a list of point clouds
def normalize_point_clouds(female_pcds, male_pcds):
    # Convert point clouds to numpy arrays
    points_list = [np.asarray(pcd.points) for pcd in female_pcds + male_pcds]
    
    # Compute the global centroid and the maximum distance from the centroid
    global_centroid = np.mean(np.concatenate(points_list, axis=0), axis=0)
    max_dist = np.max([np.max(np.linalg.norm(points - global_centroid, axis=1)) for points in points_list])
    
    # Translate and scale the points to fit within a unit circle
    female_points = [np.asarray(pcd.points) for pcd in female_pcds]
    male_points = [np.asarray(pcd.points) for pcd in male_pcds]
    normalized_points_female = [(points - global_centroid) / max_dist for points in female_points]
    normalized_points_male = [(points - global_centroid) / max_dist for points in male_points]

    # Create new point clouds with the normalized points
    normalized_pcds_female = [o3d.geometry.PointCloud() for _ in range(len(female_pcds))]
    for i in range(len(female_pcds)):
        normalized_pcds_female[i].points = o3d.utility.Vector3dVector(normalized_points_female[i])

    normalized_pcds_male = [o3d.geometry.PointCloud() for _ in range(len(male_pcds))]
    for i in range(len(male_pcds)):
        normalized_pcds_male[i].points = o3d.utility.Vector3dVector(normalized_points_male[i])
    # Return the normalized point clouds
    return normalized_pcds_female, normalized_pcds_male

# Define a function that flips a point cloud horizontally
def flip_horizontal(pcd):
    # Get the points of the point cloud as a numpy array
    points = np.asarray(pcd.points)
    # Flip the x-coordinates by multiplying them by -1
    points[:, 0] = -points[:, 0]
    # Create a new point cloud with the flipped points
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)
    # Return the flipped point cloud
    return flipped_pcd

# Define a function that augments a list of point clouds by flipping them horizontally
def augment_data(pcds):
    # Create an empty list to store augmented point clouds
    augmented_pcds = []
    # Loop over each point cloud in the list
    for pcd in pcds:
        # Append the original point cloud to the augmented list
        augmented_pcds.append(pcd)
        # Flip the point cloud horizontally and append it to the augmented list
        flipped_pcd = flip_horizontal(pcd)
        augmented_pcds.append(flipped_pcd)
    
    # Return the augmented list of point clouds
    return augmented_pcds

# Define a function that visualizes a point cloud
def visualize_point_cloud(pcd):
    # Convert it to a numpy array
    points = np.asarray(pcd.points)

    # Plot it using matplotlib with tiny points and constrained axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-1,1) # Set x-axis range
    ax.set_ylim(-1,1) # Set y-axis range
    ax.set_zlim(-1,1) # Set z-axis range
    plt.show()

# Load the data
female_pcds, male_pcds = load_data()
# Normalize the point clouds
female_pcds, male_pcds = normalize_point_clouds(female_pcds, male_pcds)

# Augment data by flipping point clouds horizontally 
female_pcds_augmented = augment_data(female_pcds)
male_pcds_augmented = augment_data(male_pcds)


# visualize_point_cloud(female_pcds[0])
# visualize_point_cloud(female_pcds_augmented[0])

# o3d.io.write_point_cloud("female_pcds_0.pcd", female_pcds[0], write_ascii=True)

# o3d.io.write_point_cloud("female_mesh_augmented_0.pcd", female_pcds_augmented[0], write_ascii=True)


# Define a function that computes the root mean square error (RMSE) between two point clouds
def rmse(pcd1, pcd2):
    # Compute the distance from each point in pcd1 to the closest point in pcd2
    dists = pcd1.compute_point_cloud_distance(pcd2)
    # Convert the distance list to a numpy array
    dists = np.asarray(dists)
    # Square the distances and take the mean
    mse = np.mean(dists**2)
    # Take the square root and return it
    return np.sqrt(mse)

cummulative_sim = 0
# Loop over each pair of original and augmented point clouds in the female meshes list
for i in range(0, len(female_pcds_augmented), 2):
    # Get the original point cloud at index i
    original_pcd = female_pcds_augmented[i]
    # Get the augmented point cloud at index i+1
    augmented_pcd = female_pcds_augmented[i+1]
    # Compute and print the RMSE between them
    similarity = rmse(original_pcd, augmented_pcd)
    cummulative_sim += similarity
    # print(f"Similarity between original and augmented point cloud {i//2}: {similarity.round(5)}")
print("cumsim:", cummulative_sim)