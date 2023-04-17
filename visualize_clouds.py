def visualize_point_cloud(pcd):
    my_cmap = plt.get_cmap('hsv')
    # Convert it to a numpy array
    #points = np.asarray(pcd)
    points = pcd.detach().numpy()

    # Plot it using matplotlib with tiny points and constrained axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:], s=0.1, cmap=my_cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    #ax.set_xlim(-1,1) # Set x-axis range
    #ax.set_ylim(-1,1) # Set y-axis range
    #ax.set_zlim(-1,1) # Set z-axis range
    plt.show()