from baseline_funcs import *


########### Data and Map Directories ##############
liDAR_data_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/Lidar_Training_Data.npy"
pose_data_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/Position_Training_Data.npy"
map_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/grid_map.npy"

########### Map Constants ##############
dx = 10
dy = 10

########### Load Data ##############
map = np.load(map_path)
map = np.pad(map, (1,1), 'constant', constant_values=(1,1))
liDAR_data = np.load(liDAR_data_path)       # assume index 0 corresponds to West
position_data = np.load(pose_data_path)
print(liDAR_data.shape)
print(position_data.shape)


"""
This function calculates the localization error between the belief after taking into 
consideration all the measurements and the true position, found in position_data.
It calculates the expected position cell (x,y) accordng to the belief and assumes that is the true location.
Only orientation 7 has nonzero probabilities, given that it is the last one to be updated.
"""
def compute_localiz_error(belief, position_data, dx, dy, idx):
    #x_bel, y_bel = np.unravel_index(np.argmax(belief.belief[:,:,7]), position_data.shape)
    x_bel = np.zeros((belief.belief.shape[0], belief.belief.shape[1]))
    y_bel = np.zeros((belief.belief.shape[0], belief.belief.shape[1]))
    for i in range(belief.belief.shape[0]):
        for j in range(belief.belief.shape[1]):
            x_bel[i,j] = i*dx+dx/2
            y_bel[i,j] = j*dy+dy/2
    x_bel = x_bel*belief.belief[:,:,idx]
    y_bel = y_bel*belief.belief[:,:,idx]
    estimatedX = sum(sum(x_bel))
    estimatedY = sum(sum(y_bel))

    x_true, y_true = np.unravel_index(np.argmax(position_data), position_data.shape)
    x_true = (x_true+1)*dx+dx/2
    y_true = (y_true+1)*dy+dy/2

    dist = np.sqrt((x_true-estimatedX)**2+(y_true-estimatedY)**2)

    return dist


"""
This function takes in the map, the 360-degrees (720 points) LiDAR scan, the true position (position_data) and the 
map size and returns the final belief and error in localization.
"""
def compute_belief_and_error(map, liDAR_data, position_data, dx, dy):

    # Initialize belief 
    belief = Belief(map, dx, dy)
    
    # Constants from data
    numOfPoints = liDAR_data.shape[0]
    pointsPerOrient = int(numOfPoints/8) # assume even

    # Find indices of LiDAR measurements for each orientation, assuming the 0 index is at -180 degrees (West)
    indices = []
    for i in range(8):
        orient_indces = [int((pointsPerOrient/2-1+(i+1)*pointsPerOrient+j)%numOfPoints) for j in range(1,pointsPerOrient+1)]
        indices.append(orient_indces)
    indices = np.array(indices)


    # Update belief with LiDAR measurements. An odometry update is necessary after each orientation change.
    error = [] # localization error after each measurement
    for i in range(8):
        for j in indices[i,:]:
            # If LiDAR measurement is invalid (-1), do not consider it
            if liDAR_data[j,0] >= 0:
                #print("yes")
                belief.update_via_perc(liDAR_data[j,0], perc_model, i)
                error.append(compute_localiz_error(belief, position_data, dx, dy, i))
        if i<7:
            belief.update_via_motion()
            error.append(compute_localiz_error(belief, position_data, dx, dy, i))
    error.append(compute_localiz_error(belief, position_data, dx, dy, 7))

    belief.plot_xy_belief()

    #print(sum(sum(sum(belief.belief))))
    return np.sum(belief.belief, axis=2), np.array(error)

"""
This function calculates the error for the final belief of every sample in the dataset
"""
def calculate_error_across_dataset(map, liDAR_dataset, positions_dataset, dx, dy):
    errors = []
    for i in range(liDAR_data.shape[0]):
        print(i)
        bel, error = compute_belief_and_error(map, liDAR_dataset[i,:,:], positions_dataset[i,:,:], dx, dy)
        errors.append(error[len(error)-1])
    
    errors = np.array(errors)

    return errors

############## Testing #####################################
# Print map
sns.heatmap(map, annot=True) 
plt.xlabel("y Coordinate")
plt.ylabel("x Coordinate")
plt.title("Map with dx=10 and dy=10")
plt.show()

# Test belief update for one sample
res, error = compute_belief_and_error(map, liDAR_data[100,:,:], position_data[100,:,:], dx, dy)
print(position_data[100,:,:])
print(res.shape)
plt.plot([i for i in range(len(error))], error)
plt.show()

# Calculate errors across dataset
errors = calculate_error_across_dataset(map, liDAR_data, position_data, dx, dy)
plt.plot([i for i in range(len(errors))], errors)
plt.show()