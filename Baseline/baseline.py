from baseline_funcs import *


########### Data and Map Directories ##############
liDAR_data_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/Lidar_Training_Data.npy"
pose_data_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/Position_Training_Data.npy"
map_path = "/Users/AlexandrosTzikas/Desktop/AA275/NN-Perception-Model/Maps/map1/grid_map.npy"

########### Map Constants ##############
dx = 20
dy = 20

########### Load Data ##############
map = np.load(map_path)
map = np.pad(map, (1,1), 'constant', constant_values=(1,1))
liDAR_data = np.load(liDAR_data_path)       # assume index 0 corresponds to West
position_data = np.load(pose_data_path)
print(liDAR_data.shape)
print(position_data.shape)


def compute_belief(map, liDAR_data, position_data, dx, dy):

    # Initialize belief 
    belief = Belief(map, dx, dy)
    
    # Constants from data
    numOfPoints = liDAR_data.shape[0]
    pointsPerOrient = int(numOfPoints/8) # assume even

    # Find indices of LiDAR measurements to be used
    indices = []
    for i in range(8):
        orient_indces = [int((pointsPerOrient/2-1+(i+1)*pointsPerOrient+j)%numOfPoints) for j in range(1,pointsPerOrient+1)]
        indices.append(orient_indces)
    indices = np.array(indices)

    for i in range(8):
        for j in indices[i,:]:
            if liDAR_data[j,0] >= 0:
                print("yes")
                belief.update_via_perc(liDAR_data[j,0],perc_model)
        belief.update_via_motion()
        #belief.update_via_motion(i,(i+1)%8)
    
    belief.plot_xy_belief()

    return np.sum(belief.belief, axis=2)

# Print map
sns.heatmap(map, annot=True) 
plt.xlabel("y Coordinate")
plt.ylabel("x Coordinate")
plt.title("Map with dx=10 and dy=10")
plt.show()

print(position_data[1,:,:])
print(position_data[1,7,3])
print(liDAR_data[1000,::])
res = compute_belief(map, liDAR_data[1000,:,:], position_data[1000,:,:], dx, dy)
print(res.shape)