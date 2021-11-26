# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:50:03 2021

@author: user
"""
import numpy as np 
import airsim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Grid:
    def __init__(self, N_x, N_y, cell_size, min_x, min_y):
        self.nx = N_x + 10//cell_size
        self.ny = N_y + 10//cell_size
        self.min_x = min_x
        self.min_y = min_y
        self.cell_size = cell_size
        self.x_spacing = np.linspace(start=min_x, stop = min_x+(self.nx)*cell_size, num=self.nx+1)
        self.y_spacing = np.linspace(start=min_y, stop = min_y+(self.ny)*cell_size, num=self.ny+1)
        self.pseudo_grid = np.array(np.meshgrid(self.x_spacing, self.y_spacing)).T.reshape(self.nx+1, self.ny+1, 2)
        # self.pseudo_grid = np.flip(self.pseudo_grid, axis=)
        self.grid = np.zeros((self.nx+1, self.ny+1))
        
        pass
    
    def update_grid(self, x, y, tol):
        # print(self.pseudo_grid)
        ix = np.array(np.isclose(self.pseudo_grid, np.array([x, y]), atol=tol).all(axis=2).nonzero()).T
        # print(ix)
        for index in ix.tolist():
            self.grid[index[1], index[0]] = 1 
        # pass
    
    def create_map(self, obstacles):
        for obstacle in obstacles:
            self.update_grid(obstacle[0], obstacle[1], 5)
        pass
    
    
    def get_occupancy(self, x, y, tol):
        self.grid = np.zeros((self.nx+1, self.ny+1))
        ix = np.array(np.isclose(self.pseudo_grid, np.array([x, y]), atol=tol).all(axis=2).nonzero()).T
        # print(ix.tolist())
        
        for index in ix.tolist():
            # print(self.pseudo_grid[index[0], index[1]])
            self.grid[index[1], index[0]] = 1/len(ix.tolist())
        return self.grid
        
    
################################
#MAP 1
# X: [-65.32, 44.12]
# Y: [-53.20, 56.75]



################################
#Save obstacle data
# client = airsim.CarClient()
# client.confirmConnection()

# objects = []
# for s in ["Cylinder.*", "OrangeBall.*", "Cone.*", "TemplateCube.*"]:
#     objects += client.simListSceneObjects(s)
    
# # print(client.simGetObjectPose("Car1"))
# object_poses = [client.simGetObjectPose(obj) for obj in objects]
# object_poses = [(obj_pos.position.x_val, obj_pos.position.y_val) for obj_pos in object_poses]
# obstacles = np.array(object_poses)   
# np.save("../Maps/map1/obstacles.npy", obstacles, allow_pickle=True)
# client.reset()


#################################
#Save map data

# grid = Grid(50, 50, 2, -65.32, -53.20)

# obstacles = np.load("../Maps/map1/obstacles.npy", allow_pickle=True)
# grid.create_map(obstacles)
# sns.heatmap(grid.grid)
# np.save("../Maps/map1-new/map50_50.npy", obstacles, allow_pickle=True)


################################
#Process data

#for 10x10 grid data
grid = Grid(10, 10, 10, -65.32, -53.20)

#for 50x50 grid data
# grid = Grid(50, 50, 2, -65.32, -53.20)

all_lidar_data = []
all_occupancy_data = []

data_dirs = ["Data_10000/", "Data_5000/"]
for data_dir in data_dirs:
    lidar_data = np.load(data_dir + "Lidar_data.npy", allow_pickle=True)
    pose_data = np.load(data_dir + "Pose_data.npy", allow_pickle=True)
    
    
    
    for i in range(len(lidar_data)):
        data = lidar_data[i]
        arr = -1*np.ones((720, 1))
        # df = pd.DataFrame(lidar_data[i])
        if len(data):
            df = pd.DataFrame()
            df["angles"] = np.rad2deg(np.arctan2(data[:, 0], data[:, 1]))
            
            df["distance"] = np.linalg.norm(data, axis=1)
            angles = np.linspace(-179.5, 180, 720, endpoint=True)
            f = lambda x: np.where(angles>x)[0].min()
            df["angles"] = df["angles"].apply(f)
            df = df.groupby("angles").mean()
            
            arr[df.index] = df.values
        occupancy = grid.get_occupancy(pose_data[i][0], pose_data[i][1], tol=grid.cell_size/2)
        
        all_lidar_data.append(arr)
        all_occupancy_data.append(occupancy)
    print(len(all_lidar_data), len(all_occupancy_data))
    
    # plt.figure()
    # sns.heatmap(occupancy)

final_dir="Final_1010/"
np.save(final_dir+"Lidar_data.npy", np.array(all_lidar_data), allow_pickle=True)
np.save(final_dir+"Pose_data.npy", np.array(all_occupancy_data), allow_pickle=True)