# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:52:20 2021

@author: user
"""
import airsim
import time
import pprint
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pptk

def parse_lidarData(data):
    points = np.array(data.point_cloud, dtype="float32").reshape(-1, 3)
    return points[:, :2]
    # return points



# X: [-65.32, 44.12]
# Y: [-53.20, 56.75]


client = airsim.CarClient()
client.confirmConnection()


objects = []
for s in ["Cylinder.*", "OrangeBall.*", "Cone.*", "TemplateCube.*"]:
    objects += client.simListSceneObjects(s)
    
# print(client.simGetObjectPose("Car1"))
object_poses = [client.simGetObjectPose(obj) for obj in objects]
object_poses = [(obj_pos.position.x_val, obj_pos.position.y_val) for obj_pos in object_poses]
obstacles = np.array(object_poses)

# print(obstacles)

# '''
lidar_data = []
pose_data = []
i = 0
while len(lidar_data) < 5000:
    candidate = np.array([np.random.uniform(-50, 34), np.random.uniform(-43, 46)])
    while np.isclose(candidate, obstacles, atol=7).all(axis=1).any():
        # print(candidate)
        candidate = np.array([np.random.uniform(-50, 34), np.random.uniform(-43, 46)])
        # print(np.isclose(candidate, obstacles, atol=8).any())
    # print(candidate)
    
    r = np.random.uniform(-np.pi, np.pi)
    pose = client.simGetVehiclePose()
    pose.position.x_val = candidate[0]
    pose.position.y_val = candidate[1]
    pose.orientation.w_val = np.cos(r/2);
    pose.orientation.x_val = 0
    pose.orientation.y_val = 0
    pose.orientation.z_val = np.sin(r/2);
    
    client.simSetObjectPose("Car1", pose)
    time.sleep(1)
    lidarData = client.getLidarData("LidarSensor1", "Car1")
    # if (len(lidarData.point_cloud)) > 100:
    lidar_data.append(parse_lidarData(lidarData))
    pose_data.append(np.hstack((candidate, r)))
    try:
        np.save("Lidar_Data.npy", lidar_data)
        np.save("Pose_Data.npy", pose_data)
    except PermissionError:
        print("failed save\n")
        continue
    
    if (i % 1000 == 0):
        print(i)
    i+=1


# '''
# pose = client.simGetVehiclePose()
# pose.position.x_val = 25
# pose.position.y_val = -44
# client.simSetObjectPose("Car1", pose)
# time.sleep(2)


# lidarData = client.getLidarData("LidarSensor1", "Car1")
# if (len(lidarData.point_cloud) > 3):
#     # print(type(lidarData.point_cloud))
#     # print(lidarData.point_cloud)
#     v = pptk.viewer(parse_lidarData(lidarData))   

# print(lidarData.segmentation)
    # time.sleep(0.5)



# object_poses

# object_scales = [client.simGetObjectScale(obj) for obj in objects]
# object_scales = [(obj_scale.x_val,obj_scale.y_val) for obj_scale in object_scales]


# anchors = np.array(object_poses) - np.array(object_scales)/2

# fig, ax = plt.subplots()
# fig.set_figheight(10)
# fig.set_figwidth(10)
# ax.set_xlim([-150, 150])
# ax.set_ylim([-150, 150])
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_title("Pseudo Occupancy Grid of Blocks Environment")

# rectangles = []
# for i in range(len(objects)):
#     rectangles.append(Rectangle(anchors[i], object_scales[i][0], object_scales[i][1], color="black"))

# ax.add_collection(PatchCollection(rectangles))
# plt.show()
# #plt.savefig("x-y occupancy grid.png")
# anchors = np.floor(anchors/10)
# anchors = anchors+5
# anchors = anchors.astype(int)

# occupancy = np.zeros((10,10))
# occupancy[anchors[(np.max(anchors, axis=1) < 10) & (np.min(anchors, axis=1) > -10)]] = 1 

# occ_mask = anchors[(np.max(anchors, axis=1) < 10) & (np.min(anchors, axis=1) > 0)].tolist() 


# client.reset()