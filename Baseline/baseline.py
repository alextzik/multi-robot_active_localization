from types import new_class
import numpy as np
from numpy.core.numeric import Inf
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

"""
The following code constructs the belief that an agent is located at any one of 
the possible points in the grid map. The belief is a 3D array with dimensions x, y, orientation
The inputs to the constructor are the following:
    - map of the environment: This is assumed to be a rectangular area that is 
                              represented as a binary numpy array, with 1 in the cells 
                              with obstacles and 0 in the cells without obstacles
    - dx, dy: x-size and y-size of grid cell

          -> y
          | |-------------------------------|    
         \ /|       |        |       |      |
          x |       |        |       |      |       
            |-------|--------|-------|------|   
            |       |        |       |      |
            |       |        |       |      |   
            |-------|--------|-------|------| 
            |       |        |       |      |   
            |       |        |       |      |   
            |       |        |       |      |
            |-------------------------------|
"""

class Belief:
    def __init__(self, map, dx, dy):
        self.dimX = map.shape[0] # x-dimension of map is x-dimension of belief-space
        self.dimY = map.shape[1] # y-dimension of map is y-dimension of belief-space
        self.numOrients = 8 # number of possible orientations 
        self.map = map # Copy of the environment map in the class
        self.dx = dx # x-size of grid cell
        self.dy = dy # y-size of grid cell

        # Perception model constants
        self.percSigma = 30 # sigma of perception model's distribution
        self.percDistThresh = 10 # perception distance after which the range sensor does not return a distance
        self.numOfBins = 21 # odd for symmetry assumed
        self.BinInterval = np.sqrt(((dx*(self.dimX+1))**2)+((dy*(self.dimY+1))**2))/self.numOfBins
    

        # Initialize belief
        self.belief = np.ones((self.dimX, self.dimY, self.numOrients)) 
        # Calculate possible cells-orientations using map and assign uniform probability to them
        self.belief = (1/(self.numOrients*sum(sum(map))))*(self.belief*(1-map[:,:,np.newaxis]))

        # Calculated expected range measurements for all positions and orientations
        self.expectedPercMeasurements = calculate_expected_perc_measurements(self.belief, map, dx, dy)
            
    # Function to update belief based on incoming range measurement
    def update_via_perc(self,actMeasurement, perc_model):
        
        # Calculate the probability of receiving actual measurement given that we are in state (x,y,o)
        probMeasGivenState = np.zeros((self.belief.shape[0], self.belief.shape[1], self.belief.shape[2]))
        probMeasGivenState = perc_model(actMeasurement*np.ones(self.belief.shape), 
                                self.expectedPercMeasurements, self.percSigma*np.ones(self.belief.shape), 
                                self.BinInterval*np.ones(self.belief.shape))

        # Calculate new belief
        self.belief = probMeasGivenState*self.belief

        # Normalize new belief
        self.belief = self.belief/sum(sum(sum(self.belief)))

    # Function to update belief based on incoming odometry measurement
    def update_via_motion(self, actMeasurement, calculate_new_odom_belief):
        prevBelief = self.belief

        # For each position, update the belief
        for i in range(self.belief.shape[0]):
            for j in range(self.belief.shape[1]):
                for o in range(self.belief.shape[2]):
                    # For each position, update the belief
                    self.belief[i,j,o] = calculate_new_odom_belief(actMeasurement, prevBelief, [i, j, o])
        self.belief = self.belief/sum(sum(sum(self.belief)))

    # Function that returns the tuple (x,y,orientation) with the highest belief
    def max_belief_idx(self):
        linIndx = np.argmax(self.belief)
        x = int(np.floor(linIndx/(self.dimY*self.numOrients)))
        
        linIndx = np.mod(linIndx, (self.dimY*self.numOrients))
        y = int(np.floor(linIndx/self.numOrients))
       
        linIndx = np.mod(linIndx, self.numOrients)
        o = linIndx
        return x,y,o

    # Plotting function that returns the probability of being at all (x,y) pairs
    def plot_xy_belief(self):
        xyBelief = np.max(self.belief, axis=2)
        sns.heatmap(xyBelief, annot=True) #np.swapaxes(xyBelief, 1, 0)
        plt.xlabel("y Coordinate")
        plt.ylabel("x Coordinate")
        plt.title("Maximum Belief at every (x,y)")
        plt.show()

##################################################################
# Functions necessary to update belief after range measurement
"""
This function takes in as input a cell and an orientation and calculates the 
the distance to the nearest object towards that orientation, given the map and the grid cell size.
The closest object could be an obstacle or the end of the map.

The orientation convention is as follows:
0: North
1: Northeast
2: East
3: Southeast
4: South
5: Southwest
6: West
7: Northwest
"""
def calculate_expected_perc_measurements(belief, map, dx, dy):

    expectedPercMeasurements = np.zeros((belief.shape[0], belief.shape[1], belief.shape[2]))

    for cellX in range(0, map.shape[0]):
        for cellY in range(0, map.shape[1]):
            for orientation in range(0, belief.shape[2]):
                dist = 0
                if orientation == 0:
                    x = cellX
                    while x!=0 and map[x, cellY]!=1:
                            x = x-1
                    if x!=cellX:
                        dist = (cellX-x)*dx-dx/2
                    # else:
                    #     dist = dx/2
                            
                elif orientation == 1:
                    i = 0
                    while cellX-i!=0 and cellY+i!=map.shape[1] and map[cellX-i, cellY+i]!=1:
                        i = i+1
                    if i!=0:
                        dist = np.sqrt((i*dx)**2 + (i*dy)**2) - np.sqrt((dx/2)**2+(dy/2)**2)
                    # else: 
                    #     np.sqrt((dx/2)**2+(dy/2)**2)

                elif orientation == 2:
                    y = cellY
                    while y!=map.shape[1] and map[cellX, y]!=1:
                        y = y + 1
                    if y!=cellY:
                        dist = (y-cellY)*dy - dy/2
                    # else:
                    #     dist = dy/2

                elif orientation == 3:
                    i = 0
                    while cellX+i!=map.shape[0] and cellY+i!=map.shape[1] and map[cellX+i, cellY+i]!=1:
                        i = i+1
                    if i!=0:
                        dist = np.sqrt((i*dx)**2 + (i*dy)**2) - np.sqrt((dx/2)**2+(dy/2)**2)
                    # else:
                    #     dist = np.sqrt((dx/2)**2+(dy/2)**2)

                elif orientation == 4:
                    x = cellX
                    while x!=map.shape[0] and map[x, cellY]!=1:
                        x = x+1
                    if x!=cellX:
                        dist = (x-cellX)*dx - dx/2
                    # else:
                    #     dist = dx/2
                
                elif orientation == 5:
                    i = 0
                    while cellX+i!=map.shape[0] and cellY-i!=0 and map[cellX+i, cellY-i]!=1:
                        i = i+1
                    if i!=0:
                        dist = np.sqrt((i*dx)**2 + (i*dy)**2) - np.sqrt((dx/2)**2+(dy/2)**2)
                    else:
                        dist = np.sqrt((dx/2)**2+(dy/2)**2)

                elif orientation == 6:
                    y = cellY
                    while y!=0 and map[cellX, y]!=1:
                        y = y-1
                    if y!=cellY:
                        dist = (cellY-y)*dy - dy/2
                    # else:
                    #     dist = dy/2
                
                elif orientation == 7:
                    i = 0
                    while cellX-i!=0 and cellY-i!=0 and map[cellX-i, cellY-i]!=1:
                        i = i+1
                    if i!=0:
                        dist = np.sqrt((i*dx)**2 + (i*dy)**2) - np.sqrt((dx/2)**2+(dy/2)**2)
                    # else:
                    #     dist = np.sqrt((dx/2)**2+(dy/2)**2)
                else:
                    print("Error")
                    dist = Inf
                
                # if dist<=0:
                #     print(cellX, cellY, orientation, dist)
                expectedPercMeasurements[cellX, cellY, orientation] = dist
    # print(expectedPercMeasurements[6,6,3])
    return expectedPercMeasurements

#######################
"""
This function takes as inputs the expected measurement (expMeasurement) and actual measurement
(actMeasurement) and outputs the probability p(actMeasurement|expMeasurement).
"""
def perc_model(actMeasurement, expMeasurement, sigma, binInterval):
    prob = scipy.stats.norm(expMeasurement, sigma)
    binIdx = (np.floor(np.abs(actMeasurement-expMeasurement))/binInterval)
    
    return prob.cdf(expMeasurement+((binIdx+1/2)*binInterval))-prob.cdf(expMeasurement+((binIdx-1/2)*binInterval))


#####################################################
# Functions necessary to update belief after odometry measurement
"""
This function takes as inputs the new odometry measurement and the state at which the new belief 
will be determined and calculates 
the new belief due to the odometry meausrement
"""
def calculate_new_odom_belief(odomMeasurement, prevBelief, state):
    newBelief = 0

    # necessary variables for vectorization
    cells = np.array([[x,y,z] for x in range(0,prevBelief.shape[0]) 
                                for y in range(0,prevBelief.shape[1]) for z in range(0,prevBelief.shape[2])])
    states = state*np.ones((prevBelief.shape[0]*prevBelief.shape[1]*prevBelief.shape[2],3))

    # vectorized result
    table = np.array(motion_model(odomMeasurement*np.ones(prevBelief.shape[0]*prevBelief.shape[1]*prevBelief.shape[2]),
                            cells[:,0], cells[:,1], cells[:,2], states[:,0], states[:,1], states[:,2]))*prevBelief.flatten()

    # new belief for the state
    newBelief = sum(table)

    return newBelief

###############          
"""
This function takes as inputs the odometry measurement and the previous and new state
and calculates the probability p(newState|prevState, odometryMeasurement)
"""
def motion_model(odomMeasurement, condStateX, condStateY, condStateO, stateX, stateY, stateO):
    return 1/(odomMeasurement*np.sqrt(2*np.pi)) *  np.exp(-((stateX-condStateX)**2
                                                    +(stateY-condStateY)**2+(stateO-condStateO)**2)/(2*(odomMeasurement**2)))

#################################################################################
##### Testing 
######### Testing script with Tim's AirSim map
map = np.load("grid_map.npy")
map = np.pad(map, (1,1), 'constant', constant_values=(1,1))

# Print map
sns.heatmap(map, annot=True) 
plt.xlabel("y Coordinate")
plt.ylabel("x Coordinate")
plt.title("Map with dx=40 and dy=10")
plt.show()


dx = 40
dy = 10

bel = Belief(map, dx, dy)
print("Initial belief") 
#print(bel.belief[:,:,0])

# Belief update using range measurements if the agent is located at [10,5,0]
# The measurements are perturbations of the true expected measurement from that state. 
print("Belief at t=1")
print("Measurement", 9.5*dx)
bel.update_via_perc(9.5*dx,perc_model)

print("Belief at t=2")
print("Measurement", 9.5*dx+0.03)
bel.update_via_perc(9.5*dx+0.03,perc_model)

print("Belief at t=3")
print("Measurement", 9.5*dx+0.5)
bel.update_via_perc(9.5*dx+0.5,perc_model)
print("")

print("Indices of maximum belief")
print(bel.max_belief_idx())

print("Max Belief value")
print(np.max(bel.belief))

print("Val of true pos")
print(bel.belief[10,5,:])
print("")

print("Belief after t=3-bel1")
bel.plot_xy_belief()
    # The belief is maximum for the states that can actually acquire these measurements
print("Some of the Maxium Beliefs")
print(bel.belief[1,1,4])
print(bel.belief[1,5,4])
print(bel.belief[10,5,0])



# We also update using the following measurements, assuming we are in position [10,5],
# but looking at a different orientation (E). The belief produced is worse than before.
# This occurs because we have turned to obtain the new measurements and hence the belief for [10,5,0] 
# from previously should now be carried to [10,5,3]
print("Belief at t=4")
print("Measurement", 5.5*dy+0.001)
bel.update_via_perc(5.5*dy+0.001,perc_model)

print("Belief at t=5")
print("Measurement", 5.5*dy+0.03)
bel.update_via_perc(5.5*dy+0.03,perc_model)

print("Belief at t=6")
print("Measurement", 5.5*dy+0.5)
bel.update_via_perc(5.5*dy+0.5,perc_model)

print("Belief after t=6, without accounting for change of orientation-bel2")
bel.plot_xy_belief()


map = np.load("grid_map.npy")
map = np.pad(map, (1,1), 'constant', constant_values=(1,1))




###########################################################
#### Belief update by considering change of orientation
bel = Belief(map, dx, dy)
print("Initial belief") 

# Belief update using range measurements if the agent is located at [10,5,0]
# The measurements are perturbations of the true expected measurement from that state. 
print("Belief at t=1")
print("Measurement", 9.5*dx)
bel.update_via_perc(9.5*dx,perc_model)

print("Belief at t=2")
print("Measurement", 9.5*dx+0.03)
bel.update_via_perc(9.5*dx+0.03,perc_model)

print("Belief at t=3")
print("Measurement", 9.5*dx+0.5)
bel.update_via_perc(9.5*dx+0.5,perc_model)
print("")
bel.plot_xy_belief()



# We also update using the following measurements, assuming we are in position [10,5],
# but looking at a different orientation (E). We account for the change of orientation 
# by now cosnidering that the previous belief for (N) now becomes the belief for (E). 
# The certainty we had for our state before now becomes the certainty for the state with the new orientation, 
# since the change of orientation is determinisitc.

# !!!!!!!!!
bel.belief[:,:,[0, 2]] = bel.belief[:,:,[2,0]]

print("Belief at t=4")
print("Measurement", 5.5*dy+0.001)
bel.update_via_perc(5.5*dy+0.001,perc_model)

print("Belief at t=5")
print("Measurement", 5.5*dy+0.03)
bel.update_via_perc(5.5*dy+0.03,perc_model)

print("Belief at t=6")
print("Measurement", 5.5*dy+0.5)
bel.update_via_perc(5.5*dy+0.5,perc_model)

print("Belief after t=6, with accounting for change of orientation-bel3")
bel.plot_xy_belief()

















######## Previous Testing Script
# ###############################################################################
# #### Testing
# map = np.zeros((7,7))
# map[:,0] = 1
# map[:,6] = 1
# map[0,:] = 1
# map[6,:] = 1
# map[2,1] = 1
# map[3,3] = 1
# map[5,4] = 1
# dx = 40
# dy = 10

# # First Example - Agent at (x=3,y=2)
# bel = Belief(map, dx, dy)
# print("Initial belief") 
# #print(bel.belief[:,:,0])

# # # Belief update using range measurements
# print("Belief at t=1")
# print("Measurement", 5*40/2+0.003)
# bel.update_via_perc(5*40/2+0.003,perc_model)

# print("Belief at t=2")
# print("Measurement", 5*40/2+0.5)
# bel.update_via_perc(5*40/2+0.5,perc_model)

# print("Belief at t=3")
# print("Measurement", 5*40/2+10)
# bel.update_via_perc(5*40/2+10,perc_model)
# print("")

# print("Indices of maximum belief")
# print(bel.max_belief_idx())

# print("Max Belief value")
# print(np.max(bel.belief))

# print("Val of true pos")
# print(bel.belief[3,2,:])
# print("")

# bel.plot_xy_belief()


# print("######################################")

# # Second example - Agent at (x=5,y=1)
# bel = Belief(map, dx, dy)
# print("Initial belief") 
# #print(bel.belief[:,:,0])

# # # # Belief update using range measurements
# print("Belief at t=1")
# bel.update_via_perc(1.5*np.sqrt((40**2)+(10**2))+0.03,perc_model)

# print("Belief at t=2")
# bel.update_via_perc(1.5*np.sqrt((40**2)+(10**2))+0.005,perc_model)

# print("Belief at t=3")
# bel.update_via_perc(1.5*np.sqrt((40**2)+(10**2)),perc_model)

# print("Indices of maximum belief")
# print(bel.max_belief_idx())
# print("Max Belief value")
# print(np.max(bel.belief))
# print("Val of true pos")
# print(bel.belief[5,1,:])


# # Comments:
# """
# We observe that the true position has a belief as large as the maximum belief and 
# if the indices of the returned maximum are different, then that point also satisfies the measurements.

# Also, to test we must provide perturbations of the expected measurement 
# without altering the orientation at the cell with which we get the measurements. To alter the orientation, 
# we would first need the belief to be updated due to the pose change (using update_odom)
# """

# # Belief updates using odometry measurements
# # print("Belief at t=4")
# # bel.update_via_motion(0.2,calculate_new_odom_belief)
# # print(bel.belief[0,0,:])

# # print("Belief at t=5")
# # bel.update_via_motion(1.3,calculate_new_odom_belief)
# # print(bel.belief[0,0,:])

# # print("Belief at t=6")
# # bel.update_via_motion(3.1,calculate_new_odom_belief)
# # print(bel.belief[0,0,:])