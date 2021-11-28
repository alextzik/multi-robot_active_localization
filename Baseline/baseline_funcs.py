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

        # Initialize necessary variables
        self.dimX = map.shape[0] # x-dimension of map is x-dimension of belief-space
        self.dimY = map.shape[1] # y-dimension of map is y-dimension of belief-space
        self.numOrients = 8 # number of possible orientations (this is equal to the number of adjacent position-cells)
        self.map = map # Copy of the environment map in the class
        self.dx = dx # x-size of grid cell
        self.dy = dy # y-size of grid cell

        # Perception model constants
        self.percSigma = 400 # sigma of perception model's distribution (30 was the initial value)
                             # Given that the agent might not be at the center of each cell, this needs to be a great number to allow 
                             # for the uncertainty in position within the cell.

        self.numOfBins = 21 # odd for symmetry assumed
        self.BinInterval = np.sqrt(((dx*(self.dimX+1))**2)+((dy*(self.dimY+1))**2))/self.numOfBins # bin size, based on maximum distance we may reach

        # Initialize belief
        self.belief = np.ones((self.dimX, self.dimY, self.numOrients)) 
        # Calculate possible cells-orientations using map and assign uniform probability to them
        self.belief = (1/(self.numOrients*sum(sum(1-map))))*(self.belief*(1-map[:,:,np.newaxis]))

        # Calculated expected range measurements for all positions and orientations
        self.expectedPercMeasurements = calculate_expected_perc_measurements(self.belief, map, dx, dy)
            
    # Function to update belief based on incoming range measurement
    """
    This functions takes in the actual LiDAR measurement, the perception model distribution and the current 
    orientation (0<=i<=7) and updates the beliefs at all cells and orientations based on the measurement. 
    Given that the current orientation is known, it assigns zero probability to all other orientations after 
    performing the update. 
    """
    def update_via_perc(self,actMeasurement, perc_model, i):
        
        # Calculate the probability of receiving actual measurement given that we are in state (x,y,o)
        probMeasGivenState = np.zeros((self.belief.shape[0], self.belief.shape[1], self.belief.shape[2]))
        probMeasGivenState = perc_model(actMeasurement*np.ones(self.belief.shape), 
                                self.expectedPercMeasurements, self.percSigma*np.ones(self.belief.shape), 
                                self.BinInterval*np.ones(self.belief.shape))

        # Calculate new belief
        self.belief = probMeasGivenState*self.belief

        # Assign zero to orientations difeerent than the current one (i)
        arr = self.belief.copy()
        self.belief = np.zeros((self.dimX, self.dimY, self.numOrients)) 
        self.belief[:,:,i] = arr[:,:,i]

        # Normalize new belief
        self.belief = self.belief/sum(sum(sum(self.belief)))

    # Function to update belief based on incoming odometry measurement. 
    """
    The odometry measurement can only be 1, meaning a change of orientation by one value clockwise
    """
    def update_via_motion(self): 
        prevBelief = self.belief.copy()

        # self.belief[:,:, j] = prevBelief[:,:, i]
        # self.belief[:,:, i] = prevBelief[:,:, j]
        for i in range (0, prevBelief.shape[2]):
            self.belief[:,:,(i+1)%prevBelief.shape[2]] = prevBelief[:,:,i]

    # Function that returns one tuple (x,y,orientation) with the highest belief
    def max_belief_idx(self):
        linIndx = np.argmax(self.belief)
        x = int(np.floor(linIndx/(self.dimY*self.numOrients)))
        
        linIndx = np.mod(linIndx, (self.dimY*self.numOrients))
        y = int(np.floor(linIndx/self.numOrients))
       
        linIndx = np.mod(linIndx, self.numOrients)
        o = linIndx
        return x,y,o

    # Plotting function that returns the probability of being at all (x,y) pairs
    """
    Given that the first orientation that updates the belief is North, the last one will be
    NorthWest. Therefore, only that will have nonzero probability, according to update_via_perc
    """
    def plot_xy_belief(self):
        xyBelief = self.belief[:,:,7]
        sns.heatmap(xyBelief, annot=True) #np.swapaxes(xyBelief, 1, 0)
        plt.xlabel("y Coordinate")
        plt.ylabel("x Coordinate")
        plt.title("Belief at every (x,y)")
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
This function takes as inputs the new odometry measurement, the previous belief 
and the state at which the new belief will be determined 
and calculates the new belief at that state due to the odometry meausrement
"""
# Currently not used!
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
# Currently not used!
def motion_model(odomMeasurement, condStateX, condStateY, condStateO, stateX, stateY, stateO):
    return 1/(odomMeasurement*np.sqrt(2*np.pi)) *  np.exp(-((stateX-condStateX)**2
                                                    +(stateY-condStateY)**2+(stateO-condStateO)**2)/(2*(odomMeasurement**2)))

