import numpy as np
from numpy.core.numeric import Inf

"""
The following code constructs the belief that an agent is located at any one of 
the possible points in the grid map. The belief is a 3D array with dimensions x, y, orientation
The inputs to the constructor are the following:
    - map of the environment: This is assumed to be a rectangular area that is 
                              represented as a binary numpy array, with 0 in the cells 
                              with obstacles and 1 in the cells without obstacles
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
        self.map = map # Copy of the environment map
        self.dx = dx # x-size of grid cell
        self.dy = dy # y-size of grid cell
        self.percSigma = 10 # sigma of perception model's distribution

        # Initialize belief
        self.belief = np.ones((self.dimX, self.dimY, self.numOrients)) 

        # Calculate possible cells-orientations using map and assign uniform probability to them
        for i in range(0, self.belief.shape[2]):
            self.belief[:,:,i] = (1/(self.numOrients*sum(sum(map))))*np.multiply(self.belief[:,:,i], map)
            
    # Function to update belief based on incoming range measurement
    def update_via_perc(self,actMeasurement, perc_model, calculate_expected_perc_measurement):
        expMeasurement = np.zeros((self.belief.shape[0], self.belief.shape[1], self.belief.shape[2]))
        for orient in range(0, self.numOrients):
            for x in range(0, self.dimX):
                for y in range(0, self.dimY):
                    # For each cell, calculate probability p(actMeasurement|expMeasurement)
                    expMeasurement[x,y,orient] = perc_model(
                        actMeasurement, calculate_expected_perc_measurement(
                            x, y, orient, self.map, self.dx, self.dy), self.percSigma)

        self.belief = np.multiply(expMeasurement, self.belief)
        self.belief = self.belief/sum(sum(sum(self.belief)))

    # Function to update belief based on incoming odometry measurement
    def update_via_motion(self, actMeasurement, calculate_new_odom_belief):
        prevBelief = self.belief

        for i in range(self.belief.shape[0]):
            for j in range(self.belief.shape[1]):
                for o in range(self.belief.shape[2]):
                    # For each position, update the belief
                    self.belief[i,j,o] = calculate_new_odom_belief(actMeasurement, i, j, o, prevBelief)


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
def calculate_expected_perc_measurement(cellX, cellY, orientation, map, dx, dy):
    
    if orientation == 0:
        x = cellX-1
        while x!=0 and map[x, cellY]!= 0:
                x = x-1
        dist = (cellX-x)*dx
            
    elif orientation == 1:
        i = 1
        while cellX-i!=0 and cellY+i!=map.shape[1] and map[cellX-i, cellY+i]!=0:
            i = i+1
        dist = np.sqrt(((cellX-i)*dx)**2 + ((i-cellY)*dy)**2)
    
    elif orientation == 2:
        y = cellY+1
        while y!=map.shape[1] and map[cellX, y]!=0:
            y = y + 1
        dist = (y-cellY)*dx

    elif orientation == 3:
        i = 1
        while cellX+i!=map.shape[0] and cellY+i!=map.shape[1] and map[cellX+1, cellY+1]!=0:
            i = i+1
        dist = np.sqrt(((i-cellX)*dx)**2 + ((i-cellY)*dy)**2)

    elif orientation == 4:
        x = cellX+1
        while x!=map.shape[0] and map[x, cellY]!=0:
            x = x+1
        dist = (x-cellX)*dx
    
    elif orientation == 5:
        i = 1
        while cellX+i!=map.shape[0] and cellY-i!=0 and map[cellX+i, cellY-i]!=0:
            i = i+1
        dist = np.sqrt(((i-cellX)*dx)**2 + ((cellY-i)*dy)**2)

    elif orientation == 6:
        y = cellY-1
        while y!=0 and map[cellX, y]!=0:
            y = y-1
        dist = (cellY-y)*dy
    
    elif orientation == 7:
        i = 1
        while cellX-1!=0 and cellY-1!=0 and map[cellX-i, cellY-i]!=0:
            i = i+1
        dist = np.sqrt(((cellX-i)*dx)**2 + ((cellY-i)*dy)**2)
    else:
        print("Error")
        dist = Inf
    return dist

###
"""
This function takes as inputs the expected measurement (expMeasurement) and actual measurement
(actMeasurement) and outputs the probability p(actMeasurement|expMeasurement).
"""
def perc_model(actMeasurement, expMeasurement, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((actMeasurement-expMeasurement)**2)/(2*(sigma**2))  )



#####################################################
# Functions necessary to update belief after odometry measurement
"""
This function takes as inputs the odometry measurement and the previous and new state
and calculates the probability p(newState|prevState, odometryMeasurement)
"""
def motion_model(odomMeasurement, condStateX, condStateY, condStateO, stateX, stateY, stateO):
    return 1/(odomMeasurement*np.sqrt(2*np.pi)) *  np.exp(-((stateX-condStateX)**2
                                                    +(stateY-condStateY)**2+(stateO-condStateO)**2)/(2*(odomMeasurement**2)))

"""
This function takes as inputs the new odometry measurement and the state at which the new belief 
will be determined and calculates 
the new belief due to the odometry meausrement
"""
def calculate_new_odom_belief(odomMeasurement, stateX, stateY, stateO, prevBelief):
    newBelief = 0
    for x in range(prevBelief.shape[0]):
        for y in range(prevBelief.shape[1]):
            for orient in range(prevBelief.shape[2]):
                newBelief += motion_model(odomMeasurement, x, y, orient,
                stateX, stateY, stateO)*prevBelief[x,y,orient]

    return newBelief
            













#### Testing
map = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
dx = 2
dy = 2

bel = Belief(map, dx, dy)
print("Initial belief") 
print(bel.belief[0,0,:])

print("Belief at t=1")
bel.update_via_perc(0.5,perc_model, calculate_expected_perc_measurement)
print(bel.belief[0,0,:])

print("Belief at t=2")
bel.update_via_perc(0.9,perc_model, calculate_expected_perc_measurement)
print(bel.belief[0,0,:])

print("Belief at t=3")
bel.update_via_perc(1.3,perc_model, calculate_expected_perc_measurement)
print(bel.belief[0,0,:])

print("Belief at t=4")
bel.update_via_motion(0.2,calculate_new_odom_belief)
print(bel.belief[0,0,:])

print("Belief at t=5")
bel.update_via_motion(1.3,calculate_new_odom_belief)
print(bel.belief[0,0,:])

print("Belief at t=6")
bel.update_via_motion(3.1,calculate_new_odom_belief)
print(bel.belief[0,0,:])