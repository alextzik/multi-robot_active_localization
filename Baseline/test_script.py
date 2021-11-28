from baseline_funcs import *

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
print(np.unravel_index(np.argmax(bel.belief), bel.belief.shape))

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
# from previously should now be carried to [10,5,2]
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
print(bel.belief[:,:,0])
bel.update_via_motion()
print()
print(bel.belief[:,:,1])
bel.update_via_motion()

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


print(sum(sum(sum(bel.belief))))
