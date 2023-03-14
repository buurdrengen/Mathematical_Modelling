import numpy as np
import matplotlib.pyplot as plt

r_earth = 6371 # [km]

data = np.loadtxt("Depression/channel_data.txt") 
H = data[:, 2]
delta_phi = (data[1:,0] - data[:-1, 0])*np.pi/180
delta_theta = (data[1:,1] - data[:-1, 1])*np.pi/180

r = r_earth*np.sqrt(delta_phi**2 + delta_theta**2) # [km] - NOT A PERFECT EQUATION BUT AN APPROXIMATION - SEE HAVERSINE

r_sum = np.cumsum(r) # Get the difference between points
r_sum = np.append(0, r_sum) # append 0 to the list so that the range corresponds to the distance from start point


r_interp = np.arange(0, np.max(r_sum), 0.25) # Create the axis which we want our function interpolated to.
h_interp = np.interp(r_interp, r_sum, H) # Interpolate the data with a linear interpolation

interpData = np.stack((r_interp, h_interp))
np.savetxt("Depression/channel_data_interp.csv", interpData.T, delimiter=",")

# plt.figure()
# plt.plot(r_sum, H)
# plt.plot(r_interp, h_interp)
# plt.grid()
# plt.show()



