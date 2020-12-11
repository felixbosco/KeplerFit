from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

from pvdata import PVData


# Parameters
file = 'tests/23033_H2CO_l08_ABD_MMS1a_120.xy.fits'
sigma = 3
distance = 3500.


# Initialize
pv_data = PVData.from_file(file)
pv_data.position_reference = pv_data.data.shape[1] // 2
pv_data.vel_unit = 'm/ s'
pv_data.pos_unit = 'deg'
pv_data.estimate_extreme_velocity_channels(sigma = sigma)

# Plot results
plt.imshow(np.ma.masked_less(pv_data.data, sigma * pv_data.noise), origin='lower')
plt.plot(pv_data.min_channels, '^', c='tab:orange', label='min')
plt.plot(pv_data.max_channels, 'v', c='tab:red', label='max')
plt.legend()
plt.show()
plt.close()

# Combine to PV-signal
pv_data.combine_extreme_channels(weak_quadrants=False)

# Plot results
plt.imshow(np.ma.masked_less(pv_data.data, sigma * pv_data.noise), origin='lower')
plt.plot(pv_data.extreme_channels, 'x', c='tab:red', label='extreme')
plt.legend()
plt.show()
plt.close()

# Transform into linear units and plot
positions, velocities = pv_data.estimate_extreme_velocities(distance=distance)
plt.plot(positions, velocities, 'o', c='k')
plt.show()


# Embed for future fun
# embed()
