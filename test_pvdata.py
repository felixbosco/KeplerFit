from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

from pvdata import PVData
from modeling import model_Keplerian


# Parameters
file = 'tests/<your_fits_file>'
sigma = 3
distance = 3500.  # in parsec
v_lsr = -53.1  # in km/ s


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
# plt.show()
plt.close()

# Combine to PV-signal
pv_data.combine_extreme_channels(weak_quadrants=False)

# Plot results
plt.imshow(np.ma.masked_less(pv_data.data, sigma * pv_data.noise), origin='lower')
plt.plot(pv_data.extreme_channels, 'x', c='tab:red', label='extreme')
plt.legend()
# plt.show()
plt.close()

# Transform into linear units and plot
positions, velocities = pv_data.estimate_extreme_velocities(distance=distance)
plt.plot(positions, velocities, 'o', c='k')
# plt.show()
plt.close()

# Model the data
model_Keplerian(positions=positions, velocities=velocities, v_lsr=v_lsr, flag_radius=100, flag_intervals=[(-700, 0)],
                plot=True, debug=True)

# Embed for future fun
# embed()
