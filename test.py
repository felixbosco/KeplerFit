from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

from pvdata import PVData


# Parameters
file = 'tests/23033_H2CO_l08_ABD_MMS1a_120.xy.fits'
sigma = 3


# Initialize
pv_data = PVData.from_file(file)
pv_data.estimate_extreme_velocity_channels(sigma = sigma)

# Plot results
plt.imshow(np.ma.masked_less(pv_data.data, sigma * pv_data.noise), origin='lower')
plt.plot(pv_data.min_channels, 'o', c='tab:orange', label='min')
plt.plot(pv_data.max_channels, 'o', c='tab:red', label='max')
plt.legend()
plt.show()

# Embed for future fun
# embed()
