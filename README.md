# KeplerFit
A small piece of code to fit a Keplerian velocity distribution model to position-velocity data. Please take a look at the example under **Basic usage**.

## Basic usage

```python
from pvdata import PVData
from modeling import model_keplerian

# Your parameters
sigma = 3
v_lsr = -100.  # km/ s
distance = 2000.  # parsec

pv_data = PVData.from_file('my_pv_data.fits', noise=5.66e-5)

# Update parameters that have not been identified correctly from the FITS header
pv_data.position_reference = pv_data.data.shape[1] // 2 
pv_data.vel_unit = 'm/ s'
pv_data.pos_unit = 'deg'

# Extract position and velocity vectors
positions, velocities = pv_data.estimate_extreme_velocities(distance=distance)

# Model the data
model_keplerian(positions=positions, velocities=velocities, v_lsr=v_lsr, 
                flag_radius=100, flag_intervals=[(-700, 0)],
                plot=True)
```

If you are using the older version, you can follow this example:
```python
from KeplerFit import PVdata, model_Keplerian
import astropy.units as u
from astropy.modeling.fitting import SLSQPLSQFitter

pvdata = PVdata('my_pv_data.fits', noise=5.66e-05*u.Jy/u.pixel, position_reference=467)
results = model_Keplerian(pvdata, 4, source_distance=2.0*u.kpc, return_stddevs=True, plot=True, 
                             flag_radius=50*u.AU,
                             #fit_method=SLSQPLSQFitter(),
                             #velocity_interval=(-10*u.km/u.s, 10*u.km/u.s),
                             flag_intervals=[(-10000*u.AU, -3000*u.AU), (3000*u.AU, 10000*u.AU)])
print(results)
```
