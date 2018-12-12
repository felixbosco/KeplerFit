# KeplerFit
A small piece of code to fit a Keplerian velocity distribution model to position-velocity data. Please take a look at the example under **Basic usage**.

## Basic usage
```python
from KeplerFit import PVdata, model_Keplerian
import astropy.units as u
from astropy.modeling.fitting import SLSQPLSQFitter

pvdata = PVdata('my_pv_data.fits', noise=5.66e-05*u.Jy/u.pixel, position_reference=467)
results = model_Keplerian(pvdata, 4, source_distance=2.0*u.kpc, return_stddevs=True, plot=True, 
                             flag_radius=50*u.AU,
                             #fit_method=SLSQPLSQFitter(),
                             flag_intervals=[(-10000*u.AU, -3000*u.AU), (3000*u.AU, 10000*u.AU)])
print(results)
```
