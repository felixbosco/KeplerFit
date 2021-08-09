# KeplerFit
A small piece of code to fit a Keplerian velocity distribution model to position-velocity (PV) data, in order to obtain
an estimate of the enclosed mass. 
Please take a look at the example under **Basic usage**.
The code and the limitations of the fitting procedure are described in detail in the work by 
[Bosco et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...629A..10B/abstract). It is recommended to visit 
Appendix A of this paper.

## Basic usage

```python
from pvdata import PVData
from modeling import model_keplerian

# Your parameters
sigma = 3
v_lsr = -100.  # km/ s
distance = 2000.  # parsec

# Load the PV data from your FITS file
pv_data = PVData.from_file('your_pv_data.fits', noise=5.66e-5)

# Update parameters that have not been identified correctly from the FITS header
pv_data.position_reference = pv_data.data.shape[1] // 2 
pv_data.vel_unit = 'm/ s'
pv_data.pos_unit = 'deg'

# Extract position and velocity vectors
positions, velocities = pv_data.estimate_extreme_velocities(distance=distance)

# Model the data
best_fit, stddevs, chi2 = model_keplerian(positions=positions, velocities=velocities, v_lsr=v_lsr, 
                                          flag_radius=100, flag_intervals=[(-700, 0)], plot=True)
```
The `PVData` class is initialized by reading in a FITS file of PV data. It reads the FITS header for extracting the 
scales of the pixels in both directions, spatial and spectral. It is recommended to check the attributes of the `PVData`
instance to see, whether the information was extracted correctly. You can then update all the bad attributes manually.

In the next step, the class method `estimate_extreme_velocities` follows the algorithm described by 
Seifried et al. (2016) for extracting at each position the most extreme velocity. This method returns two arrays of 
`positions` and `velocities`, which can then be modelled in the next step.

The function `model_keplerian` models the extracted PV data and returns a set of the best-fit parameters (`best_fit`), 
the standard deviations (`stddevs`) in each of the 
parameters and the total <img src="https://render.githubusercontent.com/render/math?math=\chi^2"> residual of the fit 
(`chi2`). Note that the standard deviations cannot be computed for each fitting algorithm as not every algorithm 
provides this information. The `stddevs` is hence returned as `None` if the computation is not available from the 
applied fitting algorithm.

The fitting parameter set contains the following parameters:
- `mass`: The enclosed mass in units of solar masses times the inclination factor 
<img src="https://render.githubusercontent.com/render/math?math=M \cdot \sin^2 i">, resulting from the fact that we 
only observe the rotational velocity under a certain inclination. The estimate hence serves as a lower limit on the 
mass. See the Appendix A of [Bosco et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26A...629A..10B/abstract), 
for further notes on how to treat the fit result.
- `v0`: The velocity offset, equivalent to the systemic velocity or 
<img src="https://render.githubusercontent.com/render/math?math=v_\mathrm{LSR}"> of the system. This parameter allows 
for adjusting the model in velocity direction, below the pixel grid.
- `r0`: The position offset. This parameter allows for adjusting the model in spatial direction, below the pixel grid.


#### Older versions
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

## Development
The code was developed by Felix Bosco at the Max Planck Institute for Astronomy (MPIA), Heidelberg
- 09 August 2021: Expand the documentation in the README.md file (Adding references and comments on the fit parameters)
- 12 December 2020: Published new version of the PVData class 
- 30 April 2020: Minor bug fixes
- ...: Minor compatibility bug fixes
- 12 December 2018: First publication of the code on GitHub

## Acknowledgements
Please acknowledge the use of the code by citing the paper by Bosco et al. (2019), see below.

## References
- Bosco, Beuther, Ahmadi et al. (2019), A&A, 629, A10, 
[Link](https://ui.adsabs.harvard.edu/abs/2019A%26A...629A..10B/abstract)
- Seifried, Sánchez-Monge, Walch, & Banerjee (2016), MNRAS, 459, 1892 
[Link](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.1892S/abstract)
