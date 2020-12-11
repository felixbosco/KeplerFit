import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as const
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.units import Unit, Quantity


@custom_model
def Keplerian1D(x, mass=1., v0=0., r0=0.):
    """Computes the Keplerian velocity at requested distances from a massive object.

    Args:
        x (array_like):
            Distances to the central object.
        mass (float, optional):
            Mass of the central object in solar masses.
        v0 (float, optional):
            Velocity offset or systemic velocity.
        r0 (float, optional):
            Position offset, the position of the central object.

    Returns:
        v (np.ndarray):
            Keplerian velocity at the positions x.
    """
    v = np.sign(x - r0) * np.sqrt(const.G * mass * const.M_sun / np.abs(x - r0) / Unit('AU')).to('km/ s').value + v0
    return v


def model_keplerian(positions, velocities, v_lsr=None, fit_method=None,
                    flag_singularity=True, flag_radius=None, flag_intervals=None,
                    return_stddevs=True, plot=False, debug=False):

    """Fit a Keplerian velocity profile to position-velocity-data.

    Args:
        positions (np.ndarray or Quantity):
            PVdata object to compute the data from.
        velocities (np.ndarray or Quantity):
            Set as multiples of PVdata.noise (for instance 3sigma)
        v_lsr (float):
            Systemic velocity in units of km/ s.
        fit_method (any, optional):
            Method to fit the model to the data.
        flag_singularity (bool, optional):
            Flag the zero position data points, to avoid running in trouble there during fitting.
        flag_radius (astropy.units.Quantity, optional):
            If given, then all data points within this given radius from the position_reference are flagged.
        flag_intervals (list of tupels of astropy.units.Quantity, optional):
            Similar to flag_radius, but arbitrary intervals may be flagged. Each interval is
            given as a tuple of two radial distances from the position_reference.
        return_stddevs (boolean, optional):
            The fit method LevMarLSQFitter is able to return the standard deviation of the fit parameters. Default is
            True.
        plot (boolean, optional):
            If True, the fit will be displayed as a matplotlib pyplot.
        debug (bool, optional):
            Stream debugging information to the terminal.

    Returns:
        best_fit (astropy.modelling.models.custom_model):
            Best fitting model.
        stddevs (numpy.array):
            Only if return_stddevs is True. The array entries correspond to the best_fit instance parameters in the
            same order.
        chi2 (float):
            chi-squared residual of the fit to the unflagged data.
    """

    # Transform Quantities to correct units
    if isinstance(positions, Quantity):
        positions = positions.to('AU').value
    if isinstance(velocities, Quantity):
        velocities = velocities.to('km/ s').value

    # Apply fall back values
    if fit_method is None:
        fit_method = LevMarLSQFitter()
    if v_lsr is None:
        v_lsr = 0

    # Create masked arrays
    xdata = np.ma.masked_array(positions, np.zeros(positions.shape, dtype=bool))
    ydata = np.ma.masked_array(velocities, np.zeros(velocities.shape, dtype=bool))

    # Mask the desired flags and intervals
    if flag_singularity:
        print('Flagging the singularity')
        singularity_mask = np.ma.masked_less(np.abs(xdata), 1e-3).mask
        xdata.mask = np.logical_or(xdata.mask, singularity_mask)
        ydata.mask = np.logical_or(ydata.mask, singularity_mask)
        print(f">> Done")
    else:
        print("Not masking the singularity")

    if flag_radius is not None:
        print(f"Flagging towards a radial distance of {flag_radius}")
        if isinstance(flag_radius, Quantity):
            flag_radius = flag_radius.to('au').value
        xdata = np.ma.masked_inside(xdata, -flag_radius, flag_radius)
        ydata.mask = np.logical_or(ydata.mask, xdata.mask)
        print(f">> Done")
        print(f"The mask is {xdata.mask}")
    else:
        print("No flag radius provided")

    if flag_intervals is not None:
        print('Flagging intervals...')
        for interval in flag_intervals:
            xdata = np.ma.masked_inside(xdata, interval[0], interval[1])
            ydata.mask = np.logical_or(ydata.mask, xdata.mask)
        print(f">> Flagged {np.sum(xdata.mask)} elements")
    else:
        print("No flag intervals provided")

    if debug:
        print('x data:', xdata)
        print('y data:', ydata)

    # Initialize the fit model
    print("Initializing the model...")
    model = Keplerian1D(mass=10., v0=v_lsr, r0=0, bounds={'mass': (0.0, None)})
    if debug:
        print(f"Initialize the model: {model}")

    # Fit the chosen model to the data
    print("Fitting the model to the data...")
    best_fit = fit_method(model, xdata.compressed(), ydata.compressed())
    if debug:
        print(fit_method.fit_info['message'])

    # Estimate chi2
    print("Computing the chi-squared value...")
    chi2 = np.sum(np.square(best_fit(xdata.compressed()) - ydata.compressed()))

    # Plot
    if plot:
        plt.plot(positions, velocities, 'o', label='data')
        plt.xlabel('Position offset (AU)')
        plt.ylabel('Velocity (km/ s)')
        plt.axhline(v_lsr, c='k', ls='--', label=r'$v_\mathrm{LSR}$')
        plt.plot(xdata, best_fit(xdata), label='model')
        plt.fill_between(xdata, best_fit(xdata), best_fit.v0, facecolor='tab:orange', alpha=.5)
        if debug:
            plt.plot(xdata, model(xdata), label='init')
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

    # Prepare the return
    stddevs = None
    if not isinstance(fit_method, LevMarLSQFitter):
        return_stddevs = False
    if return_stddevs:
        covariance = fit_method.fit_info['param_cov']
        if covariance is None:
            print(f"[ERROR] Unable to compute the covariance matrix and fit parameter uncertainties!")
        else:
            stddevs = np.sqrt(np.diag(covariance))

    return best_fit, stddevs, chi2
