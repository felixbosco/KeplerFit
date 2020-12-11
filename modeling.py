import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as const
from astropy.io import fits
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
            Mass of the central object.
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


def model_Keplerian(positions, velocities, fit_method=None,
                    flag_singularity=True, flag_radius=None, flag_intervals=None,
                    weak_quadrants=False, return_stddevs=True, plot=False, debug=False):

    """Fit a Keplerian velocity profile to position-velocity-data.

    Args:
        positions (np.ndarray or Quantity):
            PVdata object to compute the data from.
        velocities (np.ndarray or Quantity):
            Set as multiples of PVdata.noise (for instance 3sigma)
        source_distance (any):
            Distance to the source, which is necessary for computing physical distances.
        fit_method (any, optional):
            Method to fit the model to the data.
        flag_singularity (bool, optional):
            Flag the zero position data points, to avoid running in trouble there during fitting.
        flag_radius (astropy.units.Quantity, optional):
            If given, then all data points within this given radius from the position_reference are flagged.
        flag_intervals (list of tupels of astropy.units.Quantity, optional):
            Similar to flag_radius, but arbitrary intervals may be flagged. Each interval is
            given as a tuple of two radial distances from the position_reference.
        velocity_interval (any, optional):
            Velocity interval to restrict the fitting to.
        channel_interval (any, optional):
            Channel interval to restrict the fitting to.
        weak_quadrants (bool, optional):
            Fit the model to the signal in the weaker opposing quadrants.
        return_stddevs (boolean, optional):
            The fit method LevMarLSQFitter is able to return the standard deviation of the fit parameters. Default is
            True.
        plot (boolean, optional):
            If True, the fit will be displayed as a matplotlib pyplot.
        write_table_to (str, optional):
            Name of a file to write the data points to, formatted as a table.
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
    v_lsr = 0

    # Create masked arrays
    xdata = np.ma.masked_array(positions, np.zeros(positions.shape, dtype=bool))
    ydata = np.ma.masked_array(velocities, np.zeros(velocities.shape, dtype=bool))

    # Mask the desired flags and intervals
    if flag_singularity:
        print('Flagging the singularity')
        xdata = np.ma.masked_less(np.abs(xdata), 1e-6)
        ydata.mask = np.logical_or(ydata.mask, xdata.mask)
        i = np.where(np.abs(positions) < 1e-6)[0]
        xdata.mask[i] = True
        ydata.mask[i] = True
        print(f">> Done")
    if flag_radius is not None:
        print(f"Flagging towards a radial distance of {flag_radius}")
        if isinstance(flag_radius, Quantity):
            flag_radius = flag_radius.to('au').value
        xdata = np.ma.masked_less(xdata, flag_radius)
        ydata.mask = np.logical_or(ydata.mask, xdata.mask)
        print(f">> Done")
        print(f"The mask is {xdata.mask}")
    if flag_intervals is not None:
        print('Flagging intervals...')
        for interval in flag_intervals:
            xdata = np.ma.masked_inside(xdata, interval[0], interval[1])
            ydata.mask = np.logical_or(ydata.mask, xdata.mask)
        print(f">> Flagged {np.sum(xdata.mask)} elements")

    # Initialize the fit model
    # if self.start_low(indices=indices, weak_quadrants=weak_quadrants):
    #     model = Keplerian1D(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)})
    # else:
    #     model = Keplerian1D_neg(mass=10., v0=self.vLSR.value, r0=0, bounds={'mass': (0.0, None)})
    model = Keplerian1D(mass=10., v0=v_lsr, r0=0, bounds={'mass': (0.0, None)})

    # Fit the chosen model to the data
    best_fit = fit_method(model, xdata.compressed(), ydata.compressed())

    # Estimate chi2
    chi2 = np.sum(np.square(best_fit(xdata) - ydata))

    # Plot
    if plot:
        plt.plot(positions, velocities, 'o', label='data')
        plt.xlabel('Position offest (AU)')
        plt.ylabel('Velocity (km/ s)')
        plt.axhline(v_lsr, c='k', ls='--', label='$v_\mathrm{LSR}$')
        plt.plot(xdata, best_fit(xdata), label='model')
        plt.fill_between(xdata, best_fit(xdata), best_fit.v0, facecolor='tab:orange', alpha=.5)
        if debug:
            plt.plot(xdata, model(xdata), label='init')
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

    # return
    if not isinstance(fit_method, LevMarLSQFitter):
        return_stddevs = False
    if return_stddevs:
        covariance = fit_method.fit_info['param_cov']
        try:
            stddevs = np.sqrt(np.diag(covariance))
            return best_fit, stddevs, chi2
        except ValueError as e:
            if covariance is None:
                print(f"Catched an error that is due to an unsuccessful fit (covariance matrix is {covariance})")
                print(f"ValueError: {e}")
            return best_fit, chi2
    if debug:
        print(fit_method.fit_info['message'])

    return best_fit, chi2
