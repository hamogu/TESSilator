import numpy as np

import pytest

from ..lc_analysis import aic_selector
from ..logger import logger_tessilator


logger = logger_tessilator('aic_tests')

start, stop, typical_timestep = 0, 27, 0.007 # in days
period = 3.5
times = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)
y_err = 0.0005

text_x, text_y = 0.05, 0.95

flat_coeffs = [4.]
line_coeffs = [1., .1]
para_coeffs = [1.5, 0.75, 3.]



#########################################
######FUNCTIONS NEEDED TO RUN TESTS######
#########################################


def get_coords(curve, err=False):
    x = times
    y = np.zeros(len(times))
    for i, c in enumerate(curve):
        y += c * times ** (len(curve) - i - 1)
    if err:
        y = [i + np.random.normal(0, y_err) for i in y]
    return x, y


#########################################
##############RUN EACH TEST##############
#########################################
def test_aic_parabola_fail():
    """ENSURE THE DEFAULT VALUES ARE RETURNED IF THE FUNCTION CRASHES"""
    x, y = times, np.ones(len(times)+50)
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert len(coeffs) == 1
    assert(np.isclose(poly_ord, 0.0, rtol=1e-05))
    assert(np.isclose(coeffs[0], 1.0, rtol=1e-05))


@pytest.parametrize("coeff_type", [flat_coeffs, line_coeffs, para_coeffs])
@pytest.mark.parametrize("err", [True, False])
def test_aic(coeff_type, err):
    '''TRY A COMPLETELY FLAT LIGHTCURVE'''
    x, y = get_coords(coeff_type, err=err)
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert len(coeffs) == len(coeff_type)
    assert coeffs == pytest.approx(coeff_type, rel=1e-01)
