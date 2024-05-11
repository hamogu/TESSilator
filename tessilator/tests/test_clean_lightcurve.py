from ..lc_analysis import aic_selector, clean_edges_outlier, clean_edges_scatter, detrend_lc, get_time_segments, logger, make_lc, norm_choice, remove_sparse_data, sin_fit, cbv_fit_test

from ..logger import logger_tessilator
# Third party imports
import numpy as np
import os
import json


from astropy.table import Table
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq

from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
from scipy.stats import iqr

import itertools as it
from operator import itemgetter

from astropy.table import Table
from astropy.io import ascii

import pytest


@pytest.mark.parametrize(
    "test_data",
    [
        "Gaia_DR3_2314778985026776320_tests/ap_2314778985026776320_0029_1_2",
        "BD+20_2465_tests/ap_BD+20_2465_0045_3_4",
        "GESJ08065664-4703000_tests/ap_GESJ08065664-4703000_0061_3_1",
        "ABDor_tests/ap_AB_Dor_0036_4_3",
    ],
)
def test_make_lcs(test_data):
    """Run make_lc on fixed, known lightcurves and compare to fixed, known outputs"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ap_in = ascii.read(f"{test_dir}/targets_tests/{test_data}.csv")
    ap_in = ap_in[ap_in["flux"] > 0.0]
    lcname = f"{test_dir}/targets_tests/{test_data}_reg_oflux.csv"
    lcname = lcname.replace("ap_", "lc_")
    lc_test = ascii.read(lcname, converters={"pass*": bool})
    lc_exam = make_lc(ap_in)[0]
    for col in lc_test.colnames:
        if col == "nflux_dtr":
            continue
        # How precise do we expect this to be?
        # We do not want to trigger this test every time a tiny things changes.
        # So, set some limits that astrophysically make sense.
        assert lc_test[col].data == pytest.approx(lc_exam[col].data, rel=1e-4, abs=1e-8)
