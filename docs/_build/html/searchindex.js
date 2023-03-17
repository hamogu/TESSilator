Search.setIndex({"docnames": ["api/tessilator.contaminants.contamination", "api/tessilator.contaminants.flux_fraction_contaminant", "api/tessilator.contaminants.run_sql_query_contaminants", "api/tessilator.lc_analysis.aper_run", "api/tessilator.lc_analysis.clean_lc", "api/tessilator.lc_analysis.detrend_lc", "api/tessilator.lc_analysis.gauss_fit", "api/tessilator.lc_analysis.get_second_peak", "api/tessilator.lc_analysis.get_xy_pos", "api/tessilator.lc_analysis.is_period_cont", "api/tessilator.lc_analysis.make_lc", "api/tessilator.lc_analysis.run_ls", "api/tessilator.lc_analysis.sin_fit", "api/tessilator.makeplots.make_plot", "api/tessilator.maketable.get_gaia_data", "api/tessilator.maketable.table_from_coords", "api/tessilator.maketable.table_from_simbad", "api/tessilator.maketable.table_from_table", "api/tessilator.tessilator.all_sources_cutout", "api/tessilator.tessilator.all_sources_sector", "api/tessilator.tessilator.collect_contamination_data", "api/tessilator.tessilator.create_table_template", "api/tessilator.tessilator.find_xy_cont", "api/tessilator.tessilator.full_run_lc", "api/tessilator.tessilator.get_cutouts", "api/tessilator.tessilator.get_fits", "api/tessilator.tessilator.get_tess_pixel_xy", "api/tessilator.tessilator.make_2d_cutout", "api/tessilator.tessilator.make_datarow", "api/tessilator.tessilator.make_failrow", "api/tessilator.tessilator.one_cc", "api/tessilator.tessilator.one_source_cutout", "api/tessilator.tessilator.print_time_taken", "api/tessilator.tessilator.read_data", "api/tessilator.tessilator.run_test_for_contaminant", "api/tessilator.tessilator.setup_filenames", "api/tessilator.tessilator.setup_input_parameters", "api/tessilator.tessilator.test_table_large_sectors", "howtouse", "index", "install", "per_sector"], "filenames": ["api/tessilator.contaminants.contamination.rst", "api/tessilator.contaminants.flux_fraction_contaminant.rst", "api/tessilator.contaminants.run_sql_query_contaminants.rst", "api/tessilator.lc_analysis.aper_run.rst", "api/tessilator.lc_analysis.clean_lc.rst", "api/tessilator.lc_analysis.detrend_lc.rst", "api/tessilator.lc_analysis.gauss_fit.rst", "api/tessilator.lc_analysis.get_second_peak.rst", "api/tessilator.lc_analysis.get_xy_pos.rst", "api/tessilator.lc_analysis.is_period_cont.rst", "api/tessilator.lc_analysis.make_lc.rst", "api/tessilator.lc_analysis.run_ls.rst", "api/tessilator.lc_analysis.sin_fit.rst", "api/tessilator.makeplots.make_plot.rst", "api/tessilator.maketable.get_gaia_data.rst", "api/tessilator.maketable.table_from_coords.rst", "api/tessilator.maketable.table_from_simbad.rst", "api/tessilator.maketable.table_from_table.rst", "api/tessilator.tessilator.all_sources_cutout.rst", "api/tessilator.tessilator.all_sources_sector.rst", "api/tessilator.tessilator.collect_contamination_data.rst", "api/tessilator.tessilator.create_table_template.rst", "api/tessilator.tessilator.find_xy_cont.rst", "api/tessilator.tessilator.full_run_lc.rst", "api/tessilator.tessilator.get_cutouts.rst", "api/tessilator.tessilator.get_fits.rst", "api/tessilator.tessilator.get_tess_pixel_xy.rst", "api/tessilator.tessilator.make_2d_cutout.rst", "api/tessilator.tessilator.make_datarow.rst", "api/tessilator.tessilator.make_failrow.rst", "api/tessilator.tessilator.one_cc.rst", "api/tessilator.tessilator.one_source_cutout.rst", "api/tessilator.tessilator.print_time_taken.rst", "api/tessilator.tessilator.read_data.rst", "api/tessilator.tessilator.run_test_for_contaminant.rst", "api/tessilator.tessilator.setup_filenames.rst", "api/tessilator.tessilator.setup_input_parameters.rst", "api/tessilator.tessilator.test_table_large_sectors.rst", "howtouse.rst", "index.rst", "install.rst", "per_sector.rst"], "titles": ["contamination", "flux_fraction_contaminant", "run_sql_query_contaminants", "aper_run", "clean_lc", "detrend_lc", "gauss_fit", "get_second_peak", "get_xy_pos", "is_period_cont", "make_lc", "run_ls", "sin_fit", "make_plot", "get_gaia_data", "table_from_coords", "table_from_simbad", "table_from_table", "all_sources_cutout", "all_sources_sector", "collect_contamination_data", "create_table_template", "find_xy_cont", "full_run_lc", "get_cutouts", "get_fits", "get_tess_pixel_xy", "make_2d_cutout", "make_datarow", "make_failrow", "one_cc", "one_source_cutout", "print_time_taken", "read_data", "run_test_for_contaminant", "setup_filenames", "setup_input_parameters", "test_table_large_sectors", "How to use", "<strong>WELCOME TO THE TESSILATOR</strong>", "Installing tessilator software", "Using the TESSilator per sector"], "terms": {"tessil": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37], "t_target": [0, 2, 18, 19, 20, 23, 26, 28, 29, 30, 33, 38], "lc_con": [0, 18, 20, 22, 23, 31, 36, 38], "rad": [0, 1, 3, 13, 23, 30, 41], "1": [0, 1, 3, 4, 10, 13, 14, 22, 23, 25, 30, 33, 36, 38, 39, 41], "0": [0, 1, 2, 3, 4, 9, 11, 13, 14, 15, 17, 23, 30, 33, 36, 38, 41], "n_cont": [0, 20], "5": [0, 2, 9, 10, 14, 18, 20, 31, 33, 36, 38, 39], "sourc": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], "estim": 0, "flux": [0, 1, 2, 3, 4, 5, 10, 11, 13, 18, 20, 23, 29, 30, 31, 34, 38], "from": [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 16, 17, 18, 19, 20, 23, 27, 30, 31, 33, 34, 36, 39, 40, 41], "neighbour": [0, 1, 2, 4, 9, 20, 22, 34, 36, 38, 39], "astropi": [0, 2, 3, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 37, 38, 41], "tabl": [0, 2, 3, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 41], "bool": [0, 14, 18, 19, 20, 23, 30, 31, 36], "float": [0, 1, 2, 3, 4, 6, 9, 11, 12, 13, 15, 17, 23, 24, 30, 31], "int": [0, 4, 11, 18, 22, 23, 24, 25, 30, 31, 33], "none": [0, 13, 18, 24, 31, 35, 37, 38, 41], "The": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40], "purpos": 0, "thi": [0, 2, 3, 4, 5, 9, 13, 14, 17, 18, 20, 22, 23, 24, 26, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41], "function": [0, 1, 2, 3, 4, 5, 9, 10, 12, 20, 22, 23, 24, 25, 26, 28, 31, 32, 34, 36, 37, 39], "i": [0, 1, 2, 3, 4, 5, 9, 13, 14, 16, 17, 18, 19, 20, 23, 25, 26, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41], "amount": 0, "incid": 0, "tess": [0, 1, 10, 13, 22, 23, 24, 37, 38, 39], "apertur": [0, 1, 2, 3, 10, 13, 20, 22, 23, 25, 27, 30, 38], "origin": [0, 5, 10, 13], "given": [0, 1, 4, 6, 8, 19, 25, 26, 27, 29, 30, 36, 38, 41], "passband": 0, "t": [0, 1, 4, 5, 37, 40], "band": [0, 18, 29, 37], "600": 0, "1000nm": 0, "ar": [0, 4, 5, 13, 17, 18, 19, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41], "similar": 0, "gaia": [0, 2, 9, 14, 18, 22, 28, 29, 33, 37, 39, 41], "g": [0, 18, 29, 33, 36, 37, 38], "magnitud": [0, 10, 18, 29, 37, 38], "sensit": 0, "21": [0, 27], "dr3": [0, 18, 29, 33, 37, 39], "catalogu": [0, 39], "us": [0, 1, 2, 5, 10, 22, 25, 26, 27, 30, 33, 36, 37], "quantifi": [0, 1, 2, 39], "For": [0, 1, 26, 37], "each": [0, 3, 4, 5, 8, 10, 11, 15, 17, 19, 20, 26, 30, 38], "target": [0, 2, 3, 8, 9, 10, 13, 14, 16, 18, 19, 20, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 39], "input": [0, 2, 3, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 26, 29, 33, 36, 37, 39], "file": [0, 3, 13, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 33, 34, 35, 36, 37, 38, 41], "runsqlquerycontamin": 0, "return": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], "object": [0, 27], "all": [0, 7, 14, 16, 18, 19, 24, 26, 30, 31, 35, 36, 37, 38, 39], "within": [0, 2, 4, 38], "chosen": [0, 4, 38], "pixel": [0, 1, 2, 3, 8, 13, 22, 23, 24, 26, 30, 31, 37, 38], "radiu": [0, 2, 3, 13, 23, 30], "brighter": 0, "than": [0, 1, 30, 39], "g_": 0, "3": [0, 10, 13, 14, 19, 20, 23, 28, 29, 33, 35, 36, 38, 39], "rayleigh": 0, "formula": 0, "calcul": [0, 1, 3, 7, 11, 13, 18, 20, 23, 29, 31, 32, 34, 36, 38], "fraction": [0, 1], "flux_fraction_contamin": 0, "an": [0, 1, 2, 7, 13, 16, 33, 39, 41], "analyt": [0, 1], "biser": [0, 1], "millman": [0, 1], "1965": [0, 1], "equat": [0, 1], "3b": [0, 1], "10": [0, 1, 3, 4, 11, 13, 15, 18, 23, 29], "contribut": [0, 2, 18, 20, 23, 31, 34], "paramet": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39], "If": [0, 2, 9, 14, 18, 22, 24, 31, 36, 37, 38, 39, 41], "true": [0, 41], "inform": [0, 2, 5, 13, 39], "els": [0, 41], "option": [0, 2, 3, 4, 9, 11, 13, 14, 15, 18, 23, 27, 29, 30, 33, 35, 36, 38], "default": [0, 2, 3, 4, 9, 11, 13, 14, 15, 23, 27, 30, 31, 33, 35], "size": [0, 1, 13, 19, 22, 23, 25, 27, 28, 29, 30, 34, 35, 36, 38], "maximum": [0, 2, 4, 11, 15, 36, 38], "number": [0, 4, 11, 13, 18, 19, 24, 25, 29, 30, 31, 36, 37, 38, 39, 41], "store": [0, 18, 20, 21, 23, 24, 30, 31, 35, 36, 38], "extra": 0, "column": [0, 10, 14, 17, 18, 20, 22, 29, 33, 37], "t_cont": [0, 9], "contamin": [1, 2, 9, 13, 18, 20, 22, 23, 28, 29, 31, 34, 35, 36, 39], "ang_sep": 1, "": [1, 38], "d_th": 1, "5e": 1, "06": 1, "get": [1, 26, 37], "scatter": 1, "which": [1, 2, 3, 4, 5, 22, 25, 28, 30, 34, 37, 38, 39], "doubl": [1, 33], "converg": 1, "sum": 1, "infinit": 1, "limit": 1, "f_": 1, "rm": 1, "bg": 1, "e": [1, 26, 30, 33, 36, 38, 41], "sum_": 1, "n": [1, 23], "infti": 1, "bigg": 1, "frac": 1, "k": 1, "To": [1, 4], "solv": 1, "computation": 1, "summat": 1, "termin": [1, 18], "onc": [1, 26, 28], "differ": [1, 4, 9], "nth": 1, "iter": [1, 4, 5, 6, 7, 12, 13, 18, 19, 23, 24, 30, 31], "less": [1, 9, 39], "threshold": [1, 4, 9], "valu": [1, 4, 5, 6, 7, 10, 11, 12, 14, 18, 29, 35], "angular": [1, 2, 15], "distanc": [1, 2, 15], "arcsecond": [1, 15], "between": [1, 4, 18, 29], "centr": 1, "fwhm": 1, "psf": 1, "exprf": 1, "set": [1, 4, 7, 14, 23, 24, 31, 33, 35, 36, 37, 38], "65": 1, "2": [1, 3, 4, 10, 13, 14, 22, 25, 30, 33, 34, 36, 38, 39], "frac_flux_in_apertur": 1, "pix_radiu": 2, "perform": [2, 3, 5, 18, 20, 23, 31, 36, 38], "sql": [2, 33], "queri": [2, 14, 33], "identifi": [2, 7, 9, 14, 18, 22, 29, 33, 37], "analysi": [2, 4, 7, 11, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39], "specifi": [2, 31], "gener": [2, 15, 16, 17, 31, 38], "arcsec": 2, "search": [2, 39], "t_gaia": 2, "result": [2, 3, 4, 11, 13, 18, 21, 28, 30, 31], "lc_analysi": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 34, 39], "file_in": [3, 23, 27, 34], "skyrad": [3, 13, 23, 30, 41], "6": [3, 13, 23, 30, 39, 41], "8": [3, 13, 23, 30, 36, 38, 41], "xy_po": [3, 23], "photometri": [3, 10, 13, 22, 23, 25, 27, 30, 38], "imag": [3, 10, 13, 22, 26, 27, 38, 39, 41], "data": [3, 4, 5, 8, 9, 10, 11, 13, 17, 18, 19, 20, 22, 23, 28, 29, 31, 33, 35, 36, 38, 39, 41], "str": [3, 9, 15, 17, 18, 19, 20, 22, 23, 24, 28, 29, 31, 32, 34, 35, 36], "tupl": [3, 8, 13, 23, 27], "read": [3, 10, 14, 33, 37], "fit": [3, 5, 8, 22, 23, 24, 25, 26, 27, 30, 34, 38, 41], "A": [3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 21, 22, 23, 24, 25, 27, 28, 29, 31, 34, 35, 36], "form": [3, 5, 33, 34], "raw": 3, "lightcurv": [3, 4, 5, 10, 11, 13, 18, 19, 20, 23, 24, 30, 31, 36, 38, 39], "process": [3, 13, 26, 32, 38, 39], "subsequ": 3, "name": [3, 14, 16, 18, 19, 20, 22, 23, 24, 29, 31, 33, 34, 35, 36, 37, 38, 39], "contain": [3, 4, 5, 8, 9, 10, 11, 13, 14, 19, 20, 22, 26, 27, 28, 29, 30, 33, 34, 35, 36, 38], "defin": 3, "circular": 3, "area": [3, 30], "element": 3, "inner": [3, 13, 23, 30], "outer": [3, 13, 23, 30], "annulu": [3, 13], "background": [3, 10, 13, 23, 30, 39], "x": [3, 6, 8, 12, 22, 26, 27, 34, 37], "y": [3, 8, 22, 26, 27, 34, 37], "centroid": [3, 13, 22, 23], "full_phot_t": 3, "format": [3, 14, 15, 16, 17, 24, 31, 33, 36, 37, 38, 41], "f": [4, 5, 41], "mad_fac": 4, "time_fac": 4, "min_num_per_group": 4, "50": [4, 13], "remov": [4, 38], "point": [4, 10, 11, 32, 38], "like": [4, 9], "spuriou": [4, 10, 23, 38], "list": [4, 5, 6, 7, 12, 13, 15, 16, 18, 19, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 36, 39], "mani": [4, 39], "have": [4, 36, 38, 39, 40], "dai": [4, 11], "gap": [4, 39], "avoid": [4, 33], "systemat": [4, 39], "offset": 4, "ensur": [4, 26, 38], "effici": 4, "normal": [4, 5, 39], "split": 4, "string": [4, 28, 35, 36, 37, 38], "contigu": [4, 38], "must": [4, 10, 11, 14, 17, 33, 37], "been": 4, "observ": [4, 9, 39], "time": [4, 5, 10, 11, 14, 27, 32, 38, 39], "median": [4, 10, 27], "absolut": 4, "deviat": 4, "start": [4, 5, 32, 38, 39], "end": [4, 5, 32], "section": [4, 5], "mad": 4, "sector": [4, 8, 13, 18, 19, 23, 24, 25, 26, 28, 29, 30, 31, 35, 36, 37, 38, 39], "done": [4, 40], "becaus": [4, 14, 39], "often": 4, "after": [4, 13, 26, 36, 38], "larg": [4, 26, 33], "temperatur": 4, "sensor": 4, "chang": 4, "includ": [4, 10, 11], "compon": [4, 5], "just": [4, 36, 38], "signal": 4, "instrument": [4, 39], "nois": 4, "enough": 4, "datapoint": 4, "construct": [4, 6, 10], "periodogram": [4, 7, 9, 11, 13, 18, 19, 22, 23, 28, 31, 34, 35, 36, 38, 39], "coordin": [4, 5, 8, 10, 11, 14, 15, 24, 26, 31, 33, 36, 38, 39], "normalis": [4, 5, 10, 11, 13], "allow": [4, 39], "minimum": [4, 11], "start_index": 4, "indic": [4, 5, 7], "save": [4, 13, 18, 19, 24, 30, 31, 38], "end_index": 4, "d": [5, 34], "df": 5, "err": 5, "detrend": [5, 10, 11, 38, 39], "dict": [5, 9, 10, 11, 13, 28, 29, 34], "oper": 5, "clean_lc": [5, 10, 38], "routin": [5, 30], "so": [5, 26, 41], "separ": [5, 14, 15, 16, 17, 33], "can": [5, 15, 16, 26, 33, 36, 37, 38, 39, 40], "connect": [5, 39], "five": [5, 36, 38], "output": [5, 9, 15, 16, 17, 26, 27, 30, 34, 35, 36, 38, 41], "choic": [5, 14], "polynomi": [5, 10, 38], "linear": [5, 38], "quadrat": [5, 38], "depend": [5, 40], "best": [5, 12], "satisfi": 5, "aikak": [5, 38], "criterion": [5, 38], "error": [5, 10, 41], "dict_lc": 5, "dictionari": [5, 9, 11, 13, 28, 29, 34], "follow": [5, 10, 17, 18, 29, 37, 38, 39, 40, 41], "kei": [5, 10, 11, 41], "oflux": [5, 10], "nflux": [5, 10, 11], "clean": [5, 10, 11, 13, 23, 39], "enflux": [5, 10], "polyord": [5, 10], "order": [5, 10, 17, 20, 30, 37, 39], "a0": 6, "x_mean": 6, "sigma": 6, "simpl": 6, "gaussian": [6, 7], "amplitud": [6, 9, 12], "mean": 6, "uncertainti": 6, "distribut": 6, "power": [7, 13, 39], "algorithm": 7, "second": [7, 32], "highest": 7, "peak": [7, 11], "a_g": 7, "correspond": [7, 25], "around": [7, 20, 33], "a_o": 7, "other": [7, 37, 41], "part": [7, 38], "head": 8, "locat": [8, 26], "posit": [8, 13, 15, 22, 26, 27, 34], "ccd": [8, 13, 19, 23, 25, 26, 28, 29, 30, 35, 36, 37, 38], "camera": [8, 13, 19, 23, 25, 26, 28, 29, 30, 35, 36, 37, 38], "mode": [8, 26], "io": [8, 41], "celesti": [8, 26, 33], "header": [8, 17, 33], "wc": 8, "detail": [8, 20, 23, 28, 34, 41], "d_target": [9, 28, 34], "d_cont": 9, "frac_amp_cont": 9, "mai": [9, 33], "caus": [9, 39], "period": [9, 11, 13, 23, 34, 38, 39], "user": [9, 22, 36, 38, 39, 40, 41], "select": [9, 20], "measur": [9, 11, 19, 23, 35, 38, 39], "flag": [9, 38], "assess": [9, 34, 39], "actual": [9, 34], "star": [9, 23, 26, 28, 29, 30, 34], "factor": 9, "account": [9, 40], "two": [9, 37, 38], "high": [9, 39], "potenti": [9, 22], "either": [9, 14, 36, 37, 38], "b": [9, 25, 30, 33, 34], "c": [9, 33, 34, 39], "probabl": 9, "might": 9, "phot_tabl": [10, 27], "run": [10, 11, 14, 18, 23, 26, 30, 31, 33, 34, 36, 37, 39, 41], "task": 10, "produc": [10, 13, 17, 36, 38, 39], "4": [10, 13, 25, 30, 36, 38, 39], "detrend_lc": [10, 38], "aper_run": [10, 38], "py": [10, 20, 23, 26, 34, 38, 39], "mag": 10, "flux_corr": 10, "total": [10, 18, 20, 23, 31], "subtract": 10, "flux_err": 10, "cln": [10, 11], "time0": [10, 11], "rel": [10, 11, 39], "first": [10, 11], "orig": [10, 13], "p_min_thresh": [11, 13], "05": 11, "p_max_thresh": [11, 13], "100": 11, "samples_per_peak": 11, "lomb": [11, 13, 23, 38, 39], "scargl": [11, 13, 23, 38, 39], "sampl": 11, "ls_dict": [11, 13], "y0": 12, "phi": 12, "y_offset": 12, "phase": [12, 13, 38], "regular": 12, "sinusoid": 12, "midpoint": 12, "sine": 12, "curv": [12, 38], "angl": 12, "makeplot": [13, 39, 41], "im_plot": 13, "scc": [13, 19, 23, 25, 28, 29, 30, 35, 36, 41], "t_tabl": 13, "xy_ctr": [13, 27], "xy_contam": 13, "plot": [13, 18, 19, 23, 30, 31, 36, 38], "nddata": [13, 27], "cutout2d": [13, 27], "modul": [13, 23, 34, 39], "panel": 13, "displai": 13, "These": [13, 15, 38, 39], "cut": 13, "out": [13, 20, 30, 36, 37, 38], "sky": [13, 14, 26, 33, 36, 38, 39], "v": 13, "fold": [13, 38], "modifi": [13, 37], "befor": 13, "strongest": [13, 18, 20, 23, 31, 36], "shortest": 13, "longest": 13, "annuli": [13, 23, 30], "noth": [13, 18, 19, 30, 31, 37], "maket": [14, 15, 16, 17, 39], "gaia_tabl": [14, 15, 16, 17], "name_is_source_id": [14, 17, 33, 41], "correct": [14, 37, 38, 39], "comma": [14, 15, 16, 17, 33], "variabl": [14, 15, 16, 17, 33], "wai": [14, 17, 37, 38, 40, 41], "singl": [14, 33, 34, 38], "note": [14, 29, 33], "prefer": 14, "method": [14, 30, 39], "sinc": [14, 16], "unambigu": 14, "same": [14, 35, 36, 37], "icr": [14, 15], "galact": [14, 15, 33], "eclipt": [14, 15], "system": [14, 15], "slower": 14, "requir": [14, 18, 24, 25, 27, 30, 31, 34, 36, 39], "vizier": 14, "alreadi": 14, "made": [14, 18, 19, 23, 30, 31, 36, 38, 39], "ha": [14, 28, 34], "provid": [14, 15, 36, 38, 39], "equal": 14, "source_id": [14, 17, 18, 20, 26, 29, 33, 37, 41], "find": [14, 30], "common": [14, 35, 36], "coord_tabl": 15, "ang_max": 15, "type_coord": 15, "entri": [15, 17, 23, 28, 29], "need": [15, 17, 26, 36, 37, 38], "csv": [15, 16, 17, 38, 41], "consist": [15, 33], "right": [15, 18, 29, 36, 37, 38], "ascens": [15, 18, 29, 37], "declin": [15, 18, 29, 37], "degre": 15, "readi": [15, 16, 17, 33, 37, 38, 39], "further": [15, 16, 17, 18, 31, 33, 36, 38], "input_nam": 16, "charact": [16, 34], "pars": 16, "except": 16, "input_t": 17, "pre": [17, 33, 39], "quickest": 17, "import": [17, 38, 41], "properli": 17, "type": 17, "ra": [17, 18, 20, 29, 33, 37], "dec": [17, 18, 20, 29, 33, 37], "parallax": [17, 18, 20, 29, 33, 37], "gmag": [17, 18, 20, 29, 33, 37], "necessari": 17, "period_fil": [18, 19, 35, 38], "flux_con": [18, 20, 23, 31, 36, 38], "con_fil": [18, 20, 23, 31, 35, 38], "make_plot": [18, 19, 23, 30, 31, 36, 38], "choose_sec": [18, 24, 31, 38], "appar": [18, 29, 37], "log_tot_bg_star": [18, 29], "log": [18, 29, 41], "ratio": [18, 29], "log_max_bg_star": [18, 29], "largest": [18, 29], "n_contamin": [18, 29], "decid": [18, 19, 20, 23, 30, 31, 36], "here": [18, 31, 36, 38, 39, 40, 41], "download": [18, 23, 24, 31, 38, 39], "tesscut": [18, 24, 31, 38, 39], "avail": [18, 24, 31, 39], "attempt": [18, 24, 31], "final": [18, 23, 29, 30, 31, 36, 38], "program": [18, 26, 36, 39], "over": [19, 26, 39], "analys": [19, 28, 30, 33, 34, 36, 38, 39], "record": [19, 35], "configur": [19, 25, 30, 36, 38], "kwarg": 20, "collect": [20, 30], "take": [20, 32, 33, 39], "request": [20, 22, 23], "print": [20, 28, 29], "It": [20, 39], "also": 20, "descend": 20, "up": [20, 35], "most": 20, "where": [20, 23, 25, 30, 41], "keyword": [20, 23], "ad": 20, "log_tot_bg": 20, "log_max_bg": 20, "num_tot_bg": 20, "creat": [21, 26], "templat": 21, "final_t": [21, 23, 30, 31], "f_file": 22, "con_tabl": [22, 34], "cutout_s": [22, 23, 24, 31, 34], "length": [22, 23, 24, 31], "cutout": [22, 23, 24, 27, 31, 36, 38, 39], "cont_posit": 22, "np": [22, 41], "arrai": [22, 38, 41], "20": [23, 27, 31], "call": [23, 26, 31, 36, 38, 39], "rotat": [23, 39], "tess_functions2": 23, "2x2": 23, "coord": [24, 31], "target_nam": 24, "work": [24, 39, 41], "directori": 24, "skycoord": [24, 31], "manifest": 24, "sector_num": [25, 30, 36, 38], "cc": [25, 30], "fits_fil": [25, 27], "pair": 26, "row": [26, 31, 34], "full": [26, 30, 36, 38, 39, 41], "frame": [26, 30, 38, 39, 41], "simultan": [26, 30, 39], "join": [26, 41], "onli": [26, 36, 38], "calibr": [26, 30, 38, 39, 41], "when": [26, 33], "all_cc": 26, "get_gaia_data": 26, "xy_tabl": 26, "im_siz": 27, "make": [27, 30, 36, 37, 38], "2d": 27, "stack": [27, 39], "labels_cont": [28, 34], "line": [28, 29, 33, 36, 39, 40], "see": [28, 41], "getgaiadata": 28, "tess_funct": 28, "run_l": [28, 34, 38], "label": 28, "ani": [28, 37, 39, 40], "dr": [28, 29], "fail": 29, "contaminant": 29, "three": 29, "automat": [29, 33, 39, 40], "fill": [29, 39], "999": 29, "land": 30, "carri": [30, 36, 38], "chronolog": 30, "much": [30, 39], "faster": [30, 39], "do": [30, 39, 41], "vectoris": [30, 39], "one": [31, 36, 38, 39, 40], "all_sourc": 31, "finish": [32, 38, 39], "taken": 32, "hour": 32, "minut": [32, 39], "datetim": 32, "time_taken": 32, "complet": [32, 39], "t_filenam": [33, 36, 37, 38], "convert": [33, 38], "decim": 33, "prepar": [33, 37], "without": [33, 36, 38], "quickli": 33, "directli": [33, 40], "command": [33, 36, 39, 40, 41], "quotat": 33, "mark": 33, "python": [33, 39, 40, 41], "run_tess_cutout": [33, 38], "ab": 33, "doradu": 33, "toggl": [33, 36, 38], "long": 33, "veri": 33, "xy_arr": 34, "find_xy_cont": 34, "could": 34, "come": 34, "file_ref": [35, 36, 38], "give": [35, 36, 38], "convent": [35, 36], "retriev": 36, "appli": [36, 38, 39], "ye": [36, 38], "determin": [36, 38], "should": [36, 38, 39, 40], "prompt": [36, 38], "enter": [36, 38], "argument": [36, 38], "ask": [36, 38], "specif": [36, 38], "whole": [36, 37, 38, 39], "digit": [36, 38], "space": [36, 38], "entir": [36, 38], "wherea": [36, 38], "814": [36, 38], "express": [36, 38], "refer": [36, 37, 38], "initi": [36, 38], "receiv": [36, 38], "simpli": [36, 37, 38, 41], "suppli": [36, 38, 39], "otherwis": [36, 38], "howev": [36, 38], "thei": [36, 38, 39], "wrong": [36, 38], "warn": [36, 38], "messag": [36, 38], "exit": [36, 38], "check": 37, "coupl": 37, "adjust": 37, "pass": 37, "straight": [37, 38], "exactli": 37, "xpo": 37, "ypo": 37, "preced": 37, "entitl": 37, "In": [37, 39, 41], "case": 37, "table_in": 37, "There": 38, "design": [38, 39], "altern": [38, 40], "setup_input_paramet": [38, 41], "repositori": [38, 39], "exampl": 38, "all_sources_cutout": [38, 39], "everi": 38, "run_tess_sector": 38, "all_sources_sector": [38, 39], "local": 38, "machin": 38, "whether": 38, "group": 38, "cutout_target": 38, "alexand": [38, 39], "bink": [38, 39], "moritz": [38, 39], "guenther": [38, 39], "januari": 38, "2023": [38, 39], "licenc": [38, 39], "mit": [38, 39], "upon": 38, "tess_cutout": 38, "tess_large_sector": 38, "numpi": [38, 39, 40, 41], "make_lc": 38, "being": 38, "1st": 38, "2nd": 38, "infer": 38, "piec": 38, "togeth": 38, "conduct": [38, 39], "plu": 38, "sever": 38, "qualiti": [38, 39], "light": 38, "tabular": 38, "fix": 38, "constant": 38, "typic": 38, "width": 38, "half": 38, "respons": 38, "zeropoint": 38, "stop": 39, "shop": 39, "stellar": 39, "transit": 39, "exoplanet": 39, "survei": 39, "satellit": 39, "whilst": 39, "softwar": 39, "tool": 39, "mostli": 39, "variou": 39, "step": 39, "reduct": 39, "our": 39, "knowledg": 39, "autom": 39, "obtain": 39, "littl": 39, "capabl": 39, "robust": 39, "figur": 39, "public": 39, "sit": 39, "back": 39, "let": 39, "hard": 39, "photometr": 39, "seri": 39, "scan": 39, "level": [39, 41], "nearbi": 39, "poor": 39, "effect": 39, "metric": 39, "reliabl": 39, "brasseur": 39, "et": 39, "al": 39, "2019": 39, "servic": 39, "acquir": 39, "postag": 39, "stamp": 39, "sequenc": 39, "center": 39, "tesscutclass": 39, "abov": 39, "recommend": 39, "who": 39, "fast": 39, "extract": 39, "manag": 39, "With": 39, "requisit": 39, "uninterrupt": 39, "internet": 39, "approxim": 39, "want": [39, 41], "few": 39, "interest": 39, "larger": 39, "bulk": 39, "mast": 39, "archiv": 39, "multipl": 39, "due": 39, "possibl": 39, "style": 39, "author": 39, "test": 39, "million": 39, "took": 39, "week": 39, "problem": 39, "pleas": 39, "contact": 39, "alex": 39, "lead": 39, "abink": 39, "edu": 39, "packag": 39, "research": 39, "we": [39, 41], "would": [39, 41], "appreci": 39, "acknowledg": 39, "wa": 39, "instal": 39, "direct": 39, "pip": 39, "clone": 39, "github": 39, "how": 39, "insid": 39, "api": 39, "document": 39, "fixedconst": 39, "index": 39, "page": 39, "easiest": 40, "last": 40, "releas": 40, "version": 40, "your": 40, "environ": 40, "you": [40, 41], "don": 40, "them": 40, "yet": 40, "particular": 40, "matplotlib": 40, "abl": 40, "interact": 40, "onlin": 40, "interfac": 40, "link": 40, "linux": 40, "git": 40, "http": 40, "com": 40, "alexbink": 40, "o": 41, "ascii": 41, "basicconfig": 41, "filenam": 41, "fo": 41, "rmore": 41, "docuemnt": 41, "fluxcon": 41, "fileref": 41, "tfile": 41, "confil": 41, "periodfil": 41, "setup_filenam": 41, "t_large_sec_check": 41, "test_table_large_sector": 41, "ttarget": 41, "gaia_data": 41, "read_data": 41, "xy_pixel_data": 41, "get_tess_pixel_xi": 41, "write": 41, "overwrit": 41, "particualr": 41, "turn": 41, "stuff": 41, "ttaget": 41, "ther": 41, "stare": 41, "fielanem": 41, "collect_contamination_data": 41, "all_sector": 41, "_": 17}, "objects": {"tessilator": [[38, 0, 0, "-", "contaminants"], [38, 0, 0, "-", "fixedconstants"], [38, 0, 0, "-", "lc_analysis"], [38, 0, 0, "-", "makeplots"], [38, 0, 0, "-", "maketable"], [38, 0, 0, "-", "tessilator"]], "tessilator.contaminants": [[0, 1, 1, "", "contamination"], [1, 1, 1, "", "flux_fraction_contaminant"], [2, 1, 1, "", "run_sql_query_contaminants"]], "tessilator.lc_analysis": [[3, 1, 1, "", "aper_run"], [4, 1, 1, "", "clean_lc"], [5, 1, 1, "", "detrend_lc"], [6, 1, 1, "", "gauss_fit"], [7, 1, 1, "", "get_second_peak"], [8, 1, 1, "", "get_xy_pos"], [9, 1, 1, "", "is_period_cont"], [10, 1, 1, "", "make_lc"], [11, 1, 1, "", "run_ls"], [12, 1, 1, "", "sin_fit"]], "tessilator.makeplots": [[13, 1, 1, "", "make_plot"]], "tessilator.maketable": [[14, 1, 1, "", "get_gaia_data"], [15, 1, 1, "", "table_from_coords"], [16, 1, 1, "", "table_from_simbad"], [17, 1, 1, "", "table_from_table"]], "tessilator.tessilator": [[18, 1, 1, "", "all_sources_cutout"], [19, 1, 1, "", "all_sources_sector"], [20, 1, 1, "", "collect_contamination_data"], [21, 1, 1, "", "create_table_template"], [22, 1, 1, "", "find_xy_cont"], [23, 1, 1, "", "full_run_lc"], [24, 1, 1, "", "get_cutouts"], [25, 1, 1, "", "get_fits"], [26, 1, 1, "", "get_tess_pixel_xy"], [27, 1, 1, "", "make_2d_cutout"], [28, 1, 1, "", "make_datarow"], [29, 1, 1, "", "make_failrow"], [30, 1, 1, "", "one_cc"], [31, 1, 1, "", "one_source_cutout"], [32, 1, 1, "", "print_time_taken"], [33, 1, 1, "", "read_data"], [34, 1, 1, "", "run_test_for_contaminant"], [35, 1, 1, "", "setup_filenames"], [36, 1, 1, "", "setup_input_parameters"], [37, 1, 1, "", "test_table_large_sectors"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"contamin": [0, 38], "flux_fraction_contamin": 1, "run_sql_query_contamin": 2, "aper_run": 3, "clean_lc": 4, "detrend_lc": 5, "gauss_fit": 6, "get_second_peak": 7, "get_xy_po": 8, "is_period_cont": 9, "make_lc": 10, "run_l": 11, "sin_fit": 12, "make_plot": 13, "get_gaia_data": 14, "table_from_coord": 15, "table_from_simbad": 16, "table_from_t": 17, "all_sources_cutout": 18, "all_sources_sector": 19, "collect_contamination_data": 20, "create_table_templ": 21, "find_xy_cont": 22, "full_run_lc": 23, "get_cutout": 24, "get_fit": 25, "get_tess_pixel_xi": 26, "make_2d_cutout": 27, "make_datarow": 28, "make_failrow": 29, "one_cc": 30, "one_source_cutout": 31, "print_time_taken": 32, "read_data": 33, "run_test_for_contamin": 34, "setup_filenam": 35, "setup_input_paramet": 36, "test_table_large_sector": 37, "how": 38, "us": [38, 39, 40, 41], "requir": 38, "input": 38, "paramet": 38, "from": 38, "command": 38, "line": 38, "automat": 38, "run": 38, "program": 38, "insid": 38, "python": 38, "api": 38, "document": 38, "tessil": [38, 39, 40, 41], "modul": 38, "function": 38, "lc_analysi": 38, "makeplot": 38, "maket": 38, "fixedconst": 38, "welcom": 39, "TO": 39, "THE": 39, "wai": 39, "note": 39, "content": 39, "indic": 39, "tabl": 39, "instal": 40, "softwar": 40, "direct": 40, "pip": 40, "clone": 40, "github": 40, "repositori": 40, "per": 41, "sector": 41}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"WELCOME TO THE TESSILATOR": [[39, "welcome-to-the-tessilator"]], "Ways to use the tessilator": [[39, "ways-to-use-the-tessilator"]], "Notes on using the tessilator": [[39, "notes-on-using-the-tessilator"]], "Contents:": [[39, null]], "Indices and tables": [[39, "indices-and-tables"]], "Installing tessilator software": [[40, "installing-tessilator-software"]], "Direct installation using pip": [[40, "direct-installation-using-pip"]], "Cloning the GitHub repository": [[40, "cloning-the-github-repository"]], "Using the TESSilator per sector": [[41, "using-the-tessilator-per-sector"]], "contamination": [[0, "contamination"]], "flux_fraction_contaminant": [[1, "flux-fraction-contaminant"]], "run_sql_query_contaminants": [[2, "run-sql-query-contaminants"]], "aper_run": [[3, "aper-run"]], "clean_lc": [[4, "clean-lc"]], "detrend_lc": [[5, "detrend-lc"]], "gauss_fit": [[6, "gauss-fit"]], "get_second_peak": [[7, "get-second-peak"]], "get_xy_pos": [[8, "get-xy-pos"]], "is_period_cont": [[9, "is-period-cont"]], "make_lc": [[10, "make-lc"]], "run_ls": [[11, "run-ls"]], "sin_fit": [[12, "sin-fit"]], "make_plot": [[13, "make-plot"]], "get_gaia_data": [[14, "get-gaia-data"]], "table_from_coords": [[15, "table-from-coords"]], "table_from_simbad": [[16, "table-from-simbad"]], "table_from_table": [[17, "table-from-table"]], "all_sources_cutout": [[18, "all-sources-cutout"]], "all_sources_sector": [[19, "all-sources-sector"]], "collect_contamination_data": [[20, "collect-contamination-data"]], "create_table_template": [[21, "create-table-template"]], "find_xy_cont": [[22, "find-xy-cont"]], "full_run_lc": [[23, "full-run-lc"]], "get_cutouts": [[24, "get-cutouts"]], "get_fits": [[25, "get-fits"]], "get_tess_pixel_xy": [[26, "get-tess-pixel-xy"]], "make_2d_cutout": [[27, "make-2d-cutout"]], "make_datarow": [[28, "make-datarow"]], "make_failrow": [[29, "make-failrow"]], "one_cc": [[30, "one-cc"]], "one_source_cutout": [[31, "one-source-cutout"]], "print_time_taken": [[32, "print-time-taken"]], "read_data": [[33, "read-data"]], "run_test_for_contaminant": [[34, "run-test-for-contaminant"]], "setup_filenames": [[35, "setup-filenames"]], "setup_input_parameters": [[36, "setup-input-parameters"]], "test_table_large_sectors": [[37, "test-table-large-sectors"]], "How to use": [[38, "how-to-use"]], "Required input parameters": [[38, "required-input-parameters"]], "From the command line": [[38, "from-the-command-line"]], "Automatically run the program": [[38, "automatically-run-the-program"]], "Inside Python": [[38, "inside-python"]], "API documentation": [[38, "api-documentation"]], "tessilator.tessilator Module": [[38, "module-tessilator.tessilator"]], "Functions": [[38, "functions"], [38, "id1"], [38, "id2"], [38, "id3"], [38, "id4"]], "tessilator.lc_analysis Module": [[38, "module-tessilator.lc_analysis"]], "tessilator.contaminants Module": [[38, "module-tessilator.contaminants"]], "tessilator.makeplots Module": [[38, "module-tessilator.makeplots"]], "tessilator.maketable Module": [[38, "module-tessilator.maketable"]], "tessilator.fixedconstants Module": [[38, "module-tessilator.fixedconstants"]]}, "indexentries": {"contamination() (in module tessilator.contaminants)": [[0, "tessilator.contaminants.contamination"]], "flux_fraction_contaminant() (in module tessilator.contaminants)": [[1, "tessilator.contaminants.flux_fraction_contaminant"]], "run_sql_query_contaminants() (in module tessilator.contaminants)": [[2, "tessilator.contaminants.run_sql_query_contaminants"]], "aper_run() (in module tessilator.lc_analysis)": [[3, "tessilator.lc_analysis.aper_run"]], "clean_lc() (in module tessilator.lc_analysis)": [[4, "tessilator.lc_analysis.clean_lc"]], "detrend_lc() (in module tessilator.lc_analysis)": [[5, "tessilator.lc_analysis.detrend_lc"]], "gauss_fit() (in module tessilator.lc_analysis)": [[6, "tessilator.lc_analysis.gauss_fit"]], "get_second_peak() (in module tessilator.lc_analysis)": [[7, "tessilator.lc_analysis.get_second_peak"]], "get_xy_pos() (in module tessilator.lc_analysis)": [[8, "tessilator.lc_analysis.get_xy_pos"]], "is_period_cont() (in module tessilator.lc_analysis)": [[9, "tessilator.lc_analysis.is_period_cont"]], "make_lc() (in module tessilator.lc_analysis)": [[10, "tessilator.lc_analysis.make_lc"]], "run_ls() (in module tessilator.lc_analysis)": [[11, "tessilator.lc_analysis.run_ls"]], "sin_fit() (in module tessilator.lc_analysis)": [[12, "tessilator.lc_analysis.sin_fit"]], "make_plot() (in module tessilator.makeplots)": [[13, "tessilator.makeplots.make_plot"]], "get_gaia_data() (in module tessilator.maketable)": [[14, "tessilator.maketable.get_gaia_data"]], "table_from_coords() (in module tessilator.maketable)": [[15, "tessilator.maketable.table_from_coords"]], "table_from_simbad() (in module tessilator.maketable)": [[16, "tessilator.maketable.table_from_simbad"]], "table_from_table() (in module tessilator.maketable)": [[17, "tessilator.maketable.table_from_table"]], "all_sources_cutout() (in module tessilator.tessilator)": [[18, "tessilator.tessilator.all_sources_cutout"]], "all_sources_sector() (in module tessilator.tessilator)": [[19, "tessilator.tessilator.all_sources_sector"]], "collect_contamination_data() (in module tessilator.tessilator)": [[20, "tessilator.tessilator.collect_contamination_data"]], "create_table_template() (in module tessilator.tessilator)": [[21, "tessilator.tessilator.create_table_template"]], "find_xy_cont() (in module tessilator.tessilator)": [[22, "tessilator.tessilator.find_xy_cont"]], "full_run_lc() (in module tessilator.tessilator)": [[23, "tessilator.tessilator.full_run_lc"]], "get_cutouts() (in module tessilator.tessilator)": [[24, "tessilator.tessilator.get_cutouts"]], "get_fits() (in module tessilator.tessilator)": [[25, "tessilator.tessilator.get_fits"]], "get_tess_pixel_xy() (in module tessilator.tessilator)": [[26, "tessilator.tessilator.get_tess_pixel_xy"]], "make_2d_cutout() (in module tessilator.tessilator)": [[27, "tessilator.tessilator.make_2d_cutout"]], "make_datarow() (in module tessilator.tessilator)": [[28, "tessilator.tessilator.make_datarow"]], "make_failrow() (in module tessilator.tessilator)": [[29, "tessilator.tessilator.make_failrow"]], "one_cc() (in module tessilator.tessilator)": [[30, "tessilator.tessilator.one_cc"]], "one_source_cutout() (in module tessilator.tessilator)": [[31, "tessilator.tessilator.one_source_cutout"]], "print_time_taken() (in module tessilator.tessilator)": [[32, "tessilator.tessilator.print_time_taken"]], "read_data() (in module tessilator.tessilator)": [[33, "tessilator.tessilator.read_data"]], "run_test_for_contaminant() (in module tessilator.tessilator)": [[34, "tessilator.tessilator.run_test_for_contaminant"]], "setup_filenames() (in module tessilator.tessilator)": [[35, "tessilator.tessilator.setup_filenames"]], "setup_input_parameters() (in module tessilator.tessilator)": [[36, "tessilator.tessilator.setup_input_parameters"]], "test_table_large_sectors() (in module tessilator.tessilator)": [[37, "tessilator.tessilator.test_table_large_sectors"]], "module": [[38, "module-tessilator.contaminants"], [38, "module-tessilator.fixedconstants"], [38, "module-tessilator.lc_analysis"], [38, "module-tessilator.makeplots"], [38, "module-tessilator.maketable"], [38, "module-tessilator.tessilator"]], "tessilator.contaminants": [[38, "module-tessilator.contaminants"]], "tessilator.fixedconstants": [[38, "module-tessilator.fixedconstants"]], "tessilator.lc_analysis": [[38, "module-tessilator.lc_analysis"]], "tessilator.makeplots": [[38, "module-tessilator.makeplots"]], "tessilator.maketable": [[38, "module-tessilator.maketable"]], "tessilator.tessilator": [[38, "module-tessilator.tessilator"]]}})