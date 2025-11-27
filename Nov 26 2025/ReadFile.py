# A Python module containing a data-loader function to parse snapshot files and extract time, particle count, and structured arrays for further analysis.

import numpy as np
import astropy.units as u

def Read(filename):
    with open(filename, 'r') as file:
        line1 = file.readline()
        _, value = line1.split()
        time = float(value) * u.Myr

        line2 = file.readline()
        _, value = line2.split()
        count = float(value)

    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)

    return time, count, data
