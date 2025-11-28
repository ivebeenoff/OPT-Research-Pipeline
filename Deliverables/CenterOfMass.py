# Module to compute 3D center of mass positions and velocities for galaxy simulation snapshots

import numpy as np
import astropy.units as u
from ReadFile import Read


class CenterOfMass:

    def __init__(self, filename, ptype):
        self.time, self.total, self.data = Read(filename)
        self.index = np.where(self.data['type'] == ptype)
        self.m = self.data['m'][self.index]
        self.x = self.data['x'][self.index]
        self.y = self.data['y'][self.index]
        self.z = self.data['z'][self.index]
        self.vx = self.data['vx'][self.index]
        self.vy = self.data['vy'][self.index]
        self.vz = self.data['vz'][self.index]

    def COMdefine(self, a, b, c, m):
        a_com = np.sum(a * m) / np.sum(m)
        b_com = np.sum(b * m) / np.sum(m)
        c_com = np.sum(c * m) / np.sum(m)
        return a_com, b_com, c_com

    def COM_P(self, delta=0.1):
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        r_COM = np.sqrt(x_COM**2 + y_COM**2 + z_COM**2)

        x_new = self.x - x_COM
        y_new = self.y - y_COM
        z_new = self.z - z_COM
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        r_max = max(r_new) / 2.0
        change = 1000.0

        while change > delta:
            index2 = np.where(r_new < r_max)
            x2, y2, z2 = self.x[index2], self.y[index2], self.z[index2]
            m2 = self.m[index2]

            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2, y2, z2, m2)
            r_COM2 = np.sqrt(x_COM2**2 + y_COM2**2 + z_COM2**2)

            change = np.abs(r_COM - r_COM2)
            r_max /= 2.0

            x_new = self.x - x_COM2
            y_new = self.y - y_COM2
            z_new = self.z - z_COM2
            r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

            x_COM, y_COM, z_COM = x_COM2, y_COM2, z_COM2
            r_COM = r_COM2

        p_COM = np.array([x_COM, y_COM, z_COM]) * u.kpc
        return np.round(p_COM, 2)

    def COM_V(self, x_COM, y_COM, z_COM):
        rv_max = 15.0 * u.kpc

        xV = self.x * u.kpc - x_COM
        yV = self.y * u.kpc - y_COM
        zV = self.z * u.kpc - z_COM
        rV = np.sqrt(xV**2 + yV**2 + zV**2)

        indexV = np.where(rV < rv_max)
        vx_new, vy_new, vz_new = self.vx[indexV], self.vy[indexV], self.vz[indexV]
        m_new = self.m[indexV]

        vx_COM, vy_COM, vz_COM = self.COMdefine(vx_new, vy_new, vz_new, m_new)
        v_COM = np.array([vx_COM, vy_COM, vz_COM]) * u.km / u.s
        return np.round(v_COM, 2)
