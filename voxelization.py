"""The File Descibe voxelization procedure for Detector A.

Parameters of the detector A

Distances of vertical segment
a_t = 252.863 mm.
b_t = 416.545 mm.
a_b = -722.903 mm.
b_b = -1077.999 mm.

r_vs = 530 mm.
r_d = 1000 mm.
omega_vs
delta_gamma_vs = 1.05 deg.

N_c = 32
N_a = 128
N_vs = 60
"""

import numpy
from dataclasses import dataclass


def _calc_a_prime_by_a_b_x(a, b, x):
    a_prime = a + (b - a) * x
    return a_prime


@dataclass(frozen=True)
class DetectorA:
    N_vs: int = 60
    N_a: int = 128
    N_c: int = 32

    r_d: float = 1.000
    r_vs: float = 0.530
    a_t: float = 0.252863
    b_t: float = 0.416545
    a_b: float = -0.722903
    b_b: float = -1.077999

    omega_vs: float = numpy.radians(-9.101)
    delta_gamma_vs: float = numpy.radians(1.05)
    gamma_d: float = numpy.radians(0.00)
    n_id_0: int = 1

    def _calc_id_by_n_vsac(
        self,
        n_vs,
        n_a,
        n_c,
    ):
        """Calculate ID of detector base on position of vertical segment, anode and cathode."""

        n_id = n_vs + (n_a + n_c * self.N_a) * 2 * self.N_vs + self.n_id_0
        return n_id

    def _calc_n_vsac_by_id(
        self,
        n_id: int or numpy.ndarray,
    ):
        """
        Calculate position of vertical segment, anode and cathode by ID.

        Parameters
        ----------
        n_id : int or numpy.ndarray
            ID of detector.
        N_vs : int, optional
            Number of vertical segmetns. The default is 60.
        N_a : int, optional
            Number of anodes (vertical). The default is 128.
        N_c : int, optional
            Number of cathods (depth). The default is 32.

        Returns
        -------
        n_vs : int or numpy.ndarray
            from 0 to 2 N_vs not including
        n_a : int or numpy.ndarray
            from 0 to N_a not including
        n_c : int or numpy.ndarray
            from 0 to N_c not including

        Raises
        ------
        UserWarning
            If ID number is more than N_vs * N_a * N_c.
        """
        if numpy.any(n_id < self.n_id_0):
            raise UserWarning(f"ID shoul be more or equal {self.n_id_0}")
        n_vs = numpy.mod(n_id - self.n_id_0, 2 * self.N_vs)
        hh = numpy.floor_divide(n_id - self.n_id_0, 2 * self.N_vs)
        n_a = numpy.mod(hh, self.N_a)
        hh = numpy.floor_divide(hh, self.N_a)
        n_c = numpy.mod(hh, self.N_c)
        hh = numpy.floor_divide(hh, self.N_c)
        if numpy.any(hh > 0):
            raise UserWarning("Incorrect n_id")
        return n_vs, n_a, n_c

    def _calc_x_c_by_n_c(self, n_c):
        return (n_c + 0.5) / self.N_c

    def _calc_y_a_by_n_a(self, n_a):
        return (n_a + 0.5) / self.N_a

    def _calc_xy_vs_by_n_ac(
        self,
        n_c,
        n_a,
    ):
        """Calculate postion based of n_c, n_a is the frame of vertical segment."""
        x_c = self._calc_x_c_by_n_c(n_c)
        y_a = self._calc_y_a_by_n_a(n_a)
        x = self.r_vs * x_c
        a_prime_t = _calc_a_prime_by_a_b_x(self.a_t, self.b_t, x_c)
        a_prime_b = _calc_a_prime_by_a_b_x(self.a_b, self.b_b, x_c)
        y = a_prime_t + (a_prime_b - a_prime_t) * y_a
        return x, y

    def _calc_xyz_by_n_vsac(
        self,
        n_vs,
        n_a,
        n_c,
    ):
        z_vs = numpy.floor_divide(n_vs, 2) / self.N_vs
        gamma = self.gamma_d + z_vs * self.delta_gamma_vs * self.N_vs

        x, y = self._calc_xy_vs_by_n_ac(n_c, n_a)
        x_e = self.r_d * numpy.sin(gamma) + x * numpy.sin(
            gamma + self.omega_vs
        )
        z_e = self.r_d * numpy.cos(gamma) + x * numpy.cos(
            gamma + self.omega_vs
        )
        y_e = y
        np_xyz = numpy.stack((x_e, y_e, z_e), axis=0)

        return np_xyz

    def calc_xyz_by_id(
        self,
        n_id,
    ):
        n_vs, n_a, n_c = self._calc_n_vsac_by_id(n_id)

        np_xyz = self._calc_xyz_by_n_vsac(
            n_vs,
            n_a,
            n_c,
        )

        return np_xyz

    def _calc_x_c_by_xz_e(
        self,
        x_e,
        z_e,
    ):
        r_sq = numpy.square(x_e) + numpy.square(z_e)
        x_c = (
            numpy.sqrt(
                r_sq - numpy.square(self.r_d * numpy.sin(self.omega_vs))
            )
            - self.r_d * numpy.cos(self.omega_vs)
        ) / self.r_vs
        return x_c

    def _calc_n_vsac_by_xyz(
        self,
        np_xyz,
    ):
        x_e, y_e, z_e = np_xyz[0], np_xyz[1], np_xyz[2]
        x_c = self._calc_x_c_by_xz_e(
            x_e,
            z_e,
        )
        n_c = numpy.floor(self.N_c * x_c).astype(int)

        a_prime_t = _calc_a_prime_by_a_b_x(self.a_t, self.b_t, x_c)
        a_prime_b = _calc_a_prime_by_a_b_x(self.a_b, self.b_b, x_c)

        y_a = (y_e - a_prime_t) / (a_prime_b - a_prime_t)
        n_a = numpy.floor(self.N_a * y_a).astype(int)

        gamma = numpy.atan2(x_e, z_e)
        r_sq = numpy.square(x_e) + numpy.square(z_e)
        delta_gamma_prime = numpy.asin(
            numpy.sin(self.omega_vs) * x_c * self.r_vs / numpy.sqrt(r_sq)
        )

        z_vs = (
            gamma
            - self.gamma_d
            - delta_gamma_prime
            + self.delta_gamma_vs * 0.5
        ) / (self.N_vs * self.delta_gamma_vs)
        n_vs = numpy.floor(2 * self.N_vs * z_vs).astype(int)
        return (
            n_vs,
            n_a,
            n_c,
        )

    def calc_id_by_xyz(self, np_xyz):
        (
            n_vs,
            n_a,
            n_c,
        ) = self._calc_n_vsac_by_xyz(
            np_xyz,
        )
        np_id = self._calc_id_by_n_vsac(
            n_vs,
            n_a,
            n_c,
        )

        return np_id

def calc_local_voxel_position_by_id_of_detector_a(np_id, omega_vs):
    """
    Calculate local position of voxel (center of the phase) by ID number
    """
    det = DetectorA(omega_vs=omega_vs, gamma_d=0)
    np_xyz = det.calc_xyz_by_id(np_id)
    return np_xyz
    

def voxelization_of_mcstas_events_for_detector_a(
    np_event: numpy.ndarray, omega_vs
):
    """
    abs_logger_layers_dat_list_p_x_y_z_vx_vy_vz_t_id
    """
    # Here Gamma_d is zero as xyz are defined in coordinate system of arm
    det = DetectorA(omega_vs=omega_vs, gamma_d=0)
    np_xyz = np_event[:, 1:4]

    n_vs, n_a, n_c = det._calc_n_vsac_by_xyz(np_xyz.transpose())
    flag_1 = numpy.logical_and(n_vs >= 0, n_vs < 2*det.N_vs)
    flag_2 = numpy.logical_and(n_a >= 0, n_a < det.N_a)
    flag_3 = numpy.logical_and(n_c >= 0, n_c < det.N_c)
    flag = numpy.logical_and(numpy.logical_and(flag_1, flag_2), flag_3)
    np_events_reduced = np_event[flag, :]
    np_id = det._calc_id_by_n_vsac(n_vs[flag], n_a[flag], n_c[flag])
    np_xyz_voxels = det.calc_xyz_by_id(np_id).transpose()
    return np_events_reduced, np_id, np_xyz_voxels, n_vs[flag], n_a[flag], n_c[flag]


def voxelization_of_mcstas_events_for_detector_b(
    np_event: numpy.ndarray, omega_vs
):
    """
    abs_logger_layers_dat_list_p_x_y_z_vx_vy_vz_t_id
    """
    # Here Gamma_d is zero as xyz are defined in coordinate system of arm
    det = DetectorA( 
        N_vs = 16*8,
        N_a = 16,
        N_c = 32,
        r_d = 1.000,
        r_vs = 0.530,
        delta_gamma_vs = numpy.radians(-0.94),
                    
        a_t = 0.051016,
        b_t = 0.077887,
        a_b = -0.051016,
        b_b = -0.077887,
        omega_vs=omega_vs, 
        gamma_d = numpy.radians(0.00),
        n_id_0 = 491521, 
    )
    np_xyz = np_event[:, 1:4]

    n_vs, n_a, n_c = det._calc_n_vsac_by_xyz(np_xyz.transpose())
    flag_1 = numpy.logical_and(n_vs >= 0, n_vs < 2*det.N_vs)
    flag_2 = numpy.logical_and(n_a >= 0, n_a < det.N_a)
    flag_3 = numpy.logical_and(n_c >= 0, n_c < det.N_c)
    flag = numpy.logical_and(numpy.logical_and(flag_1, flag_2), flag_3)
    np_events_reduced = np_event[flag, :]
    np_id = det._calc_id_by_n_vsac(n_vs[flag], n_a[flag], n_c[flag])
    np_xyz_voxels = det.calc_xyz_by_id(np_id).transpose()
    return np_events_reduced, np_id, np_xyz_voxels, n_vs[flag], n_a[flag], n_c[flag]


def test_calc_id():
    N_ID_max = 128 * 32 * 120
    omega_vs = numpy.radians(-10)
    for val_id in range(1, N_ID_max):
        print(f"{100*val_id/(N_ID_max-1):8.2f}%", end="\r")
        det = DetectorA(omega_vs=omega_vs)
        np_xyz = det.calc_xyz_by_id(val_id)
        np_id = det.calc_id_by_xyz(np_xyz)
        if numpy.abs(val_id - np_id) > 1:
            print("ERROR       ")
            print(val_id, np_id)
            n_vs, n_a, n_c = det._calc_n_vsac_by_id(val_id)
            print("Input: ", n_vs, n_a, n_c)
            n_vs, n_a, n_c = det._calc_n_vsac_by_id(np_id)
            print("Output: ", n_vs, n_a, n_c)
            assert False
    assert True


# test_calc_id()
