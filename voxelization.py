"""The File Descibe voxelization procedure for Detector A.

Parameters of the detector A

Distances of vertical segmeh
a_t = 252.863 mm.
b_t = 416.524 mm.
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


def _calc_id_by_n_vsac(
    n_vs,
    n_a,
    n_c,
    N_vs: int = 60,
    N_a: int = 128,
    N_c: int = 32,
    n_id_0: int = 1,
):
    """Calculate ID of detector base on position of vertical segment, anode and cathode."""
    n_id = n_vs + (n_a + n_c * N_a) * 2 * N_vs + n_id_0
    return n_id


def _calc_n_vsac_by_id(
    n_id: int or numpy.ndarray,
    N_vs: int = 60,
    N_a: int = 128,
    N_c: int = 32,
    n_id_0: int = 1,
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
    if n_id < n_id_0:
        raise UserWarning(f"ID shoul be more or equal {n_id_0}")
    n_vs = numpy.mod(n_id - n_id_0, 2 * N_vs)
    hh = numpy.floor_divide(n_id - n_id_0, 2 * N_vs)
    n_a = numpy.mod(hh, N_a)
    hh = numpy.floor_divide(hh, N_a)
    n_c = numpy.mod(hh, N_c)
    hh = numpy.floor_divide(hh, N_c)
    if hh > 0:
        raise UserWarning("Incorrect n_id")
    return n_vs, n_a, n_c


def _calc_x_c_by_n_c(n_c, N_c: int = 32):
    return (n_c + 0.5) / N_c


def _calc_y_a_by_n_a(n_a, N_a: int = 128):
    return (n_a + 0.5) / N_a


def _calc_a_prime_by_a_b_x(a, b, x):
    a_prime = a + (b - a) * x
    return a_prime


def _calc_xy_vs_by_n_ac(
    n_c,
    n_a,
    N_c: int = 32,
    N_a: int = 128,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
):
    """Calculate postion based of n_c, n_a is the frame of vertical segment."""
    x_c = _calc_x_c_by_n_c(n_c, N_c=N_c)
    y_a = _calc_y_a_by_n_a(n_a, N_a=N_a)
    x = r_vs * x_c
    a_prime_t = _calc_a_prime_by_a_b_x(a_t, b_t, x_c)
    a_prime_b = _calc_a_prime_by_a_b_x(a_b, b_b, x_c)
    y = a_prime_t + (a_prime_b - a_prime_t) * y_a
    return x, y


def _calc_xyz_by_n_vsac(
    n_vs,
    n_a,
    n_c,
    omega_vs: float = numpy.radians(-6.065),
    delta_gamma_vs: float = numpy.radians(1.05),
    gamma_d: float = 0.0,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
    r_d: float = 1.000,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
):
    z_vs = numpy.floor_divide(n_vs, 2) / N_vs
    gamma = gamma_d + z_vs * delta_gamma_vs * N_vs

    x, y = _calc_xy_vs_by_n_ac(
        n_c,
        n_a,
        N_c=N_c,
        N_a=N_a,
        r_vs=r_vs,
        a_t=a_t,
        b_t=b_t,
        a_b=a_b,
        b_b=b_b,
    )
    x_e = r_d * numpy.sin(gamma) + x * numpy.sin(gamma + omega_vs)
    z_e = r_d * numpy.cos(gamma) + x * numpy.cos(gamma + omega_vs)
    y_e = y
    np_xyz = numpy.stack((x_e, y_e, z_e), axis=0)
    return np_xyz


def calc_xyz_by_id(
    n_id,
    omega_vs: float = numpy.radians(-6.065),
    delta_gamma_vs: float = numpy.radians(1.05),
    gamma_d: float = 0.0,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
    r_d: float = 1.000,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
    n_id_0: int = 1,
):
    n_vs, n_a, n_c = _calc_n_vsac_by_id(
        n_id,
        N_vs=N_vs,
        N_a=N_a,
        N_c=N_c,
        n_id_0=n_id_0,
    )

    np_xyz = _calc_xyz_by_n_vsac(
        n_vs,
        n_a,
        n_c,
        omega_vs=omega_vs,
        delta_gamma_vs=delta_gamma_vs,
        gamma_d=gamma_d,
        N_vs=N_vs,
        N_c=N_c,
        N_a=N_a,
        r_d=r_d,
        r_vs=r_vs,
        a_t=a_t,
        b_t=b_t,
        a_b=a_b,
        b_b=b_b,
    )
    return np_xyz


def _calc_x_c_by_xz_e(
    x_e,
    z_e,
    omega_vs: float = numpy.radians(-6.065),
    r_d: float = 1.000,
    r_vs: float = 0.530,
):
    r_slice = (
        numpy.sqrt(
            numpy.square(r_d)
            + numpy.square(r_vs)
            + 2 * r_d * r_vs * numpy.cos(omega_vs)
        )
        - r_d
    )
    x_c = (numpy.sqrt(numpy.square(x_e) + numpy.square(z_e)) - r_d) / (r_slice)
    return x_c


def _calc_n_vsac_by_xyz(
    np_xyz,
    omega_vs: float = numpy.radians(-6.065),
    delta_gamma_vs: float = numpy.radians(1.05),
    gamma_d: float = 0.0,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
    r_d: float = 1.000,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
):
    x_e, y_e, z_e = np_xyz[0], np_xyz[1], np_xyz[2]
    x_c = _calc_x_c_by_xz_e(
        x_e,
        z_e,
        omega_vs=omega_vs,
        r_d=r_d,
        r_vs=r_vs,
    )
    n_c = numpy.floor(N_c * x_c).astype(int)

    a_prime_t = _calc_a_prime_by_a_b_x(a_t, b_t, x_c)
    a_prime_b = _calc_a_prime_by_a_b_x(a_b, b_b, x_c)

    y_a = (y_e - a_prime_t) / (a_prime_b - a_prime_t)
    n_a = numpy.floor(N_a * y_a).astype(int)

    gamma = numpy.atan2(x_e, z_e)
    delta_gamma_prime = numpy.asin(numpy.sin(-omega_vs) * x_c)

    z_vs = (gamma - gamma_d + delta_gamma_prime) / (N_vs * delta_gamma_vs)
    n_vs = numpy.floor(2 * N_vs * z_vs).astype(int)
    return (
        n_vs,
        n_a,
        n_c,
    )


def calc_id_by_xyz(
    np_xyz,
    omega_vs: float = numpy.radians(-6.065),
    delta_gamma_vs: float = numpy.radians(1.05),
    gamma_d: float = 0.0,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
    r_d: float = 1.000,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
    n_id_0: int = 1,
):
    (
        n_vs,
        n_a,
        n_c,
    ) = _calc_n_vsac_by_xyz(
        np_xyz,
        omega_vs=omega_vs,
        delta_gamma_vs=delta_gamma_vs,
        gamma_d=gamma_d,
        N_vs=N_vs,
        N_c=N_c,
        N_a=N_a,
        r_d=r_d,
        r_vs=r_vs,
        a_t=a_t,
        b_t=b_t,
        a_b=a_b,
        b_b=b_b,
    )
    flag_testing = True
    if flag_testing:
        func_calc_chi_sq = lambda n_vs, n_a, n_c: _calc_chi_sq(
            np_xyz,
            n_vs,
            n_a,
            n_c,
            omega_vs=omega_vs,
            delta_gamma_vs=delta_gamma_vs,
            gamma_d=gamma_d,
            N_vs=N_vs,
            N_c=N_c,
            N_a=N_a,
            r_d=r_d,
            r_vs=r_vs,
            a_t=a_t,
            b_t=b_t,
            a_b=a_b,
            b_b=b_b,
        )
        n_vs, n_a, n_c = _calc_around(
            n_vs, n_a, n_c, func_calc_chi_sq, N_vs=N_vs, N_a=N_a, N_c=N_c
        )
    np_id = _calc_id_by_n_vsac(
        n_vs,
        n_a,
        n_c,
        N_vs=N_vs,
        N_a=N_a,
        N_c=N_c,
        n_id_0=n_id_0,
    )

    return np_id


def _calc_chi_sq(
    np_xyz,
    n_vs,
    n_a,
    n_c,
    omega_vs: float = numpy.radians(-6.065),
    delta_gamma_vs: float = numpy.radians(1.05),
    gamma_d: float = 0.0,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
    r_d: float = 1.000,
    r_vs: float = 0.530,
    a_t=0.252863,
    b_t=0.416524,
    a_b=-0.722903,
    b_b=-1.077999,
    n_id_0: int = 1,
):

    np_xyz_0 = _calc_xyz_by_n_vsac(
        n_vs,
        n_a,
        n_c,
        omega_vs=omega_vs,
        delta_gamma_vs=delta_gamma_vs,
        gamma_d=gamma_d,
        N_vs=N_vs,
        N_c=N_c,
        N_a=N_a,
        r_d=r_d,
        r_vs=r_vs,
        a_t=a_t,
        b_t=b_t,
        a_b=a_b,
        b_b=b_b,
    )
    chi_sq = numpy.square(np_xyz_0 - np_xyz).sum()
    return chi_sq


def _calc_around(
    n_vs,
    n_a,
    n_c,
    func_calc_chi_sq,
    N_vs: int = 60,
    N_c: int = 32,
    N_a: int = 128,
):
    l_shift = [
        (0, 0, 0),
        (-2, -1, -1),
        (0, -1, -1),
        (-2, 0, -1),
        (-2, -1, 0),
        (0, 0, -1),
        (0, -1, 0),
        (-2, 0, 0),
        (2, 1, 1),
        (0, 1, 1),
        (2, 0, 1),
        (2, 1, 0),
        (0, 0, 1),
        (0, 1, 0),
        (2, 0, 0),
    ]
    l_chi_sq = []
    for shift in l_shift:
        n_vs_t, n_a_t, n_c_t = n_vs + shift[0], n_a + shift[1], n_c + shift[2]
        # flag_vs = n_vs_t >= 0 and n_vs_t < 2 * N_vs
        # flag_a = n_a_t >= 0 and n_a_t < N_a
        # flag_c = n_c_t >= 0 and n_c_t < N_c
        # flag = flag_vs and flag_c and flag_a
        # if flag:
        l_chi_sq.append(func_calc_chi_sq(n_vs_t, n_a_t, n_c_t))
    if len(l_chi_sq) == 0:
        pass
    arg_min = numpy.argmin(l_chi_sq)
    if arg_min != 0:
        shift = l_shift[arg_min]
        n_vs_t, n_a_t, n_c_t = n_vs + shift[0], n_a + shift[1], n_c + shift[2]
        n_vs_o, n_a_o, n_c_o = _calc_around(
            n_vs_t,
            n_a_t,
            n_c_t,
            func_calc_chi_sq,
            N_vs=N_vs,
            N_c=N_c,
            N_a=N_a,
        )
    else:
        n_vs_o, n_a_o, n_c_o = n_vs, n_a, n_c
    return n_vs_o, n_a_o, n_c_o


def test_calc_id():
    N_ID_max = 120 * 32 * 60
    omega_vs = numpy.radians(-10)
    for val_id in range(1, N_ID_max):
        print(f"{100*val_id/(N_ID_max-1):8.2f}%", end="\r")
        np_xyz = calc_xyz_by_id(val_id, omega_vs=omega_vs)
        np_id = calc_id_by_xyz(np_xyz, omega_vs=omega_vs)
        if numpy.abs(val_id - np_id) > 1:
            print("ERROR       ")
            print(val_id, np_id)
            n_vs, n_a, n_c = _calc_n_vsac_by_id(val_id)
            print("Input: ", n_vs, n_a, n_c)
            n_vs, n_a, n_c = _calc_n_vsac_by_id(np_id)
            print("Output: ", n_vs, n_a, n_c)
            assert False
    assert True
