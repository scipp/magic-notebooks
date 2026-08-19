import numpy

to_ub_ess = numpy.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [-1., 0., 0.],
], dtype=float)


# to_ub_ess = numpy.array([
#     [1., 0., 0.],
#     [0., 1., 0.],
#     [0., 0., 1.],
# ], dtype=float)

def calc_wavelength(l_m, tof_ms):
    wavelength = 3.9556 * tof_ms / l_m
    return wavelength


def calc_l_total(l_incident_beam, l_scattered_beam, delta_l: float = 0.):
    l_total = l_incident_beam + l_scattered_beam - delta_l
    return l_total


def calc_tof(toa_ms, delta_t_ms: float = 0.):
    tof_ms = toa_ms-delta_t_ms
    return tof_ms


def calc_sample_position(ideal_sample_position, sample_offset):
    sample_position = ideal_sample_position + sample_offset
    return sample_position


def calc_incident_beam(source_position, tp_position, sample_position):
    v1 = sample_position - tp_position
    v2 = tp_position - source_position 
    e1 = v1/numpy.linalg.norm(v1, axis=0)
    incident_beam = v1 + e1 * numpy.linalg.norm(v2, axis=0)
    return incident_beam


def calc_scattered_beam(sample_position, event_position_global):
    scattered_beam = event_position_global - numpy.expand_dims(sample_position, axis=1)
    return scattered_beam


def calc_l_incident_beam(incident_beam):
    l_incident_beam = numpy.linalg.norm(incident_beam, axis=0)
    return l_incident_beam


def calc_l_scattered_beam(scattered_beam):
    l_scattered_beam = numpy.linalg.norm(scattered_beam, axis=0)
    return l_scattered_beam


def calc_ki(incident_beam, wavelength):
    e1 = numpy.expand_dims(incident_beam/numpy.linalg.norm(incident_beam, axis=0), axis=1)
    ki = 2.*numpy.pi*e1/numpy.expand_dims(wavelength, axis=0)
    return ki


def calc_kf(scattered_beam, wavelength):
    e1 = scattered_beam/numpy.linalg.norm(scattered_beam, axis=0)
    kf = 2.*numpy.pi*e1/numpy.expand_dims(wavelength, axis=0)
    return kf


def calc_q(ki, kf):
    q = ki-kf # The definition as in scipp
    return q


def calc_sample_rotation(sample_omega, sample_chi, sample_phi):
    omega,chi,phi = sample_omega, sample_chi, sample_phi
    zero_o = numpy.sin(numpy.zeros_like(omega))
    one_o = numpy.cos(numpy.zeros_like(omega))
    m_omega = numpy.array([
        [numpy.cos(omega), zero_o, numpy.sin(omega)],
        [zero_o, one_o, zero_o],
        [-numpy.sin(omega), zero_o, numpy.cos(omega)],
    ], dtype=float)
    zero_c = numpy.sin(numpy.zeros_like(chi))
    one_c = numpy.cos(numpy.zeros_like(chi))
    m_chi = numpy.array([
        [numpy.cos(chi), -numpy.sin(chi), zero_c],
        [numpy.sin(chi), numpy.cos(chi), zero_c],
        [zero_c, zero_c, one_c],
    ], dtype=float)

    zero_p = numpy.sin(numpy.zeros_like(phi))
    one_p = numpy.cos(numpy.zeros_like(phi))
    m_phi = numpy.array([
        [numpy.cos(phi), zero_p, numpy.sin(phi)],
        [zero_p, one_p, zero_p],
        [-numpy.sin(phi), zero_p, numpy.cos(phi)],
    ], dtype=float)
    sample_rotation = m_omega @ m_chi @ m_phi
    return sample_rotation


def calc_q_unrot(sample_rotation, q):
    q_unrot = numpy.linalg.inv(sample_rotation) @ q
    return q_unrot

np_graph_qvec = {
    'wavelength': calc_wavelength,
    'l_total': calc_l_total,
    'tof_ms': calc_tof,
    'sample_position': calc_sample_position,
    'incident_beam': calc_incident_beam,
    'scattered_beam': calc_scattered_beam,
    'l_incident_beam': calc_l_incident_beam,
    'l_scattered_beam': calc_l_scattered_beam,
    'ki': calc_ki,
    'kf': calc_kf,
    'q': calc_q,
    'sample_rotation': calc_sample_rotation,
    'q_unrot': calc_q_unrot,
}


def calc_vector_by_gamma_nu_r(gamma, nu, r):
    """
    Compute a 3D vector of length r using two angles:
    - gamma: rotation around the Y axis (azimuth), in radians
    - nu: angle between the vector and its projection on the XZ plane (elevation-like), in radians
   """
    sn, cn = numpy.sin(nu), numpy.cos(nu)
    sg, cg = numpy.sin(gamma), numpy.cos(gamma)
    return numpy.array([r * cn * sg,
                        r * sn,
                        r * cn * cg], dtype=float)


def rotate_vector_around_Y_axis(vector, angle):
    """
    Comput rotation of vector along Y axis (direction from Z axis to X axis)
    Angle is given in radians.
    """
    sa, ca = numpy.sin(angle), numpy.cos(angle)
    vx, vy, vz = vector[0], vector[1], vector[2]
    return numpy.array([vx*ca+vz*sa,
                        vy,
                        -vx*sa+vz*ca], dtype=float)


def calc_q_for_hkl(hkl, UB, R):
    """ hkl: [3, N]
    UB and R: [3, 3]
    out: [3, N]
    """
    Q = (R @ (UB @ hkl))
    return Q


def calc_gamma_nu_wavelength_for_hkl(h, k, l, UB, R):
    hkl = numpy.vstack([h, k, l])
    Q = calc_q_for_hkl(hkl, UB, R)
    Qnorm = numpy.linalg.norm(Q, axis=0)
    cos_alpha = -Q[2, :]/Qnorm
    wavelength = 2 * cos_alpha / Qnorm # 4*numpy.pi
    ki = numpy.zeros(Q.shape,dtype=float)
    ki[2, :] = 1/wavelength # 2*numpy.pi
    kf = ki - Q # definition of q like in scipp
    kf_x, kf_y, kf_z = kf[0, :], kf[1, :], kf[2, :]

    r = numpy.linalg.norm(kf, axis=0)

    gamma = numpy.rad2deg(numpy.arctan2(kf_x, kf_z))      # horizontal angle
    nu = numpy.rad2deg(numpy.arcsin(kf_y / r))
    return gamma, nu, wavelength


def calc_tth_phi_wavelength_for_hkl(h, k, l, UB, R):
    hkl = numpy.vstack([h, k, l])
    Q = calc_q_for_hkl(hkl, UB, R)
    Qnorm = numpy.linalg.norm(Q, axis=0)
    cos_alpha = -Q[2, :]/Qnorm
    wavelength = 2 * cos_alpha / Qnorm # 4*numpy.pi
    ki = numpy.zeros(Q.shape, dtype=float)
    ki[2, :] = 1/wavelength # 2*numpy.pi
    kf = ki - Q # definition of q like in scipp
    kf_x, kf_y, kf_z = kf[0, :], kf[1, :], kf[2, :]

    r = numpy.linalg.norm(kf, axis=0)

    tth = numpy.rad2deg(numpy.arccos(kf_z/r))      # diffraction angle
    phi = numpy.rad2deg(numpy.arctan2(kf_y, kf_x))
    return tth, phi, wavelength


def generate_peak_data(
    UB: numpy.ndarray, R: numpy.ndarray,
    lambda_min: float, lambda_max: float,
    gamma_min: float = 0.0, gamma_max: float = numpy.pi,
    nu_min: float = -numpy.pi/2, nu_max: float = numpy.pi/2,
    propagation_vector: numpy.ndarray = None
):
    """
    Generate synthetic diffraction peak data based on:
    - UB matrix (3x3)
    - crystal rotation matrix R (3x3)
    - wavelength range (lambda_min, lambda_max)
    - detector angular limits (gamma_min/max, nu_min/max)
    - optional propagation vector k (default: None → k = (0,0,0))

    HKL bounds are computed automatically from UB, wavelength, and angle limits.
    """

    # --- Parameter validation ---
    if gamma_min < 0.:
        raise ValueError("gamma_min cannot be below 0 radians.")
    if gamma_max > numpy.pi:
        raise ValueError("gamma_max cannot exceed π radians.")
    if gamma_min > gamma_max:
        raise ValueError("gamma_min cannot be greater than gamma_max.")

    if nu_min < -numpy.pi/2:
        raise ValueError("nu_min cannot be below -π/2 radians.")
    if nu_max > numpy.pi/2:
        raise ValueError("nu_max cannot exceed +π/2 radians.")
    if nu_min > nu_max:
        raise ValueError("nu_min cannot be greater than nu_max.")

    if lambda_min < 0.:
        raise ValueError("lambda_min cannot be below 0 Å.")
    if lambda_max > 20.:
        raise ValueError("lambda_max cannot exceed 20 Å.")
    if lambda_min > lambda_max:
        raise ValueError("lambda_min cannot be greater than lambda_max.")

    # --- Propagation vector ---
    if propagation_vector is None:
        kvec = numpy.zeros(3)
    else:
        kvec = numpy.asarray(propagation_vector, dtype=float)
        if kvec.shape != (3,):
            raise ValueError("propagation_vector must be a 3-element array.")

    # --- 0. Compute HKL bounds from Q-range ---
    gamma_vals = numpy.array([gamma_min, gamma_max])
    nu_vals = numpy.array([nu_min, nu_max])

    dirs = []
    for g in gamma_vals:
        for n in nu_vals:
            x = numpy.sin(g) * numpy.cos(n)
            y = numpy.sin(n)
            z = numpy.cos(g) * numpy.cos(n)
            dirs.append([x, y, z])
    dirs = numpy.array(dirs).T

    Q_extremes = []
    for lam in [lambda_min, lambda_max]:
        ki = numpy.array([0, 0, 1/lam])[:, None]
        kf = dirs / lam
        Q = ki - kf # according to scipp definition
        Q_extremes.append(Q)

    Q_extremes = numpy.hstack(Q_extremes)

    RUB_inv = numpy.linalg.inv(R @ UB)
    hkl_ext = RUB_inv @ Q_extremes

    h_min, h_max = numpy.floor(hkl_ext[0].min()), numpy.ceil(hkl_ext[0].max())
    k_min, k_max = numpy.floor(hkl_ext[1].min()), numpy.ceil(hkl_ext[1].max())
    l_min, l_max = numpy.floor(hkl_ext[2].min()), numpy.ceil(hkl_ext[2].max())

    # --- 1. Generate HKL grid ---
    h = numpy.arange(h_min, h_max + 1)
    k = numpy.arange(k_min, k_max + 1)
    l = numpy.arange(l_min, l_max + 1)

    H, K, L = numpy.meshgrid(h, k, l, indexing='ij')
    hkl = numpy.vstack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[:, numpy.any(hkl != 0, axis=0)]

    # --- 2. Apply propagation vector ---
    # Q = UB * (hkl + k)
    hkl_shifted = hkl + kvec[:, None]

    Q = calc_q_for_hkl(hkl_shifted, UB, R)
    Qnorm = numpy.linalg.norm(Q, axis=0)

    # --- 3. Compute wavelength ---
    cos_alpha = Q[2] / Qnorm               # ESS coordinate system: Z along incident beam
    wavelength = 2 * cos_alpha / Qnorm
    # --- 4. Apply wavelength limits ---
    mask = (wavelength >= lambda_min) & (wavelength <= lambda_max)
    hkl = hkl[:, mask]

    hkl_shifted = hkl_shifted[:, mask]
    Q = Q[:, mask]
    wavelength = wavelength[mask]

    # --- 5. Compute detector angles ---
    ki = numpy.zeros(Q.shape)
    ki[2] = 1 / wavelength
    kf = ki - Q
    kf_x, kf_y, kf_z = kf[0], kf[1], kf[2]
    r = numpy.linalg.norm(kf, axis=0)

    gamma = numpy.arctan2(kf_x, kf_z)
    nu = numpy.arcsin(kf_y / r)

    # --- 6. Apply gamma/nu limits ---
    mask_ang = (
        (gamma >= gamma_min) & (gamma <= gamma_max) &
        (nu >= nu_min) & (nu <= nu_max)
    )

    hkl = hkl[:, mask_ang]
    gamma = gamma[mask_ang]
    nu = nu[mask_ang]
    wavelength = wavelength[mask_ang]

    # --- 7. Final output as structured NumPy array ---
    dtype = [
        ('h', 'f4'),
        ('k', 'f4'),
        ('l', 'f4'),
        ('gamma', 'f8'),
        ('nu', 'f8'),
        ('wavelength', 'f8')
    ]

    result = numpy.zeros(hkl.shape[1], dtype=dtype)

    result['h'] = hkl[0]
    result['k'] = hkl[1]
    result['l'] = hkl[2]
    result['gamma'] = gamma
    result['nu'] = nu
    result['wavelength'] = wavelength

    return result


def calc_orientation_matrix(euler_alpha, euler_beta, euler_gamma, ):
    ca, cb, cg = numpy.cos(euler_alpha), numpy.cos(euler_beta), numpy.cos(euler_gamma)
    sa, sb, sg = numpy.sin(euler_alpha), numpy.sin(euler_beta), numpy.sin(euler_gamma)
    m_m = numpy.array([
        [ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg],
        [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg],
        [-sb, cb*sg, cb*cg],
    ], dtype=float)
    return m_m


def calc_cell_phi(cell_alpha, cell_beta, cell_gamma):
    ca, cb, cg = numpy.cos(cell_alpha), numpy.cos(cell_beta), numpy.cos(cell_gamma)
    cell_phi = numpy.sqrt(1. - ca*ca - cb*cb - cg*cg + 2 * ca * cb * cg)
    return cell_phi


def calc_cell_volume(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma):
    cell_phi = calc_cell_phi(cell_alpha, cell_beta, cell_gamma)
    cell_volume = cell_a * cell_b * cell_c * cell_phi
    return cell_volume


def calc_b_matrix(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma):
    cell_phi = calc_cell_phi(cell_alpha, cell_beta, cell_gamma)
    a, b, c = cell_a, cell_b, cell_c
    b_11 = numpy.sin(cell_alpha)/(a*cell_phi)
    b_12 = (numpy.cos(cell_alpha)*numpy.cos(cell_beta)-numpy.cos(cell_gamma))/(b*cell_phi*numpy.sin(cell_alpha))
    b_13 = (numpy.cos(cell_alpha)*numpy.cos(cell_gamma)-numpy.cos(cell_beta))/(c*cell_phi*numpy.sin(cell_alpha))
    b_22 = 1/(b*numpy.sin(cell_alpha))
    b_23 = -numpy.cos(cell_alpha)/(c*numpy.sin(cell_alpha))
    b_33 = 1/c
    zero = 0.
    b_matrix = numpy.array([
            [b_11, b_12, b_13],
            [zero, b_22, b_23],
            [zero, zero, b_33],
        ], dtype=float)
    return b_matrix


def calc_tth_phi_by_gamma_nu(gamma, nu):
    tth = numpy.arccos(numpy.cos(gamma) * numpy.cos(nu))
    phi = numpy.atan2(numpy.tan(nu), numpy.sin(gamma))
    return tth, phi


def calc_gamma_nu_by_tth_phi(tth, phi):
    gamma = numpy.atan2(numpy.tan(tth), numpy.cos(phi))
    nu = numpy.arcsin(numpy.sin(tth), numpy.sin(phi))
    return gamma, nu


def constraint_unit_cell_parameters_by_singony(unit_cell_parameters, singony: str = 'triclinic'):
    """Give constrained unit cell parameters based on provided singony:

    'cubic': [a, a, a, pi/2, pi/2, pi/2]
    'hexagonal': [a, a, c, pi/2, pi/2, 2/3 pi]
    'tetragonal': [a, a, c, pi/2, pi/2, pi/2]
    'orthorombic': [a, b, c, pi/2, pi/2, pi/2]
    'monoclinic': [a, b, c, pi/2, beta, pi/2]
    'triclinic': [a, b, c, alpha, beta, gamma]
    """
    ucp = unit_cell_parameters
    rad90 = numpy.pi * 0.5 * numpy.ones_like(ucp[0])
    rad120 = numpy.pi * 2. / 3. * numpy.ones_like(ucp[0])
    if singony.startswith('c'):
        ucp = numpy.array([ucp[0], ucp[0], ucp[0], rad90, rad90, rad90],
                          dtype=float)
    elif singony.startswith('h'):
        ucp = numpy.array([ucp[0], ucp[0], ucp[0], rad90, rad90, rad120],
                          dtype=float)
    elif singony.startswith('te'):
        ucp = numpy.array([ucp[0], ucp[0], ucp[2], rad90, rad90, rad90],
                          dtype=float)
    elif singony.startswith('o'):
        ucp = numpy.array([ucp[0], ucp[1], ucp[2], rad90, rad90, rad90],
                          dtype=float)
    elif singony.startswith('m'):
        ucp = numpy.array([ucp[0], ucp[1], ucp[2], rad90, ucp[4], rad90],
                          dtype=float)
    return ucp


# get ub functions
def get_ub(q_hkl):
    np_q2 = calc_sum_q1_q2(q_hkl, -1*q_hkl)
    # np_q2 = numpy.concatenate((q_hkl, np_q2), axis=1)
    # np_r = calc_sum_q1_q2(np_q2, -1*np_q2)
    # np_r = numpy.concatenate((np_q2, np_r), axis=1)
    np_r = np_q2
    flag, q1, q2, q3 = choose_min_q123(np_r)

    if not flag:
        if q1 is None:
            print("\nNon of a,b,c was found.")
            print("It looks that you do not provide the list of measured peaks.")
        elif q2 is None:
            print("\nOnly one vector in reciprocal space was found.")
            print("Provide more peaks.")
            print("\nVector 1:")
            print(f"{q1[0]:9.5f} {q1[1]:9.5f} {q1[2]:9.5f}")
            mod_q1 = numpy.sqrt(numpy.square(q1).sum())
            print(f"\nModulus is {mod_q1:9.5f} inv.Ang")
            print(f"\nDistance is {1/mod_q1:9.5f} Ang")
        elif q3 is None:
            print("\nOnly one vector in reciprocal space was found.")
            print("Provide more peaks.")
            print("\nVector 1:")
            print(f"{q1[0]:9.5f} {q1[1]:9.5f} {q1[2]:9.5f}")
            mod_q1 = numpy.sqrt(numpy.square(q1).sum())
            print(f"\nModulus is {mod_q1:9.5f} inv.Ang")
            print("\n\nVector 2:")
            print(f"{q2[0]:9.5f} {q2[1]:9.5f} {q2[2]:9.5f}")
            mod_q2 = numpy.sqrt(numpy.square(q2).sum())
            print(f"\nModulus is {mod_q2:9.5f} inv.Ang")
            q_cross = numpy.cross(q1, q2)
            mod_q_cross = numpy.sqrt(numpy.square(q_cross).sum())
            a = mod_q2/mod_q_cross
            b = mod_q1/mod_q_cross
            ang = 180. - numpy.degrees(numpy.asin(mod_q_cross/(mod_q1 * mod_q2)))
            print(f"a is {a:9.5f} Ang \nb is {b:9.5f} Ang\nAngle is {ang:9.2f} deg.")
        return None, None
    print("\nUB-matrix:")
    ub = numpy.array([
     [q1[0], q2[0], q3[0]],
     [q1[1], q2[1], q3[1]],
     [q1[2], q2[2], q3[2]],
    ], dtype=float)/(2*numpy.pi)
    print(f"{ub[0, 0]:9.5f} {ub[0, 1]:9.5f} {ub[0, 2]:9.5f}")
    print(f"{ub[1, 0]:9.5f} {ub[1, 1]:9.5f} {ub[1, 2]:9.5f}")
    print(f"{ub[2, 0]:9.5f} {ub[2, 1]:9.5f} {ub[2, 2]:9.5f}")

    ucp = calc_unit_cell_parameters_by_b_matrix(ub)
    print(f"Unit cell parameters: {ucp[0]:9.5f} {ucp[1]:9.5f} {ucp[2]:9.5f} {numpy.degrees(ucp[3]):9.5f} {numpy.degrees(ucp[4]):9.5f} {numpy.degrees(ucp[5]):9.5f}")
    return ub, ucp


def calc_sum_q1_q2(np_q1, np_q2,
                   mod_min_allowed: float = 0.03, mod_max_allowed: float = 5.):
    l_res = []
    n_q1 = np_q1.shape[1]
    n_q2 = np_q2.shape[1]
    for i1 in range(n_q1):
        for i2 in range(n_q2):
            val = np_q1[:, i1] + np_q2[:, i2]
            l_res.append(val)
    np_tot = numpy.stack(l_res, axis=1)

    np_tot_norm = numpy.sqrt(numpy.square(np_tot).sum(axis=0))
    np_flag = numpy.logical_and(
        np_tot_norm >= mod_min_allowed,
        np_tot_norm <= mod_max_allowed
        )
    np_tot = np_tot[:, np_flag]

    # np_tot = numpy.unique(np_tot, axis=0)
    # print("-------")
    # for val in np_tot.transpose():
    #     print(numpy.round(val, 2))
    return np_tot


def choose_min_q123(
    np_q,
    mod_min_allowed: float = 0.03,
    ang_min: float = numpy.radians(55),
):
    np_q_norm = numpy.sqrt(numpy.square(np_q).sum(axis=0))
    np_ind_order = numpy.argsort(np_q_norm)
    # for val in np_q.transpose():
    #     print(numpy.round(val, 2))

    # choosing q1:
    flag_q1 = False
    for ind_1, ind in enumerate(np_ind_order):
        q1 = np_q[:, ind]
        mod_q1 = np_q_norm[ind]
        if mod_q1 >= mod_min_allowed:
            flag_q1 = True
            q1_norm = q1 / numpy.expand_dims(mod_q1, axis=0)
            break
    if not flag_q1: 
        return False, None, None, None
    # print("1: ", q1, mod_q1)

    # choosing q2:
    flag_q2 = False
    for ind_2, ind in enumerate(np_ind_order[ind_1+1:]):
        q2 = np_q[:, ind]
        mod_q2 = np_q_norm[ind]
        if mod_q2 < mod_min_allowed:
            continue
        q2_norm = q2 / numpy.expand_dims(mod_q2, axis=0)
        if numpy.abs((q1_norm * q2_norm).sum()) > numpy.cos(ang_min):
            continue
        q_cross = numpy.cross(q1_norm, q2_norm)
        mod_q_cross = numpy.sqrt(numpy.square(q_cross).sum(axis=0))
        if mod_q_cross > numpy.sin(ang_min):
            q_cross = q_cross / numpy.expand_dims(mod_q_cross, axis=0)
            flag_q2 = True
            break
    if not flag_q2:
        return False, q1, None, None
    # print("2: ", q2, mod_q2)
    # choosing q3:
    flag_q3 = False
    for ind_3, ind in enumerate(np_ind_order[ind_1+1+ind_2+1:]):
        q3 = np_q[:, ind]
        mod_q3 = np_q_norm[ind]
        if mod_q3 < mod_min_allowed:
            continue
        q3_norm = q3 / numpy.expand_dims(mod_q3, axis=0)
        if numpy.abs((q1_norm * q3_norm).sum()) > numpy.cos(ang_min):
            continue
        if numpy.abs((q2_norm * q3_norm).sum()) > numpy.cos(ang_min):
            continue
        if numpy.abs((q_cross * q3_norm).sum()) < numpy.cos(ang_min):
            continue
        if numpy.abs(numpy.sum(q_cross * q3_norm)) > mod_min_allowed:
            flag_q3 = True
            break
    if not flag_q3:
        return False, q1, q2, None
    # print("3: ", q3, mod_q3, q_cross)
    return True, q1, q2, q3


def calc_unit_cell_parameters_by_b_matrix(np_b):
    abc_inv = numpy.sqrt(numpy.square(np_b).sum(axis=0))
    cos_abg_inv = (numpy.roll(np_b,shift=-1,axis=1)*numpy.roll(np_b,shift=-2,axis=1)).sum(axis=0)/(numpy.roll(abc_inv, shift=-1,axis=0)*numpy.roll(abc_inv, shift=-2,axis=0))
    sin_abg_inv = numpy.sqrt(1.-numpy.square(cos_abg_inv))
    phi_inv = numpy.sqrt(1.-numpy.square(cos_abg_inv).sum()+2.*cos_abg_inv.prod())
    vol_inv = abc_inv.prod()*phi_inv
    vol = 1./vol_inv
    abc = numpy.roll(abc_inv, shift=-1,axis=0)*numpy.roll(abc_inv, shift=-2,axis=0)*sin_abg_inv/vol_inv
    sin_abg = phi_inv/(numpy.roll(sin_abg_inv, shift=-1,axis=0)*numpy.roll(sin_abg_inv, shift=-2,axis=0))
    abg=numpy.asin(sin_abg)
    unit_cell_parameters = numpy.array([abc[0], abc[1], abc[2], abg[0], abg[1], abg[2]], dtype=float)
    return unit_cell_parameters