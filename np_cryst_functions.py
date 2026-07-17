import numpy

def calc_gamma_nu_wavelength_for_hkl(h, k, l, UB, R):
    hkl = numpy.vstack([h, k, l]).T
    Q = (R @ (UB @ hkl.T)).T
    Qnorm = numpy.linalg.norm(Q, axis=1)
    cos_alpha = -Q[:,2]/Qnorm
    wavelength = 2 * cos_alpha / Qnorm # 4*numpy.pi
    ki = numpy.zeros(Q.shape,dtype=float)
    ki[:,2] = 1/wavelength # 2*numpy.pi
    kf = ki + Q
    kf_x, kf_y, kf_z = kf[:,0], kf[:,1], kf[:,2]
    
    r = numpy.linalg.norm(kf, axis=1)

    gamma = numpy.rad2deg(numpy.arctan2(kf_x, kf_z))      # horizontal angle
    nu    = numpy.rad2deg(numpy.arcsin(kf_y / r))  
    return gamma, nu, wavelength


def calc_tth_phi_wavelength_for_hkl(h, k, l, UB, R):
    hkl = numpy.vstack([h, k, l]).T
    Q = (R @ (UB @ hkl.T)).T
    Qnorm = numpy.linalg.norm(Q, axis=1)
    cos_alpha = -Q[:,2]/Qnorm
    wavelength = 2 * cos_alpha / Qnorm # 4*numpy.pi
    ki = numpy.zeros(Q.shape,dtype=float)
    ki[:,2] = 1/wavelength # 2*numpy.pi
    kf = ki + Q
    kf_x, kf_y, kf_z = kf[:,0], kf[:,1], kf[:,2]
    
    r = numpy.linalg.norm(kf, axis=1)

    tth = numpy.rad2deg(numpy.arccos(kf_z/r))      # diffraction angle
    phi    = numpy.rad2deg(numpy.arctan2(kf_y, kf_x))  
    return tth, phi, wavelength

def generate_peak_data(UB, R, hmax, kmax, lmax, lambda_min, lambda_max):
    """
    Generate synthetic diffraction peak data based on:
    - UB matrix (3x3)
    - crystal rotation matrix R (3x3)
    - HKL range: -hmax..hmax etc.
    - wavelength range (lambda_min, lambda_max)

    Returns array with columns:
    [h, k, l, gamma_deg, nu_deg, wavelength]
    """

    # --- 1. Generate HKL grid ---
    h = numpy.arange(-int(hmax), int(hmax+1))
    k = numpy.arange(-int(kmax), int(kmax+1))
    l = numpy.arange(-int(lmax), int(lmax+1))
    H, K, L = numpy.meshgrid(h, k, l, indexing='ij')
    hkl = numpy.vstack([H.ravel(), K.ravel(), L.ravel()]).T

    # Remove (0,0,0)
    hkl = hkl[numpy.any(hkl != 0, axis=1)]

    # --- 2. Compute Q vectors in lab frame ---
    # Apply UB and then rotation R
    Q = (R @ (UB @ hkl.T)).T

    # Magnitude of Q
    Qnorm = numpy.linalg.norm(Q, axis=1)


    # --- 3. Compute wavelength from Bragg condition ---
    cos_alpha = -Q[:,2]/Qnorm

    wavelength = 2 * cos_alpha / Qnorm # 4*numpy.pi

    # wavelength = 4*numpy.pi / Qnorm

    # --- 4. Apply wavelength limits ---
    mask = (wavelength >= lambda_min) & (wavelength <= lambda_max)
    hkl = hkl[mask]
    Q = Q[mask]
    wavelength = wavelength[mask]

    # --- 5. Convert Q direction to detector angles (γ, ν) ---
    ki = numpy.zeros(Q.shape,dtype=float)
    ki[:,2] = 1/wavelength # 2*numpy.pi
    kf = ki + Q
    kf_x, kf_y, kf_z = kf[:,0], kf[:,1], kf[:,2]
    
    r = numpy.linalg.norm(kf, axis=1)

    gamma = numpy.rad2deg(numpy.arctan2(kf_x, kf_z))      # horizontal angle
    nu    = numpy.rad2deg(numpy.arcsin(kf_y / r))       # vertical angle

    # --- 6. Build final array ---
    result = numpy.column_stack([hkl[:,0], hkl[:,1], hkl[:,2], gamma, nu, wavelength])
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



def calc_sample_rotation(sample_omega, sample_chi, sample_phi):
    zero_o = numpy.sin(numpy.zeros_like(sample_omega))
    one_o = numpy.cos(numpy.zeros_like(sample_omega))
    m_omega = numpy.array([
        [numpy.cos(sample_omega), zero_o, numpy.sin(sample_omega)],
        [zero_o, one_o, zero_o],
        [-numpy.sin(sample_omega), zero_o, numpy.cos(sample_omega)],
    ],dtype=float)
    zero_c = numpy.sin(numpy.zeros_like(sample_chi))
    one_c = numpy.cos(numpy.zeros_like(sample_chi))
    m_chi = numpy.array([
        [numpy.cos(sample_chi), -numpy.sin(sample_chi), zero_c],
        [numpy.sin(sample_chi), numpy.cos(sample_chi), zero_c],
        [zero_c, zero_c, one_c],
    ],dtype=float)

    zero_p = numpy.sin(numpy.zeros_like(sample_phi))
    one_p = numpy.cos(numpy.zeros_like(sample_phi))
    m_phi = numpy.array([
        [numpy.cos(sample_phi), zero_p, numpy.sin(sample_phi)],
        [zero_p, one_p, zero_p],
        [-numpy.sin(sample_phi), zero_p, numpy.cos(sample_phi)],
    ], dtype=float)
    sample_rotation = m_omega @ m_chi @ m_phi
    return sample_rotation


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
        ],dtype=float)
    return b_matrix


def calc_tth_phi_by_gamma_nu(gamma, nu):
    tth = numpy.arccos(numpy.cos(gamma) * numpy.cos(nu))
    phi = numpy.atan2(numpy.tan(nu), numpy.sin(gamma))
    return tth, phi


def calc_gamma_nu_by_tth_phi(tth, phi):
    gamma = numpy.atan2(numpy.tan(tth), numpy.cos(phi))
    nu = numpy.arcsin(numpy.sin(tth), numpy.sin(phi))
    return gamma, nu


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
    print(f"{ub[0,0]:9.5f} {ub[0,1]:9.5f} {ub[0,2]:9.5f}")
    print(f"{ub[1,0]:9.5f} {ub[1,1]:9.5f} {ub[1,2]:9.5f}")
    print(f"{ub[2,0]:9.5f} {ub[2,1]:9.5f} {ub[2,2]:9.5f}")

    ucp = calc_unit_cell_parameters_by_b_matrix(ub)
    print(f"Unit cell parameters: {ucp[0]:9.5f} {ucp[1]:9.5f} {ucp[2]:9.5f} {numpy.degrees(ucp[3]):9.5f} {numpy.degrees(ucp[4]):9.5f} {numpy.degrees(ucp[5]):9.5f}")
    return ub, ucp
    
def calc_sum_q1_q2(np_q1, np_q2, mod_min_allowed = 0.03, mod_max_allowed = 5.):
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


def choose_min_q123(np_q, mod_min_allowed = 0.03, ang_min = numpy.radians(55)):
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