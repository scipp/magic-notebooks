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

