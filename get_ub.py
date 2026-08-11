"""
import numpy
import scipy.optimize
import magic_graphs
import magic_scipp
import scipp as sc

def get_euler_opt(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma,
        Q_vec_rot, sigma_Q_vec_rot,
        euler_alpha, euler_beta, euler_gamma, graph_hkl=magic_graphs.graph_hkl,
        refine_unit_cell=False, refine_orientation:bool=True, 
        volume_constraint:bool=False, basinhopping:bool=False,
        singony='triclinic'):

    # np_weight = numpy.asarray(weight) # values
    # np_weight = np_weight / numpy.max(np_weight)

    ea_rad = euler_alpha.to(unit="rad", copy=False).value
    eb_rad = euler_beta.to(unit="rad", copy=False).value
    eg_rad = euler_gamma.to(unit="rad", copy=False).value


    h90 = sc.scalar(90., unit="deg")
    h120 = sc.scalar(120., unit="deg")
    singony = singony.lower()
    def calc_b_matrix_by_x(x_cell):
        if singony.startswith('c'):
            ha = sc.scalar(x_cell[0], unit="angstrom")
            sc_b_matrix = graph_hkl['b_matrix'](ha,ha,ha,h90,h90,h90)
        elif singony.startswith('h'):
            ha = sc.scalar(x_cell[0], unit="angstrom")
            hc = sc.scalar(x_cell[1], unit="angstrom")
            sc_b_matrix = graph_hkl['b_matrix'](ha,ha,hc,h90,h90,h120)
        elif singony.startswith('te'):
            ha = sc.scalar(x_cell[0], unit="angstrom")
            hc = sc.scalar(x_cell[1], unit="angstrom")
            sc_b_matrix = graph_hkl['b_matrix'](ha,ha,hc,h90,h90,h90)
        elif singony.startswith('o'):
            ha = sc.scalar(x_cell[0], unit="angstrom")
            hb = sc.scalar(x_cell[1], unit="angstrom")
            hc = sc.scalar(x_cell[2], unit="angstrom")
            sc_b_matrix = graph_hkl['b_matrix'](ha,hb,hc,h90,h90,h90)
        elif singony.startswith('m'):
            ha = sc.scalar(x_cell[0], unit="angstrom")
            hb = sc.scalar(x_cell[1], unit="angstrom")
            hc = sc.scalar(x_cell[2], unit="angstrom")
            hbe = sc.scalar(x_cell[3], unit="deg")
            sc_b_matrix = graph_hkl['b_matrix'](ha,hb,hc,h90,hbe,h90)
        else:
            ha = sc.scalar(x_cell[0], unit="angstrom")
            hb = sc.scalar(x_cell[1], unit="angstrom")
            hc = sc.scalar(x_cell[2], unit="angstrom")
            hal = sc.scalar(x_cell[3], unit="deg")
            hbe = sc.scalar(x_cell[4], unit="deg")
            hga = sc.scalar(x_cell[5], unit="deg")
            sc_b_matrix = graph_hkl['b_matrix'](ha,hb,hc,hal,hbe,hga)
        return sc_b_matrix

    cell_a_ang = cell_a.to(unit="angstrom", copy=False).value
    cell_b_ang = cell_b.to(unit="angstrom", copy=False).value
    cell_c_ang = cell_c.to(unit="angstrom", copy=False).value
    cell_alpha_deg = cell_alpha.to(unit="deg", copy=False).value
    cell_beta_deg = cell_beta.to(unit="deg", copy=False).value
    cell_gamma_deg = cell_gamma.to(unit="deg", copy=False).value

    x0 = []
    if refine_orientation:
       x0.extend([ea_rad, eb_rad, eg_rad]) 
    x_cell = []
    if singony.startswith('c'):
        x_cell = [cell_a_ang, ]
    elif singony.startswith('h') or singony.startswith('te'):
        x_cell = [cell_a_ang, cell_c_ang, ]
    elif singony.startswith('o'):
        x_cell = [cell_a_ang, cell_b_ang, cell_c_ang, ]
    elif singony.startswith('m'):
        x_cell = [cell_a_ang, cell_b_ang, cell_c_ang, cell_beta_deg, ]
    else:
        x_cell = [cell_a_ang, cell_b_ang, cell_c_ang, cell_alpha_deg, cell_beta_deg, cell_gamma_deg, ]

    if refine_unit_cell:
        x0.extend(x_cell)
        
    sc_b_matrix = calc_b_matrix_by_x(x_cell)
    sc_ucp = magic_graphs.graph_ub_inv[("cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",)](sc_b_matrix)
    cell_volume = graph_hkl['cell_volume'](cell_a=sc_ucp[0], cell_b=sc_ucp[1], cell_c=sc_ucp[2],
                             cell_alpha=sc_ucp[3], cell_beta=sc_ucp[4], cell_gamma=sc_ucp[5])
    cell_volume_max = 1.2 * cell_volume

    def calc_chi_sq(x):
        i_cell = 0
        if refine_orientation:
            euler_angles = x[:3]
            i_cell = 3
        else:
            euler_angles = numpy.array([ea_rad, eb_rad, eg_rad], dtype=float)
            
        if refine_unit_cell:
            sc_b_matrix = calc_b_matrix_by_x(x[i_cell:])
        else:
            sc_b_matrix = graph_hkl['b_matrix'](cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
        
        sc_u = magic_graphs.graph_hkl_inv["u_matrix"](
            sc.scalar(euler_angles[0], unit="rad"),
            sc.scalar(euler_angles[1], unit="rad"),
            sc.scalar(euler_angles[2], unit="rad"),
            )
        sc_UB = graph_hkl["ub_matrix"](u_matrix=sc_u, b_matrix=sc_b_matrix)
        sc_hkl_int = graph_hkl["hkl_vec"](ub_matrix=sc_UB, Q_vec_rot=Q_vec_rot)
        sc_hkl_int.values = numpy.round(sc_hkl_int.values, 0)
        Q_vec_rot_ref = magic_graphs.graph_hkl_inv["Q_vec_rot"](ub_matrix=sc_UB, hkl_vec=sc_hkl_int)
        Q_vec_rot_diff = (Q_vec_rot_ref - Q_vec_rot).values/sigma_Q_vec_rot.values
        chi_sq = (numpy.square(Q_vec_rot_diff)).sum() # * numpy.expand_dims(np_weight, axis=1)
        if volume_constraint and refine_unit_cell:
            sc_ucp = magic_graphs.graph_ub_inv[("cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",)](sc_b_matrix)
            cell_volume = graph_hkl['cell_volume'](cell_a=sc_ucp[0], cell_b=sc_ucp[1], cell_c=sc_ucp[2],
                             cell_alpha=sc_ucp[3], cell_beta=sc_ucp[4], cell_gamma=sc_ucp[5])
            if cell_volume > cell_volume_max:
                chi_sq *= 7.*cell_volume.value/cell_volume_max.value
            
        return chi_sq

    if len(x0)==0:
        print('No refined parameters')
        ea_opt = numpy.array([ea_rad, eb_rad, eg_rad], dtype=float)
        sc_b_matrix
        return (sc.scalar(ea_opt[0], unit="rad"), sc.scalar(ea_opt[1], unit="rad"), sc.scalar(ea_opt[2], unit="rad")), sc_b_matrix, calc_chi_sq(x0)

    if basinhopping:
        res = scipy.optimize.basinhopping(calc_chi_sq, x0, T=10000,niter=100, interval=20, stepwise_factor=0.7, disp=True)
        x0 = res.x
        
    res = scipy.optimize.minimize(calc_chi_sq, x0, method='BFGS')# Nelder-Mead
    i_cell = 0
    if refine_orientation:
        i_cell = 3
        ea_opt = numpy.array(res.x[:i_cell], dtype=float)
    else:
        ea_opt = numpy.array([ea_rad, eb_rad, eg_rad], dtype=float)
    
    if refine_unit_cell:
        sc_b_matrix =calc_b_matrix_by_x(res.x[i_cell:])
    
    ea_opt = ea_opt%(2.*numpy.pi)
    return (sc.scalar(ea_opt[0], unit="rad"), sc.scalar(ea_opt[1], unit="rad"), sc.scalar(ea_opt[2], unit="rad")), sc_b_matrix, res.fun



def optimize_delta_t_delta_l(da):
    def calc_chi_sq(x, da, coeff):
        delta_t = x[0]
        delta_l = x[1]
        cell_a = x[2]
        ea = x[3]
        eb = x[4]
        eg = x[5]
        da.coords["delta_t"] = sc.scalar(delta_t, unit="s")
        da.coords["delta_L"] = sc.scalar(delta_l, unit="m")
        da.coords["cell_a"] = sc.scalar(cell_a, unit="Angstrom")
        da.coords["cell_b"] = sc.scalar(cell_a, unit="Angstrom")
        da.coords["cell_c"] = sc.scalar(cell_a, unit="Angstrom")
        da.coords["euler_alpha"] = sc.scalar(ea, unit="rad")
        da.coords["euler_beta"] = sc.scalar(eb, unit="rad")
        da.coords["euler_gamma"] = sc.scalar(eg, unit="rad")
        
        magic_scipp.remove_coords_in_da(da, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "hkl_vec","Q_vec_rot","Q_vec","Qx","Qy","Qz","wavelength", "tof", "Ltotal", "Q", "u_matrix", "b_matrix", "ub_matrix")
        da2 = da.transform_coords(("h_reduced", "k_reduced", "l_reduced"), graph={**magic_graphs.graph_hkl, **magic_graphs.graph_qvec})
        nd_delta_hkl = numpy.array([
            da2.coords["h_reduced"].values,
            da2.coords["k_reduced"].values,
            da2.coords["l_reduced"].values,], dtype=float)    
          
        np_weight = da2.data.values
        chi_sq = numpy.square(np_weight*(numpy.abs(nd_delta_hkl-0.5)-0.5)/numpy.expand_dims(coeff, axis=1)).sum()
        return chi_sq
    x0 = [
        da.coords["delta_t"].to(unit="s").value, 
        da.coords["delta_L"].to(unit="m").value,
        da.coords["cell_a"].to(unit="Angstrom").value,
        da.coords["euler_alpha"].to(unit="rad").value,
        da.coords["euler_beta"].to(unit="rad").value,
        da.coords["euler_gamma"].to(unit="rad").value,
        ]
    
    coeff = numpy.array([da.coords["cell_a"].value, da.coords["cell_b"].value, da.coords["cell_c"].value], dtype=float)
    print("Original chi_sq", calc_chi_sq(x0, da, coeff))
    res = scipy.optimize.minimize(calc_chi_sq, x0, args=(da,coeff, ), method="BFGS")
    print(res)
    da.coords["delta_t"] = sc.scalar(res.x[0], unit="s")
    da.coords["delta_L"] = sc.scalar(res.x[1], unit="m")
    da.coords["cell_a"] = sc.scalar(res.x[2], unit="Angstrom")
    da.coords["cell_b"] = sc.scalar(res.x[2], unit="Angstrom")
    da.coords["cell_c"] = sc.scalar(res.x[2], unit="Angstrom")
    da.coords["euler_alpha"] = sc.scalar(res.x[3], unit="rad")
    da.coords["euler_beta"] = sc.scalar(res.x[4], unit="rad")
    da.coords["euler_gamma"] = sc.scalar(res.x[5], unit="rad")
    magic_scipp.remove_coords_in_da(da, "h", "k", "l", "h_reduced", "k_reduced", "l_reduced", "hkl_vec","Q_vec_rot","Q_vec","Qx","Qy","Qz","wavelength", "tof", "Ltotal", "Q", "u_matrix", "b_matrix", "ub_matrix")
    return

"""

import numpy
import scipy.optimize
import np_cryst_functions

def get_euler_opt_by_qvec(
    unit_cell_parameters: numpy.ndarray,
    q_vec: numpy.ndarray, 
    sigma_q_vec: numpy.ndarray,
    euler_angles: numpy.ndarray,
    singony: str = 'triclinic',
    refine_unit_cell_parameters: bool = False, 
    refine_euler_angles: bool = True,
    constraint_volume: bool = False, 
    minimization_basinhopping: bool = False,
):
    """
    Joint refinement of UB matrix and unknown hkl values

    Parameters
    ----------
    unit_cell_parameters: [cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma] in angstrem and radians

    q_vec: (3, N)  Measured Q vectors in lab frame.
    sigma_Q_vec_rot : (N,3) ndarray
        Uncertainties for Q_vec_rot.
    euler_alpha, euler_beta, euler_gamma : float
        Initial Euler angles in radians.
    singony : str
        Crystal system: 'cubic', 'hexagonal', 'tetragonal', 'orthorhombic',
        'monoclinic', or 'triclinic'.
    refine_unit_cell_parameters : bool
        If True, refine unit cell parameters.
    refine_euler_angles : bool
        If True, refine Euler angles.
    constraint_volume : bool
        If True, penalize unit cells with volume above initial max.
    minimization_basinhopping : bool
        If True, run basinhopping before local minimization.

    Returns
    -------
    unit_cell_parameters_opt: (6,) ndarray
        Refined unit cell parameters
    euler_opt : (3,) ndarray
        Refined Euler angles in radians.
    b_matrix : (3,3) ndarray
        Refined reciprocal lattice matrix.
    chi_sq_min : float
        Final chi-square value.
    """

    singony = singony.lower()
    # Initial cell parameter vector
    if singony.startswith('c'):
        l_ind = [0, ]
    elif singony.startswith('h') or singony.startswith('te'):
        l_ind = [0, 2, ]
    elif singony.startswith('o'):
        l_ind = [0, 1, 2, ]
    elif singony.startswith('m'):
        l_ind = [0, 1, 2, 4, ]
    else:
        l_ind = [0, 1, 2, 3, 4, 5,]
    x_cell = [unit_cell_parameters[ind] for ind in l_ind]

    rad90 = numpy.pi * 0.5
    rad120 = numpy.pi * 2. / 3.

    def calc_b_matrix_by_x(x_cell):
        if singony.startswith('c'):
            a = x_cell[0]
            b_matrix = np_cryst_functions.calc_b_matrix(a, a, a, rad90, rad90, rad90)
        elif singony.startswith('h'):
            a, c = x_cell
            b_matrix = np_cryst_functions.calc_b_matrix(a, a, c, rad90, rad90, rad120)
        elif singony.startswith('te'):
            a, c = x_cell
            b_matrix = np_cryst_functions.calc_b_matrix(a, a, c, rad90, rad90, rad90)
        elif singony.startswith('o'):
            a, b, c = x_cell[0],x_cell[1],x_cell[2]
            b_matrix = np_cryst_functions.calc_b_matrix(a, b, c, rad90, rad90, rad90)
        elif singony.startswith('m'):
            a, b, c, beta = x_cell
            b_matrix = np_cryst_functions.calc_b_matrix(a, b, c, rad90, beta, rad90)
        else:
            a, b, c, alpha, beta, gamma = x_cell
            b_matrix = np_cryst_functions.calc_b_matrix(a, b, c, alpha, beta, gamma)
        return b_matrix

    x0 = []
    if refine_euler_angles:
        x0.extend(list(euler_angles))
    if refine_unit_cell_parameters:
        x0.extend(x_cell)

    # Initial B matrix and volume
    b_matrix_init = calc_b_matrix_by_x(x_cell)

    ucp_init = np_cryst_functions.calc_unit_cell_parameters_by_b_matrix(b_matrix_init)
    cell_volume_init = np_cryst_functions.calc_cell_volume(*ucp_init)

    cell_volume_max = 1.2 * cell_volume_init

    def calc_chi_sq(x):
        if refine_euler_angles:
            ea = numpy.array(x[:3], dtype=float)
            i_cell = 3
        else:
            ea = euler_angles
            i_cell = 0
        if refine_unit_cell_parameters:
            b_matrix = calc_b_matrix_by_x(x[i_cell:])
        else:
            b_matrix = np_cryst_functions.calc_b_matrix(*ucp_init)
        u_matrix = np_cryst_functions.calc_orientation_matrix(*ea)
        ub_matrix = u_matrix @ b_matrix
        hkl_int = numpy.linalg.inv(ub_matrix) @ q_vec
        hkl_int = numpy.round(hkl_int, 0)
        q_vec_ref = ub_matrix @ hkl_int
        q_vec_diff = (q_vec_ref - q_vec) / sigma_q_vec
        chi_sq = numpy.square(q_vec_diff).sum()
        if constraint_volume and refine_unit_cell_parameters:
            ucp = np_cryst_functions.calc_unit_cell_parameters_by_b_matrix(b_matrix)
            cell_volume = np_cryst_functions.calc_cell_volume(*ucp)
            if cell_volume > cell_volume_max:
                chi_sq *= 7.0 * (cell_volume / cell_volume_max)
        return chi_sq

    if len(x0) == 0:
        ub_matrix_init = np_cryst_functions.calc_orientation_matrix(*euler_angles) @ b_matrix_init
        return ucp_init, euler_angles, ub_matrix_init, {'fun': calc_chi_sq([]), 'message': 'No refined parameters'}

    if minimization_basinhopping:
        res_bh = scipy.optimize.basinhopping(
            calc_chi_sq, x0, T=10000, niter=100,
            interval=20, stepwise_factor=0.7, disp=True
        )
        x0 = res_bh.x
    res = scipy.optimize.minimize(calc_chi_sq, x0, method='BFGS')

    if refine_euler_angles:
        ea_opt = numpy.array(res.x[:3], dtype=float)
        i_cell = 3
    else:
        ea_opt = euler_angles
        i_cell = 0

    if refine_unit_cell_parameters:
        b_matrix_final = calc_b_matrix_by_x(res.x[i_cell:])
    else:
        b_matrix_final = b_matrix_init
    ucp_final = np_cryst_functions.calc_unit_cell_parameters_by_b_matrix(b_matrix_final)
    ea_opt = ea_opt % (2.0 * numpy.pi)
    ub_matrix_final = np_cryst_functions.calc_orientation_matrix(*ea_opt) @ b_matrix_final
    return ucp_final, ea_opt, ub_matrix_final, res
