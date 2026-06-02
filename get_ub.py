import numpy
import scipy.optimize
import magic_graphs
import magic_scipp
import scipp as sc

def get_euleur_opt(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, 
        Q_vec_rot, sigma_Q_vec_rot,
        euler_alpha, euler_beta, euler_gamma, graph_hkl=magic_graphs.graph_hkl,
        relfine_unit_cell=False,singony='triclinic'):
    """
    Joint refinement of UB matrix and unknown hkl values.

    Parameters
    ----------
    B : (3,3) array
        Reciprocal lattice matrix.
    q_list : (N,3) array
        Measured Q vectors.
    hkl_init : (N,3) array or None
        Initial guess for hkl. If None, use fractional guess from B^-1 Q.

    Returns
    -------
    UB : (3,3) array
        Refined UB matrix.
    hkl : (N,3) array
        Refined fractional hkl values.
    """
    # np_weight = numpy.asarray(weight) # values
    # np_weight = np_weight / numpy.max(np_weight)

    ea_rad = euler_alpha.to(unit="rad", copy=False).value
    eb_rad = euler_beta.to(unit="rad", copy=False).value
    eg_rad = euler_gamma.to(unit="rad", copy=False).value


    h90 = sc.scalar(90., unit="deg")
    h120 = sc.scalar(120., unit="deg")

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

    if not relfine_unit_cell:
        sc_b_matrix = graph_hkl['b_matrix'](cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
        x0 = [ea_rad,eb_rad,eg_rad]
    elif singony.startswith('c'):
        x0 = [ea_rad,eb_rad,eg_rad, cell_a_ang,]
    elif singony.startswith('h') or singony.startswith('te'):
        x0 = [ea_rad,eb_rad,eg_rad, cell_a_ang,cell_c_ang]
    elif singony.startswith('o'):
        x0 = [ea_rad,eb_rad,eg_rad, cell_a_ang,cell_b_ang, cell_c_ang]
    elif singony.startswith('m'):
        x0 = [ea_rad,eb_rad,eg_rad, cell_a_ang,cell_b_ang, cell_c_ang, cell_beta_deg]
    else:
        x0 = [ea_rad,eb_rad,eg_rad, cell_a_ang, cell_b_ang, cell_c_ang, cell_alpha_deg, cell_beta_deg, cell_gamma_deg]
        

    def calc_chi_sq(x):
        euleur_angles = x[:3]
        if relfine_unit_cell:
            sc_b_matrix = calc_b_matrix_by_x(x[3:])
        else:
            sc_b_matrix = graph_hkl['b_matrix'](cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
        sc_u = magic_graphs.graph_hkl_inv["u_matrix"](
            sc.scalar(euleur_angles[0], unit="rad"),
            sc.scalar(euleur_angles[1], unit="rad"),
            sc.scalar(euleur_angles[2], unit="rad"),
            )
        sc_UB = graph_hkl["ub_matrix"](u_matrix=sc_u, b_matrix=sc_b_matrix)
        sc_hkl_int = graph_hkl["hkl_vec"](ub_matrix=sc_UB, Q_vec_rot=Q_vec_rot)
        sc_hkl_int.values = numpy.round(sc_hkl_int.values,0) 
        Q_vec_rot_ref = magic_graphs.graph_hkl_inv["Q_vec_rot"](ub_matrix=sc_UB, hkl_vec=sc_hkl_int)
        Q_vec_rot_diff = (Q_vec_rot_ref - Q_vec_rot).values/sigma_Q_vec_rot.values
        chi_sq = (numpy.square(Q_vec_rot_diff)).sum() # * numpy.expand_dims(np_weight, axis=1)
        return chi_sq


    res = scipy.optimize.minimize(calc_chi_sq, x0, method='BFGS')# Nelder-Mead
    # res = scipy.optimize.basinhopping(calc_chi_sq, x0)
    ea_opt = res.x[:3]
    if relfine_unit_cell:
        sc_b_matrix =calc_b_matrix_by_x(res.x[3:])
    else:
        ea_opt = res.x[:3]%(2.*numpy.pi)
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