import numpy
import scipy
from scippneutron.conversion import graph
import scipp as sc
from voxelization import calc_local_voxel_position_by_id_of_detector_a, DetectorA

def get_sc_rotation_matrix(r_matrix):
    quternion_r_matrix = scipy.spatial.transform.Rotation.from_matrix(r_matrix).as_quat()
    r_out = sc.spatial.rotation(value=quternion_r_matrix / numpy.linalg.norm(quternion_r_matrix))
    return r_out 
    
def calc_incident_beam_magic(source_position, tp_position, sample_position):
    tp = tp_position.to(unit='m', copy=False)
    v1 = sample_position.to(unit='m', copy=False) - tp
    v2 = tp - source_position.to(unit='m', copy=False)
    e1 = v1/sc.norm(v1)
    incident_beam = v1 + e1 * sc.norm(v2)
    return incident_beam

def calc_event_position_global(position, event_position_local, gamma):

    zero_o = sc.sin(sc.zeros_like(gamma))
    one_o = sc.cos(sc.zeros_like(gamma))
    m_gamma = [
        [sc.cos(gamma), zero_o, sc.sin(gamma)],
        [zero_o, one_o, zero_o],
        [-sc.sin(gamma), zero_o, sc.cos(gamma)],
    ]
    rotation_gamma = get_sc_rotation_matrix(m_gamma)
    position_rotated = rotation_gamma * event_position_local.to(unit='m', copy=False)
    event_position_global = position.to(unit='m', copy=False) + position_rotated
    return event_position_global


def calc_tof(toa, delta_t=sc.scalar(value=3, unit="ms")):
    tof = toa.to(unit='s', copy=False) - delta_t.to(unit='s', copy=False)
    return tof

def calc_Ltotal(L1, L2, delta_L=sc.scalar(value=0, unit="m")):
    Ltotal = L1.to(unit='m', copy=False) + L2.to(unit='m', copy=False) - delta_L.to(unit='m', copy=False)
    return Ltotal

def calc_sample_rotation(sample_omega, sample_chi, sample_phi):
    zero_o = sc.sin(sc.zeros_like(sample_omega))
    one_o = sc.cos(sc.zeros_like(sample_omega))
    m_omega = [
        [sc.cos(sample_omega), zero_o, sc.sin(sample_omega)],
        [zero_o, one_o, zero_o],
        [-sc.sin(sample_omega), zero_o, sc.cos(sample_omega)],
    ]
    zero_c = sc.sin(sc.zeros_like(sample_chi))
    one_c = sc.cos(sc.zeros_like(sample_chi))
    m_chi = [
        [sc.cos(sample_chi), -sc.sin(sample_chi), zero_c],
        [sc.sin(sample_chi), sc.cos(sample_chi), zero_c],
        [zero_c, zero_c, one_c],
    ]

    zero_p = sc.sin(sc.zeros_like(sample_phi))
    one_p = sc.cos(sc.zeros_like(sample_phi))
    m_phi = [
        [sc.cos(sample_phi), zero_p, sc.sin(sample_phi)],
        [zero_p, one_p, zero_p],
        [-sc.sin(sample_phi), zero_p, sc.cos(sample_phi)],
    ]
    rm_o = get_sc_rotation_matrix(m_omega)
    rm_c = get_sc_rotation_matrix(m_chi)
    rm_p = get_sc_rotation_matrix(m_phi)
    sample_rotation = rm_o * rm_c * rm_p
    return sample_rotation

def calc_orientation_matrix(euler_alpha, euler_beta, euler_gamma, ):
    ca, cb, cg = sc.cos(euler_alpha), sc.cos(euler_beta), sc.cos(euler_gamma)
    sa, sb, sg = sc.sin(euler_alpha), sc.sin(euler_beta), sc.sin(euler_gamma)
    m_m = [
        [ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg],
        [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg],
        [-sb, cb*sg, cb*cg],
    ]
    orientation_matrix = get_sc_rotation_matrix(m_m)
    return orientation_matrix

def _calc_cell_phi(cell_alpha, cell_beta, cell_gamma):
    ca, cb, cg = sc.cos(cell_alpha), sc.cos(cell_beta), sc.cos(cell_gamma)
    cell_phi = sc.sqrt(1. - ca*ca - cb*cb - cg*cg + 2 * ca * cb * cg)
    return cell_phi

def calc_cell_volume(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma):
    cell_phi = _calc_cell_phi(cell_alpha, cell_beta, cell_gamma)
    cell_volume = cell_a.to(unit="angstrom") * cell_b.to(unit="angstrom") * cell_c.to(unit="angstrom") * cell_phi
    return cell_volume

def calc_b_matrix(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma):
    cell_phi = _calc_cell_phi(cell_alpha, cell_beta, cell_gamma)
    a, b, c = cell_a.to(unit="angstrom").value, cell_b.to(unit="angstrom").value, cell_c.to(unit="angstrom").value
    b_11 = sc.sin(cell_alpha)/(a*cell_phi)
    b_12 = (sc.cos(cell_alpha)*sc.cos(cell_beta)-sc.cos(cell_gamma))/(b*cell_phi*sc.sin(cell_alpha))
    b_13 = (sc.cos(cell_alpha)*sc.cos(cell_gamma)-sc.cos(cell_beta))/(c*cell_phi*sc.sin(cell_alpha))
    b_22 = 1/(b*sc.sin(cell_alpha))
    b_23 = -sc.cos(cell_alpha)/(c*sc.sin(cell_alpha))
    b_33 = 1/c
    zero = 0.
    b_matrix = sc.spatial.linear_transform(
        value=[
            [b_11, b_12, b_13],
            [zero, b_22, b_23],
            [zero, zero, b_33],
        ],
        unit='1/angstrom',
    )
    return b_matrix

def calc_h_reduced(h):
    h_reduced = h%1
    h_reduced.values[h_reduced.values > 0.5] -= 1.
    return h_reduced

def calc_k_reduced(k):
    k_reduced = calc_h_reduced(k)
    return k_reduced

def calc_l_reduced(l):
    l_reduced = calc_h_reduced(l)
    return l_reduced

def Q_vec_rot_from_elastic_Q_vec(*, Q_vec: sc.Variable, sample_rotation: sc.Variable) -> sc.Variable:
    return (sc.spatial.inv(sample_rotation) * Q_vec)

def hkl_vec_from_elastic_Q_vec_rot(*, Q_vec_rot: sc.Variable, ub_matrix: sc.Variable) -> sc.Variable:
    return (sc.spatial.inv(ub_matrix) * Q_vec_rot) / (2 * numpy.pi)

def Q_vec_rot_from_elastic_hkl_vec(*, hkl_vec: sc.Variable, ub_matrix: sc.Variable) -> sc.Variable:
    return (ub_matrix * hkl_vec * 2 * numpy.pi)



def calc_sample_position(ideal_sample_position, sample_offset):
    sample_position = ideal_sample_position.to(unit='m', copy=False) + sample_offset.to(unit='m', copy=False)
    return sample_position

def calc_unit_cell_parameters_by_b_matrix(b_matrix):
    np_b = b_matrix.to(unit="1/Angstrom").values
    abc_inv = numpy.sqrt(numpy.square(np_b).sum(axis=0))
    cos_abg_inv = (numpy.roll(np_b,shift=-1,axis=1)*numpy.roll(np_b,shift=-2,axis=1)).sum(axis=0)/(numpy.roll(abc_inv, shift=-1,axis=0)*numpy.roll(abc_inv, shift=-2,axis=0))
    sin_abg_inv = numpy.sqrt(1.-numpy.square(cos_abg_inv))
    phi_inv = numpy.sqrt(1.-numpy.square(cos_abg_inv).sum()+2.*cos_abg_inv.prod())
    vol_inv = abc_inv.prod()*phi_inv
    vol = 1./vol_inv
    abc = numpy.roll(abc_inv, shift=-1,axis=0)*numpy.roll(abc_inv, shift=-2,axis=0)*sin_abg_inv/vol_inv
    sin_abg = phi_inv/(numpy.roll(sin_abg_inv, shift=-1,axis=0)*numpy.roll(sin_abg_inv, shift=-2,axis=0))
    abg=numpy.degrees(numpy.asin(sin_abg))
    cell_a = sc.scalar(abc[0], unit="Angstrom")
    cell_b = sc.scalar(abc[1], unit="Angstrom")
    cell_c = sc.scalar(abc[2], unit="Angstrom")
    cell_alpha = sc.scalar(abg[0], unit="deg")
    cell_beta = sc.scalar(abg[1], unit="deg")
    cell_gamma = sc.scalar(abg[2], unit="deg")
    return (cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)

def calc_sin_theta(two_theta):
    return sc.sin(0.5*two_theta)

def calc_d_space(sin_theta, wavelength):
    return 0.5*wavelength/sin_theta

def calc_norm_q(Q_vec):
    return sc.norm(Q_vec)

def calc_voxel_id_vsac(detector_number, N_vs, N_a, N_c, ID_0):
    np_id = detector_number.values
    
    det = DetectorA(
        N_vs = N_vs.value,
        N_a = N_a.value,
        N_c = N_c.value,
        r_d = 0,
        r_vs = 0,
        a_t = 0,
        b_t = 0,
        a_b = 0,
        b_b = 0,
        omega_vs = 0,
        delta_gamma_vs = 0,
        gamma_d = 0,
        n_id_0 = ID_0.value,
    )
    n_vs, n_a, n_c = det._calc_n_vsac_by_id(np_id)
    voxel_ID_VS = sc.array(
                        dims=detector_number.dims,
                        values=n_vs,
                    )
    voxel_ID_a = sc.array(
                        dims=detector_number.dims,
                        values=n_a,
                    )
    voxel_ID_c = sc.array(
                        dims=detector_number.dims,
                        values=n_c,
                    )
    return {'voxel_ID_VS':voxel_ID_VS, 'voxel_ID_a':voxel_ID_a, 'voxel_ID_c':voxel_ID_c}
    

def calc_event_position_local_by_pixel_id(voxel_ID_VS, voxel_ID_a, voxel_ID_c, N_vs, N_a, N_c, r_d, r_vs, a_t,b_t, a_b, b_b, casette_omega, casette_delta_gamma, ID_0):
    np_vs = voxel_ID_VS.values
    np_a = voxel_ID_a.values
    np_c = voxel_ID_c.values
    det = DetectorA(
        N_vs = N_vs.value,
        N_a = N_a.value,
        N_c = N_c.value,
        r_d = r_d.to(unit='m').value,
        r_vs = r_vs.to(unit='m').value,
        a_t = a_t.to(unit='m').value,
        b_t = b_t.to(unit='m').value,
        a_b = a_b.to(unit='m').value,
        b_b = b_b.to(unit='m').value,
        omega_vs = casette_omega.to(unit='rad', copy=False).value,
        delta_gamma_vs = casette_delta_gamma.to(unit='rad', copy=False).value,
        gamma_d = numpy.radians(0.),
        n_id_0 = ID_0.value,
    )
    np_xyz = det._calc_xyz_by_n_vsac(np_vs, np_a, np_c)
    # np_xyz = det.calc_xyz_by_id(np_id)
    event_position_local = sc.vectors(
                        dims=voxel_ID_VS.dims,
                        values=np_xyz.transpose(),
                        unit="m",
                    )
    return event_position_local

def calc_gamma_nu_event(event_position_local, gamma):
    np_xyz = event_position_local.values.transpose()
    np_norm = numpy.linalg.norm(np_xyz,axis=0)
    np_gamma = numpy.atan2(np_xyz[0], np_xyz[2]) + gamma.to(unit="rad").value
    np_nu = numpy.asin(np_xyz[1]/np_norm)
    event_gamma = sc.array(
        dims=event_position_local.dims, 
        values=np_gamma, 
        unit='rad')

    event_nu = sc.array(
        dims=event_position_local.dims, 
        values=np_nu, 
        unit='rad')
    return {"event_gamma":event_gamma, "event_nu":event_nu, }
    
scipp_graph = {**graph.beamline.beamline(scatter=True), **graph.tof.elastic_hkl(start='tof')}

def calc_scattered_beam(sample_position, event_position_global):
    return scipp_graph["scattered_beam"](position=event_position_global, sample_position=sample_position)

graph_detector = {
    "event_position_global": calc_event_position_global,
    ("voxel_ID_VS", "voxel_ID_a", "voxel_ID_c"):calc_voxel_id_vsac,
    ("event_gamma", "event_nu"): calc_gamma_nu_event,
    "event_position_local": calc_event_position_local_by_pixel_id,
    
}

graph_qvec = {
    "incident_beam": calc_incident_beam_magic,
    "tof": calc_tof,
    "Ltotal": calc_Ltotal,
    "sample_rotation": calc_sample_rotation,
    "Q_vec_rot": Q_vec_rot_from_elastic_Q_vec,
    "L1": scipp_graph["L1"],
    "L2": scipp_graph["L2"],
    "scattered_beam": calc_scattered_beam,
    "Q_vec": scipp_graph["Q_vec"],
    "two_theta": scipp_graph["two_theta"],
    "sin_theta": calc_sin_theta,
    "d_space": calc_d_space,
    "norm_Q": calc_norm_q,
    "wavelength": scipp_graph["wavelength"],
    ("Qx","Qy","Qz"): scipp_graph[("Qx","Qy","Qz")],
    "sample_position": calc_sample_position,
}
graph_hkl = {

    "cell_volume": calc_cell_volume,
    "b_matrix": calc_b_matrix,
    "u_matrix": calc_orientation_matrix,
    "h_reduced": calc_h_reduced,
    "k_reduced": calc_k_reduced,
    "l_reduced": calc_l_reduced,
    "hkl_vec": hkl_vec_from_elastic_Q_vec_rot,
    "ub_matrix": scipp_graph["ub_matrix"],
    ("h","k","l"): scipp_graph[("h","k","l")],
}

graph_hkl_inv = {
    "Q_vec_rot": Q_vec_rot_from_elastic_hkl_vec,
    "b_matrix": graph_hkl["b_matrix"],
    "u_matrix": graph_hkl["u_matrix"],
    "ub_matrix": graph_hkl["ub_matrix"],
}

graph_ub_inv = {
    ("cell_a","cell_b","cell_c","cell_alpha","cell_beta","cell_gamma"): calc_unit_cell_parameters_by_b_matrix,
}

def calc_incident_beam_cave_monitor(source_position, tp_position, cave_monitor_position):
    incident_beam = graph_qvec["incident_beam"](source_position, tp_position, cave_monitor_position)
    return incident_beam

def calc_wavelength(L1, delta_L, tof):
    Ltotal = L1.to(unit='m', copy=False) - delta_L.to(unit='m', copy=False)
    wavelength = graph_qvec["wavelength"](Ltotal=Ltotal, tof=tof)
    return wavelength


graph_cave_monitor = {
    "tof": graph_qvec["tof"],
    "L1": graph_qvec["L1"],
    "incident_beam": calc_incident_beam_cave_monitor,
    "wavelength": calc_wavelength,
}