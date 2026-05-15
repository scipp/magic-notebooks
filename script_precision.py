import numpy
import os

# Local libraries
import read_h5
import magic_graphs

import plopp
def calc_aver_std(np_val, np_weight):
    sum_w = np_weight.sum()
    aver_val = (np_val * np_weight).sum() / sum_w
    std_val = numpy.sqrt(
        (numpy.square(np_val - aver_val) * np_weight).sum() / sum_w
    )
    return aver_val, std_val

def calc_flag(np_val, aver_val, std_val):
    return numpy.logical_and(aver_val+3*std_val>np_val, np_val>aver_val-3*std_val)

def calc_flags(np_weight,*np_vals):
    flag_vals = numpy.ones(shape=np_weight.shape, dtype=bool)
    for np_val in np_vals:
        aver_val, std_val = calc_aver_std(np_val, np_weight)
        flag_val = calc_flag(np_val, aver_val, std_val)
        flag_vals = numpy.logical_and(flag_vals, flag_val)
    return flag_vals

def calc_iteration_aver_std(np_weight,*np_vals):
    flag_vals = calc_flags(np_weight,np_vals)
    if flag_vals.sum() >= 0.7*flag_vals.size:
        l_aver, l_std = [],[]
        for np_val in np_vals:
            aver_val, std_val = calc_aver_std(np_val, np_weight)
            l_aver.append(aver_val)
            l_std.append(std_val)
        return l_aver, l_std
    np_weight_2=np_weight[flag_vals]
    l_np_vals_2 = []
    for np_val in np_vals:
        np_vals_2 = np_val[flag_vals]
        l_np_vals_2.append(np_vals_2)
    return calc_iteration_aver_std(np_weight,*l_np_vals_2)




l_name = [
    "data_n2_gamma30_nu0",
    "data_n3_gamma30_nu0",
    "data_n2_gamma30_nu15",
    "data_n3_gamma30_nu15",
    "data_n2_gamma30_nu30m",
    "data_n3_gamma30_nu30m",
]
for name in l_name:
    f_nexus_data = f"../magic_detector_a/{name:}/mccode.h5"
    print(80 * "*")
    print(name)
    print()

    d_out = read_h5.read_h5_to_dict(f_nexus_data, flag_voxelization=True)
    data_event = d_out["data_event"]
    data_event = data_event.transform_coords(
        ("voxel_ID_a_detector_a", "nu_event"), magic_graphs.graph_qvec
    )

    data_event.coords["voxel_ID_c_detector_a"]
    np_nu = numpy.degrees(data_event.coords["nu_event"].values)
    np_gamma = numpy.degrees(data_event.coords["gamma_event"].values)
    np_w = data_event.data.values
    np_xyz = data_event.coords["detector_event_position_local_mcstas"].values
    np_r = numpy.linalg.norm(np_xyz, axis=1)
    np_nu_mcstas = numpy.degrees(numpy.asin(np_xyz[:, 1] / np_r))
    np_gamma_mcstas = numpy.degrees(numpy.atan2(np_xyz[:, 0], np_xyz[:, 2]))
    l_aver, l_std = calc_iteration_aver_std(np_w, np_gamma, np_nu)
    aver_gamma, std_gamma = l_aver[0], l_std[0]
    aver_nu, std_nu = l_aver[1], l_std[1]
    # aver_nu, std_nu = calc_aver_std(np_nu, np_w)
    # aver_gamma, std_gamma = calc_aver_std(np_gamma, np_w)
    print('1')
    print(
        f"Aver. gamma is {aver_gamma:.3f} deg. \nStandard deviation is {std_gamma:.3f} deg.\n"
    )
    print(
        f"Aver. nu is {aver_nu:.3f} deg. \nStandard deviation is {std_nu:.3f} deg.\n"
    )
    print('2')

    flag_gamma = numpy.logical_and(aver_gamma+std_gamma>np_gamma,np_gamma>aver_gamma-std_gamma)
    flag_nu = numpy.logical_and(aver_nu+std_nu>np_nu,np_nu>aver_nu-std_nu)
    flag_event = numpy.logical_and(flag_gamma,flag_nu)
    np_nu=np_nu[flag_event]
    np_gamma=np_gamma[flag_event]
    np_w=np_w[flag_event]
    np_nu_mcstas=np_nu_mcstas[flag_event]
    np_gamma_mcstas=np_gamma_mcstas[flag_event]

    aver_nu, std_nu = calc_aver_std(np_nu, np_w)
    aver_gamma, std_gamma = calc_aver_std(np_gamma, np_w)

    aver_nu_mcstas, std_nu_mcstas = calc_aver_std(np_nu_mcstas, np_w)
    aver_gamma_mcstas, std_gamma_mcstas = calc_aver_std(np_gamma_mcstas, np_w)

    print(
        f"Aver. gamma is {aver_gamma:.3f} deg. \nStandard deviation is {std_gamma:.3f} deg.\n"
    )
    print(
        f"Aver. gamma_mcstas is {aver_gamma_mcstas:.3f} deg. \nStandard deviation is {std_gamma_mcstas:.3f} deg.\n"
    )
    print(
        f"Aver. nu is {aver_nu:.3f} deg. \nStandard deviation is {std_nu:.3f} deg.\n"
    )
    print(
        f"Aver. nu_mcstas is {aver_nu_mcstas:.3f} deg. \nStandard deviation is {std_nu_mcstas:.3f} deg.\n"
    )
    print(80 * "*")