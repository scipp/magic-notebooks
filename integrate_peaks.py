import numpy
import scipp as sc
import magic_graphs


def naive_integration(da, integration_box, scale: float = 1.0):
    ls_out = ["# Naive integration\n"]
    da = da.transform_coords(
        ("two_theta", "h", "k", "l"),
        graph={
            **magic_graphs.graph_detector,
            **magic_graphs.graph_qvec,
            **magic_graphs.graph_hkl,
        },
    )
    flag_not_border = ~da.masks["detector_border"]
    np_hkl = numpy.array(
        [
            da[flag_not_border].coords["h"].values,
            da[flag_not_border].coords["k"].values,
            da[flag_not_border].coords["l"].values,
        ],
        dtype=float,
    )
    np_hkl = numpy.unique(numpy.round(np_hkl, 0).astype(int), axis=1)

    l_fsq_exp = []
    l_fsq_mod = []
    l_hkl = []
    l_ratio = []
    l_wavelength = []
    l_tth = []
    N_hkl = np_hkl.shape[1]
    i_hkl = 0
    for hkl in np_hkl.transpose():
        i_hkl += 1
        print(f"Progress: {100*i_hkl/N_hkl:.2f}", end="\r")
        flag_h = sc.abs(da.coords["h"] - hkl[0]) < integration_box[0]
        flag_k = sc.abs(da.coords["k"] - hkl[1]) < integration_box[1]
        flag_l = sc.abs(da.coords["l"] - hkl[2]) < integration_box[2]
        flag_hkl = sc.logical_and(flag_h, sc.logical_and(flag_k, flag_l))
        da_one_hkl = da[flag_hkl]
        np_wavelength = da_one_hkl.coords["wavelength"].values
        wavelength = numpy.mean(np_wavelength)
        np_tth = da_one_hkl.coords["two_theta"].values
        tth = numpy.mean(np_tth)
        sin_sq = numpy.square(numpy.sin(0.5 * tth))
        val = sc.sum(da_one_hkl.data)
        iint = val.value
        siint = numpy.sqrt(val.variance)
        fsq_exp = scale * iint * sin_sq / (numpy.power(wavelength, 4))
        # flag_hkl_model = (np_hkl_model == numpy.expand_dims(hkl,axis=1)).prod(axis=0).astype(bool)
        # if flag_hkl_model.sum() != 1:
        #     print("Reflection ", hkl, " is not found")
        # else:
        #    fsq_mod = np_fsq_model[flag_hkl_model][0]
        #    ratio = fsq_mod/fsq_exp
        l_fsq_exp.append(fsq_exp)
        #    l_fsq_mod.append(fsq_mod)
        l_hkl.append(hkl)
        #    l_ratio.append(ratio)
        l_wavelength.append((wavelength, numpy.std(np_wavelength)))
        l_tth.append((numpy.degrees(tth), numpy.degrees(numpy.std(np_tth))))
    # scale_new = scale * numpy.median(l_ratio)
    # print(f"Scale should be around {scale_new:}")
    np_fsq_exp = numpy.array(l_fsq_exp)  # *scale_new/scale
    # np_fsq_mod = numpy.array(l_fsq_mod, dtype=float)
    np_wavelength = numpy.array(l_wavelength, dtype=float)

    np_tth = numpy.array(l_tth, dtype=float)
    np_hkl_int = numpy.array(l_hkl, dtype=int).transpose()
    # np_diff_fsq_norm = numpy.abs((np_fsq_exp-np_fsq_mod)/np_fsq_mod)
    # ls_out.append("  h   k   l Fsq_exp Fsq_mod      RF2 Wavelength  STD    2Theta       STD")
    # ls_out_bad = []
    # for hkl, hh1, hh2, hh3, hh4, hh5 in zip(np_hkl_int.transpose(), np_fsq_exp, np_fsq_mod, np_diff_fsq_norm, np_wavelength, np_tth):
    #     s_common = f"{hkl[0]:3} {hkl[1]:3} {hkl[2]:3} {hh1:7.2f} {hh2:7.2f} {hh3*100:7.2f}% {hh4[0]:7.5f} {hh4[1]:7.5f} {hh5[0]:9.2f} {hh5[1]:9.2f}"
    #     if hh3 > 0.1:
    #         ls_out_bad.append(s_common+"  ! Bad integration")
    #     else:
    #         ls_out.append(s_common)
    # ls_out.extend(ls_out_bad)
    # rf2 = np_diff_fsq_norm.mean()*100
    # rf2_without_bad = np_diff_fsq_norm[np_diff_fsq_norm<=0.1].mean()*100
    # ls_out.append(f"\n Agreement factor RF2 is {rf2:.2f}% (all)")
    # ls_out.append(f"                         {rf2_without_bad:.2f}% (without bad)")
    # print("\n".join(ls_out))
    return np_hkl_int, np_fsq_exp, np_wavelength, np_tth
