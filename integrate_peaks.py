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
    if "detector_border" in da.masks.keys():
        flag_not_border = ~da.masks["detector_border"]
        np_hkl = numpy.array(
            [
                da[flag_not_border].coords["h"].values,
                da[flag_not_border].coords["k"].values,
                da[flag_not_border].coords["l"].values,
            ],
            dtype=float,
        )
    else:
        np_hkl = numpy.array(
            [
            da.coords["h"].values,
            da.coords["k"].values,
            da.coords["l"].values,
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
        l_fsq_exp.append(fsq_exp)
        l_hkl.append(hkl)
        l_wavelength.append((wavelength, numpy.std(np_wavelength)))
        l_tth.append((numpy.degrees(tth), numpy.degrees(numpy.std(np_tth))))
    np_fsq_exp = numpy.array(l_fsq_exp)  # *scale_new/scale
    np_wavelength = numpy.array(l_wavelength, dtype=float)

    np_tth = numpy.array(l_tth, dtype=float)
    np_hkl_int = numpy.array(l_hkl, dtype=int).transpose()
    return np_hkl_int, np_fsq_exp, np_wavelength, np_tth
