import scipp as sc
import numpy
import h5py
import voxelization


def read_h5_to_dict(
    f_nexus,
    s_source: str = "_sourceMantid",
    s_sample: str = "_sampleMantid",
    s_detector_a: str = "_arm_detector_a",
    flag_voxelization: bool = True,
):
    d_out = {}
    delta_L_deafault = sc.scalar(0.0, unit="m")
    delta_t_default = sc.scalar(3.0, unit="ms").to(unit="s", copy=False)
    with h5py.File(f_nexus) as fid:
        components = fid["entry1"]["instrument"]["components"]
        l_key_source = [hh for hh in components.keys() if s_source in hh]

        l_key_sample = [hh for hh in components.keys() if s_sample in hh]

        l_key_detector = [hh for hh in components.keys() if s_detector_a in hh]

        l_components = components.keys()

        flag_source = False
        if len(l_key_sample) != 0:
            source = components[l_key_sample[0]]
            source_position = source["Position"][()]
            flag_source = True

        flag_sample = False
        if len(l_key_sample) != 0:
            sample = components[l_key_sample[0]]
            sample_position = sample["Position"][()]
            flag_sample = True

        flag_detector = False
        if len(l_key_detector) != 0:
            detector = components[l_key_detector[0]]
            detector_position = detector["Position"][()]
            detector_rotation = detector["Rotation"][()]
            flag_detector = True

        data = fid["entry1"]["data"]  #
        l_data = data.keys()
        l_component_data_names = take_component_data_names(
            l_components, l_data
        )
        for component_name, data_name in l_component_data_names:
            component = components[component_name]
            data_component = data[data_name]

            d_out[data_name] = read_data_component_to_dict(
                component, data_component
            )
            d_out[data_name]["component_name"] = component_name

        d_components = {}
        for component_name in l_components:
            component = components[component_name]
            d_component = get_type_position_rotation_of_component(component)
            component_name_short = "_".join(component_name.split("_")[1:])
            d_components[component_name_short] = d_component
        d_out["components"] = d_components

        simulation_param = fid["entry1"]["simulation"]["Param"]
        d_simulation_param = {}
        for s_key in simulation_param.keys():
            try:
                d_simulation_param[s_key] = float(
                    simulation_param[s_key][()][0].decode("ascii")
                )
            except:
                pass
        d_out["simulation_parameters"] = d_simulation_param

        sample_omega = d_simulation_param.get("sample_omega", 0.0)
        sample_chi = d_simulation_param.get("sample_chi", 0.0)
        sample_phi = d_simulation_param.get("sample_phi", 0.0)
        gamma_detector_a = d_simulation_param.get("gamma_detector_a", 0.0)
        omega_vs = d_simulation_param.get("omega_casette", 0.0)

        # neutron_up = simulation_param["isFlip"][()]
        l_detector_a_key = [
            key
            for i_key, key in enumerate(l_data)
            if "abs_logger_layers_dat_list"
        ]
        if len(l_detector_a_key) > 0:
            s_key = l_detector_a_key[0]
            data_events = data[s_key]["events"][()]
            if flag_voxelization:
                data_events, np_id, _, _, _, _ = (
                    voxelization.voxelization_of_mcstas_events_for_detector_a(
                        data_events,
                        numpy.radians(omega_vs),
                    )
                )
            l_param = s_key.split("_")[5:]
            ind_p = l_param.index("p")
            ind_x = l_param.index("x")
            ind_z = l_param.index("z")
            ind_t = l_param.index("t")

            da = sc.DataArray(
                data=sc.array(
                    dims=["event"],
                    values=data_events[:, ind_p],
                    variances=(data_events[:, ind_p] ** 2),
                ),
                coords={
                    # "detector_radius": sc.scalar(detector_radius, unit="m"),
                    "delta_L": delta_L_deafault,
                    "delta_t": delta_t_default,
                    "sample_offset": sc.vector([0.0, 0.0, 0.0], unit="m"),
                    "detector_position": sc.vector(
                        detector_position, unit="m"
                    ),
                    "detector_event_position_local_mcstas": sc.vectors(
                        dims=[
                            "event",
                        ],
                        values=data_events[:, ind_x : (ind_z + 1)],
                        unit="m",
                    ),
                    "toa": sc.array(
                        dims=["event"], values=data_events[:, ind_t], unit="s"
                    ),
                    "omega_vs_detector_a": sc.scalar(omega_vs, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                    "gamma_detector_a": sc.scalar(
                        gamma_detector_a, unit="deg."
                    ).to(unit="rad", copy=False),
                    "sample_omega": sc.scalar(sample_omega, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                    "sample_chi": sc.scalar(sample_chi, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                    "sample_phi": sc.scalar(sample_phi, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                },
            )
            if flag_voxelization:
                da.coords["voxel_ID_detector_a"] = sc.array(
                    dims=["event"],
                    values=np_id,
                )

            if "arm_w6" in d_components.keys():
                da.coords["source_position"] = sc.vector(
                    value=d_components["arm_w6"]["position"], unit="m"
                )
            elif flag_source:
                da.coords["source_position"] = sc.vector(
                    value=source_position, unit="m"
                )

            if "arm_egs2" in d_components.keys():
                da.coords["tp_position"] = sc.vector(
                    value=d_components["arm_egs2"]["position"], unit="m"
                )
            if flag_sample:
                da.coords["ideal_sample_position"] = sc.vector(
                    sample_position, unit="m"
                )

            d_out["data_event"] = da
        if "tof_egs2_1_plot" in d_out.keys():  # cave monitor
            data_monitor = d_out["tof_egs2_1_plot"]
            da = sc.DataArray(
                data=sc.array(
                    dims=["counts"],
                    values=data_monitor["data"],
                    variances=(data_monitor["errors"] ** 2),
                ),
                coords={
                    "source_position": sc.vector(
                        value=d_components["arm_w6"]["position"], unit="m"
                    ),
                    "tp_position": sc.vector(
                        value=d_components["arm_egs2"]["position"], unit="m"
                    ),
                    "cave_monitor_position": sc.vector(
                        value=d_components["tof_egs2_1"]["position"], unit="m"
                    ),
                    "delta_L": delta_L_deafault,
                    "delta_t": delta_t_default,
                    "toa": sc.array(
                        dims=["counts"],
                        values=data_monitor["time_of_flight"],
                        unit="micros",
                    ).to(unit="s", copy=False),
                },
            )
            d_out["data_cave_monitor"] = da
    return d_out


def take_component_data_names(l_component, l_data):
    l_res = []
    for component_name in l_component:
        hh = "_".join(component_name.split("_")[1:])
        for data_name in l_data:
            if data_name.startswith(hh):
                l_res.append((component_name, data_name))
    return l_res


def read_data_component_to_dict(component, data_component):
    component_type = get_component_type(component)
    if component_type == "PSD_monitor":
        d_out = read_psd_to_dict(component, data_component)
    elif component_type == "DivPos_monitor":
        d_out = read_divpos_to_dict(component, data_component)
    elif component_type == "TOF_monitor":
        d_out = read_tof_to_dict(component, data_component)
    elif component_type == "Union_abs_logger_nD":
        d_out = read_union_abs_logger_to_dict(component, data_component)
    else:
        d_out = read_common_to_dict(component, data_component)
    return d_out


def read_psd_to_dict(component, data_component):
    d_out = read_common_to_dict(component, data_component)
    if "X_position__cm_" in data_component.keys():
        d_out["X_position"] = data_component["X_position__cm_"][()]
    if "Y_position__cm_" in data_component.keys():
        d_out["Y_position"] = data_component["Y_position__cm_"][()]
    if "X_divergence__deg_" in data_component.keys():
        d_out["X_divergence"] = data_component["X_divergence__deg_"][()]
    if "Y_divergence__deg_" in data_component.keys():
        d_out["Y_divergence"] = data_component["Y_divergence__deg_"][()]
    return d_out


def read_divpos_to_dict(component, data_component):
    d_out = read_common_to_dict(component, data_component)
    d_out["divergence"] = data_component["divergence__deg_"][()]
    d_out["pos"] = data_component["pos__m_"][()]
    return d_out


def read_tof_to_dict(component, data_component):
    d_out = read_common_to_dict(component, data_component)
    d_out["time_of_flight"] = data_component["Time_of_flight___gms_"][()]
    return d_out


def read_union_abs_logger_to_dict(component, data_component):
    """Read info about events."""
    d_out = {}
    d_out["events"] = data_component["events"][()]
    return d_out


def read_common_to_dict(component, data_component):
    values = data_component["data"][()]
    errors = data_component["errors"][()]
    ncount = data_component["ncount"][()]
    d_component = get_type_position_rotation_of_component(component)
    d_out = {
        "data": values,
        "errors": errors,
        "ncount": ncount,
    }
    d_out.update(d_component)
    return d_out


def get_component_type(component):
    component_type = component["Component_type"][()][0].decode("ascii")
    return component_type


def get_type_position_rotation_of_component(component):
    component_type = get_component_type(component)
    position = component["Position"][()]
    rotation = component["Rotation"][()]
    d_out = {
        "component_type": component_type,
        "position": position,
        "rotation": rotation,
    }
    return d_out
