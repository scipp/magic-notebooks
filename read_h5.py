import scipp as sc
import numpy
import h5py
import voxelization


def read_h5_to_dict(f_nexus):
    d_out = {}
    delta_L_deafault = sc.scalar(0.0, unit="m")
    delta_t_default = sc.scalar(3.0, unit="ms").to(unit="s", copy=False)
    with h5py.File(f_nexus) as fid:
        components = fid["entry1"]["instrument"]["components"]
        l_key_sample = [
            hh for hh in components.keys() if "_sampleMantid" in hh
        ]
        l_key_source = [
            hh for hh in components.keys() if "_sourceMantid" in hh
        ]
        l_key_detector = [
            hh for hh in components.keys() if "_arm_detector_a" in hh
        ]

        l_components = components.keys()

        sample = components[l_key_sample[0]]
        sample_position = sample["Position"][()]

        detector = components[l_key_detector[0]]
        detector_position = detector["Position"][()]
        detector_rotation = detector["Rotation"][()]
        # detector_radius = float(detector["Geometry"].attrs["radius"])

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
        sample_omega = float(
            simulation_param["sample_rotation_y"][()][0].decode("ascii")
        )
        gamma_detector_a = float(
            simulation_param["detector_a_gamma"][()][0].decode("ascii")
        )
        omega_vs = float(
            simulation_param["omega_casette"][()][0].decode("ascii")
        )
        neutron_up = simulation_param["isFlip"][()]

        if "abs_logger_layers_dat_list_p_x_y_z_vx_vy_vz_t" in l_data:
            data_events = data[
                "abs_logger_layers_dat_list_p_x_y_z_vx_vy_vz_t"
            ]["events"][()]

            data_events, np_id, np_xyz_voxel = (
                voxelization.voxelization_of_mcstas_events_for_detector_a(
                    data_events,
                    numpy.radians(omega_vs),
                    numpy.radians(gamma_detector_a),
                )
            )
            da = sc.DataArray(
                data=sc.array(
                    dims=["event"],
                    values=data_events[:, 0],
                    variances=(data_events[:, 0] ** 2),
                ),
                coords={
                    # "detector_radius": sc.scalar(detector_radius, unit="m"),
                    "delta_L": delta_L_deafault,
                    "delta_t": delta_t_default,
                    "source_position": sc.vector(
                        value=d_components["arm_w6"]["position"], unit="m"
                    ),
                    "tp_position": sc.vector(
                        value=d_components["arm_egs2"]["position"], unit="m"
                    ),
                    "ideal_sample_position": sc.vector(
                        sample_position, unit="m"
                    ),
                    "sample_offset": sc.vector([0.0, 0.0, 0.0], unit="m"),
                    "detector_position": sc.vector(
                        detector_position, unit="m"
                    ),
                    "detector_event_position_local": sc.vectors(
                        dims=[
                            "event",
                        ],
                        values=np_xyz_voxel,
                        unit="m",
                    ),
                    "detector_event_position_local_mcstas": sc.vectors(
                        dims=[
                            "event",
                        ],
                        values=data_events[:, 1:4],
                        unit="m",
                    ),
                    "velocity_local": sc.vectors(
                        dims=[
                            "event",
                        ],
                        values=data_events[:, 4:7],
                        unit="m/s",
                    ),
                    "toa": sc.array(
                        dims=["event"], values=data_events[:, 7], unit="s"
                    ),
                    "voxel_ID_detector_a": sc.array(
                        dims=["event"],
                        values=np_id,
                    ),
                    "gamma_detector_a": sc.scalar(
                        gamma_detector_a, unit="deg."
                    ).to(unit="rad", copy=False),
                    "sample_omega": sc.scalar(sample_omega, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                    "sample_chi": sc.scalar(0.0, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                    "sample_phi": sc.scalar(0.0, unit="deg.").to(
                        unit="rad", copy=False
                    ),
                },
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
    d_out["X_position"] = data_component["X_position__cm_"][()]
    d_out["Y_position"] = data_component["Y_position__cm_"][()]
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
