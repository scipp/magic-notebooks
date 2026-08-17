import numpy as np
import scipp as sc
import h5py as h5
import shutil
import os

# Read detector data from McStas file
# Note: This processes all events and may take time for large files
import read_h5


def sample_event_indices(
        event_probability: np.ndarray, 
        number_event: int
):
    # Normalize probabilities
    p = np.asarray(event_probability, dtype=float)
    p /= p.sum()

    # Indices of the probability array
    indices = np.arange(len(p))

    # Sample indices according to probabilities
    event_ind = np.random.choice(indices, size=number_event, p=p)
    return event_ind


def replace_dataset(entry, name, values):
    """Replace a dataset in an HDF5 group, preserving attributes."""
    attrs = dict(entry[name].attrs)
    del entry[name]
    dset = entry.create_dataset(name, data=values)
    dset.attrs.update(attrs)


def detector_to_coda_format(
    dg_detector,
    detector_name,
    det_numbers:int,
    number_event: int = 100,
):
    """Convert scipp DataGroup from read_h5 to CODA HDF5 format.

    Parameters
    ----------
    dg_detector: scipp DataGroup containing detector data
    detector_name: 'detector_a' or 'detector_b'

    Returns
    -------
    dict: Dictionary with CODA-compatible data structure

    Keys: 
    - 'event_id',
    - 'toa',
    - 'event_time_zero',
    - 'event_time_offset',
    - 'event_index'.
    """
    result = {}
    # The read_h5 function returns a DataGroup with 'detector_a' or 'detector_b' key
    if detector_name == 'detector_a':
        detector_data = dg_detector['detector_a'] if (
            'detector_a' in dg_detector
        ) else (
            list(dg_detector.values())[0]
        )

    elif detector_name == 'detector_b':
        detector_data = dg_detector['detector_b'] if (
            'detector_b' in dg_detector
        ) else (
            list(dg_detector.values())[0]
        )
    else:
        raise ValueError(f"Unknown detector_name: {detector_name}")
    hh = sc.array(values=det_numbers, dims=('detector_number',))
    detector_data =  detector_data.group(hh)

    # Extract event_id array (this is the main data we need)
    event_probability = detector_data.bins.data.bins.concat().value.values
    event_ind = sample_event_indices(event_probability, number_event,)

    toa = detector_data.bins.coords["toa"].bins.concat().value[event_ind]
    # IMPORTANT! we need to sort the arrays below according to toa,
    # so that the event_index does not get messed up!    
    event_id = sc.sort(
        (
            sc.bins_like(detector_data, sc.array(
                dims=detector_data.dims, values=det_numbers)).bins.concat().value[event_ind]
        ),
        key=toa,
    )

    # Get the shape (number of events)
    n_events = event_id.size
    print(f"  {detector_name}: {n_events} events")

    # Prepare CODA-compatible data
    # The event_id values are the main payload for the CODA detector data
    toa = toa.to(unit='ns')
    result['toa'] = toa
    result['event_id'] = event_id

    unit = toa.unit
    period = (1.0 / sc.scalar(14.0, unit="Hz")).to(unit=unit)
    start = sc.datetime("2026-01-01T12:00:00.000000000")

    event_time_zero = sc.sort(
        (period * (toa.to(unit="ns", copy=False) // period)).to(dtype=int) + start,
        key=toa,
    )

    event_time_offset = sc.sort(toa % period.to(unit=toa.unit), key=toa)
    result['event_time_offset'] = event_time_offset

    event_index = sc.DataArray(
        data=sc.ones_like(event_time_offset),
        coords={"event_time_zero": event_time_zero},
    ).group("event_time_zero")

    event_index = sc.cumsum(event_index.bins.size())
    event_index.values = np.concatenate([[0], event_index.values[:-1]])
    result['event_index'] = event_index
    result['detector_rotation_value'] = detector_data.coords['gamma']
    return result


def mcstas_to_coda(
    mcstas_data_file: str,
    template_coda_file: str,
    outfile: str,
    number_event_detector_a: int = 1000,
    number_event_detector_b: int = 100,
):
    """Store the events from a McStas MAGiC simulation in a CODA HDF5 file.

    Parameters
    ----------
    mcstas_data_file : str
        Data file containing simulated McStas events.
    template_coda_file : str
        CODA HDF5 file containing geometry and instrument info, used as template.
    outfile : str
        Output CODA HDF5 file to be written.
    """
    print(f"Writing {outfile} file")
    # Copy template file and update with detector data
    shutil.copyfile(template_coda_file, outfile)

    # Read detector data from McStas file
    print('Reading detector_a...')
    dg_detector_a = read_h5.read_detector_a_from_nexus(mcstas_data_file)

    # Try to read detector_b (may not be available)
    try:
        print('Reading detector_b...')
        dg_detector_b = read_h5.read_detector_b_from_nexus(mcstas_data_file)
        if len(dg_detector_b.keys()) == 0:
            dg_detector_b = None
    except Exception as e:
        print(f'Could not read detector_b: {e}')
        dg_detector_b = None

    # Convert to CODA format
    print('Converting detector_a to CODA format...')
    det_numbers_a = get_det_numbers(outfile, label_detector='a')
    data_a = detector_to_coda_format(
        dg_detector_a,
        'detector_a',
        det_numbers=det_numbers_a,
        number_event=number_event_detector_a,
    )
    replace_detector_event(outfile, data_a, label_detector='a')

    # If detector_b is available
    data_b = None
    if dg_detector_b is not None:
        print('Converting detector_b to CODA format...')
        det_numbers_b = get_det_numbers(outfile, label_detector='b')
        data_b = detector_to_coda_format(
            dg_detector_b,
            'detector_b',
            det_numbers=det_numbers_b,
            number_event=number_event_detector_b,
        )
        replace_detector_event(outfile, data_b, label_detector='b')


    # Monitor data
    # print('Converting cave monitor to CODA format...')
    # replace_monitor_event(outfile, data_cave_monitor)
    
    # Remove user info
    print('Removing user info...')
    with h5.File(outfile, "r+") as f:
        for subgroup in f['entry']:
            if f[f'entry/{subgroup}'].attrs.get('NX_class') == 'NXuser':
                del f[f'entry/{subgroup}']

    # Show output structure
    print(f'\nOutput structure: ')
    with h5.File(outfile, 'r') as f:
        print("Output structure:")
        for name, obj in f.items():
            print_structure(obj)
    print(f'\nSuccessfully wrote {outfile}')


def shortname(obj):
    return obj.name.split('/')[-1]  # take only the final part


def print_structure(obj, indent=0):
    pad = "  " * indent
    name = shortname(obj)

    if isinstance(obj, h5.Group):
        print(f"{pad}[G] {name}")

        # Print group attributes
        if obj.attrs:
            print(f"{pad}  Attributes:")
            for key, val in obj.attrs.items():
                print(f"{pad}    - {key}: {val}")

        # Recursively print children
        for child in obj.values():
            print_structure(child, indent + 1)

    else:  # Dataset
        print(f"{pad}[D] {name}: shape={obj.shape}, dtype={obj.dtype}")

        # Print dataset attributes
        if obj.attrs:
            print(f"{pad}  Attributes:")
            for key, val in obj.attrs.items():
                print(f"{pad}    - {key}: {val}")


def get_det_numbers(
    f_nexus: str,
    label_detector: str = 'a',
):
    """Get detector numbers from NeXuS file.
    """
    with h5.File(f_nexus, 'r') as f:
        det_group = f[f'entry/instrument/magic_detector_{label_detector:}']
        det_numbers = det_group["detector_number"][()]
    return det_numbers


def replace_detector_event(
    f_nexus: str,
    data_detector: dict,
    label_detector: str = 'a',
):
    """Replace data in NeXuS file by data given in data_detector.
    """

    with h5.File(f_nexus, 'r+') as f:
        det_group = f[f'entry/instrument/magic_detector_{label_detector:}']
        det_event_data = det_group[f'magic_detector_{label_detector:}_event_data']

        replace_dataset(
            det_event_data,
            'event_id',
            data_detector['event_id'].values,
        )
        replace_dataset(
            det_event_data,
            'event_time_offset',
            data_detector['event_time_offset'].values,
        )
        replace_dataset(
            det_event_data,
            'event_time_zero',
            data_detector['event_index'].coords['event_time_zero'].values.astype(int),
        )
        replace_dataset(
            det_event_data,
            'event_index',
            data_detector['event_index'].data.values,
        )
        
        det_group_rotation = f[f'entry/instrument/detector_{label_detector:}_rotation']
        replace_dataset(
            det_group_rotation['value'],
            'value',
            data_detector['detector_rotation_value'].value,
        )
            


def replace_monitor_event(
    f_nexus: str,
    data_detector: dict,
):
    """Replace data in NeXuS file by data given in data_cave_monitor.
    IK: Not tested yet.
    """

    with h5.File(f_nexus, 'r+') as f:
        if monitor_entry_path is not None:
            monitor_data = f[f"{monitor_entry_path}/monitor_3_events"]
            replace_dataset(
                monitor_data, name="event_id", values=np.zeros_like(event_id.values)
            )
            replace_dataset(
                monitor_data,
                name="event_time_offset",
                values=event_time_offset.to(
                    unit=monitor_data["event_time_offset"].attrs["units"], copy=False
                ).values,
            )
            replace_dataset(monitor_data, name="event_index", values=event_index.values)
            replace_dataset(
                monitor_data,
                name="event_time_zero",
                values=event_index.coords["event_time_zero"]
                .to(unit=monitor_data["event_time_zero"].attrs["units"], copy=False)
                .values.astype(int),
            )


if __name__ == '__main__':
    directory_path = "/dmsc/scipp/magic"
    mcstas_data_file = directory_path + "/mccode.h5"
    template_coda_file = '/ess/raw/coda/999999/raw/coda_magic_999999_00014893.hdf'
    outfile = directory_path + "/mccode.nxs"

    # mcstas_data_file = "mccode.h5"
    # template_coda_file = 'coda_magic_999999_00016485.hdf'
    # outfile = "mccode.nxs"
    
    number_event_detector_a = 1000
    number_event_detector_b = 10
    mcstas_to_coda(
        mcstas_data_file=mcstas_data_file,
        template_coda_file=template_coda_file,
        outfile=outfile,
        number_event_detector_a=number_event_detector_a,
        number_event_detector_b=number_event_detector_b,
    )