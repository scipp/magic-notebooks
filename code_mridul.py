import numpy as np
import scipp as sc
import scippnexus as snx


def _detector_positions(detector):
    events = next(x for x in detector.values() if isinstance(x, sc.DataArray))
    detector_number = events.coords["detector_number"].to(dtype="int64")
    vertices = detector["pixel_shape"]["vertices"].to(unit="m")

    # not sure about this, just a mean of all 8 vertices
    # local_position = sc.mean(
    #     vertices.fold(
    #         dim="vertex",
    #         sizes={"detector_number": detector_number.size, "corner": 8},
    #     ),
    #     dim="corner",
    # )
    vf = vertices.fold(
        dim="vertex",
        sizes={"detector_number": detector_number.size, "faces": 4, 'side':2},
    )
    lfp = sc.mean(vf, dim='faces')
    hh = sc.flatten(lfp, dims=('detector_number', 'side'), to='detector_number')
    local_position = hh[2*np.arange(hh.size//2) + np.arange(hh.size//2)%2]

    transform = detector["depends_on"].compute()

    # mccode.nxs contains one static rotation value stored as an NXlog.
    # IK: it is correct only if detector is not moving during data collection,
    #     otherwise it should be outside of this function
    transform = transform["time", 0].data

    result = sc.DataArray(
        transform * local_position,
        coords={"detector_number": detector_number},
    )
    return result


def event_positions(nexus, event_ids):
    ids = sc.array(
        dims=["event"],
        values=event_ids,
        dtype="int64",
        unit=None,
    )
    instrument = nexus["entry"]["instrument"]

    positions = sc.concat(
        [
            _detector_positions(instrument["magic_detector_a"]),
            _detector_positions(instrument["magic_detector_b"]),
        ],
        dim="detector_number",
    )

    detector_number = positions.coords["detector_number"]

    return sc.DataArray(
        sc.lookup(positions, "detector_number", mode="previous")(ids),
        coords={"event_id": ids},
    )
