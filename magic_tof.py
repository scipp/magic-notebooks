import numpy
import scipp as sc
import tof

def model_MAGiC(psc_nu=154, psc_opening_angle=105, wavelength_band_min:float=0.5, pulses:int=1, neutrons:int=1_000_000):
    psc_nu_allowed = [14,28,70,112, 154]
    if not (psc_nu in psc_nu_allowed):
        np_hh = numpy.abs(numpy.array(psc_nu_allowed) - psc_nu)
        psc_nu = psc_nu_allowed[numpy.argmin(np_hh)]
        print(f"PSC_nu is {psc_nu:.2f}")
    
    Hz = sc.Unit('Hz')
    deg = sc.Unit('deg')
    meter = sc.Unit('m')
    detla_t23 = 0.5*2.5 *0.001 #seconds

    source = tof.Source(facility='ess', neutrons=neutrons, pulses=pulses)

    psc1_pos = 6.229
    psc2_pos = 6.244
    sc_pos = 6.735
    bm_bunker_pos_1 = 7.823
    bm_bunker_pos_2 = 7.823
    bc_pos = 79.9
    bm_cave_pos = 157.903 
    # sample_pos = 159.403 
    detector_pos = 160.403 


    psc_slit_a = 8.6
    psc_slit_b = 105
    sc_slit = 20.6
    sc_nu = 14
    bc_slit = 180
    bc_nu = 14


    if psc_opening_angle <= psc_slit_a:
        i_slit = 0
    elif psc_opening_angle <= psc_slit_b:
        i_slit = 1
    else: 
        psc_opening_angle = psc_slit_b
        i_slit = 1
        raise UserWarning(f"PSC opening angle exceeds maximum value. Set to maximum ({psc_slit_b} deg.).")


    sc_phase = wavelength_band_min * sc_pos / 3956 * sc_nu * 360 + 0.5*sc_slit + detla_t23 * sc_nu * 360
    bc_phase = wavelength_band_min * bc_pos / 3956 * bc_nu * 360 + 0.5*bc_slit + detla_t23 * bc_nu * 360

    if i_slit == 0:
        psc_values = 0
        psc1_phase = wavelength_band_min * psc1_pos / 3956 * psc_nu * 360 + 0.5*psc_slit_a + detla_t23 * psc_nu * 360
        psc2_phase = wavelength_band_min * psc2_pos / 3956 * psc_nu * 360 + 0.5*psc_slit_a + detla_t23 * psc_nu * 360
    else:
        psc_values = -90
        psc1_phase = wavelength_band_min * psc1_pos / 3956 * psc_nu * 360 + 0.5*psc_slit_b + detla_t23 * psc_nu * 360
        psc2_phase = wavelength_band_min * psc2_pos / 3956 * psc_nu * 360 + 0.5*psc_slit_b + detla_t23 * psc_nu * 360

    ch_PSC1 = tof.Chopper(
        frequency=psc_nu * Hz,
        open=sc.array(
            dims=['cutout'],
            values=[-0.5*psc_slit_a+psc_values, 90-0.5*psc_slit_b+psc_values],
            unit='deg',
        ),
        close=sc.array(
            dims=['cutout'],
            values=[0.5*psc_slit_a+psc_values,  90+0.5*psc_slit_b+psc_values],
            unit='deg',
        ),
        phase=psc1_phase * deg,
        distance=psc1_pos * meter,
        name="PSC-1",
    )

    ch_PSC2 = tof.Chopper(
        frequency=psc_nu * Hz,
        open=sc.array(
            dims=['cutout'],
            values=[-0.5*psc_slit_a-psc_values, 270-0.5*psc_slit_b-psc_values],
            unit='deg',
        ),
        close=sc.array(
            dims=['cutout'],
            values=[0.5*psc_slit_a-psc_values,  270+0.5*psc_slit_b-psc_values],
            unit='deg',
        ),
        phase=psc2_phase* deg,
        distance=psc2_pos * meter,
        name="PSC-2",
    )

    ch_SC = tof.Chopper(
        frequency=sc_nu * Hz,
        open=sc.array(
            dims=['cutout'],
            values=[-0.5*sc_slit, ],
            unit='deg',
        ),
        close=sc.array(
            dims=['cutout'],
            values=[0.5*sc_slit, ],
            unit='deg',
        ),
        phase=sc_phase * deg,
        distance=sc_pos * meter,
        name="SC",
    )

    ch_BC = tof.Chopper(
        frequency=bc_nu * Hz,
        open=sc.array(
            dims=['cutout'],
            values=[-0.5*bc_slit, ],
            unit='deg',
        ),
        close=sc.array(
            dims=['cutout'],
            values=[0.5*bc_slit, ],
            unit='deg',
        ),
        phase=bc_phase * deg,
        distance=bc_pos * meter,
        name="BC",
    )

    choppers = [ch_PSC1, ch_PSC2, 
                ch_SC, 
                ch_BC
                ]

    detectors = [
        tof.Detector(distance=bm_bunker_pos_1 * meter, name='Bunker BM-1'),
        tof.Detector(distance=bm_bunker_pos_2 * meter, name='Bunker BM-2'),
        tof.Detector(distance=bm_cave_pos * meter, name='Cave BM'),
        tof.Detector(distance=detector_pos * meter, name='Detector'),
    ]

    model = tof.Model(source=source, choppers=choppers, detectors=detectors)
    return model