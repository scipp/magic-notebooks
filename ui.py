# ui.py
import os
from ipyfilechooser import FileChooser
import ipywidgets as widgets

dir_path_data = "/Users/iuriikibalin/Downloads/sim_c60"

if not os.path.isdir(dir_path_data):
    dir_path_data = "/"

fc = FileChooser(
    # '/Users/iuriikibalin/Downloads/C60_n3',
    dir_path_data,
    title="Select NEXuS file with Single crystal diffraction data",
    select_default=False,
)

dir_path_vanadium = "/Users/iuriikibalin/Downloads/sim_vanadium"
if not os.path.isdir(dir_path_vanadium):
    dir_path_vanadium = "/"

fc_vanadium = FileChooser(
    dir_path_vanadium,
    title="Select NEXuS file with Vanadium measurements for normatlization (optional)",
    select_default=False,
)


load_events_button = widgets.Button(description="Load", button_style="success")

display_button = widgets.Button(
    description="Display",
    button_style="info",
    layout=widgets.Layout(display="none"),  # hidden initially
)

hide_output_data_button = widgets.Button(
    description="Hide",
    button_style="info",
    layout=widgets.Layout(display="none"),  # hidden initially
)


state_dropdown = widgets.Dropdown(
    options=[
        "magic_data",
        "detector_a_event",
        "detector_a_event_normalized",
        "detector_a_event_hist",
        "detector_a_event_hist_normalized",
        "detector_b_event",
        "detector_b_event_normalized",
        "detector_b_event_hist",
        "detector_b_event_hist_normalized",
        "monitor_cave",
        "da_peaks",
    ],
    value="magic_data",
    description="",
    disabled=False,
    layout=widgets.Layout(display="none"),  # hidden initially
)

display_da_button = widgets.Button(
    description="Show data",
    button_style="info",
    layout=widgets.Layout(display="none"),  # hidden initially
)

display_peaks_button = widgets.Button(
    description="Display",
    button_style="info",
    layout=widgets.Layout(display="none"),  # hidden initially
)

hide_peaks_button = widgets.Button(
    description="Hide",
    button_style="info",
    layout=widgets.Layout(display="none"),  # hidden initially
)


# find_peaks_button = widgets.Button(
#     description="Find Peaks",
#     button_style="success",
#     layout=widgets.Layout(display="none"),  # hidden initially
# )

find_peaks_hist_button = widgets.Button(
    description="Find Peaks",
    button_style="success",
    # layout=widgets.Layout(display="none"),  # hidden initially
)


fig_rbuttons = widgets.RadioButtons(
    options=[
        "3D Laue Pattern",
        "2D Pattern",
        "Monitor Data",
        "3D Normalization Data",
        "2D Normalization Data",
    ],
    value="3D Laue Pattern",  # Defaults to 'pineapple'
    #    layout={'width': 'max-content'}, # If the items' names are long
    description="Data to display:",
    disabled=False,
    layout=widgets.Layout(display="none"),  # hidden initially
)


norm_rbuttons = widgets.ToggleButtons(
    options=["No Normalization", "Per Vanadium"],  # "Per Monitor",
    description="Choose Normalization Condition",
    disabled=False,
    button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=["No normalization", "Per Vanadium"],  # "Per Monitor",
    # layout=widgets.Layout(display="none"),  # hidden initially
    #     icons=['check'] * 3
)

cell_a = widgets.BoundedFloatText(
    value=14.04078,
    min=0,
    max=60.0,
    step=0.0001,
    description="a:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)
cell_b = widgets.BoundedFloatText(
    value=14.04078,
    min=0,
    max=60.0,
    step=0.0001,
    description="b:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)
cell_c = widgets.BoundedFloatText(
    value=14.04078,
    min=0,
    max=60.0,
    step=0.0001,
    description="c:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)
cell_alpha = widgets.BoundedFloatText(
    value=90,
    min=0.1,
    max=179.9,
    step=0.01,
    description="Alpha:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)
cell_beta = widgets.BoundedFloatText(
    value=90,
    min=0.1,
    max=179.9,
    step=0.0001,
    description="Beta:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)
cell_gamma = widgets.BoundedFloatText(
    value=90,
    min=0.1,
    max=179.9,
    step=0.01,
    description="Gamma:",
    disabled=False,
    # layout=widgets.Layout(display="none"),  # hidden initially
)

indexation_button = widgets.Button(
    description="Indexate peaks (find UB)",
    button_style="success",
    # layout=widgets.Layout(display="none"),  # hidden initially
)


integration_button = widgets.Button(
    description="Integration peaks",
    button_style="success",
    # layout=widgets.Layout(display="none"),  # hidden initially
)


# output = widgets.Output()
output_data = widgets.Output()
peaks_output = widgets.Output()  # separate area for peaks table
indexation_output = widgets.Output()  # separate area for indexation
integration_output = widgets.Output()  # separate area for indexation

# ui_fc = widgets.HBox(
#     [fc, fc_vanadium], layout=widgets.Layout(align_items="center")
# )

ui_hh = widgets.VBox(
    [
        display_button,
        hide_output_data_button,
    ],
    layout=widgets.Layout(align_items="center"),
)


ui_control_display = widgets.HBox(
    [fig_rbuttons, ui_hh], layout=widgets.Layout(align_items="center")
)

ui_da_data = widgets.HBox(
    [state_dropdown, display_da_button],
    layout=widgets.Layout(align_items="center"),
)


ui_peaks_button = widgets.HBox(
    [display_peaks_button, hide_peaks_button],
    layout=widgets.Layout(align_items="center"),
)


ui_abc = widgets.HBox(
    [
        cell_a,
        cell_b,
        cell_c,
    ],
    layout=widgets.Layout(align_items="center"),
)

ui_angles = widgets.HBox(
    [
        cell_alpha,
        cell_beta,
        cell_gamma,
    ],
    layout=widgets.Layout(align_items="center"),
)

spinner = widgets.HTML("""
    <div style="
        border: 6px solid #f3f3f3;
        border-top: 6px solid #3498db;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 10px auto;
    "></div>

    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """)
