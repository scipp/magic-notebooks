from ipyfilechooser import FileChooser
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

import magic_graphs
import read_h5
import plopp
def get_widget():
    # ---------------------------------------------------------
    # File chooser widget
    # ---------------------------------------------------------
    fc = FileChooser(
        '/',
    title='Select a data file',
    select_default=True
    )

    plot_button = widgets.Button(
        description="Plot file",
        button_style='success'
    )

    output = widgets.Output()

    # ---------------------------------------------------------
    # Callback: load + plot
    # ---------------------------------------------------------
    def plot_file(b):
        with output:
            clear_output()
            file_path = fc.selected

            if file_path is None:
                print("No file selected")
                return
    
            try:
                d_out = read_h5.read_h5_to_dict(file_path)
            except Exception as e:
                print(f"Error reading file: {e}")
                return

            data_event = d_out["data_event"]
            data_event = data_event.transform_coords(
                ('detector_event_position_local',),
                graph=magic_graphs.graph_qvec,
                rename_dims=False
            )

            # IMPORTANT: store the plopp figure
            fig = plopp.scatter3d(
                data_event[::10000],
                pos='detector_event_position_local',
                size=0.01
            )


            peak_button = widgets.Button(
                description="Give list of peaks",
                button_style='success'
                )

            # IMPORTANT: display the figure explicitly
            display(fig)
            print(f"Plotted: {file_path}")
            display(peak_button)
            peak_button.on_click(peak_find)

    def peak_find(b):
        print(b)
    plot_button.on_click(plot_file)

    # ---------------------------------------------------------
    # Display UI
    # ---------------------------------------------------------
    display(fc, plot_button, output)
