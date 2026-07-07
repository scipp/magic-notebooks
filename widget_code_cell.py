from IPython.utils.capture import capture_output

import ipywidgets
import ast
from IPython.display import display

import traceback

def execute_with_last_expr(code, global_ns):
    """
    Executes code and returns the value of the last expression (if any).
    """
    tree = ast.parse(code)

    # If last node is an expression, separate it
    if isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body[-1].value)
        exec(compile(ast.Module(tree.body[:-1], []), "<cell>", "exec"), global_ns)
        return eval(compile(last_expr, "<cell>", "eval"), global_ns)
    else:
        exec(code, global_ns)
        return None
    
# Global namespace shared with the rest of your application
EXEC_GLOBALS = globals()


def make_code_cell(initial_text:str="", placeholder:str="Write Python code here...\n\n", width:str="95%", height:str='200px'):
    """
    Creates a mini notebook-like code cell:
    - Text area for Python code
    - Run button
    - Output area
    """
    css = ipywidgets.HTML("""
    <style>
    textarea {
        font-family: monospace !important;
        font-size: 14px;
    }
    </style>
    """)
    code_area = ipywidgets.Textarea(
        value=initial_text,
        placeholder=placeholder,
        layout=ipywidgets.Layout(width=width, height=height),
        style={"font_family": "monospace"}
    )

    run_button = ipywidgets.Button(
        description="Run",
        button_style="success",
        tooltip="Execute the code",
        icon="play"
    )


    output = ipywidgets.Output(layout=ipywidgets.Layout(border="1px solid #ccc", padding='6px', width=width))


    def run_code(_):
        output.clear_output()
        run_button.disabled = True
        run_button.button_style = "warning"
        run_button.icon = "spinner"

        code = code_area.value


        try:
            with output:
                # exec(code, EXEC_GLOBALS)
                result = execute_with_last_expr(code, EXEC_GLOBALS)
                if result is not None:
                    display(result)
        except Exception:
            with output:
                traceback.print_exc()
        finally:
            run_button.disabled = False
            run_button.button_style = "success"
            run_button.icon = "play"

    run_button.on_click(run_code)

    return ipywidgets.VBox([
        ipywidgets.HTML("<b>Python Code Cell</b>"),
        css,
        code_area,
        run_button,
        output
    ])

# code_cell = make_code_cell()