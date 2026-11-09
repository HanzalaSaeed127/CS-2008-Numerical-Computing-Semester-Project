from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp

from modules.expression_parser import ExpressionError, make_function, derivative
from modules.roots import bisection, fixed_point, newton, secant
from modules.interpolation import lagrange, newton_divided_difference, cubic_spline, polynomial_fit
from modules.integration import midpoint, trapezoidal, simpson, simpson_three_eighths, gaussian, newton_cotes
from modules.differentiation import finite_difference
from modules.ode_solvers import solve_ode, solve_ode_scipy
from modules.linear_algebra import gaussian_elimination, _lu, cholesky, iterative_solve, condition_number
from modules.approximation import least_squares, gram_schmidt, qr, svd
from modules.optimization import bounded_minimize, multivariable_minimize, make_multivariable_function
from modules.eigenvalues import power_iteration, inverse_power_iteration, eig
from modules.arithmetic_errors import error_metrics, ieee754, machine_info, rounding_demo
from modules.formatting import format_number, format_dataframe, matrix_latex
from ui.equation_editor import equation_editor
from ui.data_tables import points_editor, matrix_editor, vector_editor


st.set_page_config(page_title="Numerica", page_icon="∑", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@keyframes ambient-glow { 0%, 100% { opacity:.55; transform:scale(1); } 50% { opacity:.8; transform:scale(1.08); } }
@keyframes reveal-on-scroll { from { opacity:0; transform:translateY(16px) scale(.985); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes page-drop { 0% { opacity:0; transform:translateY(-28px) scale(.985); filter:blur(3px); } 65% { opacity:1; transform:translateY(4px) scale(1.002); filter:blur(0); } 100% { opacity:1; transform:translateY(0) scale(1); } }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 78% 8%, rgba(86,76,210,.28), transparent 28rem), radial-gradient(circle at 18% 82%, rgba(34,190,187,.12), transparent 24rem), #080b1c; color:#eef3ff; }
[data-testid="stHeader"] { background:rgba(8,11,28,.72); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(13,20,48,.96),rgba(8,12,30,.98)); border-right:1px solid rgba(135,154,255,.16); }
[data-testid="stSidebar"] * { color:#e7edff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px; transition:background .2s ease, transform .2s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(91,224,205,.12); transform:translateX(4px); }
.main .block-container { padding-top:1rem; padding-bottom:1rem; animation:page-drop .55s cubic-bezier(.22,.8,.24,1) both; transform-origin:top center; }
[data-testid="stVerticalBlock"] { gap:.55rem; }
[data-testid="stHorizontalBlock"] { gap:.75rem; }
.hero { position:relative; overflow:hidden; background:linear-gradient(120deg,rgba(20,33,78,.92),rgba(54,48,143,.78)); border:1px solid rgba(136,161,255,.35); box-shadow:0 12px 40px rgba(34,27,105,.34), inset 0 1px rgba(255,255,255,.12); color:white; padding:1rem 1.8rem; border-radius:16px; margin-bottom:.7rem; animation:reveal-on-scroll .7s ease both; }
.hero::after { content:""; position:absolute; width:15rem; height:15rem; right:-5rem; top:-9rem; border-radius:50%; background:rgba(102,238,218,.22); filter:blur(20px); animation:ambient-glow 6s ease-in-out infinite; }
.hero h1 { position:relative; z-index:1; margin:0; font-size:2rem; letter-spacing:.02em; } .hero p { position:relative; z-index:1; color:#cbd8ff; margin:.25rem 0 0; }
.main h1, .main h2, .main h3 { margin-top:.35rem; margin-bottom:.2rem; }
.main [data-testid="stCaptionContainer"] { margin-bottom:.1rem; }
.main [data-testid="stWidgetLabel"] { margin-bottom:.1rem; }
.main [data-testid="stMarkdownContainer"], .main [data-testid="stDataFrame"], .main [data-testid="stMetric"] { animation:reveal-on-scroll .55s ease both; animation-timeline:view(); animation-range:entry 0% cover 18%; }
.card, div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(26,35,75,.78),rgba(17,21,52,.78)); border:1px solid rgba(136,161,255,.22); border-radius:14px; padding:1rem 1.2rem; box-shadow:0 8px 26px rgba(2,5,20,.22), inset 0 1px rgba(255,255,255,.06); }
.input-panel, .result-panel { background:rgba(15,21,51,.58); border:1px solid rgba(136,161,255,.2); border-radius:14px; padding:.45rem .7rem .2rem; }
.result-panel { background:linear-gradient(145deg,rgba(26,35,75,.72),rgba(17,21,52,.62)); }
div[data-testid="stMetric"] { padding:12px; transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
div[data-testid="stMetric"]:hover { transform:translateY(-4px) scale(1.02); border-color:rgba(94,235,211,.62); box-shadow:0 12px 30px rgba(19,174,166,.18); }
.muted { color:#aab8df; font-size:.9rem; }
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div { background:rgba(30,34,65,.82) !important; border-color:rgba(142,157,221,.26) !important; color:#f0f4ff !important; transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within, [data-baseweb="input"] > div:focus-within { border-color:#62e8d1 !important; box-shadow:0 0 0 1px #62e8d1, 0 0 18px rgba(98,232,209,.16) !important; }
input:hover, textarea:hover, [data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover { transform:scale(1.008); }
button[kind="primary"] { background:linear-gradient(110deg,#16aa9d,#5de5cb); border:0; color:#071827; font-weight:700; box-shadow:0 5px 18px rgba(53,220,195,.2); transition:transform .2s ease, box-shadow .2s ease; }
button[kind="primary"]:hover { transform:translateY(-2px) scale(1.025); box-shadow:0 9px 24px rgba(53,220,195,.35); }
button:not([kind="primary"]) { transition:transform .2s ease, border-color .2s ease; }
button:not([kind="primary"]):hover { transform:translateY(-1px); border-color:#62e8d1 !important; }
[data-testid="stExpander"] { background:rgba(20,27,60,.58); border:1px solid rgba(136,161,255,.2); border-radius:12px; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
}
</style>""", unsafe_allow_html=True)

DISPLAY_PRECISION = {"4 decimals": 4, "6 decimals": 6, "8 decimals": 8, "10 decimals": 10, "Scientific": 8}
precision_mode = st.sidebar.selectbox("Display precision", list(DISPLAY_PRECISION), index=2)
DISPLAY_DIGITS = DISPLAY_PRECISION[precision_mode]

def parse_numbers(text, name="values"):
    try:
        values = [float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers.") from exc
    if not values: raise ValueError(f"Enter at least one value for {name}.")
    return np.array(values)

def parse_matrix(text):
    try:
        rows = [[float(v.strip()) for v in row.split(",")] for row in text.strip().splitlines() if row.strip()]
    except ValueError as exc:
        raise ValueError("Matrix entries must be numeric and comma-separated by row.") from exc
    if not rows or len({len(row) for row in rows}) != 1: raise ValueError("Matrix rows must have equal length.")
    return np.array(rows, dtype=float)

def show_error(exc):
    st.error(str(exc))

def plot_xy(x, y, title, xlabel="x", ylabel="y", scatter=None):
    x, y = np.asarray(x), np.asarray(y)
    figure = themed_plot(x, y, title, xlabel, ylabel, scatter)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

def themed_plot(
    x, y, title, xlabel="x", ylabel="y", scatter=None, marker_labels=None,
    x_range=None, y_range=None, animation_duration_ms=1000,
):
    x, y = np.asarray(x), np.asarray(y)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Computed", line={"color": "#5de5cb", "width": 3}))
    if scatter is not None:
        figure.add_trace(go.Scatter(x=[], y=[], mode="markers+text" if marker_labels else "markers",
                                    text=marker_labels, textposition="top center",
                                    name="Roots" if marker_labels else "Data",
                                    marker={"color": "#ffb86b", "size": 10}))
    frame_indices = np.unique(np.linspace(2, len(x), 31, dtype=int))
    frames = [
        go.Frame(
            data=[
                go.Scatter(x=x[:index], y=y[:index], mode="lines", line={"color": "#5de5cb", "width": 3}),
                *([go.Scatter(x=scatter[0], y=scatter[1],
                              text=marker_labels, textposition="top center",
                              mode="markers+text" if marker_labels else "markers",
                              marker={"color": "#ffb86b", "size": 8})] if scatter is not None else []),
            ],
            traces=list(range(2 if scatter is not None else 1)),
            name=str(index),
        )
        for index in frame_indices
    ]
    figure.frames = frames
    figure.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,17,42,.72)",
        font={"color": "#eaf0ff"}, hovermode="x unified",
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
        xaxis={"gridcolor": "rgba(151,168,230,.16)", "zerolinecolor": "#91a8ff", "zerolinewidth": 2,
               "showline": True, "linecolor": "rgba(151,168,230,.5)"},
        yaxis={"gridcolor": "rgba(151,168,230,.16)", "zerolinecolor": "#91a8ff", "zerolinewidth": 2,
               "showline": True, "linecolor": "rgba(151,168,230,.5)"},
        updatemenus=[{"type": "buttons", "showactive": False, "x": .5, "xanchor": "center", "y": 1.16,
                      "buttons": [{"label": "▶ Draw", "method": "animate",
                                   "args": [None, {"frame": {
                                       "duration": animation_duration_ms / max(1, len(frames)),
                                       "redraw": False, "easing": "linear"},
                                                   "transition": {"duration": 0}, "fromcurrent": True}]}]}],
    )
    if x_range is not None:
        figure.update_xaxes(range=x_range)
    if y_range is not None:
        figure.update_yaxes(range=y_range)
    return figure

def root_convergence_plot(histories):
    figure = go.Figure()
    colors = ["#5de5cb", "#ffb86b"]
    for index, (method, history) in enumerate(histories):
        iterations = [row["iteration"] for row in history]
        errors = [max(abs(row["error"]), 1e-16) for row in history]
        figure.add_trace(go.Scatter(
            x=iterations, y=errors, mode="lines+markers", name=method,
            line={"color": colors[index % len(colors)], "width": 3},
        ))
    figure.update_layout(
        title="Error by iteration",
        xaxis_title="Iteration",
        yaxis_title="Error",
        yaxis_type="log",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,17,42,.72)",
        font={"color": "#eaf0ff"}, hovermode="x unified",
        margin={"l": 55, "r": 20, "t": 55, "b": 45},
        xaxis={"gridcolor": "rgba(151,168,230,.16)", "zerolinecolor": "#91a8ff"},
        yaxis={"gridcolor": "rgba(151,168,230,.16)", "zerolinecolor": "#91a8ff"},
        legend={"title": {"text": "Method"}},
    )
    return figure

def result_table(rows):
    frame = pd.DataFrame(rows)
    display = format_dataframe(frame, DISPLAY_DIGITS, precision_mode == "Scientific")
    st.dataframe(display, width="stretch", height=min(260, 38 + len(display) * 35), hide_index=True)
    st.download_button("Download CSV", display.to_csv(index=False), "numerica-results.csv", "text/csv")

def explanation(title, body):
    with st.expander(f"About {title}"):
        st.markdown(body)

def comparison_button(key):
    state_key = f"show_{key}_comparison"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    if not st.session_state[state_key]:
        if st.button("Compare two methods", key=f"{key}_show_comparison"):
            st.session_state[state_key] = True
            st.rerun()
        return False
    if st.button("Hide comparison", key=f"{key}_hide_comparison"):
        st.session_state[state_key] = False
        st.rerun()
    return True

def solve_root_method(method_name, fn, expression, lo, hi, tolerance, max_iter):
    if method_name == "Bisection":
        return bisection(fn, lo, hi, tolerance, max_iter)
    if method_name == "Newton–Raphson":
        return newton(fn, (lo + hi) / 2, tolerance, max_iter, derivative(expression))
    if method_name == "Secant":
        return secant(fn, lo, hi, tolerance, max_iter)
    return fixed_point(lambda x: x - fn(x), (lo + hi) / 2, tolerance, max_iter, fn)

def render_root_comparison():
    if "show_root_comparison" not in st.session_state:
        st.session_state.show_root_comparison = False
    if not st.session_state.show_root_comparison:
        if st.button("Compare two methods", type="secondary", key="show_comparison_button"):
            st.session_state.show_root_comparison = True
            st.rerun()
        return

    with st.container(border=True):
        st.markdown("## Compare two methods")
        close_left, close_right = st.columns([6, 1])
        close_right.button("Hide comparison", key="hide_comparison_button",
                           on_click=lambda: st.session_state.update(show_root_comparison=False))
        st.info("Choose two methods in the upper panels. The single lower panel compares their results.")
        method_left, method_right = st.columns(2)
        available_methods = ["Bisection", "Newton–Raphson", "Secant", "Fixed-point"]
        method_a = method_left.selectbox("Method 1", available_methods, key="inline_method_a")
        method_b = method_right.selectbox("Method 2", available_methods, index=1, key="inline_method_b")
        st.divider()
        st.markdown("### Shared equation and settings")
        expression, parsed = equation_editor(
            "Comparison f(x)", "x^3 - x - 2", key="inline_comparison_f",
            examples=["x^3 - x - 2", "cos(x) - x", "x^2 - 2"],
        )
        setting_left, setting_mid, setting_right = st.columns(3)
        lo = setting_left.number_input("Left endpoint", value=1.0, key="inline_comparison_lo")
        hi = setting_mid.number_input("Right endpoint", value=2.0, key="inline_comparison_hi")
        reference = setting_right.number_input("Reference root", value=1.5213797068, format="%.10f", key="inline_comparison_reference")
        tolerance = st.number_input("Tolerance", min_value=1e-14, value=1e-8, format="%.1e", key="inline_comparison_tol")
        max_iter = st.number_input("Maximum iterations", min_value=1, value=100, key="inline_comparison_iter")
        if st.button("Compare both methods", type="primary", key="inline_compare_button"):
            if parsed is None:
                st.stop()
            if method_a == method_b:
                raise ValueError("Choose two different methods to compare.")
            if lo >= hi:
                raise ValueError("Left endpoint must be smaller than right endpoint.")
            fn = make_function(expression)
            results = []
            histories = []
            for method_name in (method_a, method_b):
                outcome = solve_root_method(method_name, fn, expression, lo, hi, tolerance, int(max_iter))
                histories.append((method_name, outcome.iterations))
                absolute_error = abs(outcome.root - reference)
                relative_error = absolute_error / abs(reference) if reference else None
                results.append({
                    "method": method_name, "root": outcome.root, "iterations": len(outcome.iterations),
                    "absolute error": absolute_error,
                    "relative error (%)": None if relative_error is None else relative_error * 100,
                    "status": "Converged" if outcome.converged else "Not converged",
                })
            st.divider()
            st.markdown("### Comparison results")
            result_table(results)
            st.plotly_chart(
                root_convergence_plot(histories),
                width="stretch", config={"displaylogo": False},
            )
            st.caption("Each line shows how the method's error changes over its iterations. Lower values indicate a closer approximation.")

st.sidebar.markdown("## NUMERICA")
sections = {
    "Root finding": ["Bisection", "Newton–Raphson", "Secant", "Fixed-point"],
    "Interpolation & fitting": ["Interpolation", "Polynomial fit"],
    "Integration": ["Midpoint", "Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Newton–Cotes", "Gauss–Legendre"],
    "Differentiation & ODE": ["Finite differences", "ODE solvers"],
    "Linear algebra": ["Linear systems", "Factorizations", "Iterative solvers", "Condition number"],
    "Approximation": ["Least squares", "QR / Gram–Schmidt / SVD"],
    "Optimization & eigen": ["Optimization", "Eigenvalues"],
    "Arithmetic": ["Error analysis", "Rounding & truncation", "IEEE 754"],
}
section_pages = {"Integration": "Numerical integration"}
def sync_integration_method():
    st.session_state.method_Integration = st.session_state.integration_method

def select_section_method(section):
    selected = st.session_state[f"method_{section}"]
    st.session_state.selected_page = section_pages.get(section, selected)
    if section == "Integration":
        st.session_state.integration_method = selected
    st.session_state.pop("page_override", None)

st.sidebar.markdown("### Select Method")
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Bisection"
if "integration_method" not in st.session_state:
    st.session_state.integration_method = "Midpoint"
for section, methods in sections.items():
    st.sidebar.markdown(f"**{section}**")
    st.sidebar.selectbox(
        f"{section} method", methods,
        index=methods.index(st.session_state.integration_method)
        if section == "Integration" and st.session_state.integration_method in methods
        else (methods.index(st.session_state.selected_page)
              if st.session_state.selected_page in methods else 0),
        key=f"method_{section}", on_change=select_section_method, args=(section,),
    )
if st.sidebar.button("Graph Setting", type="secondary", key="graph_settings_button"):
    st.session_state.page_override = "Graph Setting"
    st.rerun()
page = st.session_state.get("page_override", st.session_state.selected_page)
st.sidebar.markdown("---")

st.markdown('<div class="hero"><h1>Numerica</h1></div>', unsafe_allow_html=True)

try:
    if page == "Graph Setting":
        st.header("Graph Setting")
        st.caption("Configure the graph view, then press Plot. The curve stays hidden until you press Draw.")
        expression, parsed = equation_editor("f(x)", "sin(x) * exp(-x/4)", key="graphing_f",
                                              examples=["sin(x) * exp(-x/4)", "x^2 - 2x", "exp(-x^2)"])
        left, right = st.columns(2)
        lo = left.number_input("Start x", value=-10.0, key="graphing_lo")
        hi = right.number_input("End x", value=10.0, key="graphing_hi")
        y_left, y_right = st.columns(2)
        auto_y = st.checkbox("Automatic y-axis range", value=True, key="graphing_auto_y")
        y_min = y_left.number_input("Minimum y", value=-10.0, disabled=auto_y, key="graphing_y_min")
        y_max = y_right.number_input("Maximum y", value=10.0, disabled=auto_y, key="graphing_y_max")
        points = st.slider("Plot resolution", min_value=100, max_value=1000, value=400, step=50)
        if st.button("Plot", type="primary", key="graphing_plot"):
            if parsed is None:
                st.stop()
            if lo >= hi:
                raise ValueError("Start x must be smaller than end x.")
            if not auto_y and y_min >= y_max:
                raise ValueError("Minimum y must be smaller than maximum y.")
            grid = np.linspace(lo, hi, points)
            values = make_function(expression)(grid)
            st.plotly_chart(themed_plot(
                grid, values, "Function plot",
                x_range=(lo, hi), y_range=None if auto_y else (y_min, y_max),
                animation_duration_ms=1000,
            ), width="stretch",
                            config={"displaylogo": False})
            st.caption("Use the centered ▶ Draw control above the chart to animate the curve.")
        explanation("Graph Setting", "Set a wider x range or choose a manual y range to avoid excessive zoom. Plotly's interactive zoom remains available after drawing.")

    elif page == "Method comparison":
        st.header("Method comparison")
        st.caption("Choose two methods to compare for the same equation.")
        expression, parsed = equation_editor("f(x)", "x^3 - x - 2", key="comparison_f",
                                              examples=["x^3 - x - 2", "cos(x) - x", "x^2 - 2"])
        available_methods = ["Bisection", "Newton–Raphson", "Secant", "Fixed-point"]
        method_left, method_right = st.columns(2)
        method_a = method_left.selectbox("Method 1", available_methods, key="comparison_method_a")
        method_b = method_right.selectbox("Method 2", available_methods, index=1, key="comparison_method_b")
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Left endpoint", value=1.0, key="comparison_lo")
        hi = c2.number_input("Right endpoint", value=2.0, key="comparison_hi")
        reference = c3.number_input("Reference root", value=1.5213797068, format="%.10f")
        tolerance = st.number_input("Tolerance", min_value=1e-14, value=1e-8, format="%.1e", key="comparison_tol")
        max_iter = st.number_input("Maximum iterations", min_value=1, value=100, key="comparison_iter")
        if st.button("Compare methods", type="primary", key="compare_methods"):
            if parsed is None:
                st.stop()
            if method_a == method_b:
                raise ValueError("Choose two different methods to compare.")
            fn = make_function(expression)
            results = []
            histories = []
            for method_name in (method_a, method_b):
                if method_name == "Bisection":
                    outcome = bisection(fn, lo, hi, tolerance, int(max_iter))
                elif method_name == "Newton–Raphson":
                    outcome = newton(fn, (lo + hi) / 2, tolerance, int(max_iter), derivative(expression))
                elif method_name == "Secant":
                    outcome = secant(fn, lo, hi, tolerance, int(max_iter))
                else:
                    fixed_function = lambda x: x - fn(x)
                    outcome = fixed_point(fixed_function, (lo + hi) / 2, tolerance, int(max_iter), fn)
                histories.append((method_name, outcome.iterations))
                absolute_error = abs(outcome.root - reference)
                relative_error = absolute_error / abs(reference) if reference else None
                results.append({
                    "method": method_name, "root": outcome.root, "iterations": len(outcome.iterations),
                    "absolute error": absolute_error,
                    "relative error (%)": None if relative_error is None else relative_error * 100,
                    "status": "Converged" if outcome.converged else "Not converged",
                })
            result_table(results)
            st.info("Relative error is measured against the reference root you entered.")
            st.plotly_chart(root_convergence_plot(histories),
                            width="stretch", config={"displaylogo": False})
            st.caption("Each line shows how the method's error changes over its iterations. Lower values indicate a closer approximation.")
        explanation("Method comparison", "Use the same equation and interval for both methods. Enter a known root if you want to compare the errors.")

    elif page in {"Bisection", "Newton–Raphson", "Secant", "Fixed-point"}:
        st.header(page); st.caption("Enter an equation, set the parameters, and run the method.")
        input_col, result_col = st.columns([1, 1.35], gap="large")
        with input_col:
            with st.container(border=True):
                expression, parsed = equation_editor("f(x)", "x^3 - x - 2", key=f"{page}_f", examples=["x^3 - x - 2", "sin(x) - x/2", "2x + 3"])
                tol = st.number_input("Tolerance", min_value=1e-14, value=1e-8, format="%.1e", key=f"{page}_tol")
                max_iter = st.number_input("Maximum iterations", min_value=1, value=100, key=f"{page}_max_iter")
                if parsed is None:
                    st.stop()
                fn = make_function(expression)
                if page == "Bisection":
                    a, b = st.columns(2); lo, hi = a.number_input("Left endpoint", value=1.0), b.number_input("Right endpoint", value=2.0)
                    run_method = st.button("Run method", type="primary", key=f"{page}_run")
                    if run_method: result = bisection(fn, lo, hi, tol, max_iter)
                elif page == "Newton–Raphson":
                    x0 = st.number_input("Initial guess", value=1.5)
                    run_method = st.button("Run method", type="primary", key=f"{page}_run")
                    if run_method: result = newton(fn, x0, tol, max_iter, derivative(expression))
                elif page == "Secant":
                    x0, x1 = st.columns(2); a, b = x0.number_input("First guess", value=1.0), x1.number_input("Second guess", value=2.0)
                    run_method = st.button("Run method", type="primary", key=f"{page}_run")
                    if run_method: result = secant(fn, a, b, tol, max_iter)
                else:
                    g, g_parsed = equation_editor("g(x)", "(x + 2)^(1/3)", key="fixed_point_g", examples=["(x + 2)^(1/3)", "cos(x)"])
                    if g_parsed is None:
                        st.stop()
                    x0 = st.number_input("Initial guess", value=1.0)
                    run_method = st.button("Run method", type="primary", key=f"{page}_run")
                    if run_method: result = fixed_point(make_function(g), x0, tol, max_iter, fn)
        with result_col:
            with st.container(border=True):
                st.subheader("Result")
                if "result" in locals():
                    m1, m2, m3 = st.columns(3)
                    m1.latex(r"x \approx " + format_number(result.root, DISPLAY_DIGITS, precision_mode == "Scientific"))
                    m2.metric("|f(x)|", format_number(abs(result.value), DISPLAY_DIGITS, precision_mode == "Scientific"))
                    m3.metric("Iterations", len(result.iterations))
                    if result.converged:
                        st.success(result.message)
                    else:
                        st.warning(result.message)
                    result_table(result.iterations)
                    if result.iterations:
                        plot_xy([r["iteration"] for r in result.iterations], [r["error"] for r in result.iterations], "Convergence history", "Iteration", "Error")
                else:
                    st.info("Enter the parameters and run the method to see the result here.")
        explanation(page, "Iterative root-finding methods seek an x for which **f(x) = 0**. Convergence is reported only when the requested tolerance is met.")
        render_root_comparison()

    elif page in {"Interpolation", "Polynomial fit"}:
        st.header(page);         x, y = points_editor(key=f"{page}_points")
        if page == "Interpolation":
            method = st.selectbox("Interpolation method", ["Lagrange", "Newton divided difference", "Cubic spline"], key="interpolation_method")
            points = st.text_input("Evaluate at", "0, 0.5, 1, 1.5, 2, 2.5, 3")
            if st.button("Run method", type="primary"):
                q = parse_numbers(points, "evaluation points")
                if method == "Lagrange": values = lagrange(x, y, q)
                elif method == "Cubic spline": values = cubic_spline(x, y, q)
                else: values, coeff = newton_divided_difference(x, y, q)
                result_table([{"x": a, "y": b} for a, b in zip(q, values)]); plot_xy(q, values, method, scatter=(x, y))
            explanation("Interpolation", "Interpolation methods estimate values between known data points.")
            if comparison_button("interpolation"):
                st.subheader("Compare interpolation methods")
                compare_a, compare_b = st.columns(2)
                interpolation_methods = ["Lagrange", "Newton divided difference", "Cubic spline"]
                method_a = compare_a.selectbox("Method 1", interpolation_methods, key="interpolation_compare_a")
                method_b = compare_b.selectbox("Method 2", interpolation_methods, index=1, key="interpolation_compare_b")
                if st.button("Compare interpolation results", type="primary", key="interpolation_compare_run"):
                    if method_a == method_b:
                        raise ValueError("Choose two different interpolation methods.")
                    q_compare = parse_numbers(points, "evaluation points")
                    outputs = []
                    for candidate in (method_a, method_b):
                        if candidate == "Lagrange":
                            candidate_values = lagrange(x, y, q_compare)
                        elif candidate == "Cubic spline":
                            candidate_values = cubic_spline(x, y, q_compare)
                        else:
                            candidate_values = newton_divided_difference(x, y, q_compare)[0]
                        outputs.append(candidate_values)
                    result_table([
                        {"x": point, method_a: value_a, method_b: value_b,
                         "absolute difference": abs(value_a - value_b)}
                        for point, value_a, value_b in zip(q_compare, outputs[0], outputs[1])
                    ])
                    plot_xy(q_compare, outputs[0], f"{method_a} vs {method_b}", scatter=(q_compare, outputs[1]))
        else:
            degree = st.number_input("Polynomial degree", min_value=0, max_value=max(0, len(x)-1), value=min(2, len(x)-1))
            if st.button("Run method", type="primary"):
                coeff, values = polynomial_fit(x, y, int(degree)); grid = np.linspace(min(x), max(x), 200)
                st.metric("Residual norm", f"{np.linalg.norm(values-y):.4g}"); st.code(np.array2string(coeff, precision=6))
                plot_xy(grid, np.polyval(coeff, grid), "Polynomial approximation", scatter=(x, y))
            explanation("Polynomial fit", "Polynomial fitting methods are compared by their approximation residuals.")

    elif page in {"Numerical integration", "Integration comparison"}:
        st.header(page)
        input_col, result_col = st.columns([1, 1.35], gap="large")
        with input_col:
            with st.container(border=True):
                expression, parsed = equation_editor("f(x)", "x^2", key=f"{page}_integrand", examples=["x^2", "sin(x)", "exp(-x^2)"])
                a, b = st.columns(2); lo, hi = a.number_input("Lower limit", value=0.0), b.number_input("Upper limit", value=1.0)
                method = st.selectbox(
                    "Integration method",
                    ["Midpoint", "Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Newton–Cotes", "Gauss–Legendre"],
                    key="integration_method",
                    on_change=sync_integration_method,
                )
                n = st.number_input("Subintervals / quadrature points", min_value=1, value=6)
                run_integration = st.button("Run method", type="primary", key="integration_run")
        if parsed is None:
            st.stop()
        with result_col:
            with st.container(border=True):
                st.subheader("Result")
                if run_integration:
                    fn = make_function(expression); n = int(n)
                    if method == "Midpoint": value, nodes = midpoint(fn, lo, hi, n)
                    elif method == "Trapezoidal": value, nodes = trapezoidal(fn, lo, hi, n)
                    elif method == "Simpson 1/3": value, nodes = simpson(fn, lo, hi, n)
                    elif method == "Simpson 3/8": value, nodes = simpson_three_eighths(fn, lo, hi, n)
                    elif method == "Newton–Cotes": value, nodes, _ = newton_cotes(fn, lo, hi, n)
                    else: value, nodes = gaussian(fn, lo, hi, n)
                    st.metric("Approximate integral", f"{value:.12g}"); plot_xy(np.linspace(lo, hi, 300), fn(np.linspace(lo, hi, 300)), "Integrand", scatter=(nodes, fn(nodes)))
                else:
                    st.info("Choose an integration method and run it to see the result here.")
            if page == "Integration comparison":
                rows = []
                for label, compute in [("Midpoint", midpoint), ("Trapezoidal", trapezoidal), ("Simpson 1/3", simpson), ("Simpson 3/8", simpson_three_eighths), ("Gauss–Legendre", gaussian)]:
                    try:
                        output = compute(fn, lo, hi, n if label not in {"Simpson 1/3"} else (n if n % 2 == 0 else n + 1))[0]
                        rows.append({"method": label, "estimate": output})
                    except ValueError as exc:
                        rows.append({"method": label, "estimate": str(exc)})
                result_table(rows)
        explanation(method, "Composite rules approximate the area by evaluating the integrand at structured nodes. Simpson's 1/3 rule requires an even number of subintervals; Simpson's 3/8 rule requires a multiple of three.")
        if comparison_button("integration"):
            st.subheader("Compare integration methods")
            compare_a, compare_b = st.columns(2)
            integration_methods = ["Midpoint", "Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Newton–Cotes", "Gauss–Legendre"]
            method_a = compare_a.selectbox("Method 1", integration_methods, key="integration_compare_a")
            method_b = compare_b.selectbox("Method 2", integration_methods, index=1, key="integration_compare_b")
            if st.button("Compare integration results", type="primary", key="integration_compare_run"):
                if method_a == method_b:
                    raise ValueError("Choose two different integration methods.")
                fn_compare = make_function(expression)
                values = []
                for candidate in (method_a, method_b):
                    if candidate == "Midpoint": estimate = midpoint(fn_compare, lo, hi, int(n))[0]
                    elif candidate == "Trapezoidal": estimate = trapezoidal(fn_compare, lo, hi, int(n))[0]
                    elif candidate == "Simpson 1/3": estimate = simpson(fn_compare, lo, hi, int(n if n % 2 == 0 else n + 1))[0]
                    elif candidate == "Simpson 3/8": estimate = simpson_three_eighths(fn_compare, lo, hi, int(n))[0]
                    elif candidate == "Newton–Cotes": estimate = newton_cotes(fn_compare, lo, hi, int(n))[0]
                    else: estimate = gaussian(fn_compare, lo, hi, int(n))[0]
                    values.append(estimate)
                result_table([
                    {"method": method_a, "estimate": values[0]},
                    {"method": method_b, "estimate": values[1]},
                    {"absolute difference": abs(values[0] - values[1])},
                ])
                st.metric("Relative difference (%)", f"{abs(values[0] - values[1]) / max(abs(values[0]), 1e-15) * 100:.6g}")

    elif page == "Finite differences":
        st.header(page)
        input_col, result_col = st.columns([1, 1.35], gap="large")
        with input_col:
            with st.container(border=True):
                expression, parsed = equation_editor("f(x)", "sin(x)", key="difference_f", examples=["sin(x)", "x^2", "exp(x)"])
                x, h = st.columns(2); point = x.number_input("Evaluation point", value=1.0); step = h.number_input("Step size", value=1e-4, min_value=1e-12, format="%.1e")
                scheme = st.selectbox("Differentiation scheme", ["forward", "backward", "central", "second"], key="difference_scheme")
                run_difference = st.button("Run method", type="primary", key="difference_run")
        if parsed is None:
            st.stop()
        with result_col:
            with st.container(border=True):
                st.subheader("Result")
                if run_difference:
                    value = finite_difference(make_function(expression), point, step, scheme)
                    st.latex(f"f'({format_number(point, DISPLAY_DIGITS)}) \\approx {format_number(value, DISPLAY_DIGITS, precision_mode == 'Scientific')}")
                else:
                    st.info("Run the method to see the derivative here.")
        if comparison_button("finite_difference"):
            st.subheader("Compare differentiation schemes")
            compare_a, compare_b = st.columns(2)
            schemes = ["forward", "backward", "central", "second"]
            scheme_a = compare_a.selectbox("Scheme 1", schemes, index=0, key="difference_compare_a")
            scheme_b = compare_b.selectbox("Scheme 2", schemes, index=2, key="difference_compare_b")
            if st.button("Compare differentiation results", type="primary", key="difference_compare_run"):
                if scheme_a == scheme_b:
                    raise ValueError("Choose two different differentiation schemes.")
                fn_compare = make_function(expression)
                value_a = finite_difference(fn_compare, point, step, scheme_a)
                value_b = finite_difference(fn_compare, point, step, scheme_b)
                result_table([
                    {"scheme": scheme_a, "derivative": value_a},
                    {"scheme": scheme_b, "derivative": value_b},
                ])
                st.metric("Absolute difference", f"{abs(value_a - value_b):.6g}")
                st.metric("Relative difference (%)", f"{abs(value_a - value_b) / max(abs(value_a), 1e-15) * 100:.6g}")
    elif page in {"ODE solvers", "ODE comparison"}:
        st.header(page); expression, parsed = equation_editor("f(x, y)", "y - x^2 + 1", variables=("x", "y"), key=f"{page}_ode", examples=["y - x^2 + 1", "x + y", "y"])
        if parsed is None:
            st.stop()
        c1, c2 = st.columns(2)
        x0, y0 = c1.number_input("x₀", value=0.0), c2.number_input("y₀", value=0.5); xend = c1.number_input("Final x", value=2.0); steps = c2.number_input("Steps", min_value=1, value=20)
        method = st.selectbox("ODE method", ["Euler", "Heun", "RK2", "RK4", "SciPy RK45"], key="ode_method")
        if st.button("Run method", type="primary"):
            fn = make_function(expression, ("x", "y"))
            if method == "SciPy RK45": x, y = solve_ode_scipy(fn, x0, y0, xend, int(steps))
            else: x, y = solve_ode(fn, x0, y0, xend, int(steps), method)
            result_table([{"x": a, "y": b} for a, b in zip(x, y)]); plot_xy(x, y, f"{method} solution")
            if page == "ODE comparison":
                compare = []
                for candidate in ["Euler", "Heun", "RK2", "RK4"]:
                    _, values = solve_ode(fn, x0, y0, xend, int(steps), candidate)
                    compare.extend({"x": a, "y": b, "method": candidate} for a, b in zip(x, values))
                st.line_chart(pd.DataFrame(compare).pivot(index="x", columns="method", values="y"))
        if comparison_button("ode"):
            st.subheader("Compare ODE solvers")
            compare_a, compare_b = st.columns(2)
            ode_methods = ["Euler", "Heun", "RK2", "RK4", "SciPy RK45"]
            method_a = compare_a.selectbox("Solver 1", ode_methods, key="ode_compare_a")
            method_b = compare_b.selectbox("Solver 2", ode_methods, index=1, key="ode_compare_b")
            if st.button("Compare ODE results", type="primary", key="ode_compare_run"):
                if method_a == method_b:
                    raise ValueError("Choose two different ODE solvers.")
                fn_compare = make_function(expression, ("x", "y"))
                trajectories = []
                for candidate in (method_a, method_b):
                    if candidate == "SciPy RK45":
                        x_values, y_values = solve_ode_scipy(fn_compare, x0, y0, xend, int(steps))
                    else:
                        x_values, y_values = solve_ode(fn_compare, x0, y0, xend, int(steps), candidate)
                    trajectories.append((x_values, y_values))
                difference = np.abs(trajectories[0][1] - trajectories[1][1])
                result_table([
                    {"solver": method_a, "final y": trajectories[0][1][-1]},
                    {"solver": method_b, "final y": trajectories[1][1][-1]},
                ])
                st.metric("Maximum absolute difference", f"{np.max(difference):.6g}")
                st.metric("Final relative difference (%)", f"{difference[-1] / max(abs(trajectories[0][1][-1]), 1e-15) * 100:.6g}")
                plot_xy(trajectories[0][0], trajectories[0][1], f"{method_a} vs {method_b}",
                        scatter=(trajectories[1][0], trajectories[1][1]))
    elif page in {"Linear systems", "Factorizations", "Iterative solvers", "Condition number"}:
        st.header(page)
        A = matrix_editor(key=f"{page}_matrix")
        b = vector_editor(key=f"{page}_vector")
        if page == "Condition number":
            norm = st.selectbox("Norm", ["2", "1", "inf"], key="condition_norm")
            if st.button("Analyze conditioning", type="primary"):
                norm_value = {"2": None, "1": 1, "inf": np.inf}[norm]
                value = condition_number(A, norm_value)
                st.metric("Condition number", f"{value:.6g}")
                st.info("Large condition numbers indicate that small input perturbations may cause large changes in the solution.")
        elif page == "Linear systems":
            pivot = st.checkbox("Partial pivoting", True)
            if st.button("Solve", type="primary"):
                solution, steps = gaussian_elimination(A, b, pivot); st.metric("Residual norm", f"{np.linalg.norm(A@solution-b):.3e}"); st.write("Solution", solution); st.dataframe(pd.DataFrame(steps))
        elif page == "Factorizations":
            method = st.selectbox("Factorization", ["LU", "Cholesky"], key="factorization_method")
            if st.button("Factor", type="primary"):
                if method == "LU": L, U = _lu(A); st.write("L", L); st.write("U", U)
                else: st.write("L", cholesky(A))
        else:
            method = st.selectbox("Iterative method", ["Jacobi", "Gauss-Seidel"], key="iterative_method"); tol = st.number_input("Tolerance", value=1e-8, format="%.1e")
            if st.button("Solve", type="primary"):
                solution, history, converged = iterative_solve(A, b, method, tol)
                st.write("Solution", solution)
                if converged:
                    st.success("Converged")
                else:
                    st.warning("Not converged")
                result_table(history)
    elif page in {"Least squares", "QR / Gram–Schmidt / SVD"}:
        st.header(page)
        A = matrix_editor(key=f"{page}_matrix", default=pd.DataFrame([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], columns=["c1", "c2"]))
        b = vector_editor(key=f"{page}_vector", default=pd.DataFrame({"b": [1.0, 2.0, 2.0]}))
        if st.button("Run method", type="primary"):
            if page == "Least squares": x, residuals, rank, singular = least_squares(A, b); st.write("Coefficients", x); st.metric("Rank", rank); st.metric("Residual norm", f"{np.linalg.norm(A@x-b):.3e}")
            else:
                method = st.selectbox("Approximation method", ["QR", "Gram–Schmidt", "SVD"], key="approximation_method")
                if method == "QR": q, r = qr(A); st.write("Q", q); st.write("R", r)
                elif method == "Gram–Schmidt": q, r = gram_schmidt(A); st.write("Q", q); st.write("R", r)
                else: u, s, vh = svd(A); st.write("U", u); st.write("Singular values", s); st.write("Vᵀ", vh)
    elif page == "Optimization":
        st.header(page); mode = st.radio("Problem type", ["One-dimensional bounded", "Two-dimensional unconstrained"], horizontal=True)
        expression, parsed = equation_editor(
            "f(x)" if mode.startswith("One") else "f(x, y)",
            "(x-2)^2 + 1" if mode.startswith("One") else "(x-2)^2 + (y+1)^2",
            variables=("x",) if mode.startswith("One") else ("x", "y"),
            key="optimization_objective",
            examples=["(x-2)^2 + 1", "sin(x)^2 + x^2"] if mode.startswith("One") else ["(x-2)^2 + (y+1)^2", "x^2 + y^2"],
        )
        if parsed is None:
            st.stop()
        if mode.startswith("One"):
            a, b = st.columns(2); lo, hi = a.number_input("Lower bound", value=-5.0), b.number_input("Upper bound", value=5.0)
        else:
            a, b = st.columns(2); x0, y0 = a.number_input("Initial x", value=0.0), b.number_input("Initial y", value=0.0)
        if st.button("Optimize", type="primary"):
            if mode.startswith("One"):
                result = bounded_minimize(make_function(expression), lo, hi)
                st.metric("Optimum x", format_number(result.x, DISPLAY_DIGITS))
                st.metric("Objective", format_number(result.fun, DISPLAY_DIGITS))
                if result.success:
                    st.success(result.message)
                else:
                    st.warning(result.message)
                grid = np.linspace(lo, hi, 300); plot_xy(grid, make_function(expression)(grid), "Objective function")
            else:
                result = multivariable_minimize(make_multivariable_function(expression), [x0, y0])
                st.write("Optimum", [format_number(value, DISPLAY_DIGITS) for value in result.x])
                st.metric("Objective", format_number(result.fun, DISPLAY_DIGITS))
                if result.success:
                    st.success(result.message)
                else:
                    st.warning(result.message)
    elif page == "Eigenvalues":
        st.header(page); A = matrix_editor(key="eigen_matrix", default=pd.DataFrame([[2.0, 1.0], [1.0, 2.0]], columns=["c1", "c2"])); method = st.selectbox("Eigenvalue method", ["Power iteration", "Inverse power iteration", "NumPy eigensolver"], key="eigenvalue_method")
        shift = st.number_input("Shift", value=0.0) if method == "Inverse power iteration" else 0.0
        if st.button("Compute", type="primary"):
            if method == "Power iteration":
                value, vector, history, converged = power_iteration(A)
                st.metric("Dominant eigenvalue", f"{value:.10g}")
                st.write("Eigenvector", vector)
                result_table(history)
                if converged:
                    st.success("Converged")
                else:
                    st.warning("Not converged")
            elif method == "Inverse power iteration":
                value, vector, history, converged = inverse_power_iteration(A, shift=shift)
                st.metric("Eigenvalue near shift", f"{value:.10g}")
                st.write("Eigenvector", vector)
                result_table(history)
                if converged:
                    st.success("Converged")
                else:
                    st.warning("Not converged")
            else:
                values, vectors = eig(A); st.write("Eigenvalues", values); st.write("Eigenvectors", vectors)
    elif page == "Error analysis":
        st.header(page); exact = st.number_input("Exact value", value=1.0); approx = st.number_input("Approximation", value=.99)
        if st.button("Analyze", type="primary"):
            metrics = error_metrics(exact, approx); st.metric("Absolute error", f"{metrics['absolute']:.4g}"); st.metric("Percentage error", "undefined (exact value is zero)" if metrics["percentage"] is None else f"{metrics['percentage']:.4g}%")
        st.write(machine_info())
    elif page == "Rounding & truncation":
        st.header(page); value = st.number_input("Value", value=3.1415926535, format="%.10f"); digits = st.number_input("Decimal places", min_value=0, max_value=12, value=3)
        if st.button("Compare", type="primary"):
            data = rounding_demo(value, digits)
            result_table([data])
            st.info("Rounding chooses the nearest representable decimal; truncation discards digits toward zero.")
    else:
        st.header(page); value = st.number_input("Value", value=1.0); precision = st.selectbox("Precision", ["single", "double"], key="ieee_precision")
        if st.button("Inspect", type="primary"):
            data = ieee754(value, precision); st.json(data)
except (ValueError, ExpressionError, FloatingPointError) as exc:
    show_error(exc)
