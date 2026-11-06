from __future__ import annotations

import streamlit as st

from modules.expression_parser import ExpressionError, parse_expression


def equation_editor(label, default, variables=("x",), key="equation", examples=None):
    examples = examples or [default]
    state_key = f"{key}_value"
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    options = ["Custom"] + [item for item in examples if item != default]
    selected = st.selectbox("Example", options, key=f"{key}_example")
    left, right = st.columns([5, 1])
    if selected != "Custom":
        st.session_state[state_key] = selected
    if right.button("Reset", key=f"{key}_reset"):
        st.session_state[state_key] = default
    expression = left.text_input(label, key=state_key, help="Use natural notation such as 2x, x^2, sin(x), or ln(x).")
    try:
        parsed = parse_expression(expression, variables)
    except ExpressionError as exc:
        st.error(f"Invalid expression: {exc}")
        return expression, None
    st.success("Valid expression")
    st.latex(f"{label.split('(')[0].strip()} = {__import__('sympy').latex(parsed)}")
    return expression, parsed
