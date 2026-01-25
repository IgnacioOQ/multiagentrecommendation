import streamlit as st
import sys
import os
import streamlit.components.v1 as components

# Page Layout
st.set_page_config(layout="wide", page_title="Master Plan Visualization")

# Title
st.title("Recommender Systems: Master Plan")

# Path Handling to find planner/visualize_html.py
current_dir = os.path.dirname(os.path.abspath(__file__))
planner_dir = os.path.dirname(current_dir)
sys.path.append(planner_dir)

try:
    import visualize_html
except ImportError as e:
    st.error(f"Failed to import visualize_html: {e}")
    st.stop()

# Generate HTML
target_file = os.path.join(planner_dir, "MASTER_PLAN.md")
if not os.path.exists(target_file):
    st.error(f"MASTER_PLAN.md not found at {target_file}")
    st.stop()

with st.spinner("Generating Visualization..."):
    html_content = visualize_html.generate_html(target_file)

# Render HTML
# We set a large height to accommodate the d3 tree
components.html(html_content, height=1000, scrolling=True)

st.sidebar.markdown("### Info")
st.sidebar.info("This view provides a live, interactive visualization of the underlying Master Plan markdown file.")
