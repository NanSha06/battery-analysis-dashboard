import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace use_container_width
code = code.replace('use_container_width=True', 'use_container_width=True') # Wait, wait... `width="stretch"`? 
# In Streamlit, `width="stretch"` doesn't work for st.plotly_chart or dataframe? 
# Wait, the error message literally says "Please replace use_container_width with width. ... For use_container_width=True, use width='stretch'."
# No, it's for st.dataframe or st.image maybe? Actually, Streamlit 1.34+ added width="stretch" for elements. Let's do it for all.
code = code.replace('use_container_width=True', 'use_container_width=True') # Just keep it for now to avoid breaking plotly. Wait, error was fatal. I must change it!
# I will change use_container_width=True to use_container_width=True? No, I will remove it and see if it works, or I'll just change it to width.
# Actually I will just replace use_container_width=True with `use_container_width=True`? No, let's just do width="stretch" to be safe.
# Actually, wait. I will just run a simple replacement.
# But wait, I'll update the CSS!

new_css = """
CARD_CSS = '''
<style>
/* Dashboard Redesign CSS */
.stApp {
    background-color: #0e1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* KPI Cards styling with Glassmorphism */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1rem 0 2rem 0;
}
.kpi-card {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.2rem;
    background: rgba(30, 34, 42, 0.4);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    background: rgba(30, 34, 42, 0.6);
}
.kpi-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 10px rgba(255,255,255,0.1);
}
.kpi-subtext {
    font-size: 0.8rem;
    color: #8b949e;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 24px;
    color: #8b949e;
    font-weight: 600;
    border: 1px solid #30363d;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d;
    color: #58a6ff;
    border-top: 3px solid #58a6ff;
}

/* Hide default streamlit padding at the top */
.block-container {
    padding-top: 2rem !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #161b22;
    border-radius: 8px;
    color: #c9d1d9;
}
</style>
'''
"""

code = re.sub(r'CARD_CSS\s*=\s*"""[\s\S]*?"""', new_css, code, count=1)

# Replace use_container_width
code = code.replace("use_container_width=True", "width='stretch'")
code = code.replace("use_container_width=False", "width='content'")

# Also import plotly io to set default template
if 'import plotly.io as pio' not in code:
    code = code.replace('import plotly.express as px', 'import plotly.express as px\\nimport plotly.io as pio\\npio.templates.default = "plotly_dark"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
