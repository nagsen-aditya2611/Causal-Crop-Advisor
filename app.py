import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Causal Crop Decision Support System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load Models (Cached for Performance)
# --------------------------------------------------
@st.cache_resource
def load_models():
    try:
        return {
            "rf": joblib.load("data/models/rf_causal.pkl"),
            "xgb": joblib.load("data/models/xgb_causal.pkl"),
            "lgb": joblib.load("data/models/lgb_causal.pkl"),
        }
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

models = load_models()
rf_model = models["rf"]
xgb_model = models["xgb"]
lgb_model = models["lgb"]

FEATURE_COLUMNS = [
    "avg_ssm", "avg_tavg", "avg_rad", "avg_cwb",
    "avg_ndvi", "avg_fpar", "awc", "bulk_density", "drainage"
]

# --------------------------------------------------
# Helper Function for Safe Prediction (Prevents Crashes)
# --------------------------------------------------
def safe_predict(model, df_input, raw_array):
    """
    Attempts to predict using DataFrame. If column names mismatch what 
    the model expects, falls back to raw NumPy array to prevent crashing.
    """
    try:
        if hasattr(model, "feature_names_in_"):
            expected_names = list(model.feature_names_in_)
            if set(expected_names).issubset(set(df_input.columns)):
                return float(model.predict(df_input[expected_names])[0])
        return float(model.predict(df_input)[0])
    except Exception:
        return float(model.predict(raw_array)[0])

# --------------------------------------------------
# Hide Streamlit Default Menus
# --------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
section[data-testid="stSidebar"]{ display:none; }
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌾 Causal Crop Advisor")
st.caption("Helping Farmers Focus on the Factors that Truly Influence Crop Yield")
st.divider()

# --------------------------------------------------
# Field Information Inputs
# --------------------------------------------------
st.header("📝 Field Information")
st.caption("ℹ️ Crop, Year and Area are used for report metadata — not for yield prediction.")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Crop", ["Wheat"])
    harvest_year = st.number_input("Harvest Year", min_value=1990, max_value=2100, value=2020, step=1)
    harvest_area = st.number_input("Harvest Area (ha)", min_value=0.1, value=100.0, step=1.0)
    awc = st.number_input("Available Water Capacity", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
    bulk_density = st.number_input("Bulk Density (g/cm³)", min_value=0.5, max_value=2.5, value=1.30, step=0.05)
    drainage = st.selectbox("Drainage Class (1=Poor, 5=Excellent)", [1, 2, 3, 4, 5], index=2)

with col2:
    avg_ssm = st.number_input("Average Soil Moisture", min_value=0.0, max_value=1.0, value=0.32, step=0.01)
    avg_tavg = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=55.0, value=24.5, step=0.1)
    avg_rad = st.number_input("Solar Radiation (MJ/m²/day)", min_value=0.0, max_value=40.0, value=18.5, step=0.1)
    avg_cwb = st.number_input("Climatic Water Balance (mm)", min_value=-200.0, max_value=200.0, value=-8.0, step=0.5)
    avg_ndvi = st.number_input("Average NDVI", min_value=-1.0, max_value=1.0, value=0.45, step=0.01)
    avg_fpar = st.number_input("Average FPAR", min_value=0.0, max_value=1.0, value=0.55, step=0.01)

st.divider()
run = st.button("🌾 Run Causal Analysis", use_container_width=True, type="primary")

if not run:
    st.stop()

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------
raw_values = [[
    avg_ssm, avg_tavg, avg_rad, avg_cwb,
    avg_ndvi, avg_fpar, awc, bulk_density, drainage
]]

input_df = pd.DataFrame(raw_values, columns=FEATURE_COLUMNS)
input_np = np.array(raw_values)

# --------------------------------------------------
# Generate Predictions (Fail-safe)
# --------------------------------------------------
with st.spinner("🔍 Running causal model predictions..."):
    rf_pred  = safe_predict(rf_model, input_df, input_np)
    xgb_pred = safe_predict(xgb_model, input_df, input_np)
    lgb_pred = safe_predict(lgb_model, input_df, input_np)

predicted_yield = xgb_pred

# Calculate Yield Category
if predicted_yield >= 5.5:
    category = "🟢 High Yield Potential"
elif predicted_yield >= 4.0:
    category = "🟡 Moderate Yield Potential"
else:
    category = "🔴 Low Yield Potential"

# --------------------------------------------------
# Causal Impact Dashboard
# --------------------------------------------------
st.divider()
st.header("📊 Causal Impact Dashboard")

row1 = st.columns(3)
with row1[0]:
    st.success(f"### 💧 Soil Moisture\n\n**Importance:** 🔴 Very High\n\nStatus: {'⚠ Needs Attention' if avg_ssm < 0.30 else '✅ Good'}")
with row1[1]:
    st.success(f"### 🌡 Temperature\n\n**Importance:** 🟠 High\n\nStatus: {'⚠ High' if avg_tavg > 30 else '✅ Normal'}")
with row1[2]:
    st.success("### ☀ Solar Radiation\n\n**Importance:** 🟡 Medium\n\nStatus: ✅ Stable")

row2 = st.columns(3)
with row2[0]:
    st.success(f"### 🌦 Water Balance\n\n**Importance:** 🟠 High\n\nStatus: {'⚠ Water Deficit' if avg_cwb < -10 else '✅ Balanced'}")
with row2[1]:
    st.success(f"### 🌍 Water Capacity\n\n**Importance:** 🟡 Medium\n\nStatus: {'⚠ Low' if awc < 0.15 else '✅ Good'}")
with row2[2]:
    st.success(f"### 🪨 Soil Condition\n\n**Importance:** 🟠 High\n\nStatus: {'⚠ Compacted' if bulk_density > 1.45 else '✅ Healthy'}")

st.success(f"### 🚜 Drainage\n\n**Importance:** 🟠 High\n\nStatus: {'⚠ Poor' if drainage <= 2 else '✅ Good'}")

# --------------------------------------------------
# Variables Needing Attention
# --------------------------------------------------
st.divider()
st.header("⭐ Variables That Need Attention")
st.write("Based on causal analysis, the following factors have the strongest direct influence on crop yield.")

variables = [
    ("💧 Soil Moisture", "Very High Impact", "Maintain adequate soil moisture throughout the growing season."),
    ("🌡 Temperature", "High Impact", "Monitor temperature during critical crop growth stages."),
    ("☀ Solar Radiation", "Medium Impact", "Adequate sunlight is essential for photosynthesis."),
    ("🌦 Climatic Water Balance", "High Impact", "Balance between rainfall and evapotranspiration affects yield."),
    ("🌍 Available Water Capacity", "Medium Impact", "Higher water holding capacity improves crop resilience."),
    ("🪨 Bulk Density", "Medium Impact", "Avoid excessive soil compaction for healthy root growth."),
    ("🚜 Drainage Class", "High Impact", "Proper drainage prevents waterlogging and crop stress.")
]

for icon, impact, advice in variables:
    st.info(f"### {icon}\n\n**Impact Level:** {impact}\n\n{advice}")

# --------------------------------------------------
# Yield Prediction Section
# --------------------------------------------------
st.divider()
st.header("🌾 Estimated Crop Yield")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌲 Random Forest", f"{rf_pred:.2f} t/ha")
with col2:
    st.metric("⚡ XGBoost (Best)", f"{xgb_pred:.2f} t/ha")
with col3:
    st.metric("💡 LightGBM", f"{lgb_pred:.2f} t/ha")

st.success(f"## Final Recommendation\n\n### Expected Yield\n# 🌾 {predicted_yield:.2f} tonnes/hectare\n\n**Field Status:** {category}")

# --------------------------------------------------
# Recommendations
# --------------------------------------------------
st.divider()
st.header("👨‍🌾 Recommended Actions")

recommendations = []
if avg_ssm < 0.30: recommendations.append("💧 Increase irrigation to improve soil moisture.")
if avg_tavg > 30: recommendations.append("🌡 High temperature detected. Consider heat mitigation strategies.")
if avg_cwb < -10: recommendations.append("🌦 Water deficit detected. Improve water management practices.")
if drainage <= 2: recommendations.append("🚜 Improve field drainage to reduce waterlogging risk.")
if bulk_density > 1.45: recommendations.append("🪨 High soil compaction detected. Consider soil loosening practices.")
if awc < 0.15: recommendations.append("🌍 Low water holding capacity. Organic matter addition is recommended.")

if not recommendations:
    st.success("Current field conditions look favourable. Continue regular irrigation, soil management, and crop monitoring.")
else:
    for rec in recommendations:
        st.warning(rec)

# --------------------------------------------------
# Why These Recommendations
# --------------------------------------------------
st.divider()
with st.expander("📖 Why were these recommendations generated?"):
    st.markdown("""
This advisory system is based on **causal inference** rather than standard correlation.

Instead of assuming every feature contributes equally, the pipeline selects features with a **direct causal link** to yield.

Machine learning models predict yield strictly from these causally selected features:
- Soil Moisture, Temperature, Solar Radiation, Climatic Water Balance
- NDVI, FPAR, Available Water Capacity, Bulk Density, Drainage Class
""")

# --------------------------------------------------
# Enhanced PDF Report Generator
# --------------------------------------------------
st.divider()
st.header("📄 Download Farmer Report")

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, title="Causal Crop Advisor Report")
    styles = getSampleStyleSheet()
    story = []

    # 1. Header & Title
    story.append(Paragraph("<b>CAUSAL CROP ADVISOR REPORT</b>", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    # 2. Field Summary
    story.append(Paragraph("<b>Field Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"Crop: <b>{crop}</b> | Year: <b>{harvest_year}</b> | Area: <b>{harvest_area} ha</b>", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    # 3. Input Parameters Table
    story.append(Paragraph("<b>Analyzed Field Parameters</b>", styles["Heading2"]))
    input_table_data = [
        ["Parameter", "Input Value", "Parameter", "Input Value"],
        ["Soil Moisture", f"{avg_ssm:.2f}", "NDVI", f"{avg_ndvi:.2f}"],
        ["Temperature (°C)", f"{avg_tavg:.1f}", "FPAR", f"{avg_fpar:.2f}"],
        ["Solar Rad. (MJ/m²)", f"{avg_rad:.1f}", "Water Capacity (AWC)", f"{awc:.2f}"],
        ["Water Balance (mm)", f"{avg_cwb:.1f}", "Bulk Density (g/cm³)", f"{bulk_density:.2f}"],
        ["Drainage Class", f"{drainage}", "", ""]
    ]
    t_input = Table(input_table_data, colWidths=[1.8 * inch, 1.0 * inch, 1.8 * inch, 1.0 * inch])
    t_input.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t_input)
    story.append(Spacer(1, 0.15 * inch))

    # 4. Yield Predictions Table
    story.append(Paragraph("<b>Yield Predictions & Status</b>", styles["Heading2"]))
    pred_table = [
        ["Model", "Yield (t/ha)"],
        ["Random Forest", f"{rf_pred:.2f}"],
        ["XGBoost (Selected Best)", f"{xgb_pred:.2f}"],
        ["LightGBM", f"{lgb_pred:.2f}"],
    ]
    t_pred = Table(pred_table, colWidths=[2.8 * inch, 2.8 * inch])
    t_pred.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t_pred)
    story.append(Paragraph(f"<br/><b>Status Category:</b> {category}", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    # 5. Recommended Actions
    story.append(Paragraph("<b>Recommended Actions</b>", styles["Heading2"]))
    if not recommendations:
        story.append(Paragraph("Field environmental conditions meet standard baseline targets. Continue regular irrigation, nutrient, and crop monitoring.", styles["BodyText"]))
    else:
        for rec in recommendations:
            story.append(Paragraph("• " + rec, styles["BodyText"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

st.download_button(
    "⬇ Download PDF Report",
    generate_pdf(),
    file_name=f"Farmer_Report_{harvest_year}.pdf",
    mime="application/pdf"
)

st.divider()
st.caption("Developed as part of a Causal Crop Yield Forecasting project.")
