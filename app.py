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

# ⚠️ Feature order used during training
FEATURE_COLUMNS = [
    "avg_ssm", "avg_tavg", "avg_rad", "avg_cwb",
    "avg_ndvi", "avg_fpar", "awc", "bulk_density", "drainage"
]

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
# Predictions Input (Pandas DataFrame prevents column misalignment)
# --------------------------------------------------
input_data = pd.DataFrame([[
    avg_ssm, avg_tavg, avg_rad, avg_cwb,
    avg_ndvi, avg_fpar, awc, bulk_density, drainage
]], columns=FEATURE_COLUMNS)

with st.spinner("🔍 Running causal model predictions..."):
    rf_pred = float(rf_model.predict(input_data)[0])
    xgb_pred = float(xgb_model.predict(input_data)[0])
    lgb_pred = float(lgb_model.predict(input_data)[0])

predicted_yield = xgb_pred

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

if predicted_yield >= 5.5:
    category = "🟢 High Yield Potential"
elif predicted_yield >= 4.0:
    category = "🟡 Moderate Yield Potential"
else:
    category = "🔴 Low Yield Potential"

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
# PDF Report Generator
# --------------------------------------------------
st.divider()
st.header("📄 Download Farmer Report")

buffer = io.BytesIO()
doc = SimpleDocTemplate(buffer, title="Causal Crop Advisor Report")
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>CAUSAL CROP ADVISOR REPORT</b>", styles["Title"]))
story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>Field Summary</b>", styles["Heading2"]))
story.append(Paragraph(f"Crop: {crop} | Year: {harvest_year} | Area: {harvest_area} ha", styles["BodyText"]))
story.append(Spacer(1, 0.1 * inch))

story.append(Paragraph("<b>Yield Predictions</b>", styles["Heading2"]))
pred_table = [
    ["Model", "Yield (t/ha)"],
    ["Random Forest", f"{rf_pred:.2f}"],
    ["XGBoost (Selected Best)", f"{xgb_pred:.2f}"],
    ["LightGBM", f"{lgb_pred:.2f}"],
]
t = Table(pred_table, colWidths=[2.5 * inch, 2.5 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(t)
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>Recommended Actions</b>", styles["Heading2"]))
if not recommendations:
    story.append(Paragraph("Field conditions are favourable. Continue normal management.", styles["BodyText"]))
else:
    for rec in recommendations:
        story.append(Paragraph("• " + rec, styles["BodyText"]))

doc.build(story)
pdf_bytes = buffer.getvalue()
buffer.close()

st.download_button(
    "⬇ Download PDF Report",
    pdf_bytes,
    file_name=f"Farmer_Report_{harvest_year}.pdf",
    mime="application/pdf"
)

st.divider()
st.caption("Developed as part of a Causal Crop Yield Forecasting project.")
