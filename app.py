import streamlit as st

import joblib
import numpy as np

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Causal Crop Decision Support System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

rf_model = joblib.load("models/rf_causal.pkl")
xgb_model = joblib.load("models/xgb_causal.pkl")
lgb_model = joblib.load("models/lgb_causal.pkl")

# --------------------------------------------------
# Hide Streamlit default menu & sidebar
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"]{
    display:none;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🌾 Causal Crop Advisor")

st.caption(
    "Helping Farmers Focus on the Factors that Truly Influence Crop Yield"
)

st.divider()

# ==========================================
# FARM INFORMATION
# ==========================================

st.header("📝 Field Information")

col1, col2 = st.columns(2)

with col1:

    crop = st.selectbox(
        "Crop",
        ["Wheat"]
    )

    harvest_year = st.number_input(
        "Harvest Year",
        value=2020
    )

    harvest_area = st.number_input(
        "Harvest Area (ha)",
        value=100.0
    )

    awc = st.number_input(
        "Available Water Capacity",
        value=0.18
    )

    bulk_density = st.number_input(
        "Bulk Density",
        value=1.30
    )

    drainage = st.selectbox(
        "Drainage Class",
        [1,2,3,4,5]
    )

with col2:

    avg_ssm = st.number_input(
        "Average Soil Moisture",
        value=0.32
    )

    avg_tavg = st.number_input(
        "Average Temperature (°C)",
        value=24.5
    )

    avg_rad = st.number_input(
        "Solar Radiation",
        value=18.5
    )

    avg_cwb = st.number_input(
        "Climatic Water Balance",
        value=-8.0
    )
    avg_ndvi = st.number_input(
    "Average NDVI",
    value=0.45
    )

    avg_fpar = st.number_input(
    "Average FPAR",
    value=0.55
    )

st.divider()

run = st.button(
    "🌾 Run Causal Analysis",
    use_container_width=True
)

if not run:
    st.stop()

# ==========================================
# CAUSAL IMPACT DASHBOARD
# ==========================================

st.divider()

st.header("📊 Causal Impact Dashboard")

dashboard = [
    ["💧 Soil Moisture", "Very High", "⚠ Needs Attention" if avg_ssm < 0.30 else "✅ Good"],
    ["🌡 Temperature", "High", "⚠ High" if avg_tavg > 30 else "✅ Normal"],
    ["☀ Solar Radiation", "Medium", "✅ Stable"],
    ["🌦 Climatic Water Balance", "High", "⚠ Water Deficit" if avg_cwb < -10 else "✅ Balanced"],
    ["🌍 Water Capacity", "Medium", "⚠ Low" if awc < 0.15 else "✅ Good"],
    ["🪨 Bulk Density", "Medium", "⚠ Compacted" if bulk_density > 1.45 else "✅ Normal"],
    ["🚜 Drainage", "High", "⚠ Poor" if drainage <= 2 else "✅ Good"],
]

# ==========================================
# CAUSAL IMPACT CARDS
# ==========================================

row1 = st.columns(3)

with row1[0]:
    st.success("""
### 💧 Soil Moisture

**Importance:** 🔴 Very High

Status:
{}

""".format("⚠ Needs Attention" if avg_ssm < 0.30 else "✅ Good"))

with row1[1]:
    st.success("""
### 🌡 Temperature

**Importance:** 🟠 High

Status:
{}

""".format("⚠ High" if avg_tavg > 30 else "✅ Normal"))

with row1[2]:
    st.success("""
### ☀ Solar Radiation

**Importance:** 🟡 Medium

Status:

✅ Stable
""")

row2 = st.columns(3)

with row2[0]:
    st.success("""
### 🌦 Water Balance

**Importance:** 🟠 High

Status:
{}

""".format("⚠ Water Deficit" if avg_cwb < -10 else "✅ Balanced"))

with row2[1]:
    st.success("""
### 🌍 Water Capacity

**Importance:** 🟡 Medium

Status:
{}

""".format("⚠ Low" if awc < 0.15 else "✅ Good"))

with row2[2]:
    st.success("""
### 🪨 Soil Condition

**Importance:** 🟠 High

Status:
{}

""".format("⚠ Compacted" if bulk_density > 1.45 else "✅ Healthy"))

st.success("""
### 🚜 Drainage

**Importance:** 🟠 High

Status:
{}
""".format("⚠ Poor" if drainage <= 2 else "✅ Good"))


# ==========================================
# CAUSAL ANALYSIS RESULTS
# ==========================================

st.divider()

st.header("⭐ Variables That Need Attention")

st.write(
    "Based on the causal analysis, the following factors have the strongest influence on crop yield."
)

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

    st.info(f"""
### {icon}

**Impact Level:** {impact}

{advice}
""")

# Prepare Model Input

input_data = np.array([[
    avg_ssm,
    avg_tavg,
    avg_rad,
    avg_cwb,
    avg_ndvi,
    avg_fpar,
    awc,
    bulk_density,
    drainage
]])

# Predictions

rf_pred = rf_model.predict(input_data)[0]
xgb_pred = xgb_model.predict(input_data)[0]
lgb_pred = lgb_model.predict(input_data)[0]
    

# ==========================================
# YIELD PREDICTION
# ==========================================

st.divider()

st.header("🌾 Estimated Crop Yield")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌲 Random Forest",
        f"{rf_pred:.2f} t/ha"
    )

with col2:
    st.metric(
        "⚡ XGBoost",
        f"{xgb_pred:.2f} t/ha"
    )

with col3:
    st.metric(
        "💡 LightGBM",
        f"{lgb_pred:.2f} t/ha"
    )

predicted_yield = xgb_pred

# Yield Category

if predicted_yield >= 5.5:
    category = "🟢 High Yield Potential"

elif predicted_yield >= 4:
    category = "🟡 Moderate Yield Potential"

else:
    category = "🔴 Low Yield Potential"

st.success(f"""

## Final Recommendation

### Expected Yield

# 🌾 {predicted_yield:.2f} tonnes/hectare

**Field Status**

{category}

""")

# ==========================================
# WHY THIS YIELD?
# ==========================================

st.divider()

st.header("🧠 Why was this Yield Predicted?")

st.write(
    "The prediction is based on the causally important variables identified during the causal analysis."
)

col1, col2 = st.columns(2)

with col1:

    st.info(f"""
### 💧 Soil Moisture

Current Value: **{avg_ssm:.2f}**

Importance: 🔴 Very High

{ "⚠ Low soil moisture may reduce crop yield."
if avg_ssm < 0.30
else "✅ Soil moisture is within a favourable range."}
""")

    st.info(f"""
### 🌡 Temperature

Current Value: **{avg_tavg:.1f} °C**

Importance: 🟠 High

{ "⚠ Elevated temperature may stress the crop."
if avg_tavg > 30
else "✅ Temperature is suitable for crop growth."}
""")

with col2:

    st.info(f"""
### 🌦 Climatic Water Balance

Current Value: **{avg_cwb:.1f}**

Importance: 🟠 High

{ "⚠ Water deficit detected."
if avg_cwb < -10
else "✅ Water balance is acceptable."}
""")

    st.info(f"""
### 🌱 Vegetation Condition

NDVI : **{avg_ndvi:.2f}**

FPAR : **{avg_fpar:.2f}**

Importance: 🟡 Moderate

Healthy vegetation generally contributes to improved crop productivity.
""")


# ==========================================
# FARMER RECOMMENDATIONS
# ==========================================

st.divider()

st.header("👨‍🌾 Recommended Actions")

recommendations = []

if avg_ssm < 0.30:
    recommendations.append(
        "💧 Increase irrigation to improve soil moisture."
    )

if avg_tavg > 30:
    recommendations.append(
        "🌡 High temperature detected. Consider heat mitigation strategies."
    )

if avg_cwb < -10:
    recommendations.append(
        "🌦 Water deficit detected. Improve water management practices."
    )

if drainage <= 2:
    recommendations.append(
        "🚜 Improve field drainage to reduce waterlogging risk."
    )

if bulk_density > 1.45:
    recommendations.append(
        "🪨 High soil compaction detected. Consider soil loosening practices."
    )

if awc < 0.15:
    recommendations.append(
        "🌍 Low water holding capacity. Organic matter addition is recommended."
    )

if len(recommendations) == 0:

    st.success("""
Current field conditions look favourable.

Continue maintaining:

• Irrigation

• Soil health

• Nutrient management

• Crop monitoring
""")

else:

    for rec in recommendations:

        st.warning(rec)


# ==========================================
# WHY THESE RECOMMENDATIONS?
# ==========================================

st.divider()

with st.expander("📖 Why were these recommendations generated?"):

    st.markdown("""
This advisory system is based on **causal inference** rather than simple correlations.

Instead of assuming that every available variable contributes equally to crop yield,
the analysis identifies variables with a **direct causal influence** on yield.

The recommendation engine focuses on the following causally important variables:

- 💧 Average Soil Moisture
- 🌡 Average Temperature
- ☀ Solar Radiation
- 🌦 Climatic Water Balance
- 🌿 NDVI
- 🌱 FPAR
- 🌍 Available Water Capacity
- 🪨 Bulk Density
- 🚜 Drainage Class

Machine Learning models are then used to estimate crop yield using only these
causally selected variables, improving both interpretability and practical decision making.
""")

# ==========================================
# DOWNLOAD PDF REPORT
# ==========================================

st.divider()

st.header("📄 Download Farmer Report")

buffer = io.BytesIO()

doc = SimpleDocTemplate(buffer)

styles = getSampleStyleSheet()

story = []

story.append(Paragraph("<b>CAUSAL CROP ADVISOR REPORT</b>", styles["Title"]))

story.append(Paragraph("<br/>", styles["BodyText"]))

story.append(Paragraph(f"<b>Crop:</b> {crop}", styles["BodyText"]))
story.append(Paragraph(f"<b>Harvest Year:</b> {harvest_year}", styles["BodyText"]))
story.append(Paragraph(f"<b>Harvest Area:</b> {harvest_area} ha", styles["BodyText"]))

story.append(Paragraph("<br/>", styles["BodyText"]))

story.append(Paragraph("<b>Yield Prediction</b>", styles["Heading2"]))

story.append(Paragraph(f"Random Forest : {rf_pred:.2f} t/ha", styles["BodyText"]))
story.append(Paragraph(f"XGBoost : {xgb_pred:.2f} t/ha", styles["BodyText"]))
story.append(Paragraph(f"LightGBM : {lgb_pred:.2f} t/ha", styles["BodyText"]))

story.append(Paragraph("<br/>", styles["BodyText"]))

story.append(Paragraph("<b>Final Recommendation</b>", styles["Heading2"]))
story.append(Paragraph(f"{category}", styles["BodyText"]))

story.append(Paragraph("<br/>", styles["BodyText"]))

story.append(Paragraph("<b>Recommended Actions</b>", styles["Heading2"]))

if len(recommendations) == 0:

    story.append(
        Paragraph(
            "Current field conditions look favourable. Continue maintaining irrigation, soil health and crop monitoring.",
            styles["BodyText"]
        )
    )

else:

    for rec in recommendations:

        story.append(
            Paragraph("• " + rec, styles["BodyText"])
        )

story.append(Paragraph("<br/>", styles["BodyText"]))

story.append(Paragraph("<b>Causally Important Variables</b>", styles["Heading2"]))

important_features = [
    "Average Soil Moisture",
    "Average Temperature",
    "Solar Radiation",
    "Climatic Water Balance",
    "NDVI",
    "FPAR",
    "Available Water Capacity",
    "Bulk Density",
    "Drainage Class"
]

for feat in important_features:

    story.append(
        Paragraph("• " + feat, styles["BodyText"])
    )

doc.build(story)

pdf = buffer.getvalue()

buffer.close()

st.download_button(
    "⬇ Download PDF Report",
    pdf,
    file_name="Farmer_Report.pdf",
    mime="application/pdf"
)


st.divider()

st.caption(
    "Developed as part of a Causal Crop Yield Forecasting project using Causal Inference and Machine Learning."
)
