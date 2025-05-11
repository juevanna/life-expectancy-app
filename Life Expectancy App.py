import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import geopandas as gpd
import pycountry
from sklearn.preprocessing import StandardScaler


lr_model  = joblib.load("model/linear_regression_model.pkl")
rf_model  = joblib.load("model/random_forest_model_compressed.pkl")
gb_model  = joblib.load("model/gradient_boosting_model.pkl")
xgb_model = joblib.load("model/xgboost_model.pkl")
nn_model  = joblib.load("model/neural_network_model.pkl")
svr_model = joblib.load("model/svr_model.pkl")
scaler    = joblib.load("model/scaler.pkl")


df = pd.read_csv("dataset/pivoted_dataset.csv")



def get_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

df["ISO3"] = df["GEO_NAME_SHORT"].apply(get_iso3)

country_iso_map = {
    "Bolivia (Plurinational State of)": "BOL",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Democratic Republic of the Congo": "COD",
    "Micronesia (Federated States of)": "FSM",
    "Iran (Islamic Republic of)": "IRN",
    "Republic of Korea": "KOR"
}
df["ISO3"] = df["ISO3"].fillna(df["GEO_NAME_SHORT"].map(country_iso_map))

latest = (
    df
    .sort_values("DIM_TIME")
    .groupby("ISO3", as_index=False)
    .last()
)

FEATURE_LIST = [
    'Air pollution deaths (age-standardized) - RATE_PER_100000_N',
    'Alcohol consumption (age 15+) - RATE_PER_CAPITA_N',
    'Births attended by health personnel - RATE_PER_100_N',
    'DTP3 immunization coverage (age 1) - RATE_PER_100_N',
    'Density of doctors - RATE_PER_10000_N',
    'Density of nurses and midwives - RATE_PER_10000_N',
    'Density of pharmacists - RATE_PER_10000_N',
    'Development assistance to medical research and basic health - MONEY_N',
    'Family planning satisfied with modern methods - RATE_PER_100_N',
    'General government expenditure on domestic health - RATE_PER_100_N',
    'HIV infections - RATE_PER_1000_N',
    'Hepatitis B surface antigen (HBsAg) in children (under 5) - RATE_PER_100_N',
    'Homicide deaths - RATE_PER_100000_N',
    'Hypertension in adults (age 30 to 79) - RATE_PER_100_N',
    'Intimate partner violence prevalence among ever partnered women in the previous 12 months (%) - RATE_PER_100_N',
    'Intimate partner violence prevalence among ever partnered women in their lifetime (%) - RATE_PER_100_N',
    'Malaria cases - RATE_PER_1000_N',
    'Maternal mortality ratio - RATE_PER_100000_N',
    'Mean particulates (PM2.5) in urban areas - RATE_N',
    'Mortality rate (neonetal) - RATE_PER_1000_N',
    'Mortality rate (under 5) - RATE_PER_1000_N',
    'Non-communicable diseases deaths (age 30 to 70) - RATE_PER_100_N',
    'Obesity in adults (age 18+) - RATE_PER_100_N',
    'Obesity in children (age 5 to 19) - RATE_PER_100_N',
    'People requiring interventions against Neglected Tropical Diseases (NTDs) - COUNT_N',
    'Poisoning deaths (unintentional) - RATE_PER_100000_N',
    'Population using hand-washing facilities with soap and water - PERCENT_POP_N',
    'Population using safely managed drinking-water services - PERCENT_POP_N',
    'Population using safely managed sanitation services - PERCENT_POP_N',
    'Population with primary reliance on clean fuels - PERCENT_POP_N',
    'Road traffic deaths - RATE_PER_100000_N',
    'Safely treated domestic wastewater flows - RATE_PER_100_N',
    'Stunting in children (under 5) - RATE_PER_100_N',
    'Suicide deaths - RATE_PER_100000_N',
    'Tobacco use - PERCENT_POP_N',
    'Tuberculosis cases - RATE_PER_100000_N',
    'Unsafe water, sanitation and hygiene services deaths - RATE_PER_100000_N',
    'Wasting in children (under 5) - RATE_PER_100_N'
]


feature_means = df[FEATURE_LIST].mean().to_dict()

world = gpd.read_file(
    r"naturalearth_lowres/ne_110m_admin_0_countries.shp"
)

if 'iso_a3' in world.columns:
    world['ISO3'] = world['iso_a3']
elif 'ISO_A3' in world.columns:
    world['ISO3'] = world['ISO_A3']
elif 'ADM0_A3' in world.columns:
    world['ISO3'] = world['ADM0_A3']
else:
    st.error("No ISO3 field found in world shapefile!")


for c in ('iso_a3','ISO_A3','ADM0_A3'):
    if c in world.columns:
        world.drop(columns=[c], inplace=True)


st.title("Life Expectancy Prediction")
tab1, tab2, tab3 = st.tabs(["Manual Input", "World Map", "Feature Importance"])
with tab1:
    st.header("Country-Specific Life Expectancy Prediction")

    
    model_choice = st.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting", 
         "XGBoost", "Neural Network", "Support Vector Regression"],
        key="tab1_model"
)
    country_choice = st.selectbox(
        "Select Country",
        sorted(latest["GEO_NAME_SHORT"].unique()),
        key="tab1_country"
)


    country_row = latest[latest["GEO_NAME_SHORT"] == country_choice].iloc[0]

    cols = st.columns(4)
    data = {}
    for i, feat in enumerate(FEATURE_LIST):
        label = feat.split(" - ")[0]
        val = country_row.get(feat, np.nan)
        if pd.isna(val):
            val = feature_means[feat]
        data[feat] = cols[i % 4].number_input(label, min_value=0.0, value=float(val))
    input_df = pd.DataFrame(data, index=[0])


    X_raw = input_df.values
    if model_choice in ("Linear Regression", "Neural Network", "Support Vector Regression"):
        X_pred = scaler.transform(X_raw)
    else:
        X_pred = X_raw


    model_dict = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model,
        "Neural Network": nn_model,
        "Support Vector Regression": svr_model
}
    raw_pred = model_dict[model_choice].predict(X_pred)[0]
    pred_value = max(0.0, min(float(raw_pred), 120.0))

    life_cols = [c for c in latest.columns if "life expectancy" in c.lower()]
    if life_cols:
        actual = country_row[life_cols[0]]
    else:
        actual = None


    st.header(f"{country_choice} Predicted Life Expectancy")


    st.markdown(f"## {pred_value:.2f} years")


    if actual is not None:
        st.markdown(f"*This is a {abs(pct):.2f}% overall {direction}*")

with tab2:
    st.header("Global Predictions Map")

    
    map_model_choice = st.selectbox(
        "Choose Model for Map",
        [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Neural Network",
            "Support Vector Regression"
        ],
        key="tab2_model"
    )

    latest_feats = latest[FEATURE_LIST].copy()
    latest_feats.fillna(latest_feats.mean(), inplace=True)
    X_ctry = latest_feats.values


    if map_model_choice in ("Linear Regression", "Neural Network", "Support Vector Regression"):
        X_ctry = scaler.transform(X_ctry)

    model_dict = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model,
        "Neural Network": nn_model,
        "Support Vector Regression": svr_model
    }
    raw_preds = model_dict[map_model_choice].predict(X_ctry)
    latest["predicted_life_expectancy"] = np.round(raw_preds, 2)

    geo = world.merge(latest, on="ISO3", how="left")

    geo["predicted_life_expectancy_str"] = geo["predicted_life_expectancy"]\
        .map(lambda x: f"{x:.2f}" if pd.notna(x) else None)
    geo["tooltip_text"] = geo["predicted_life_expectancy_str"]
    geo.loc[geo["ISO3"] == "GRL", "tooltip_text"] = "No data available"


    layer = pdk.Layer(
        "GeoJsonLayer",
        geo.__geo_interface__,
        get_fill_color=[["*", ["predicted_life_expectancy"], 2], 100, 200, 100, 180],
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=10, longitude=0, zoom=1)
    tooltip = {
        "html": "<b>{GEO_NAME_SHORT}</b><br/>Predicted: {tooltip_text} years",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
with tab3:
    st.header("Feature Correlation with Life Expectancy")

    # 1) Build the latest-per-country DataFrame with the true target
    data = (
        df
        .sort_values("DIM_TIME")
        .groupby("ISO3", as_index=False)
        .last()
        .set_index("ISO3")
    )
    target_col = "Life expectancy (at birth)"

    # 2) Compute Pearson correlations
    corrs = {
        feat: data[[feat, target_col]]
                .dropna()[feat]
                .corr(data[[feat, target_col]].dropna()[target_col])
        for feat in FEATURE_LIST
    }
    corr_series = pd.Series(corrs).sort_values()

    # 3) Plot dynamically with high DPI and bar-labels
    fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
    bars = ax.barh(corr_series.index, corr_series.values, color="steelblue")
    ax.set_title("Pearson Correlation with Life Expectancy", fontsize=16)
    ax.set_xlabel("Correlation coefficient", fontsize=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=12)

    # 4) Annotate each bar with its value
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (0.01 if w >= 0 else -0.01),
            bar.get_y() + bar.get_height()/2,
            f"{w:.2f}",
            va="center",
            fontsize=10
        )

    st.pyplot(fig, use_container_width=True)

