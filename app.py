import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NO₂ Downscaling — Delhi",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_USER = "yash16trivedi16-cell"
REPO_NAME = "satellite-based-air-quality-downscaling"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/outputs"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"

# Output image filenames, as tracked in the repo's /outputs folder
IMG = {
    "raw_no2": f"{RAW_BASE}/delhi_no2_map.png",
    "monthly_no2": f"{RAW_BASE}/delhi_monthly_no2.png",
    "aligned": f"{RAW_BASE}/all_datasets_aligned.png",
    "model_comparison": f"{RAW_BASE}/model_comparison.png",
    "feature_importance": f"{RAW_BASE}/feature_importance_v2.png",
    "training_history": f"{RAW_BASE}/unet_training_history.png",
    "final_map": f"{RAW_BASE}/delhi_no2_final_map.png",
    "test_results": f"{RAW_BASE}/final_test_results.png",
}


def safe_image(url, caption=None, use_container_width=True):
    """Show an image, or a friendly placeholder if it can't be fetched."""
    try:
        st.image(url, caption=caption, use_container_width=use_container_width)
    except Exception:
        st.info(f"Image not available: {caption or url}")


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("🛰️ Project Navigator")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📡 Dataset & Pipeline",
        "🤖 Models & Results",
        "🗺️ Final NO₂ Map",
        "📚 Key Learnings",
        "ℹ️ About & Links",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Repository**")
st.sidebar.markdown(f"[GitHub ↗]({REPO_URL})")
st.sidebar.markdown("**Geographic Scope:** Delhi, India")
st.sidebar.caption("28.4°N–28.9°N, 76.8°E–77.4°E")

# ----------------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Downscaling of Satellite-Based Air Quality Maps")
    st.caption("Vocational Training Project — Machine Learning & Geospatial Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Test R²", "0.9342", "U-Net v2")
    col2.metric("Input Resolution", "~3.5 km/px", "TROPOMI raw")
    col3.metric("Output Resolution", "~500 m/px", "113 × 135 grid")
    col4.metric("Models Built", "5", "LR → XGB → U-Net")

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("What is this project?")
        st.write(
            "Satellites like Sentinel-5P TROPOMI monitor NO₂ pollution from space, "
            "but each pixel covers a huge area — about 3.5 km × 3.5 km. That's too "
            "coarse to tell which neighborhoods are polluted and which are clean."
        )
        st.write(
            "This project uses machine learning to **downscale** that coarse data "
            "into a sharp, high-resolution NO₂ map of Delhi — combining the satellite "
            "signal with elevation, land use, population density, and road network data "
            "to predict pollution at roughly 500 m resolution."
        )
        st.write(
            "Five models were built and compared, ending with a **U-Net convolutional "
            "neural network** that explains over 93% of NO₂ variation on unseen test data."
        )
    with c2:
        st.subheader("Why Delhi?")
        st.markdown(
            """
            - One of the world's most polluted capitals — strong NO₂ signal
            - Dense CPCB ground-monitoring network for validation
            - Diverse land use: industrial, residential, forest, river
            - Well studied in literature, enabling comparison
            """
        )

    st.markdown("---")
    st.subheader("Before → After")
    safe_image(IMG["final_map"], caption="Raw satellite NO₂ (left) vs. ML-downscaled high-resolution map (right)")

# ----------------------------------------------------------------------------
# PAGE: DATASET & PIPELINE
# ----------------------------------------------------------------------------
elif page == "📡 Dataset & Pipeline":
    st.title("📡 Dataset & Pipeline")

    st.subheader("Datasets used")
    st.table(
        {
            "Dataset": ["TROPOMI NO₂", "Elevation (DEM)", "Land Use", "Population Density", "Road Network"],
            "Source": [
                "Sentinel-5P / Google Earth Engine",
                "NASA SRTM / GEE",
                "MODIS MCD12Q1 / GEE",
                "WorldPop / GEE",
                "OpenStreetMap / OSMnx",
            ],
            "Resolution": ["~3.5 km", "30 m", "500 m", "100 m", "Vector"],
            "Purpose": [
                "Target variable (annual + 12 monthly maps)",
                "Terrain / pollution trapping",
                "Urban / industrial / vegetation class",
                "Human activity proxy",
                "Vehicle emission proxy",
            ],
        }
    )

    st.markdown("---")
    st.subheader("Pipeline")
    st.code(
        """
Phase 1: Data Collection        → GEE (NO2, elevation, land use, population) + OSMnx (roads)
Phase 2: Preprocessing          → reproject to 113×135 grid, gap-fill, normalize, road density
Phase 3: Tabular ML             → Linear Regression → XGBoost v1 → XGBoost v2 (+ roads)
Phase 4: Deep Learning          → 16×16 patch extraction (stride=8) → U-Net v1 → U-Net v2
Phase 5: Evaluation             → seasonal train/val/test split, final map generation
        """,
        language="text",
    )

    st.markdown("---")
    st.subheader("All features, aligned to a common grid")
    safe_image(IMG["aligned"], caption="Elevation, land use, population, and roads reprojected to the reference grid")

    st.subheader("Seasonal NO₂ variation")
    safe_image(IMG["monthly_no2"], caption="12-month NO₂ maps — winter months show stronger pollution due to temperature inversions")

# ----------------------------------------------------------------------------
# PAGE: MODELS & RESULTS
# ----------------------------------------------------------------------------
elif page == "🤖 Models & Results":
    st.title("🤖 Models & Results")

    st.subheader("Validation set comparison")
    st.dataframe(
        {
            "Model": ["Linear Regression", "XGBoost v1", "XGBoost v2 (+roads)", "U-Net v1", "U-Net v2"],
            "R²": [0.5075, 0.9028, 0.9196, 0.8982, 0.9013],
            "RMSE": [0.1874, 0.0833, 0.0757, 0.0765, 0.0763],
            "MAE": [0.1487, 0.0550, 0.0503, 0.0578, 0.0576],
            "Notes": ["Baseline", "+40% vs baseline", "Best tabular model", "Spatial model", "Improved spatial model"],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.success("🏆 **U-Net v2** achieved **R² = 0.9342 on unseen test data** — the best overall model.")

    tab1, tab2, tab3 = st.tabs(["Model Comparison Chart", "Feature Importance", "U-Net Training History"])
    with tab1:
        safe_image(IMG["model_comparison"], caption="XGBoost vs Linear Regression performance")
    with tab2:
        safe_image(IMG["feature_importance"], caption="Land use dominates at 82.6% importance")
        st.markdown(
            """
            | Feature | Importance | Interpretation |
            |---|---|---|
            | Land Use | 82.6% | Industrial zones dominate pollution levels |
            | Elevation | 9.96% | Terrain trapping effect |
            | Population | 7.46% | Human activity contribution |
            | Road Density | 6.42% | Vehicle emissions (partly captured by land use) |
            """
        )
    with tab3:
        safe_image(IMG["training_history"], caption="Training vs validation loss across epochs")

    st.markdown("---")
    st.subheader("Final test on unseen data")
    safe_image(IMG["test_results"], caption="U-Net v2 predictions vs. ground truth on held-out test months")

# ----------------------------------------------------------------------------
# PAGE: FINAL MAP
# ----------------------------------------------------------------------------
elif page == "🗺️ Final NO₂ Map":
    st.title("🗺️ Final High-Resolution NO₂ Map of Delhi")
    st.write(
        "Generated by stitching together predictions from overlapping 16×16 patches "
        "using the trained U-Net v2 model, producing a smooth 113×135 pixel NO₂ surface."
    )
    safe_image(IMG["final_map"], caption="Final downscaled NO₂ map — Delhi, India")
    safe_image(IMG["raw_no2"], caption="Original raw TROPOMI NO₂ input, for comparison")

# ----------------------------------------------------------------------------
# PAGE: KEY LEARNINGS
# ----------------------------------------------------------------------------
elif page == "📚 Key Learnings":
    st.title("📚 Key Learnings")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Science")
        st.markdown(
            """
            - Always visualize data before modelling — raw satellite values look
              black in image viewers due to tiny decimal magnitudes
            - Never apply one cleaning rule to every dataset — population can be
              legitimately zero (parks, rivers), but NO₂ cannot
            - Normalization is critical — unscaled elevation (0–900 m) would
              dominate over NO₂ (0.00012–0.00022)
            """
        )
        st.subheader("Machine Learning")
        st.markdown(
            """
            - Tabular models (XGBoost) can outperform complex models on limited data
            - U-Net is architecturally superior for spatial tasks, but needs
              sufficient training data
            - Patch-based training (16×16, stride=8, 50% overlap) solved the
              limited-sample problem
            - Seasonal train/val/test splits are essential for satellite time series
            """
        )
    with c2:
        st.subheader("Deep Learning")
        st.markdown(
            """
            - More parameters than training samples → overfitting risk
            - Dropout + a smaller architecture often beats a larger, overfit one
            - Early stopping mattered — v1 stopped at epoch 17, v2 at epoch 46
            - Averaging overlapping patch predictions smooths the final map
            """
        )
        st.subheader("Geospatial")
        st.markdown(
            """
            - All datasets must share the same CRS and resolution before ML
            - Land use resolution (500 m) is the most practical reference grid
            - Road density (metres/pixel) is a strong proxy for vehicle emissions
            """
        )

    st.markdown("---")
    st.subheader("Common mistakes avoided")
    st.table(
        {
            "Mistake": [
                "Testing on training data",
                "Skipping cloud-gap handling",
                "Not aligning data grids",
                "Jumping to deep learning first",
                "Forgetting to normalize",
            ],
            "Fix applied": [
                "Held-out 15% test set, never seen during training",
                "5×5 neighbourhood averaging to fill gaps",
                "Reprojected everything to the same CRS & resolution",
                "Built Linear Regression → XGBoost baselines first",
                "MinMaxScaler applied to all features",
            ],
        }
    )

# ----------------------------------------------------------------------------
# PAGE: ABOUT
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.write(
        "This dashboard was built as the deployment/demo component of a 6th-semester "
        "Vocational Training project: **Downscaling of Satellite-Based Air Quality Maps**."
    )

    st.subheader("Links")
    st.markdown(f"- **GitHub Repository:** [{REPO_URL}]({REPO_URL})")
    st.markdown("- **Notebooks:** `01_download_no2_data.ipynb` → `05_unet_model.ipynb`")
    st.markdown("- **Data source:** Copernicus Sentinel-5P (open access)")

    st.subheader("How this app works")
    st.write(
        "This is a lightweight Streamlit front end — it doesn't rerun the ML pipeline "
        "itself, but presents the results, figures, and metrics produced by the notebooks "
        "in the repository, pulled directly from the `outputs/` folder on GitHub."
    )

    st.subheader("Tech stack")
    st.markdown(
        """
        `Python` · `TensorFlow` · `XGBoost` · `scikit-learn` · `Rasterio` ·
        `GeoPandas` · `OSMnx` · `Google Earth Engine` · `Streamlit`
        """
    )

