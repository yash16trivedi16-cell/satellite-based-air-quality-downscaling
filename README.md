# 🛰️ Downscaling of Satellite-Based Air Quality Maps Using Machine Learning

> A complete end-to-end ML pipeline that transforms coarse-resolution TROPOMI satellite NO₂ data into high-resolution air quality maps over Delhi, India — achieving **R² = 0.9284** on unseen test data.

---

## 📌 Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models Built](#models-built)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Future Work](#future-work)

---

## 🚀 Live Demo

Explore the interactive dashboard here: **[Live App →](https://satellite-based-air-quality-downscaling-fhykhcj7axtlxccdqbqhzb.streamlit.app/)**

The dashboard includes 6 pages — Overview, Dataset & Pipeline, Models & Results, Final NO₂ Map, Key Learnings, and About & Links — with interactive Plotly charts and the final high-resolution Delhi NO₂ map.

---

## Project Overview

Satellites like **Sentinel-5P TROPOMI** monitor air pollution from space, but their data is coarse — each pixel covers approximately 3.5km × 3.5km. This makes it impossible to understand air quality at the neighborhood level.

This project uses **machine learning** to "downscale" this coarse satellite data — transforming blurry, low-resolution NO₂ maps into sharp, high-resolution maps that reveal street-level pollution patterns.

Think of it like using AI to sharpen a blurry photograph — except the photograph is a pollution map, and the sharpening is done using geographic and demographic context.

```
Before (Satellite gives us):          After (Our model produces):
┌─────────────────────────┐           ┌─────────────────────────┐
│   ░░░▒▒▒███▒▒░░░░░░░   │           │  ░▒▒███████▒▒░░░░░░░░  │
│   ░░▒▒███████▒▒░░░░░   │   →  ML   │  ░▒████████████▒░░░░░  │
│   ░░░▒▒▒███▒▒░░░░░░░   │           │  ▒███░░░░███▒▒░░░░░░░  │
└─────────────────────────┘           └─────────────────────────┘
   57×68 pixels (~3.5km/px)              113×135 pixels (~500m/px)
```

---

## Problem Statement

Individual tools exist for satellite data processing, ML, and geospatial visualization — but there is no comprehensive, end-to-end solution that ties all of these together specifically for air quality downscaling. This project fills that gap by building a complete, transferable ML framework demonstrated over Delhi as a case study.

The methodology is **transferable** — the same pipeline can be applied to any city in the world.

---

## Research Question

> *Can machine learning improve the spatial resolution of TROPOMI NO₂ satellite observations over urban areas, and which model architecture best captures the spatial patterns of urban air pollution?*

**Geographic Scope:** Delhi, India (28.4°N–28.9°N, 76.8°E–77.4°E)

**Why Delhi?**
- One of the world's most polluted capitals — strong NO₂ signal
- Dense CPCB ground monitoring network for validation
- Diverse land use — industrial, residential, forests, rivers
- Well-studied in literature, allowing comparison

---

## Dataset

Five datasets were collected and integrated:

| Dataset | Source | Resolution | Purpose |
|---|---|---|---|
| **TROPOMI NO₂** | Sentinel-5P / Google Earth Engine | ~3.5km | Target variable (annual + 12 monthly maps) |
| **Elevation (DEM)** | NASA SRTM / Google Earth Engine | 30m | Terrain effect on pollution trapping |
| **Land Use** | MODIS MCD12Q1 / Google Earth Engine | 500m | Urban/industrial/vegetation classification |
| **Population Density** | WorldPop / Google Earth Engine | 100m | Human activity proxy |
| **Road Network** | OpenStreetMap / OSMnx | Vector | Vehicle emission proxy |

**Why these features?**

- **Elevation:** Pollution accumulates in low-lying areas due to temperature inversions. Cold, dense air traps pollutants close to the ground in valleys.
- **Land Use:** Industrial zones directly produce large NO₂ emissions. Forests act as sinks. This feature contributed **82.6% of XGBoost's predictive power.**
- **Population Density:** More people = more vehicles, cooking, and energy use = more NO₂.
- **Road Density:** Computed as total road length (metres) per grid pixel — directly represents vehicle emissions.

---

## Project Structure

```
Downscaling-of-satellite-based-air-quality-map/
│
├── data/
│   ├── raw/                    ← Downloaded satellite & map files
│   │   ├── delhi_no2_2023.tif          (annual mean NO₂)
│   │   ├── delhi_no2_2023_jan.tif      (monthly NO₂ × 12)
│   │   ├── delhi_elevation.tif
│   │   ├── delhi_landuse.tif
│   │   ├── delhi_population.tif
│   │   └── delhi_roads.gpkg
│   │
│   └── processed/
│       └── ml_dataset.csv              (15,244 clean pixels, 5 features)
│
├── notebooks/
│   ├── 01_download_no2_data.ipynb      ← GEE authentication & NO₂ download
│   ├── 02_download_auxiliary_data.ipynb← Roads, elevation, landuse, population
│   ├── 03_preprocess_data.ipynb        ← Alignment, gap-filling, normalization
│   ├── 04_ml_models.ipynb              ← Linear Regression & XGBoost
│   └── 05_unet_model.ipynb             ← U-Net CNN training & evaluation
│
├── outputs/
│   ├── delhi_no2_map.png               ← Raw satellite visualization
│   ├── delhi_monthly_no2.png           ← 12-month seasonal visualization
│   ├── all_datasets_aligned.png        ← All features at same resolution
│   ├── model_comparison.png            ← XGBoost vs Linear Regression
│   ├── feature_importance_v2.png       ← Which features matter most
│   ├── unet_training_history.png       ← Training/validation loss curves
│   ├── delhi_no2_final_map.png         ← Final downscaled NO₂ map
│   └── final_test_results.png          ← U-Net predictions vs actual
│
├── src/                                ← (Reserved for modular scripts)
├── .gitignore
└── README.md
```

---

## Pipeline Overview

```
Phase 1: Data Collection
─────────────────────────────────────────────────────
Google Earth Engine → TROPOMI NO₂ (annual + monthly)
Google Earth Engine → Elevation, Land Use, Population
OpenStreetMap      → Road Network
                            ↓
Phase 2: Preprocessing
─────────────────────────────────────────────────────
Reproject all datasets → reference grid (113×135)
Fill missing values    → 5×5 neighbourhood averaging
Normalize              → MinMaxScaler (0 to 1)
Compute road density   → road length per pixel (metres)
Combine                → ml_dataset.csv (15,244 rows)
                            ↓
Phase 3: Tabular ML Models (04_ml_models.ipynb)
─────────────────────────────────────────────────────
Linear Regression  → baseline
XGBoost v1         → elevation + population + landuse
XGBoost v2         → + road density
                            ↓
Phase 4: Deep Learning (05_unet_model.ipynb)
─────────────────────────────────────────────────────
Download 12 monthly NO₂ maps
Extract overlapping 16×16 patches (stride=8)
Stratified seasonal split → train/val/test
Build U-Net CNN v1 → evaluate
Build U-Net CNN v2 → smaller, dropout, more patches
Final test evaluation
Generate full Delhi NO₂ map
```

---

## Models Built

### 1. Linear Regression (Baseline)
The simplest possible model — draws a straight line through the relationship between features and NO₂. Used as the benchmark every subsequent model must beat.

### 2. XGBoost v1
Gradient boosted decision trees trained on elevation, population, and land use. Builds 200 trees sequentially, each correcting the mistakes of the previous one. Dramatically outperforms linear regression.

### 3. XGBoost v2 (+ Road Density)
XGBoost v1 extended with road density as an additional feature. Road density contributes 6.4% feature importance — smaller than land use (82.6%) because land use already partially encodes road infrastructure.

### 4. U-Net CNN v1
A convolutional neural network with encoder-decoder architecture and skip connections. Unlike XGBoost (which treats each pixel independently), U-Net understands spatial relationships between neighbouring pixels. Trained on 32×32 patches from 8 training months.

```
Architecture:
Input (32×32×4) → Encoder → Bottleneck → Decoder → Output (32×32×1)
                    ↕ skip connections ↕
Filters: 32 → 64 → 128 → 64 → 32
Total parameters: 471,841
```

### 5. U-Net CNN v2 (Improved)
Redesigned with:
- Smaller 16×16 patches → 1,560 training patches (vs 336 in v1)
- Fewer filters (16→32→64) → fewer parameters, less overfitting
- Dropout layers (10-20%) → prevents memorisation
- Extended training (100 epochs with early stopping)

```
Architecture:
Input (16×16×4) → Encoder → Bottleneck → Decoder → Output (16×16×1)
                    ↕ skip connections ↕
Filters: 16 → 32 → 64 → 32 → 16
Total parameters: ~60,000
Best epoch: 39
```

---

## Results

### Model Comparison (Validation Set)

| Model | R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | 0.5075 | 0.1874 | 0.1487 | Baseline |
| XGBoost v1 | 0.9028 | 0.0833 | 0.0550 | +40% vs baseline |
| XGBoost v2 + roads | 0.9196 | 0.0757 | 0.0503 | Best tabular |
| U-Net v1 | 0.8982 | 0.0765 | 0.0578 | Spatial model |
| U-Net v2 | 0.9013 | 0.0763 | 0.0576 | Improved spatial |

### Final Test Result (Unseen Data)

| Model | Val R² | Test R² | Verdict |
|---|---|---|---|
| **U-Net v2** | **0.9013** | **0.9284** | **Best overall** 🏆 |

> The U-Net v2 achieved **R² = 0.9284 on completely unseen test data** — meaning the model explains 92.84% of NO₂ variation across Delhi. The test score being higher than validation is attributed to the test months (August, December) having stronger seasonal pollution signals than validation months (May, November).

### Feature Importance (XGBoost v2)

| Feature | Importance | Interpretation |
|---|---|---|
| Land Use | 82.6% | Industrial zones dominate pollution levels |
| Elevation | 9.96% | Terrain trapping effect |
| Population | 7.46% | Human activity contribution |
| Road Density | 6.42% | Vehicle emissions (partially captured by land use) |

### Seasonal Patterns

Analysis of 12 monthly NO₂ maps revealed clear seasonal variation:

```
High NO₂ months (Oct–Feb):
→ Winter temperature inversions trap pollutants
→ Crop burning in neighbouring states
→ Increased heating and cooking fires

Low NO₂ months (May–Aug):  
→ Monsoon rains wash away pollutants
→ Hot air rises and disperses NO₂
→ Less heating required
```

This finding motivated a **stratified seasonal train/val/test split** to ensure fair evaluation across all seasons.

---

## Key Learnings

**Data Science:**
- Always visualize data before modelling — raw satellite data looks black in image viewers because of tiny decimal values
- Never apply the same cleaning rule blindly to all datasets — population can legitimately be zero (parks, rivers), NO₂ cannot
- Normalization is critical — without it, elevation (0–900m) would dominate over NO₂ (0.00012–0.00022)

**Machine Learning:**
- Tabular models (XGBoost) can outperform complex models on limited data
- U-Net is architecturally superior for spatial tasks — but needs sufficient training data
- Patch-based training (16×16 patches, stride=8, 50% overlap) solved the limited sample problem
- Seasonal data splits are essential for time-series satellite data

**Deep Learning:**
- More parameters than training samples → overfitting risk
- Dropout + smaller architecture often outperforms large overfit models
- Early stopping is essential — U-Net v1 stopped at epoch 17, v2 at epoch 46
- Averaging overlapping patch predictions smooths the final map

**Geospatial:**
- All datasets must be reprojected to the same CRS and resolution before ML
- Land use resolution (500m) is the most practical reference grid — finer than NO₂ but not unrealistically fine
- Road density (metres/pixel) is a powerful proxy for vehicle emissions

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sneha-trivedii/Downscaling-of-satellite-based-air-quality-map.git
cd Downscaling-of-satellite-based-air-quality-map
```

### 2. Set Up Environment

```bash
conda create -n airquality python=3.10
conda activate airquality
pip install numpy pandas xarray rasterio geopandas matplotlib
pip install scikit-learn xgboost tensorflow earthengine-api
pip install osmnx folium scipy
```

### 3. Authenticate Google Earth Engine

```python
import ee
ee.Authenticate()
ee.Initialize(project='your-gee-project-id')
```

### 4. Run Notebooks in Order

```
01_download_no2_data.ipynb       → Downloads annual + monthly NO₂
02_download_auxiliary_data.ipynb → Downloads supporting datasets
03_preprocess_data.ipynb         → Aligns, cleans, normalizes
04_ml_models.ipynb               → Trains XGBoost models
05_unet_model.ipynb              → Trains U-Net and generates final map
```

> **Note:** Raw data files (`.tif`, `.gpkg`) are excluded from version control due to size. Run notebooks 01–02 to regenerate them.

### 5. Or Just Try the Live Demo

No setup needed — explore the results directly: **[Live App →](https://satellite-based-air-quality-downscaling-fhykhcj7axtlxccdqbqhzb.streamlit.app/)**

---

## Requirements

```
Python        3.10
TensorFlow    2.21.0
numpy         ≥1.24
pandas        ≥2.0
rasterio      ≥1.3
geopandas     ≥0.13
scikit-learn  ≥1.3
xgboost       ≥2.0
osmnx         ≥1.6
scipy         ≥1.10
matplotlib    ≥3.7
earthengine-api ≥0.1.370
```

---

## Future Work

**Model Improvements:**
- CNN+LSTM hybrid for temporal prediction (predicting future monthly NO₂)
- Test on multiple Indian cities to validate transferability
- Incorporate meteorological features (wind, temperature, humidity)
- Experiment with larger U-Net architectures with more training data

**Data Improvements:**
- Multi-year data (2020–2024) for more training samples
- Ground station validation against CPCB sensor readings
- Higher resolution auxiliary data (building footprints, traffic counts)

**Deployment:**
- ~~Streamlit dashboard for interactive Delhi NO₂ visualization~~ ✅ **Live:** [satellite-based-air-quality-downscaling-fhykhcj7axtlxccdqbqhzb.streamlit.app](https://satellite-based-air-quality-downscaling-fhykhcj7axtlxccdqbqhzb.streamlit.app/)
- Automated monthly pipeline to generate fresh maps from new TROPOMI data
- Export final maps as GeoTIFF for use in QGIS and Google Earth Engine

---

## License

This project is for academic and research purposes. Data sources retain their original licenses:
- TROPOMI NO₂: Copernicus/ESA (open access)
- SRTM Elevation: NASA (public domain)
- MODIS Land Use: NASA (public domain)
- WorldPop: Creative Commons Attribution 4.0
- OpenStreetMap: Open Database License (ODbL)

---

*Built with Python, TensorFlow, Google Earth Engine, and a lot of debugging 🐛*
