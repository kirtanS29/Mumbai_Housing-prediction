# 🏠 Mumbai Housing Price Prediction

This project predicts housing prices in **Mumbai, India** using machine learning models. It is deployed as an **interactive Streamlit app** where users can input property details (area, bedrooms, amenities, resale, etc.) and get a **predicted price**.

---

##  View the Live App
Check out the live version here:  
**[Mumbai Housing Price Predictor](https://mumbaihousing.streamlit.app/)**

---

##  Why Mumbai and not California?

We specifically chose **Mumbai** because:

- **High real estate demand** – Mumbai is India’s **financial capital** and has one of the most **volatile and expensive property markets** in the country.  
- **Diverse housing spectrum** – The dataset includes apartments with varying amenities, resale conditions, and sizes, making it a **good candidate for ML modeling**.  
- **Relevance to Indian context** – Most existing datasets (like those for California housing) are already well-studied in Kaggle/ML tutorials. Using Mumbai data provides **fresh insights** that are more locally relevant.  
- **Challenge** – Mumbai’s housing prices are not only determined by size and amenities but also **location, accessibility, and social factors**, which makes the prediction task **harder and more interesting**.  

---

##  Model & Performance

We trained an **ensemble Voting Regressor** combining:

- Random Forest Regressor  
- XGBoost Regressor  
- Polynomial Regression + Linear Regression  

**Performance (on real dataset):** 
- **R² Score: 0.43**

---

##  What does 0.43 R² mean?

- The model explains only **43% of the variance** in housing prices.  
- This is relatively **low compared to ideal models**, but it reflects the **complexity of Mumbai’s real estate market**.  
- For comparison, the California Housing dataset usually achieves **R² between 0.65 and 0.75**, since its data is **cleaner, more consistent, and less volatile**.

---

##  Challenges with Dataset

- **Time-variant data**  
  - Housing prices change year by year due to inflation, infrastructure growth, and regulatory shifts.  
  - The dataset lacks the **year of data collection**, which limits predictive accuracy.

- **Location granularity**  
  - Significant price variation within neighborhoods of Mumbai.  
  - Dataset includes `Location`, but lacks proximity or geospatial markers like distance to metro stations or commercial hubs.

- **Amenity-based modeling**  
  - Used `Amenity_Count` as a feature.  
  - However, not all amenities impact pricing uniformly.

---

##  Limitations of ML in Real Estate

- Real estate prices are **influenced by non-quantitative factors** such as:
  - **Policy/regulatory shifts** (e.g., stamp duty changes, development plans)
  - **Market sentiment and buyer psychology**
  - **Cash/kala market influence** (especially in Mumbai)
  - **Scarcity and exclusivity**
  - **Speculative forecasts** and developer-led pricing
- These factors are **not captured in standard datasets**, which limits the performance of purely data-driven models.

---

##  Future Improvements

- Collect updated datasets with **year & timestamp** features.  
- Add **geospatial data** (latitude/longitude, distance to metro, seafront, etc.).  
- Include **economic indicators** (interest rates, rental yields, inflation).  
- Consider **time-series modeling** instead of static regression.  
- Hyperparameter tuning through **cross-validation** and use of **advanced models** (LightGBM, CatBoost).

---

##  Tech Stack

- **Python**  
- **scikit-learn, XGBoost, joblib** – for model training  
- **Streamlit** – for interactive UI  
- **pandas, NumPy, matplotlib** – for data manipulation and visualization  
- **Google Gemini API** – for AI-generated explanations

---

##  Installation

```bash
git clone https://github.com/yourusername/mumbai_housing_prediction.git
cd mumbai_housing_prediction

# Create a virtual environment (Python 3.11 recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

Usage

Open the Streamlit UI.

Input details like area, bedrooms, location, resale status, amenities.

Get an instant housing price prediction.

(Optional) View AI-based explanations through the Gemini API.

Results Summary

Dataset: Mumbai.csv

Preprocessing: Outlier filtering via IQR

Feature engineering: Amenity_Count, Location_AvgPrice

Model: Voting Regressor (RF + XGB + Polynomial + Linear)

R² Score: 0.43

⚠️ Note: The modest performance is expected due to missing temporal context and inherent city-specific volatility.

Contribution

Contributions welcome! Please fork the repository and submit a pull request.
Potential contributions:

Enrich the dataset

Enhance model accuracy

Improve UI/UX with interactive visuals
