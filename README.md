# ML for Research Paper

This repository contains the code and resources for a research project investigating the determinants of economic growth, with a particular focus on the role of Artificial Intelligence (AI) innovation and various socio-economic factors. The project workflow is structured into three main stages: data acquisition, machine learning modeling, and data visualization.

## Project Structure

*   `jupyter/`: Contains Jupyter notebooks detailing the data acquisition, machine learning, and visualization processes.
    *   `Get_Data_API_WDI.ipynb`: Data acquisition from World Bank Data360 API and Our World in Data.
    *   `ML_Model.ipynb`: Machine learning model training, evaluation, and feature importance analysis.
    *   `Visualization.ipynb`: Interactive data visualizations, including 3D plots and choropleth maps.
*   `assets/`: Stores supplementary files, including static images of model results and visualizations.
    *   `RF Model.png`: Visualization related to the Random Forest model.
    *   `SHAP RF.png`: Visualization of SHAP values for Random Forest model interpretability.
*   `data/`: (Expected) Directory for raw and processed data files.
*   `draft/`: (Expected) Directory for draft research papers or reports.

## Overview of the Workflow

### 1. Data Acquisition (`Get_Data_API_WDI.ipynb`)

This notebook is responsible for gathering a comprehensive dataset of economic, social, and technological indicators. It leverages the World Bank Data360 API and other publicly available datasets (e.g., Our World in Data) to collect a broad range of variables, including:

*   **Dependent Variables:** GDP annual growth, GDP per capita, etc.
*   **Institutional Indicators:** Measures of governance quality (e.g., Control of Corruption, Government Effectiveness, Rule of Law).
*   **Digital Transformation & Investment Indicators:** Research and Development expenditure, Government expenditure on education.
*   **Intellectual Capital:**
    *   **Human Capital:** School enrollment, educational attainment, labor force with advanced education, ICT skills.
    *   **Structural Capital:** Patent applications, scientific and technical journal articles, high-technology exports, manufacturing value added.
    *   **Relational Capital:** Foreign direct investment, trade, charges for intellectual property, international tourism receipts.
*   **AI Absorptive Capacity:** Infrastructure (fixed broadband, secure internet servers, internet users, mobile cellular subscriptions) and Market Advantage (ICT service/goods exports/imports).
*   **Control Variables:** Industry value added, urban population, total population, inflation, gross capital formation, electric power consumption, and access to electricity.

The notebook cleans and combines this diverse data into a single `combined_wdi_data.csv` file, and includes additional features like an "Advanced Economy" flag.

### 2. Machine Learning Modeling (`ML_Model.ipynb`)

This notebook applies various machine learning techniques to analyze the collected data and identify the key drivers of GDP per capita growth.

*   **Data Preparation:** Further data cleaning, feature selection, and the creation of lagged variables (e.g., `ln_gdp_pc_lag`).
*   **Model Training:** Employs `RandomForestRegressor` and `XGBRegressor` models.
*   **Model Evaluation:** Utilizes K-Fold cross-validation to ensure robust performance assessment (R-squared, RMSE, MAE).
*   **Feature Importance Analysis:** Identifies the most significant features influencing GDP per capita growth across different model configurations.
*   **Cluster Analysis:** Performs K-Means clustering on key dimensions (Human Capital, AI Innovation, Economic Freedom) to categorize countries and explore distinct development pathways.

### 3. Data Visualization (`Visualization.ipynb`)

This notebook provides visual insights to complement the statistical and machine learning analyses.

*   **3D Surface Plots:** Visualizes complex relationships, such as the interplay between Economic Freedom, Human Capital, and AI Index, with the surface colored by log GDP per capita.
*   **Choropleth Maps:** Generates interactive geographical maps to illustrate the spatial distribution of mean AI documents, economic freedom, and Human Capital Index across countries.

## Getting Started

To replicate the analysis or explore the project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AnhQuocVo/ML-for-Research-Paper.git
    cd ML-for-Research-Paper
    ```
2.  **Install dependencies:** Ensure you have Python installed. The required libraries (e.g., `pandas`, `requests`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `plotly`) can be installed via pip:
    ```bash
    pip install -r requirements.txt # (assuming a requirements.txt will be created)
    ```
    *(Note: A `requirements.txt` file might need to be generated based on the notebooks' imports.)*
3.  **Run the notebooks:** Execute the Jupyter notebooks in sequence:
    *   `Get_Data_API_WDI.ipynb` to fetch and preprocess the data.
    *   `ML_Model.ipynb` to perform machine learning analysis.
    *   `Visualization.ipynb` to generate visualizations.

## Visual Assets

The `assets` folder contains key figures generated during the analysis, such as:
![Random Forest Model](./assets/RF%20Model.png)
*   `RF Model.png`: A visual representation of the Random Forest model's output or performance.

![SHAP](./assets/SHAP%20RF.png)
*   `SHAP RF.png`: Illustrates the SHAP values, providing insights into individual feature contributions to model predictions.

## Conclusion
 This research employed a machine learning-driven approach to elucidate the complex determinants of GDP per capita
  growth across a diverse global panel. Leveraging advanced regression models, including Random Forest and XGBoost, and
  rigorously evaluating them through K-fold cross-validation, our findings illuminate the critical interplay of various
  socio-economic and technological factors.

  Our analysis consistently identified industry_value and foreign direct investment (FDI) as key drivers, underscoring
  the enduring significance of industrial development and global capital integration for economic expansion. Crucially,
  measures of AI innovation (proxied by ai_doc1) emerged as a significant predictor, highlighting the increasingly vital
  role of technological advancement in Artificial Intelligence in shaping national economic trajectories.

  Beyond direct economic and technological inputs, the strength of a nation's institutional framework proved paramount.
  Indicators reflecting market openness, regulatory efficiency, effective governance size, and the rule of law
  consistently demonstrated substantial influence on GDP per capita growth. Concurrently, human capital, as measured by
  indices like HC_Index and educational attainment proxies, reaffirms its foundational importance, indicating that a
  skilled and knowledgeable workforce remains a cornerstone of economic prosperity. The inclusion of lagged GDP per
  capita further confirmed the historical dependence in economic growth patterns.

  Visualizations, such as the 3D surface plot depicting the combined influence of Economic Freedom, Human Capital, and
  AI Index on log GDP per capita, graphically illustrated that countries excelling in these dimensions tend to achieve
  higher economic output. Furthermore, choropleth maps provided compelling spatial evidence, showcasing that regions
  with elevated levels of AI innovation, human capital, and economic freedom often correspond with the world's most
  economically advanced nations.

  In conclusion, this study offers robust empirical evidence that sustainable GDP per capita growth is fostered by a
  synergistic combination of robust industrial activity, open and efficient institutional governance, strategic human
  capital development, and proactive engagement with frontier technologies like Artificial Intelligence. These insights
  provide actionable guidance for policymakers aiming to cultivate resilient and prosperous economies in the 21st
  century.