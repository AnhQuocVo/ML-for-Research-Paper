# ICT and Intellectual Capital Impact on Economic Growth - Complete ML Pipeline
# Author: AI Assistant
# Purpose: Analyze the impact of ICT and IC on GDP using PCA, Random Forest, and SHAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
import importlib.util

warnings.filterwarnings("ignore")

# Ensure 'shap' is installed
try:
    import shap
except ImportError:
    raise ImportError(
        "The 'shap' package is required. Please install it using 'pip install shap'."
    )

# Optional: Check if umap-learn is available using importlib
UMAP_AVAILABLE = importlib.util.find_spec("umap") is not None
if not UMAP_AVAILABLE:
    print("UMAP not available. Will use t-SNE only for visualization.")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ICTAnalysisPipeline:
    """
    Complete pipeline for analyzing ICT and Intellectual Capital impact on economic growth
    """

    def __init__(self, df):
        """
        Initialize the pipeline with the dataset

        Parameters:
        df (pd.DataFrame): Panel dataset with all required variables
        """
        self.df = df.copy()
        self.scalers = {}
        self.pca_models = {}
        self.indices = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        """
        Step 1: Data preparation and feature engineering
        """
        print("=== Step 1: Data Preparation ===")

        # Create log GDP
        self.df["log_gdp"] = np.log(self.df["gdp"])
        print(f"Created log_gdp variable. Shape: {self.df.shape}")

        # Institutional variables: use the mean of six governance indicators as a single variable
        self.institutional_vars = ["cc", "ge", "pv", "rl", "rq", "va"]
        self.df["institutional_mean"] = self.df[self.institutional_vars].mean(axis=1)
        self.institutional_vars = ["institutional_mean"]
        # Define variable groups for PCA
        self.ict_vars = ["inet_usr", "mob_sub", "ict_exp", "sec_srv", "gr_cap"]
        self.ic_vars = [
            "edu_exp",
            "ter_enr",
            "sci_art",
            "fdi",
            "trade",
            "institutional_mean",
        ]

        # Additional predictors
        self.other_vars = ["labor", "hdi", "pop", "infl"]

        print(f"ICT variables: {self.ict_vars}")
        print(f"IC variables: {self.ic_vars}")
        print(f"Institutional variables: {self.institutional_vars}")
        print(f"Other predictors: {self.other_vars}")

        return self

    def standardize_and_create_indices(self, include_institutional_in_ic=False):
        """
        Step 2: Standardize variables and create PCA-based indices

        Parameters:
        include_institutional_in_ic (bool): Whether to include institutional vars in IC index
        """
        print("\n=== Step 2: Standardization and PCA Index Creation ===")

        # Decision on institutional variables
        if include_institutional_in_ic:
            print("Including institutional variables in IC Index")
            self.ic_vars.extend(self.institutional_vars)
            self.use_separate_institutional = False
        else:
            print("Creating separate Institutional Quality Index")
            self.use_separate_institutional = True

        # Create indices using PCA
        self._create_pca_index("ICT", self.ict_vars)
        self._create_pca_index("IC", self.ic_vars)

        if self.use_separate_institutional:
            self._create_pca_index("Institutional", self.institutional_vars)

        # Standardize other variables
        other_scaler = StandardScaler()
        self.df[self.other_vars] = other_scaler.fit_transform(self.df[self.other_vars])
        self.scalers["other"] = other_scaler

        print("Standardization and index creation completed!")
        return self

    def _create_pca_index(self, index_name, variables):
        """
        Helper function to create PCA-based index
        """
        print(f"\nCreating {index_name} Index from: {variables}")

        # Standardize variables
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(self.df[variables])
        self.scalers[index_name] = scaler

        # Apply PCA
        pca = PCA(n_components=1)
        index_values = pca.fit_transform(standardized_data)

        # Store results
        self.pca_models[index_name] = pca
        self.indices[f"{index_name}_Index"] = index_values.flatten()
        self.df[f"{index_name}_Index"] = index_values.flatten()

        # Print PCA results
        explained_var = pca.explained_variance_ratio_[0]
        print(f"{index_name} Index explained variance: {explained_var:.3f}")

        # Show component loadings
        loadings = pca.components_[0]
        loading_df = pd.DataFrame(
            {
                "Variable": variables,
                "Loading": loadings,
                "Abs_Loading": np.abs(loadings),
            }
        ).sort_values("Abs_Loading", ascending=False)

        print(f"{index_name} Index - Variable Loadings:")
        for _, row in loading_df.iterrows():
            print(f"  {row['Variable']}: {row['Loading']:.3f}")

    def prepare_model_data(self):
        """
        Step 3: Prepare data for machine learning model
        """
        print("\n=== Step 3: Model Data Preparation ===")

        # Define features for the model
        model_features = ["ICT_Index", "IC_Index"] + self.other_vars

        if self.use_separate_institutional:
            model_features.append("Institutional_Index")

        # Prepare X and y
        X = self.df[model_features].copy()
        y = self.df["log_gdp"].copy()

        print(f"Model features: {model_features}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

        return self

    def train_random_forest(self, n_estimators=100):
        """
        Step 4: Train Random Forest model
        """
        print("\n=== Step 4: Random Forest Training ===")

        # Initialize Random Forest
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=5, scoring="r2"
        )
        print(
            f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "Feature": self.X_train.columns,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        print("\nFeature Importance (Random Forest):")
        for _, row in feature_importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

        return self

    def calculate_shap_values(self):
        """
        Step 5: Calculate SHAP values for model interpretation
        """
        print("\n=== Step 5: SHAP Analysis ===")

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values for test set
        self.shap_values = self.explainer.shap_values(self.X_test)

        print(f"SHAP values shape: {self.shap_values.shape}")
        print("SHAP analysis completed!")

        return self

    def create_visualizations(self):
        """
        Step 6: Create all visualizations (each in a separate figure)
        """
        print("\n=== Step 6: Creating Visualizations ===")

        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_vars = (
            self.ict_vars
            + self.ic_vars
            + self.institutional_vars
            + self.other_vars
            + ["log_gdp"]
        )
        corr_matrix = self.df[correlation_vars].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title(
            "Correlation Heatmap of All Variables", fontsize=20, fontweight="bold"
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # 2. SHAP Summary Plot (Bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.show()

        # 3. SHAP Summary Beeswarm Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("SHAP Summary Plot (Beeswarm)", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.show()

        # 4. Index Distributions
        plt.figure(figsize=(10, 8))
        indices_to_plot = ["ICT_Index", "IC_Index"]
        if self.use_separate_institutional:
            indices_to_plot.append("Institutional_Index")
        for idx in indices_to_plot:
            plt.hist(self.df[idx], alpha=0.7, label=idx, bins=30)
        plt.xlabel("Index Value", fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.title("Distribution of PCA Indices", fontsize=20, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 5. Actual vs Predicted
        plt.figure(figsize=(10, 8))
        y_pred_test = self.model.predict(self.X_test)
        plt.scatter(self.y_test, y_pred_test, alpha=0.6, s=60)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual log(GDP)", fontsize=16)
        plt.ylabel("Predicted log(GDP)", fontsize=16)
        plt.title("Actual vs Predicted log(GDP)", fontsize=20, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 6. Feature Importance Comparison
        plt.figure(figsize=(12, 8))
        feature_imp = pd.DataFrame(
            {
                "Feature": self.X_train.columns,
                "RF_Importance": self.model.feature_importances_,
                "SHAP_Importance": np.abs(self.shap_values).mean(0),
            }
        )
        x = np.arange(len(feature_imp))
        width = 0.35
        plt.bar(
            x - width / 2,
            feature_imp["RF_Importance"],
            width,
            label="Random Forest",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            feature_imp["SHAP_Importance"],
            width,
            label="SHAP",
            alpha=0.8,
        )
        plt.xlabel("Features", fontsize=16)
        plt.ylabel("Importance", fontsize=16)
        plt.title("Feature Importance Comparison", fontsize=20, fontweight="bold")
        plt.xticks(x, feature_imp["Feature"], rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 7. t-SNE Visualization
        plt.figure(figsize=(10, 8))
        self._create_tsne_plot()
        plt.tight_layout()
        plt.show()

        # 8. GDP vs Indices Scatter
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.df["ICT_Index"], self.df["log_gdp"], alpha=0.6, label="ICT Index", s=60
        )
        plt.scatter(
            self.df["IC_Index"], self.df["log_gdp"], alpha=0.6, label="IC Index", s=60
        )
        plt.xlabel("Index Value", fontsize=16)
        plt.ylabel("log(GDP)", fontsize=16)
        plt.title("GDP vs PCA Indices", fontsize=20, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Create separate SHAP waterfall plot
        self._create_shap_waterfall()

        return self

    def _create_tsne_plot(self):
        """
        Create t-SNE visualization
        """
        # Prepare data for dimensionality reduction
        tsne_features = ["ICT_Index", "IC_Index"]
        if self.use_separate_institutional:
            tsne_features.append("Institutional_Index")

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(self.df[tsne_features])

        # Create scatter plot colored by GDP
        scatter = plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=self.df["log_gdp"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="log(GDP)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(
            "t-SNE Visualization (colored by log GDP)", fontsize=14, fontweight="bold"
        )
        plt.grid(True, alpha=0.3)

    def _create_shap_waterfall(self):
        """
        Create SHAP waterfall plot for one observation
        """
        # Select a random observation from test set
        idx = np.random.randint(0, len(self.X_test))

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[idx],
                base_values=self.explainer.expected_value,
                data=self.X_test.iloc[idx],
            ),
            max_display=10,
            show=False,
        )
        plt.title(
            f"SHAP Waterfall Plot (Observation {idx})", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def perform_clustering_analysis(self, n_clusters=4):
        """
        Step 7: Perform K-means clustering analysis
        """
        print(f"\n=== Step 7: K-means Clustering Analysis (k={n_clusters}) ===")

        # Prepare clustering features
        cluster_features = ["ICT_Index", "IC_Index"]
        if self.use_separate_institutional:
            cluster_features.append("Institutional_Index")

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.df[cluster_features])
        self.df["Cluster"] = clusters

        # Analyze clusters
        print("Cluster Analysis Results:")
        cluster_analysis = (
            self.df.groupby("Cluster")
            .agg(
                {
                    "log_gdp": ["mean", "std", "count"],
                    "ICT_Index": "mean",
                    "IC_Index": "mean",
                    "gdp": "mean",
                }
            )
            .round(3)
        )

        if self.use_separate_institutional:
            cluster_analysis[("Institutional_Index", "mean")] = self.df.groupby(
                "Cluster"
            )["Institutional_Index"].mean()

        print(cluster_analysis)

        # Visualize clusters
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Cluster visualization
        axes[0].scatter(
            self.df["ICT_Index"],
            self.df["IC_Index"],
            c=clusters,
            cmap="tab10",
            alpha=0.7,
        )
        axes[0].scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Centroids",
        )
        axes[0].set_xlabel("ICT Index")
        axes[0].set_ylabel("IC Index")
        axes[0].set_title("K-means Clustering Results")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # GDP by cluster
        cluster_gdp = (
            self.df.groupby("Cluster")["log_gdp"].mean().sort_values(ascending=False)
        )
        axes[1].bar(
            range(len(cluster_gdp)),
            cluster_gdp.values,
            color=plt.cm.tab10(range(len(cluster_gdp))),
        )
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Mean log(GDP)")
        axes[1].set_title("Average log(GDP) by Cluster")
        axes[1].set_xticks(range(len(cluster_gdp)))
        axes[1].set_xticklabels([f"Cluster {i}" for i in cluster_gdp.index])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.df["Cluster"] = clusters

        # In tên từng quốc gia trong mỗi cụm
        for c in sorted(self.df["Cluster"].unique()):
            print(f"\nCluster {c}:")
            print(self.df.loc[self.df["Cluster"] == c, "country_name"].unique())

        return self

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
        print("=" * 80)

        print("\n1. DATA OVERVIEW:")
        print(f"   - Dataset shape: {self.df.shape}")
        print("Target variable: log(GDP)")
        print(f"   - ICT variables: {len(self.ict_vars)} variables")
        print(f"   - IC variables: {len(self.ic_vars)} variables")
        print(f"   - Institutional variables: {len(self.institutional_vars)} variables")

        print("\n2. PCA INDEX RESULTS:")
        for index_name, pca_model in self.pca_models.items():
            explained_var = pca_model.explained_variance_ratio_[0]
            print(f"   - {index_name} Index: {explained_var:.1%} variance explained")

        print("\n3. MODEL PERFORMANCE:")
        y_pred_test = self.model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        print(f"   - Test R²: {test_r2:.4f}")
        print(f"   - Test RMSE: {test_rmse:.4f}")

        print("\n4. KEY FINDINGS:")
        # Top 3 most important features
        feature_importance = pd.DataFrame(
            {
                "Feature": self.X_train.columns,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        print("   - Top 3 most important features:")
        for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
            print(f"     {i + 1}. {row['Feature']}: {row['Importance']:.4f}")

        # Correlation with GDP
        correlations = (
            self.df[["ICT_Index", "IC_Index", "log_gdp"]]
            .corr()["log_gdp"]
            .sort_values(ascending=False)
        )
        print(
            f"\n   - ICT Index correlation with log(GDP): {correlations['ICT_Index']:.4f}"
        )
        print(
            f"   - IC Index correlation with log(GDP): {correlations['IC_Index']:.4f}"
        )

        if self.use_separate_institutional:
            inst_corr = self.df[["Institutional_Index", "log_gdp"]].corr()["log_gdp"][
                "Institutional_Index"
            ]
            print(
                f"   - Institutional Index correlation with log(GDP): {inst_corr:.4f}"
            )

        print("\n5. POLICY IMPLICATIONS:")
        if correlations["ICT_Index"] > correlations["IC_Index"]:
            print(
                "   - ICT infrastructure shows stronger association with economic growth"
            )
            print("   - Prioritize digital connectivity and technology adoption")
        else:
            print(
                "   - Intellectual Capital shows stronger association with economic growth"
            )
            print("   - Prioritize education, R&D, and knowledge-based investments")

        print("\n" + "=" * 80)

        return self

    def run_complete_pipeline(
        self, include_institutional_in_ic=False, n_estimators=100, n_clusters=4
    ):
        """
        Run the complete analysis pipeline

        Parameters:
        include_institutional_in_ic (bool): Whether to include institutional vars in IC index
        n_estimators (int): Number of trees in Random Forest
        n_clusters (int): Number of clusters for K-means
        """
        print("STARTING COMPLETE ICT & INTELLECTUAL CAPITAL ANALYSIS PIPELINE")
        print("=" * 80)

        # Execute pipeline steps
        (
            self.prepare_data()
            .standardize_and_create_indices(include_institutional_in_ic)
            .prepare_model_data()
            .train_random_forest(n_estimators)
            .calculate_shap_values()
            .create_visualizations()
            .perform_clustering_analysis(n_clusters)
            .generate_summary_report()
        )

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        return self


# ===========================
# USAGE EXAMPLE
# ===========================


def run_analysis(df):
    """
    Main function to run the complete analysis

    Parameters:
    df (pd.DataFrame): Your panel dataset

    Returns:
    ICTAnalysisPipeline: Fitted pipeline object
    """

    # Initialize and run pipeline
    pipeline = ICTAnalysisPipeline(df)

    # Run complete analysis
    # Set include_institutional_in_ic=True to include institutional vars in IC Index
    # Set include_institutional_in_ic=False to create separate Institutional Index
    pipeline.run_complete_pipeline(
        include_institutional_in_ic=False,  # Change to True if you want institutional vars in IC
        n_estimators=200,  # Number of trees in Random Forest
        n_clusters=4,  # Number of clusters for K-means
    )

    return pipeline


# ===========================
# ADDITIONAL UTILITY FUNCTIONS
# ===========================


def compare_institutional_approaches(df):
    """
    Compare results with institutional variables included vs separate
    """
    print("COMPARING INSTITUTIONAL VARIABLE APPROACHES")
    print("=" * 60)

    # Approach 1: Separate institutional index
    print("\n--- Approach 1: Separate Institutional Index ---")
    pipeline1 = ICTAnalysisPipeline(df)
    pipeline1.run_complete_pipeline(
        include_institutional_in_ic=False, n_estimators=100, n_clusters=4
    )

    # Approach 2: Institutional in IC index
    print("\n--- Approach 2: Institutional Variables in IC Index ---")
    pipeline2 = ICTAnalysisPipeline(df)
    pipeline2.run_complete_pipeline(
        include_institutional_in_ic=True, n_estimators=100, n_clusters=4
    )

    # Compare model performance
    y_pred1 = pipeline1.model.predict(pipeline1.X_test)
    y_pred2 = pipeline2.model.predict(pipeline2.X_test)

    r2_1 = r2_score(pipeline1.y_test, y_pred1)
    r2_2 = r2_score(pipeline2.y_test, y_pred2)

    print("MODEL PERFORMANCE COMPARISON:")
    print(f"Separate Institutional Index R²: {r2_1:.4f}")
    print(f"Institutional in IC Index R²: {r2_2:.4f}")

    if r2_1 > r2_2:
        print("RECOMMENDATION: Use separate Institutional Quality Index")
    else:
        print("RECOMMENDATION: Include institutional variables in IC Index")

    return pipeline1, pipeline2


# ===========================
# RUN THE ANALYSIS
# ===========================


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv(
        r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\processed_dataset.csv"
    )  # Load your dataset here
    pipeline = run_analysis(df)

    # To compare approaches:
    # pipeline_separate, pipeline_combined = compare_institutional_approaches(df)

    print("\nTo run the analysis, use:")
    print("pipeline = run_analysis(df)")
    print("\nOr to compare approaches:")
    print("pipeline_separate, pipeline_combined = compare_institutional_approaches(df)")
