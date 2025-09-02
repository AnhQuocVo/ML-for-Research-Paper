import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

import warnings

warnings.filterwarnings("ignore")

# Thiết lập matplotlib để hiển thị tiếng Việt
plt.rcParams["font.size"] = 10
plt.style.use("seaborn-v0_8")

# Đọc dữ liệu
# Đọc dữ liệu
# Sử dụng biến cấu hình cho đường dẫn file để tăng tính linh hoạt
DATA_PATH = r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\C3_Menthology\01_machine learning method\data_csv\process_missing.csv"  # Thay bằng tên file hoặc đường dẫn tương đối

df = pd.read_csv(DATA_PATH, encoding="utf-8")
# Giả sử bạn đã load dữ liệu vào biến df


def prepare_data(df):
    """
    Task 1.1: Chuẩn hóa dữ liệu
    """
    print("=== TASK 1.1: CHUẨN HÓA DỮ LIỆU ===")

    # Các biến cần chuẩn hóa
    vars_to_standardize = [
        "inet_usr",
        "mob_sub",
        "ict_exp",
        "sec_srv",
        "gr_cap",
        "edu_exp",
        "ter_enr",
        "sci_art",
        "fdi",
        "trade",
        "cc",
        "ge",
        "pv",
        "rl",
        "rq",
        "va",
    ]

    # Tạo bản sao để xử lý
    df_processed = df.copy()

    # Loại bỏ các hàng có giá trị null trong các biến quan trọng
    df_processed = df_processed.dropna(subset=vars_to_standardize + ["gdp", "hdi"])

    # Log transform cho GDP
    df_processed["log_gdp"] = np.log(df_processed["gdp"])
    print("Đã áp dụng log transform cho GDP")

    # Chuẩn hóa các biến định lượng
    scaler = StandardScaler()
    df_processed[vars_to_standardize] = scaler.fit_transform(
        df_processed[vars_to_standardize]
    )

    print(f"Đã chuẩn hóa {len(vars_to_standardize)} biến")
    print(f"Số lượng quan sát sau xử lý: {len(df_processed)}")

    # Chỉ giữ lại các cột cần thiết
    cols_to_keep = ["year", "country_name", "log_gdp", "hdi"] + vars_to_standardize
    df_final = df_processed[cols_to_keep].copy()

    return df_final, scaler


def create_pca_indices(df):
    """
    Task 1.2: Xây dựng các chỉ số tổng hợp bằng PCA
    """
    print("\n=== TASK 1.2: XÂY DỰNG CÁC CHỈ SỐ TỔNG HỢP BẰNG PCA ===")

    # Định nghĩa các nhóm biến
    groups = {
        "Institution_Index": ["cc", "ge", "pv", "rl", "rq", "va"],
        "ICT": ["inet_usr", "mob_sub", "ict_exp", "sec_srv", "gr_cap"],
        "HC": ["edu_exp", "ter_enr"],
        "SC_base": ["sci_art"],  # Sẽ thêm Institution_Index sau
        "RC": ["fdi", "trade"],
    }

    pca_results = {}
    df_with_indices = df.copy()

    # Thực hiện PCA cho từng nhóm
    for group_name, variables in groups.items():
        if group_name == "SC_base":
            continue  # Xử lý riêng sau

        print(f"\n--- Phân tích PCA cho {group_name} ---")

        # Loại bỏ các hàng có giá trị null
        group_data = df[variables].dropna()

        if len(group_data) == 0:
            print(f"Không có dữ liệu cho nhóm {group_name}")
            continue

        # Thực hiện PCA
        pca = PCA(n_components=1)
        pc1_values = pca.fit_transform(group_data)

        # Lưu kết quả
        pca_results[group_name] = {
            "pca_model": pca,
            "pc1_values": pc1_values.flatten(),
            "explained_variance": pca.explained_variance_ratio_[0],
            "loadings": pca.components_[0],
            "variables": variables,
            "valid_indices": group_data.index,
        }

        # Thêm PC1 vào dataframe
        df_with_indices.loc[group_data.index, group_name] = pc1_values.flatten()

        print(f"Explained variance: {pca.explained_variance_ratio_[0]:.3f}")
        print(f"Loadings: {dict(zip(variables, pca.components_[0]))}")

    # Tạo SC (Scientific Capital) bằng cách kết hợp sci_art và Institution_Index
    print("\n--- Phân tích PCA cho SC (Scientific Capital) ---")
    sc_vars = ["sci_art", "Institution_Index"]
    sc_data = df_with_indices[sc_vars].dropna()

    if len(sc_data) > 0:
        pca_sc = PCA(n_components=1)
        sc_pc1 = pca_sc.fit_transform(sc_data)

        pca_results["SC"] = {
            "pca_model": pca_sc,
            "pc1_values": sc_pc1.flatten(),
            "explained_variance": pca_sc.explained_variance_ratio_[0],
            "loadings": pca_sc.components_[0],
            "variables": sc_vars,
            "valid_indices": sc_data.index,
        }

        df_with_indices.loc[sc_data.index, "SC"] = sc_pc1.flatten()
        print(f"Explained variance: {pca_sc.explained_variance_ratio_[0]:.3f}")
        print(f"Loadings: {dict(zip(sc_vars, pca_sc.components_[0]))}")

    # Tạo IC (Integrated Capital) bằng cách kết hợp HC, SC, RC
    print("\n--- Phân tích PCA cho IC (Integrated Capital) ---")
    ic_vars = ["HC", "SC", "RC"]
    ic_data = df_with_indices[ic_vars].dropna()

    if len(ic_data) > 0:
        pca_ic = PCA(n_components=1)
        ic_pc1 = pca_ic.fit_transform(ic_data)

        pca_results["IC"] = {
            "pca_model": pca_ic,
            "pc1_values": ic_pc1.flatten(),
            "explained_variance": pca_ic.explained_variance_ratio_[0],
            "loadings": pca_ic.components_[0],
            "variables": ic_vars,
            "valid_indices": ic_data.index,
        }

        df_with_indices.loc[ic_data.index, "IC"] = ic_pc1.flatten()
        print(f"Explained variance: {pca_ic.explained_variance_ratio_[0]:.3f}")
        print(f"Loadings: {dict(zip(ic_vars, pca_ic.components_[0]))}")

    return df_with_indices, pca_results


def analyze_pca_components(pca_results):
    """
    Task 2.1: Phân tích chi tiết các thành phần PCA
    """
    print("\n=== TASK 2.1: PHÂN TÍCH CHI TIẾT CÁC THÀNH PHẦN PCA ===")

    # Tạo figure cho các biểu đồ
    plt.figure(figsize=(20, 15))

    # 1. Biểu đồ explained variance
    plt.subplot(2, 3, 1)
    groups = list(pca_results.keys())
    variances = [pca_results[group]["explained_variance"] for group in groups]

    bars = plt.bar(groups, variances, color="skyblue", alpha=0.7)
    plt.title("Explained Variance by PC1 for Each Group")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(rotation=45)

    # Thêm giá trị lên các bar
    for bar, variance in zip(bars, variances):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{variance:.3f}",
            ha="center",
            va="bottom",
        )

    # 2. Heatmap loadings
    plt.subplot(2, 3, 2)

    # Chuẩn bị dữ liệu cho heatmap
    all_vars = []
    all_loadings = []
    group_labels = []

    for group_name, result in pca_results.items():
        for var, loading in zip(result["variables"], result["loadings"]):
            all_vars.append(var)
            all_loadings.append(loading)
            group_labels.append(group_name)

    # Tạo DataFrame cho heatmap
    loading_df = pd.DataFrame(
        {"Variable": all_vars, "Loading": all_loadings, "Group": group_labels}
    )

    # Pivot để tạo heatmap
    heatmap_data = loading_df.pivot_table(
        values="Loading", index="Variable", columns="Group", fill_value=0
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        cbar_kws={"label": "Loading Values"},
    )
    plt.title("PCA Loadings Heatmap")
    plt.tight_layout()

    # 3. Biểu đồ loadings cho từng nhóm
    for i, (group_name, result) in enumerate(pca_results.items()):
        if i >= 4:  # Chỉ hiển thị 4 nhóm đầu tiên
            break

        plt.subplot(2, 3, i + 3)

        loadings = result["loadings"]
        variables = result["variables"]

        # Sắp xếp theo giá trị tuyệt đối
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        sorted_loadings = loadings[sorted_idx]
        sorted_vars = [variables[i] for i in sorted_idx]

        colors = ["red" if x < 0 else "blue" for x in sorted_loadings]
        bars = plt.barh(
            range(len(sorted_vars)), sorted_loadings, color=colors, alpha=0.7
        )

        plt.yticks(range(len(sorted_vars)), sorted_vars)
        plt.xlabel("Loading Values")
        plt.title(f"{group_name} - PC1 Loadings")
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Thêm giá trị lên các bar
        for bar, loading in zip(bars, sorted_loadings):
            plt.text(
                loading + (0.01 if loading > 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{loading:.3f}",
                ha="left" if loading > 0 else "right",
                va="center",
            )

    plt.tight_layout()
    plt.show()

    # In kết quả chi tiết
    print("\n--- CHI TIẾT KẾT QUẢ PCA ---")
    for group_name, result in pca_results.items():
        print(f"\n{group_name}:")
        print(f"  Explained Variance: {result['explained_variance']:.3f}")
        print("  Variables and Loadings:")
        for var, loading in zip(result["variables"], result["loadings"]):
            print(f"    {var}: {loading:.3f}")


def analyze_relationships(df, _):
    """
    Phân tích mối liên hệ giữa các chỉ số tổng hợp
    """
    print("\n--- PHÂN TÍCH MỐI LIÊN HỆ GIỮA CÁC CHỈ SỐ ---")

    # Tạo DataFrame chỉ với các chỉ số tổng hợp
    indices = ["IC", "ICT", "Institution_Index"]
    df_indices = df[indices].dropna()

    # Tính correlation và covariance
    correlation_matrix = df_indices.corr()
    covariance_matrix = df_indices.cov()

    print("Correlation Matrix:")
    print(correlation_matrix)
    print("\nCovariance Matrix:")
    print(covariance_matrix)

    # Vẽ biểu đồ
    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Scatterplot matrix
    pd.plotting.scatter_matrix(df_indices, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title("Scatterplot Matrix of Composite Indices")
    axes[0, 0].set_title("Scatterplot Matrix of Composite Indices")

    # Distribution plots
    axes[0, 1].hist(df_indices["IC"], alpha=0.7, label="IC", bins=30)
    axes[0, 1].hist(df_indices["ICT"], alpha=0.7, label="ICT", bins=30)
    axes[0, 1].hist(
        df_indices["Institution_Index"], alpha=0.7, label="Institution_Index", bins=30
    )
    axes[0, 1].set_title("Distribution of Composite Indices")
    axes[0, 1].legend()

    # Correlation heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        ax=axes[1, 0],
        fmt=".3f",
    )
    axes[1, 0].set_title("Correlation Heatmap")

    # Covariance heatmap
    sns.heatmap(covariance_matrix, annot=True, cmap="viridis", ax=axes[1, 1], fmt=".3f")
    axes[1, 1].set_title("Covariance Heatmap")

    plt.tight_layout()
    plt.show()

    return correlation_matrix, covariance_matrix


def country_analysis_and_visualization(df):
    """
    Task 2.2: Phân tích theo quốc gia và trực quan hóa
    """
    print("\n=== TASK 2.2: PHÂN TÍCH THEO QUỐC GIA VÀ TRỰC QUAN HÓA ===")

    # Chuẩn bị dữ liệu cho phân tích
    analysis_df = df[
        ["country_name", "year", "hdi", "Institution_Index", "IC", "ICT"]
    ].dropna()

    # K-means clustering
    print("\n--- K-MEANS CLUSTERING ---")
    kmeans_data = analysis_df[["hdi", "Institution_Index"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(kmeans_data)
    analysis_df.loc[:, "cluster"] = clusters

    print(f"Đã phân {len(analysis_df)} quan sát thành 3 nhóm")
    print("Số lượng quốc gia trong mỗi nhóm:")
    print(analysis_df["cluster"].value_counts().sort_index())

    # Tạo các biểu đồ
    plt.figure(figsize=(20, 15))

    # 1. K-means clustering result
    plt.subplot(2, 3, 1)
    colors = ["red", "blue", "green"]
    for i in range(3):
        cluster_data = analysis_df[analysis_df["cluster"] == i]
        plt.scatter(
            cluster_data["Institution_Index"],
            cluster_data["hdi"],
            c=colors[i],
            label=f"Cluster {i}",
            alpha=0.6,
        )

    # Vẽ centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1], c="black", marker="x", s=200, linewidths=3
    )

    plt.xlabel("Institution Index")
    plt.ylabel("HDI")
    plt.title("K-means Clustering (k=3)\nBased on HDI and Institution Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. HDI vs IC
    plt.subplot(2, 3, 2)
    plt.scatter(analysis_df["hdi"], analysis_df["IC"], alpha=0.6, c="purple")
    plt.xlabel("HDI")
    plt.ylabel("IC (Integrated Capital)")
    plt.title("HDI vs Integrated Capital")
    plt.grid(True, alpha=0.3)

    # Thêm trendline
    z = np.polyfit(analysis_df["hdi"], analysis_df["IC"], 1)
    p = np.poly1d(z)
    plt.plot(analysis_df["hdi"], p(analysis_df["hdi"]), "r--", alpha=0.8)

    # 3. HDI vs ICT
    plt.subplot(2, 3, 3)
    plt.scatter(analysis_df["hdi"], analysis_df["ICT"], alpha=0.6, c="orange")
    plt.xlabel("HDI")
    plt.ylabel("ICT Index")
    plt.title("HDI vs ICT Index")
    plt.grid(True, alpha=0.3)

    # Thêm trendline
    z = np.polyfit(analysis_df["hdi"], analysis_df["ICT"], 1)
    p = np.poly1d(z)
    plt.plot(analysis_df["hdi"], p(analysis_df["hdi"]), "r--", alpha=0.8)

    # 4. 3D Scatter plot
    ax4 = plt.subplot(2, 3, 4, projection="3d")

    # Sử dụng HDI để tạo colormap
    scatter = ax4.scatter(
        analysis_df["IC"],
        analysis_df["ICT"],
        analysis_df["hdi"],
        c=analysis_df["hdi"],
        cmap="viridis",
        alpha=0.6,
    )

    ax4.set_xlabel("IC (Integrated Capital)")
    ax4.set_ylabel("ICT Index")
    ax4.set_zlabel("HDI")
    ax4.set_title("3D Relationship: IC, ICT, and HDI")

    # Thêm colorbar
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.5)
    cbar.set_label("HDI Values")

    # 5. Cluster distribution by year
    ax5 = plt.subplot(2, 3, 5)
    year_cluster = analysis_df.groupby(["year", "cluster"]).size().unstack(fill_value=0)
    year_cluster.plot(kind="bar", stacked=True, ax=ax5, color=colors)
    plt.title("Cluster Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Countries")
    plt.legend(title="Cluster")
    plt.xticks(rotation=45)

    # 6. Top countries by IC
    plt.subplot(2, 3, 6)
    top_countries = analysis_df.nlargest(10, "IC")
    plt.barh(range(len(top_countries)), top_countries["IC"], color="skyblue")
    plt.yticks(range(len(top_countries)), top_countries["country_name"])
    plt.xlabel("IC (Integrated Capital)")
    plt.title("Top 10 Countries by Integrated Capital")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # In thông tin chi tiết về clusters
    print("\n--- CHI TIẾT VỀ CÁC CLUSTER ---")
    for i in range(3):
        cluster_data = analysis_df[analysis_df["cluster"] == i]
        print(f"\nCluster {i}:")
        print(f"  Số quan sát: {len(cluster_data)}")
        print(f"  HDI trung bình: {cluster_data['hdi'].mean():.3f}")
        print(
            f"  Institution Index trung bình: {cluster_data['Institution_Index'].mean():.3f}"
        )
        print(f"  IC trung bình: {cluster_data['IC'].mean():.3f}")
        print(f"  ICT trung bình: {cluster_data['ICT'].mean():.3f}")
        print("  Một số quốc gia tiêu biểu:")
        sample_countries = cluster_data["country_name"].unique()[:5]
        for country in sample_countries:
            print(f"    - {country}")

    return analysis_df, kmeans


def main():
    """
    Hàm chính để thực hiện toàn bộ phân tích
    """
    print("PHÂN TÍCH DỮ LIỆU QUỐC GIA VỚI PCA VÀ K-MEANS")
    print("=" * 50)

    # Lưu ý: Bạn cần thay thế phần này bằng cách đọc dữ liệu thực tế
    # df = pd.read_csv(r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\process_missing.csv")

    # Thực hiện phân tích
    try:
        # Task 1.1: Chuẩn hóa dữ liệu
        df_processed, _ = prepare_data(df)

        # Task 1.2: Tạo các chỉ số tổng hợp
        df_with_indices, pca_results = create_pca_indices(df_processed)

        # Task 2.1: Phân tích chi tiết PCA
        analyze_pca_components(pca_results)

        # Phân tích mối liên hệ
        correlation_matrix, _ = analyze_relationships(df_with_indices, pca_results)

        # Task 2.2: Phân tích theo quốc gia
        final_df, kmeans_model = country_analysis_and_visualization(df_with_indices)

        print("\n=== HOÀN THÀNH PHÂN TÍCH ===")
        print(f"Dữ liệu cuối cùng có {len(final_df)} quan sát")
        print("Các chỉ số tổng hợp đã được tạo:")
        for group_name, result in pca_results.items():
            print(
                f"  - {group_name}: Explained variance = {result['explained_variance']:.3f}"
            )

        return df_with_indices, pca_results, final_df, kmeans_model

    except Exception as e:
        print(f"Lỗi trong quá trình phân tích: {str(e)}")
        return None, None, None, None


# Chạy phân tích
if __name__ == "__main__":
    df_with_indices, pca_results, final_df, kmeans_model = main()
