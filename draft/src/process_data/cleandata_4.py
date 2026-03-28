import pandas as pd
import warnings
import numpy as np
# (imports removed, already at top of file)

warnings.filterwarnings("ignore")


def process_dataset(df):
    """
    Xử lý dataset theo các yêu cầu:
    1. Filter 2010-2023
    2. Đếm missing values theo quốc gia và cột
    3. Loại bỏ quốc gia có >20% missing trên bất kỳ cột nào
    4. Fill missing values theo chiều dọc (nội suy)
    5. Fill missing values theo chiều ngang (KNN - optional)
    """

    print("=== BƯỚC 1: FILTER DỮ LIỆU 2000-2025 ===")
    # Filter data từ 2010-2023
    df_filtered = df[(df["year"] >= 2010) & (df["year"] <= 2025)].copy()

    # Đếm số dòng trước khi loại bỏ duplicate
    before_drop_duplicates = len(df_filtered)
    # Xóa duplicate row (giữ lại dòng đầu tiên)
    df_filtered = df_filtered.drop_duplicates()
    after_drop_duplicates = len(df_filtered)
    print(f"Số dòng sau khi filter: {after_drop_duplicates}")
    print(f"Số quốc gia: {df_filtered['country_name'].nunique()}")
    print(f"Năm từ {df_filtered['year'].min()} đến {df_filtered['year'].max()}")
    print(
        "Số dòng Duplicate đã loại bỏ:", before_drop_duplicates - after_drop_duplicates
    )

    # === BƯỚC 1.1 (OPTION): LỌC BỚT CỘT ===
    print("=== BƯỚC 1.1 (OPTION): LỌC BỚT CỘT ===")
    drop_cols = [
        "p_a",
        "hte",
        "suscom",
        "nfa",
        "nbden",
        "gii",
        "eco_fre",
        "rdrate", "hci"
    ]
    df_filtered = df_filtered.drop(columns=drop_cols, errors="ignore")
    print(f"Đã loại bỏ các cột: {drop_cols}")

    # Các cột numeric cần kiểm tra missing (chỉ lấy các cột còn lại)
    # Lấy tất cả các cột numeric (trừ country_name, country_code, year)
    exclude_cols = ["country_name", "year"]
    for col in df_filtered.columns:
        if col not in exclude_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")
    all_numeric_cols = [col for col in df_filtered.columns if col not in exclude_cols]
    numeric_cols = [col for col in all_numeric_cols if col in df_filtered.columns]

    # Đếm missing values theo quốc gia cho từng cột
    missing_by_country = {}
    for country in df_filtered["country_name"].unique():
        country_data = df_filtered[df_filtered["country_name"] == country]
        total_years = len(country_data)

        missing_count = {}
        missing_pct = {}
        for col in numeric_cols:
            missing = country_data[col].isna().sum()
            missing_count[col] = missing
            missing_pct[col] = (missing / total_years) * 100

        missing_by_country[country] = {
            "total_years": total_years,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

    # Tạo DataFrame để hiển thị missing percentages
    missing_df = pd.DataFrame(
        {country: data["missing_pct"] for country, data in missing_by_country.items()}
    ).T

    print("Top 10 quốc gia có nhiều missing data nhất (% missing trung bình):")
    avg_missing = missing_df.mean(axis=1).sort_values(ascending=False)
    print(avg_missing.head(10))

    print("\n=== BƯỚC 3: LOẠI BỎ QUỐC GIA CÓ {missing_threshold}% === MISSING ===")
    # Cho phép tùy chỉnh ngưỡng missing %
    missing_threshold = (80)

    # Tìm quốc gia có bất kỳ cột nào missing > missing_threshold%
    countries_to_remove = []
    col_remove_count = {
        col: 0 for col in numeric_cols
    }  # Thống kê số quốc gia bị loại bởi từng cột

    for country, data in missing_by_country.items():
        max_missing_pct = max(data["missing_pct"].values())
        if max_missing_pct > missing_threshold:
            countries_to_remove.append((country, max_missing_pct))
            # Đếm cho từng cột
            for col, pct in data["missing_pct"].items():
                if pct > missing_threshold:
                    col_remove_count[col] += 1

    # Sắp xếp theo thứ tự từ nhiều đến ít missing
    countries_to_remove.sort(key=lambda x: x[1], reverse=True)

    print(f"Số quốc gia bị loại bỏ: {len(countries_to_remove)}")
    if countries_to_remove:
        print("Danh sách quốc gia bị loại (theo thứ tự % missing giảm dần):")
        for country, pct in countries_to_remove[:10]:  # Hiển thị 10 đầu
            print(f"  {country}: {pct:.1f}% missing")

    # Thống kê số quốc gia bị loại bởi từng cột
    print("\nSố quốc gia bị loại bởi từng cột (có > {missing_threshold} % missing):")
    for col in numeric_cols:
        print(f"  {col}: {col_remove_count[col]} quốc gia")

    # Lọc bỏ các quốc gia này
    countries_to_keep = [
        country
        for country, _ in missing_by_country.items()
        if country not in [c[0] for c in countries_to_remove]
    ]

    df_clean = df_filtered[df_filtered["country_name"].isin(countries_to_keep)].copy()
    print(f"Số quốc gia còn lại: {len(countries_to_keep)}")
    print(f"Số dòng còn lại: {len(df_clean)}")
    print(
        f"Tổng missing values sau khi loại bỏ: {df_clean[numeric_cols].isna().sum().sum()}"
    )

    print("\n=== BƯỚC 4: FILL MISSING VALUES THEO CHIỀU DỌC (NỘI SUY) ===")
    # Fill missing values theo từng quốc gia (nội suy theo thời gian)
    df_interpolated = df_clean.copy()

    fill_stats = {}  # Lưu số lượng và tỷ lệ fill cho từng quốc gia

    for country in df_interpolated["country_name"].unique():
        mask = df_interpolated["country_name"] == country
        country_data = df_interpolated[mask].copy()

        # Sắp xếp theo năm
        country_data = country_data.sort_values("year")

        # Đếm missing trước khi nội suy
        missing_before = country_data[numeric_cols].isna().sum().sum()

        # Nội suy cho từng cột, chỉ fill đoạn liên tiếp không quá 20% tổng số dòng
        total_years = len(country_data)
        for col in numeric_cols:
            max_gap = int(np.floor(0.2 * total_years))
            # Nếu max_gap < 1 thì không fill
            if max_gap < 1:
                continue
            country_data[col] = country_data[col].interpolate(
                method="linear", axis=0, limit=max_gap, limit_direction="both"
            )

        # Đếm missing sau khi nội suy
        missing_after = country_data[numeric_cols].isna().sum().sum()

        # Số lượng giá trị đã fill
        filled_count = missing_before - missing_after
        total_values = len(country_data) * len(numeric_cols)
        fill_ratio = filled_count / total_values * 100 if total_values > 0 else 0

        fill_stats[country] = {
            "filled_count": filled_count,
            "fill_ratio_pct": fill_ratio,
            "total_values": total_values,
        }

        df_interpolated.loc[mask, country_data.columns] = country_data
        # Đếm số lượng giá trị đã fill theo từng cột
        fill_count_by_col = {col: 0 for col in numeric_cols}
        for country in fill_stats:
            # Lấy dữ liệu trước và sau nội suy cho từng quốc gia
            mask = df_clean["country_name"] == country
            country_data_before = df_clean[mask].sort_values("year")[numeric_cols]
            country_data_after = df_interpolated[mask].sort_values("year")[numeric_cols]
            for col in numeric_cols:
                filled = country_data_before[col].isna() & (
                    ~country_data_after[col].isna()
                )
                fill_count_by_col[col] += filled.sum()

        # Hiển thị số lượng giá trị đã fill theo từng cột
        print("\nSố lượng giá trị đã fill theo từng cột (sau nội suy):")
        for col in numeric_cols:
            print(f"  {col}: {fill_count_by_col[col]}")

        # Hiển thị top 10 quốc gia có tỷ lệ fill cao nhất
        fill_stats_df = pd.DataFrame(fill_stats).T
        fill_stats_df = fill_stats_df.sort_values("fill_ratio_pct", ascending=False)
        print("\nTop 10 quốc gia có tỷ lệ giá trị được fill cao nhất sau nội suy:")
        print(fill_stats_df[["filled_count", "fill_ratio_pct"]].head(10))

        # Kiểm tra missing sau nội suy
        missing_after_interp = df_interpolated[numeric_cols].isna().sum().sum()
        print(f"Tổng missing values sau nội suy: {missing_after_interp}")

    print("\n=== BƯỚC 5: FILL MISSING VALUES THEO CHIỀU NGANG (KNN - OPTIONAL) ===")
    df_final = df_interpolated.copy()

    print(f"Số quốc gia: {df_final['country_name'].nunique()}")
    print(f"Số năm: {df_final['year'].nunique()}")
    print(f"Tổng số dòng: {len(df_final)}")
    print(f"Tổng missing values: {df_final[numeric_cols].isna().sum().sum()}")

    # Hiển thị missing values theo cột
    print("\nMissing values theo từng cột:")
    for col in numeric_cols:
        missing = df_final[col].isna().sum()
        pct = (missing / len(df_final)) * 100
        print(f"  {col}: {missing} ({pct:.2f}%)")

    print("Missing Valuse còn lại", df_final[numeric_cols].isna().sum().sum())

    # Danh sách các quốc gia còn lại
    print("\nDanh sách country_name còn lại sau xử lý:")
    print(sorted(df_final["country_name"].unique()))

    return df_final, missing_by_country, countries_to_remove


df = pd.read_csv(
    r"C:\Users\VAQ\OneDrive\Desktop\RP_2 ngươi bạn\C3_Menthology\01_machine learning method\data\Mege_dataset.csv"
)
df_processed, missing_stats, removed_countries = process_dataset(df)

print(df_processed.shape)

# === PHASE 2: Loại bỏ quốc gia có bất kỳ cột nào missing > 70% sau khi fill KNN ===
missing_threshold_2 = 100  # Bạn có thể điều chỉnh ngưỡng này

# Tìm các quốc gia có bất kỳ cột nào missing > missing_threshold
countries_to_remove_phase2 = []
for country in df_processed["country_name"].unique():
    country_data = df_processed[df_processed["country_name"] == country]
    for col in [c for c in df_processed.columns if c not in ["country_name", "year"]]:
        pct_missing = country_data[col].isna().mean() * 100
        if pct_missing > missing_threshold_2:
            countries_to_remove_phase2.append(country)
            break

countries_to_keep_phase2 = [
    c
    for c in df_processed["country_name"].unique()
    if c not in countries_to_remove_phase2
]
df_processed_phase2 = df_processed[
    df_processed["country_name"].isin(countries_to_keep_phase2)
].copy()

print(
    f"\n=== PHASE 2: Sau khi loại bỏ quốc gia có bất kỳ cột nào missing > {missing_threshold_2}% ==="
)
print(f"Số quốc gia còn lại: {len(countries_to_keep_phase2)}")
print(f"Số dòng còn lại: {len(df_processed_phase2)}")
print(f"Tổng missing values còn lại: {df_processed_phase2.isna().sum().sum()}")

# Lưu kết quả phase 2 nếu muốn
# df_processed_phase2.to_csv('processed_dataset_phase2.csv', index=False)

# Sử dụng hàm
# df = pd.read_csv('your_dataset.csv')  # Uncomment và thay đổi path
# df_processed, missing_stats, removed_countries = process_dataset(df)

# Lưu kết quả
# df_processed.to_csv('processed_dataset.csv', index=False)

print("Code đã sẵn sàng! Uncomment các dòng cuối để chạy với dataset của bạn.")
