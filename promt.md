Task 1: 

**1.1. Chuẩn hóa dữ liệu**
- Sử dụng StandardScaler để chuẩn hóa các biến định lượng: inet_usr, mob_sub, ict_exp, sec_srv, gr_ca, edu_exp, ter_enr, sci_art, fdi, trade, cc, ge, pv, rl, rq, va. Việc chuẩn hóa giúp các biến có cùng thang đo, thuận tiện cho phân tích đa biến.
- Áp dụng phép biến đổi log tự nhiên (log-transform) cho biến gdp để giảm ảnh hưởng của các giá trị ngoại lai và phân phối lệch.

**1.2. Xây dựng các chỉ số tổng hợp bằng PCA**
- Thực hiện phân tích thành phần chính (PCA) cho từng nhóm biến sau, lấy thành phần chính đầu tiên (PC1) làm chỉ số tổng hợp đại diện cho nhóm:
  - Institution_Index: cc, ge, pv, rl, rq, va
  - ICT: inet_usr, mob_sub, ict_exp, sec_srv, gr_ca
  - HC (Human Capital): edu_exp, ter_enr
  - SC (Scientific Capital): sci_art, Institution_Index (dùng giá trị PC1 của Institution_Index ở trên)
  - RC (Resource Capital): fdi, trade
  - IC (Integrated Capital): HC, SC, RC (dùng giá trị PC1 của các nhóm trên)
- Lưu lại giá trị PC1 của từng nhóm để sử dụng cho các phân tích tiếp theo.

Task 2:

**2.1. Phân tích chi tiết các thành phần PCA**
- Phân tích trọng số (loadings) của từng biến trong PC1 để xác định biến nào đóng góp lớn nhất vào chỉ số tổng hợp.
- Đánh giá tỷ lệ phương sai giải thích (explained variance) của PC1 trong từng nhóm biến để đảm bảo chỉ số tổng hợp đủ đại diện.
- Kiểm tra mối liên hệ giữa các chỉ số tổng hợp (IC, ICT, Institution_Index) bằng các phương pháp:
  - Vẽ scatterplot và phân phối (distribution plot) để trực quan hóa mối quan hệ giữa các chỉ số.
  - Tính hệ số tương quan (correlation) và hiệp phương sai (covariance) giữa các chỉ số.
  - Sử dụng biểu đồ bar hoặc heatmap để thể hiện mức độ đóng góp (importance) của từng biến thành phần vào PC1 dựa trên loadings của PCA.

**2.2. Phân tích theo quốc gia và trực quan hóa**
- Áp dụng thuật toán K-means (k=3) để phân nhóm các quốc gia dựa trên chỉ số hdi và Institution_Index (không chuẩn hóa), sau đó vẽ biểu đồ thể hiện các nhóm quốc gia.
- Vẽ scatterplot để thể hiện mối quan hệ giữa hdi và các chỉ số tổng hợp như IC, ICT.
- Vẽ biểu đồ 3D (hoặc scatter 3D) để thể hiện đồng thời mối quan hệ giữa IC, ICT, hdi cho nhiều quốc gia, sử dụng màu sắc để biểu diễn giá trị của một biến (ví dụ: hdi).

---
**Lưu ý:**
- Mỗi bước cần ghi rõ biến đầu vào, phương pháp xử lý, và mục tiêu phân tích.
- Kết quả PCA cần được giải thích rõ ràng về ý nghĩa kinh tế/xã hội của từng thành phần chính.
- Các biểu đồ trực quan nên có chú thích rõ ràng để dễ dàng so sánh giữa các quốc gia và nhóm chỉ số.

