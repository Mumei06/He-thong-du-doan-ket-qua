# Hệ thống dự đoán kết quả học tập của sinh viên (v3)

## Chạy nhanh
```
pip install -r requirements.txt
streamlit run app.py
```
Lần chạy đầu ứng dụng tự huấn luyện mô hình (vài giây). Muốn lưu mô hình để các lần sau khởi động nhanh hơn, chạy một lần:
```
python huan_luyen.py
```

## Các chương trình
| File | Công dụng |
|---|---|
| `cau_hinh.py` | Tên cột, giai đoạn, đường dẫn dùng chung |
| `tao_du_lieu_v3.py` | Tạo dữ liệu mô phỏng 1000 sinh viên (có nhiễu, có quan hệ giữa các cột) và lớp mẫu 120 sinh viên |
| `mo_hinh.py` | Huấn luyện và đánh giá mô hình cho từng giai đoạn |
| `huan_luyen.py` | Chạy huấn luyện, lưu `mo_hinh_v3.pkl` và `ket_qua_danh_gia.csv` |
| `so_sanh_mo_hinh.py` | So sánh 8 mô hình theo từng giai đoạn, lưu `ket_qua_so_sanh.csv` |
| `du_lieu_that.py` | Đánh giá trên dữ liệu thật UCI Student Performance (đậu hoặc rớt) |
| `phan_tich.py` | Giải thích dự đoán, gợi ý cải thiện, phân tích cả lớp, xuất Excel |
| `lich_su.py` | Lưu lịch sử dự đoán vào SQLite (`lich_su.db`, tự tạo khi dùng) |
| `app.py` | Giao diện Streamlit |

## Giao diện có 5 tab
- **Dự đoán**: chọn giai đoạn của học kỳ ở thanh bên (Đầu kỳ, Sau bài tập, Sau giữa kỳ, Sau cuối kỳ), nhập điểm đã có. Ứng dụng cho xếp loại dự đoán, xác suất từng loại, điểm tổng dự kiến kèm khoảng sai số, cảnh báo nguy cơ rớt môn theo ngưỡng chỉnh được, giải thích vì sao (đóng góp của từng thuộc tính) và gợi ý cần làm gì để đạt mục tiêu (hết nguy cơ, đạt Khá, đạt Giỏi).
- **Dashboard lớp**: dùng lớp mẫu hoặc tải CSV cả lớp lên, xem phân bố xếp loại, danh sách sinh viên nguy cơ cao, tải báo cáo Excel hoặc CSV.
- **Lịch sử**: xem các lần dự đoán đã lưu, theo dõi tiến bộ của một sinh viên qua các giai đoạn.
- **Mô hình và đánh giá**: bảng chỉ số theo giai đoạn, ma trận nhầm lẫn, hệ số mô hình, so sánh các mô hình, giải thích từng chỉ số.
- **Dữ liệu thật (UCI)**: kết quả trên dữ liệu thật và ô thử dự đoán đậu hoặc rớt.

## Dữ liệu thật
`du_lieu/student-mat.csv` là bộ UCI Student Performance (môn Toán, 395 học sinh), Cortez, P. (2008), UCI Machine Learning Repository, https://doi.org/10.24432/C5TG7T, giấy phép CC BY 4.0.

## Chạy lại toàn bộ (không bắt buộc)
```
python tao_du_lieu_v3.py
python huan_luyen.py
python so_sanh_mo_hinh.py
python du_lieu_that.py
```
Kết quả có thể lệch nhẹ nếu phiên bản scikit-learn khác. Các file `ket_qua_*.csv` đi kèm là kết quả đã chạy sẵn.
