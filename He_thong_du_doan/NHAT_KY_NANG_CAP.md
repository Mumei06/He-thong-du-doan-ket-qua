# Nhật ký nâng cấp v3

Bản v3 gộp thành một hệ thống duy nhất (không còn chia phần Người 1 và Người 2), nâng cấp từ bản v2.

## ĐÃ làm

| # | Nâng cấp | File |
|---|---|---|
| 1 | **Dữ liệu mô phỏng mới**: 1000 sinh viên, mỗi sinh viên có năng lực và thói quen học ẩn, cộng nhiễu, các cột tương quan 0,5 đến 0,7 (bản cũ sinh độc lập). Kèm lớp mẫu 120 sinh viên. | `tao_du_lieu_v3.py`, `du_lieu/` |
| 2 | **Dự đoán sớm theo 4 giai đoạn** (Đầu kỳ, Sau bài tập, Sau giữa kỳ, Sau cuối kỳ), mỗi giai đoạn một mô hình dùng đúng thông tin đã có. Accuracy CV: 0,527, 0,658, 0,770, 0,992. | `mo_hinh.py`, `huan_luyen.py`, `cau_hinh.py` |
| 3 | **Điểm tổng dự kiến** bằng hồi quy tuyến tính, kèm khoảng sai số 90% từ phần dư kiểm định chéo. | `mo_hinh.py` |
| 4 | **Ngưỡng cảnh báo tối ưu** cho lớp Yếu, chọn bằng `cross_val_predict` trên tập học, chỉnh được trên giao diện. | `mo_hinh.py`, `app.py` |
| 5 | **Giải thích dự đoán**: đóng góp của từng thuộc tính vào nguy cơ Yếu so với sinh viên trung bình. | `phan_tich.py`, `app.py` |
| 6 | **Gợi ý "nếu thì"**: cần điểm tiếp theo bao nhiêu, hoặc tăng giờ học và chuyên cần bao nhiêu để hết nguy cơ, đạt Khá, đạt Giỏi. | `phan_tich.py`, `app.py` |
| 7 | **Dashboard cả lớp**: dùng lớp mẫu hoặc tải CSV, phân bố xếp loại, danh sách nguy cơ, tải báo cáo Excel (3 sheet) và CSV. | `phan_tich.py`, `app.py` |
| 8 | **Lịch sử dự đoán** bằng SQLite, lọc theo mã sinh viên, biểu đồ tiến bộ qua các giai đoạn, tải CSV. | `lich_su.py`, `app.py` |
| 9 | **Dữ liệu thật UCI Student Performance** (395 học sinh): đánh giá 3 giai đoạn, so sánh 5 mô hình, ô thử dự đoán đậu hoặc rớt. | `du_lieu_that.py`, `app.py` |
| 10 | **So sánh 8 mô hình theo từng giai đoạn** thay vì một giai đoạn. | `so_sanh_mo_hinh.py` |
| 11 | App hiển thị độ chính xác của giai đoạn đang chọn, cảnh báo khi nhập giá trị ngoài khoảng dữ liệu, tự huấn luyện lại nếu không nạp được file mô hình. | `app.py` |

## KHÔNG làm

- Đăng nhập và phân quyền giảng viên, sinh viên (cần cơ chế xác thực thật, chỉ làm giả lập thì không có giá trị).
- Dùng thêm bộ UCI môn Bồ Đào Nha (student-por.csv).
- Hiệu chỉnh xác suất (calibration).
- Không kèm sẵn file `.pkl`: chạy `python huan_luyen.py` để tạo trên máy bạn, tránh lỗi khác phiên bản scikit-learn.

## Kết quả cần nhớ

| Giai đoạn | Accuracy kiểm tra | Accuracy CV 5×3 | AUC Yếu |
|---|---|---|---|
| Đầu kỳ | 0,475 | 0,527 | 0,894 |
| Sau bài tập | 0,620 | 0,658 | 0,969 |
| Sau giữa kỳ | 0,785 | 0,770 | 0,986 |
| Sau cuối kỳ | 0,995 | 0,992 | 1,000 |

Mức sàn (đoán bừa) là 0,355. Dữ liệu thật: chưa có điểm 0,711 (ngang mức sàn 0,671), có điểm kỳ 1 là 0,848, có điểm kỳ 1 và 2 là 0,906 (Accuracy CV 5×3).

## Cần hiểu đúng

- Giai đoạn Sau cuối kỳ gần tuyệt đối vì nhãn là hàm của các điểm (bước tính xếp loại, không phải dự báo).
- Dữ liệu mô phỏng do nhóm tự sinh nên các con số chủ yếu cho thấy xu hướng theo giai đoạn. Con số trên dữ liệu thật mới có ý nghĩa dự báo thực tế.
- Gợi ý "nếu thì" phản ánh quan hệ mô hình học được, không chứng minh quan hệ nhân quả.
