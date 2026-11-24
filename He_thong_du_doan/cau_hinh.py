"""
Cấu hình chung của hệ thống dự đoán kết quả học tập (v3).
Mọi chương trình khác đều đọc tên cột, giai đoạn và đường dẫn từ file này.
"""
import os

THU_MUC = os.path.dirname(os.path.abspath(__file__))
DU_LIEU = os.path.join(THU_MUC, "du_lieu")

FILE_SINH_VIEN = os.path.join(DU_LIEU, "sinh_vien_v3.csv")
FILE_LOP_MAU = os.path.join(DU_LIEU, "lop_mau.csv")
FILE_UCI = os.path.join(DU_LIEU, "student-mat.csv")
FILE_MO_HINH = os.path.join(THU_MUC, "mo_hinh_v3.pkl")
FILE_MO_HINH_THAT = os.path.join(THU_MUC, "mo_hinh_du_lieu_that.pkl")
FILE_DANH_GIA = os.path.join(THU_MUC, "ket_qua_danh_gia.csv")
FILE_SO_SANH = os.path.join(THU_MUC, "ket_qua_so_sanh.csv")
FILE_DU_LIEU_THAT = os.path.join(THU_MUC, "ket_qua_du_lieu_that.csv")
FILE_LICH_SU = os.path.join(THU_MUC, "lich_su.db")

# ---------- Dữ liệu mô phỏng (xếp loại 4 lớp) ----------
CAC_COT = ["gio_hoc", "chuyen_can", "diem_bai_tap", "diem_giua_ky", "diem_cuoi_ky"]
TEN_COT = {
    "gio_hoc": "Số giờ học mỗi ngày",
    "chuyen_can": "Điểm chuyên cần (0-100)",
    "diem_bai_tap": "Điểm bài tập",
    "diem_giua_ky": "Điểm giữa kỳ",
    "diem_cuoi_ky": "Điểm cuối kỳ",
}
# Khoảng giá trị và bước nhập của từng cột
KHOANG = {
    "gio_hoc": (0.0, 12.0, 0.5),
    "chuyen_can": (0.0, 100.0, 1.0),
    "diem_bai_tap": (0.0, 10.0, 0.1),
    "diem_giua_ky": (0.0, 10.0, 0.1),
    "diem_cuoi_ky": (0.0, 10.0, 0.1),
}
MAC_DINH = {"gio_hoc": 3.0, "chuyen_can": 80.0, "diem_bai_tap": 7.0,
            "diem_giua_ky": 6.5, "diem_cuoi_ky": 6.5}

THU_TU_LOP = ["Gioi", "Kha", "Trung binh", "Yeu"]          # từ cao xuống thấp
HANG_LOP = {"Yeu": 0, "Trung binh": 1, "Kha": 2, "Gioi": 3}  # càng cao càng tốt
TEN_LOP = {"Gioi": "Giỏi", "Kha": "Khá", "Trung binh": "Trung bình", "Yeu": "Yếu"}

# Các giai đoạn trong học kỳ: càng về sau càng biết nhiều điểm thành phần
GIAI_DOAN = [
    {"ma": "dau_ky", "ten": "Đầu kỳ",
     "cot": ["gio_hoc", "chuyen_can"],
     "mo_ta": "Mới biết thói quen học và chuyên cần. Đây là dự báo sớm nhất, độ chắc chắn thấp nhất."},
    {"ma": "sau_bai_tap", "ten": "Sau bài tập",
     "cot": ["gio_hoc", "chuyen_can", "diem_bai_tap"],
     "mo_ta": "Đã có điểm bài tập."},
    {"ma": "sau_giua_ky", "ten": "Sau giữa kỳ",
     "cot": ["gio_hoc", "chuyen_can", "diem_bai_tap", "diem_giua_ky"],
     "mo_ta": "Đã có điểm giữa kỳ. Vẫn còn kịp can thiệp trước kỳ thi cuối."},
    {"ma": "sau_cuoi_ky", "ten": "Sau cuối kỳ",
     "cot": CAC_COT,
     "mo_ta": "Đã có đủ điểm, đây là bước tính xếp loại chứ không còn là dự báo."},
]
TEN_GIAI_DOAN = [g["ten"] for g in GIAI_DOAN]

# Công thức điểm tổng và ngưỡng xếp loại (dùng khi tạo dữ liệu)
def diem_tong(chuyen_can, bai_tap, giua_ky, cuoi_ky):
    return chuyen_can / 100 + bai_tap * 0.2 + giua_ky * 0.3 + cuoi_ky * 0.4


def xep_loai(diem):
    if diem >= 8.0:
        return "Gioi"
    if diem >= 6.5:
        return "Kha"
    if diem >= 5.0:
        return "Trung binh"
    return "Yeu"

# ---------- Dữ liệu thật UCI (đậu / rớt môn Toán) ----------
COT_THAT = ["studytime", "failures", "absences", "G1", "G2"]
TEN_COT_THAT = {
    "studytime": "Thời gian tự học hằng tuần (1: dưới 2 giờ, 2: 2-5 giờ, 3: 5-10 giờ, 4: trên 10 giờ)",
    "failures": "Số lần rớt môn trước đây (0-3)",
    "absences": "Số buổi vắng",
    "G1": "Điểm kỳ 1 (thang 20)",
    "G2": "Điểm kỳ 2 (thang 20)",
}
GIAI_DOAN_THAT = [
    {"ma": "khong_diem", "ten": "Chưa có điểm", "cot": ["studytime", "failures", "absences"]},
    {"ma": "co_g1", "ten": "Có điểm kỳ 1", "cot": ["studytime", "failures", "absences", "G1"]},
    {"ma": "co_g2", "ten": "Có điểm kỳ 1 và 2", "cot": COT_THAT},
]
