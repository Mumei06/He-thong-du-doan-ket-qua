"""
Tạo dữ liệu mô phỏng v3 cho 1000 sinh viên, có nhiễu và có quan hệ giữa các cột.

Khác bản cũ: mỗi sinh viên có hai đặc điểm ẩn là năng lực và thói quen học. Cả hai cùng ảnh
hưởng tới giờ học, chuyên cần và các điểm số, cộng thêm nhiễu ngẫu nhiên. Nhờ vậy thói quen học
và điểm sớm có liên hệ với kết quả cuối, và việc dự đoán sớm mới có ý nghĩa.
Nhãn xếp loại vẫn tính từ điểm tổng theo công thức đã nêu trong báo cáo.

Chạy: python tao_du_lieu_v3.py
"""
import numpy as np
import pandas as pd

from cau_hinh import FILE_LOP_MAU, FILE_SINH_VIEN, diem_tong, xep_loai
import os

N = 1000
rng = np.random.default_rng(2026)

nang_luc = rng.normal(0, 1, N)
thoi_quen = 0.35 * nang_luc + np.sqrt(1 - 0.35 ** 2) * rng.normal(0, 1, N)


def cat(v, lo, hi):
    return np.clip(v, lo, hi)


gio_hoc = cat(3.0 + 1.0 * thoi_quen + 0.2 * nang_luc + rng.normal(0, 0.8, N), 0.0, 8.0)
chuyen_can = cat(84 + 7 * thoi_quen + 2 * nang_luc + rng.normal(0, 6, N), 40, 100)
bai_tap = cat(6.6 + 0.9 * nang_luc + 0.7 * thoi_quen + rng.normal(0, 0.9, N), 0, 10)
giua_ky = cat(6.2 + 1.15 * nang_luc + 0.5 * thoi_quen + rng.normal(0, 1.1, N), 0, 10)
cuoi_ky = cat(6.2 + 1.25 * nang_luc + 0.55 * thoi_quen + rng.normal(0, 1.0, N), 0, 10)

df = pd.DataFrame({
    "ma_sinh_vien": [f"SV{i + 1:04d}" for i in range(N)],
    "gio_hoc": gio_hoc.round(1),
    "chuyen_can": chuyen_can.round(0).astype(int),
    "diem_bai_tap": bai_tap.round(1),
    "diem_giua_ky": giua_ky.round(1),
    "diem_cuoi_ky": cuoi_ky.round(1),
})
df["diem_tong"] = [round(diem_tong(r.chuyen_can, r.diem_bai_tap, r.diem_giua_ky, r.diem_cuoi_ky), 2)
                   for r in df.itertuples()]
df["xep_loai"] = df["diem_tong"].map(xep_loai)

os.makedirs(os.path.dirname(FILE_SINH_VIEN), exist_ok=True)
df.to_csv(FILE_SINH_VIEN, index=False, encoding="utf-8-sig")
print("Da tao", FILE_SINH_VIEN, df.shape)
print(df["xep_loai"].value_counts().to_string())
print(df[["gio_hoc", "chuyen_can", "diem_bai_tap", "diem_giua_ky", "diem_cuoi_ky", "diem_tong"]]
      .describe().round(2).loc[["mean", "std", "min", "max"]].to_string())
print(df[["gio_hoc", "chuyen_can", "diem_bai_tap", "diem_giua_ky", "diem_cuoi_ky"]].corr().round(2).to_string())

# Lớp mẫu 120 sinh viên (bỏ cột nhãn) để thử chức năng Dashboard
mau = df.sample(120, random_state=7).sort_values("ma_sinh_vien")
mau[["ma_sinh_vien", "gio_hoc", "chuyen_can", "diem_bai_tap", "diem_giua_ky", "diem_cuoi_ky"]].to_csv(
    FILE_LOP_MAU, index=False, encoding="utf-8-sig")
print("Da tao", FILE_LOP_MAU)
