"""
Huấn luyện mô hình v3 cho bốn giai đoạn của học kỳ, đánh giá và lưu kết quả.
Chạy: python huan_luyen.py
Tạo: mo_hinh_v3.pkl và ket_qua_danh_gia.csv
"""
import warnings

import joblib

from cau_hinh import FILE_DANH_GIA, FILE_MO_HINH
from mo_hinh import danh_gia_tat_ca, doc_du_lieu, huan_luyen_tat_ca

warnings.filterwarnings("ignore")

df = doc_du_lieu()
cac_mo_hinh = huan_luyen_tat_ca(df)
kq = danh_gia_tat_ca(cac_mo_hinh, df)

import pandas as pd
pd.set_option("display.width", 200)
print(kq.round(4).to_string(index=False))
kq.to_csv(FILE_DANH_GIA, index=False, encoding="utf-8-sig")
joblib.dump(cac_mo_hinh, FILE_MO_HINH)
print("Da luu", FILE_MO_HINH, "va", FILE_DANH_GIA)
