"""Lưu lịch sử dự đoán vào SQLite để theo dõi tiến bộ của từng sinh viên theo thời gian."""
import datetime
import sqlite3

import pandas as pd

from cau_hinh import CAC_COT, FILE_LICH_SU

COT_LICH_SU = ["thoi_gian", "ma_sinh_vien", "giai_doan"] + CAC_COT + ["xep_loai", "xac_suat_yeu",
                                                                     "diem_du_kien", "nguy_co"]


def _ket_noi():
    con = sqlite3.connect(FILE_LICH_SU)
    con.execute("""CREATE TABLE IF NOT EXISTS lich_su (
        id INTEGER PRIMARY KEY AUTOINCREMENT, thoi_gian TEXT, ma_sinh_vien TEXT, giai_doan TEXT,
        gio_hoc REAL, chuyen_can REAL, diem_bai_tap REAL, diem_giua_ky REAL, diem_cuoi_ky REAL,
        xep_loai TEXT, xac_suat_yeu REAL, diem_du_kien REAL, nguy_co TEXT)""")
    return con


def them(ban_ghi):
    """ban_ghi: dict có các khóa trong COT_LICH_SU (cột điểm chưa biết để None). thoi_gian tự thêm nếu thiếu."""
    ban_ghi = dict(ban_ghi)
    ban_ghi.setdefault("thoi_gian", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    con = _ket_noi()
    with con:
        con.execute(f"INSERT INTO lich_su ({', '.join(COT_LICH_SU)}) VALUES ({', '.join('?' * len(COT_LICH_SU))})",
                    [ban_ghi.get(c) for c in COT_LICH_SU])
    con.close()


def doc():
    con = _ket_noi()
    df = pd.read_sql_query(f"SELECT {', '.join(COT_LICH_SU)} FROM lich_su ORDER BY id DESC", con)
    con.close()
    return df


def xoa_het():
    con = _ket_noi()
    with con:
        con.execute("DELETE FROM lich_su")
    con.close()
