"""
Phân tích sau dự đoán: giải thích, gợi ý cải thiện ("nếu thì"), phân tích cả lớp và xuất Excel.
"""
import io

import numpy as np
import pandas as pd

from cau_hinh import GIAI_DOAN, HANG_LOP, KHOANG, TEN_COT, TEN_LOP, THU_TU_LOP

# Giới hạn trên khi gợi ý thói quen, để không gợi ý những mức không thực tế (học trên 8 giờ mỗi ngày)
GIOI_HAN_GOI_Y = {"gio_hoc": 8.0, "chuyen_can": 100.0}


def tao_hang(gia_tri, cot):
    return pd.DataFrame([[gia_tri[c] for c in cot]], columns=cot)


def du_doan(m, hang):
    """Dự đoán xếp loại, xác suất và điểm tổng dự kiến (kèm khoảng sai số 90%)."""
    pc, hq = m["phan_loai"], m["hoi_quy"]
    p = pd.Series(pc.predict_proba(hang[m["cot"]])[0], index=pc.classes_)
    diem = float(hq.predict(hang[m["cot"]])[0])
    lo, hi = m["khoang_sai_so"]
    return {
        "lop": p.idxmax(), "xac_suat": p, "p_yeu": float(p.get("Yeu", 0.0)),
        "diem": float(np.clip(diem, 0, 10)),
        "diem_thap": float(np.clip(diem + lo, 0, 10)), "diem_cao": float(np.clip(diem + hi, 0, 10)),
    }


def dong_gop(m, hang, lop="Yeu"):
    """
    Đóng góp của từng thuộc tính vào điểm số của lớp `lop` so với một sinh viên trung bình.
    Với hồi quy logistic, điểm số là tổng (hệ số x giá trị đã chuẩn hoá), nên chia được theo từng cột.
    Dương nghĩa là thuộc tính đó đẩy sinh viên về phía lớp này.
    """
    pc = m["phan_loai"]
    z = pc[0].transform(hang[m["cot"]])[0]
    k = list(pc.classes_).index(lop)
    return pd.Series(pc[-1].coef_[k] * z, index=m["cot"])


def _dat_muc_tieu(xs, classes, muc_tieu, nguong):
    classes = list(classes)
    p_yeu = xs[:, classes.index("Yeu")]
    if muc_tieu == "an_toan":
        return p_yeu < nguong
    lop = np.array(classes)[xs.argmax(axis=1)]
    return np.array([HANG_LOP[c] for c in lop]) >= HANG_LOP[muc_tieu]


def tim_toi_thieu(m, hang, ten_cot, tu, muc_tieu, nguong):
    """
    Tìm giá trị nhỏ nhất của cột `ten_cot` (từ `tu` trở lên) để đạt mục tiêu, giữ nguyên các cột khác.
    Trả về (giá trị, trạng thái) với trạng thái là "da_dat", "can" hoặc "khong_du".
    """
    _, hi, buoc = KHOANG[ten_cot]
    hi = GIOI_HAN_GOI_Y.get(ten_cot, hi)
    gia_tri = np.arange(tu, hi + 1e-9, buoc)
    bang = pd.concat([hang] * len(gia_tri), ignore_index=True)
    bang[ten_cot] = gia_tri
    pc = m["phan_loai"]
    ok = _dat_muc_tieu(pc.predict_proba(bang[m["cot"]]), pc.classes_, muc_tieu, nguong)
    if not ok.any():
        return None, "khong_du"
    i = int(np.argmax(ok))
    return float(gia_tri[i]), ("da_dat" if i == 0 and tu == hang[ten_cot].iloc[0] else "can")


def goi_y(cac_mo_hinh, chi_so_gd, gia_tri, muc_tieu, nguong):
    """
    Gợi ý cần làm gì để đạt mục tiêu. Hai nhóm:
    - Điểm kế tiếp: dùng mô hình của giai đoạn sau để tìm điểm tiếp theo cần đạt.
    - Thói quen (giờ học, chuyên cần): dùng mô hình của giai đoạn hiện tại.
    Trả về danh sách dict gồm: ten, hien_tai, can_dat, trang_thai, loai.
    """
    g = GIAI_DOAN[chi_so_gd]
    m = cac_mo_hinh[g["ma"]]
    hang = tao_hang(gia_tri, m["cot"])
    kq = []
    if chi_so_gd + 1 < len(GIAI_DOAN):
        m_tiep = cac_mo_hinh[GIAI_DOAN[chi_so_gd + 1]["ma"]]
        cot_tiep = [c for c in m_tiep["cot"] if c not in m["cot"]][0]
        hang_tiep = tao_hang(gia_tri, [c for c in m_tiep["cot"] if c != cot_tiep])
        hang_tiep[cot_tiep] = KHOANG[cot_tiep][0]
        v, tt = tim_toi_thieu(m_tiep, hang_tiep, cot_tiep, KHOANG[cot_tiep][0], muc_tieu, nguong)
        kq.append({"ten": TEN_COT[cot_tiep], "hien_tai": None, "can_dat": v,
                   "trang_thai": "khong_du" if v is None else "can", "loai": "diem_tiep"})
        # Với điểm kế tiếp, mức thấp nhất đã đạt nghĩa là chắc chắn đạt mục tiêu dù điểm đó thấp
        if v is not None and v <= KHOANG[cot_tiep][0] + 1e-9:
            kq[-1]["trang_thai"] = "da_dat"
    if chi_so_gd < len(GIAI_DOAN) - 1:
        for c in ["gio_hoc", "chuyen_can"]:
            v, tt = tim_toi_thieu(m, hang, c, float(hang[c].iloc[0]), muc_tieu, nguong)
            kq.append({"ten": TEN_COT[c], "hien_tai": float(hang[c].iloc[0]), "can_dat": v,
                       "trang_thai": tt, "loai": "thoi_quen"})
    return kq


def phan_tich_lop(df, m, nguong):
    """Dự đoán cho cả lớp. Trả về bảng kết quả đã sắp xếp theo xác suất Yếu giảm dần."""
    pc, hq = m["phan_loai"], m["hoi_quy"]
    X = df[m["cot"]]
    P = pc.predict_proba(X)
    kq = df.copy()
    kq["xep_loai_du_doan"] = [TEN_LOP[c] for c in np.array(pc.classes_)[P.argmax(axis=1)]]
    kq["xac_suat_yeu"] = P[:, list(pc.classes_).index("Yeu")].round(3)
    kq["diem_tong_du_kien"] = np.clip(hq.predict(X), 0, 10).round(2)
    kq["nguy_co_rot_mon"] = np.where(kq["xac_suat_yeu"] >= nguong, "Có", "Không")
    return kq.sort_values("xac_suat_yeu", ascending=False).reset_index(drop=True)


def tong_hop_lop(kq):
    """Số lượng và tỷ lệ từng xếp loại dự đoán."""
    dem = kq["xep_loai_du_doan"].value_counts().reindex([TEN_LOP[c] for c in THU_TU_LOP], fill_value=0)
    return pd.DataFrame({"Xếp loại dự đoán": dem.index, "Số sinh viên": dem.values,
                         "Tỷ lệ (%)": (dem.values / max(len(kq), 1) * 100).round(1)})


def xuat_excel(kq, ten_giai_doan, nguong):
    """Xuất báo cáo Excel gồm ba sheet: Tổng hợp, Nguy cơ, Chi tiết. Trả về bytes."""
    from openpyxl.styles import Alignment, Font, PatternFill

    tong_hop = tong_hop_lop(kq)
    thong_tin = pd.DataFrame({
        "Chỉ tiêu": ["Giai đoạn dự đoán", "Ngưỡng cảnh báo (xác suất Yếu từ)", "Số sinh viên",
                     "Số sinh viên có nguy cơ rớt môn", "Tỷ lệ có nguy cơ (%)", "Điểm tổng dự kiến trung bình"],
        "Giá trị": [ten_giai_doan, f"{nguong:.0%}", len(kq), int((kq["nguy_co_rot_mon"] == "Có").sum()),
                    round((kq["nguy_co_rot_mon"] == "Có").mean() * 100, 1), round(kq["diem_tong_du_kien"].mean(), 2)],
    })
    nguy_co = kq[kq["nguy_co_rot_mon"] == "Có"]
    bo = io.BytesIO()
    with pd.ExcelWriter(bo, engine="openpyxl") as w:
        thong_tin.to_excel(w, sheet_name="Tổng hợp", index=False)
        tong_hop.to_excel(w, sheet_name="Tổng hợp", index=False, startrow=len(thong_tin) + 3)
        nguy_co.to_excel(w, sheet_name="Nguy cơ", index=False)
        kq.to_excel(w, sheet_name="Chi tiết", index=False)
        dau = PatternFill("solid", fgColor="D9E2F3")
        do = PatternFill("solid", fgColor="F8D7DA")
        for ws in w.book.worksheets:
            for hang in ws.iter_rows(min_row=1, max_row=1):
                for o in hang:
                    o.font, o.fill = Font(bold=True), dau
            for cot in ws.columns:
                dai = max(len(str(o.value)) if o.value is not None else 0 for o in cot)
                ws.column_dimensions[cot[0].column_letter].width = min(max(dai + 2, 10), 42)
            ws.freeze_panes = "A2"
        if "nguy_co_rot_mon" in kq.columns:
            vi_tri = list(kq.columns).index("nguy_co_rot_mon") + 1
            ws = w.book["Chi tiết"]
            for hang in ws.iter_rows(min_row=2):
                if hang[vi_tri - 1].value == "Có":
                    for o in hang:
                        o.fill = do
        ws = w.book["Tổng hợp"]
        r = len(thong_tin) + 4
        for o in ws[r]:
            o.font, o.fill = Font(bold=True), dau
        ws.freeze_panes = None
        ws["A1"].alignment = Alignment(horizontal="left")
    return bo.getvalue()
