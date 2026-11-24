"""
Hệ thống dự đoán kết quả học tập của sinh viên (v3) - giao diện Streamlit.
Chạy:  streamlit run app.py

Mô hình đọc từ mo_hinh_v3.pkl (tạo bằng huan_luyen.py). Nếu chưa có hoặc không nạp được,
ứng dụng tự huấn luyện lại từ du_lieu/sinh_vien_v3.csv.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

import du_lieu_that as DT
import lich_su as LS
import mo_hinh as MH
import phan_tich as PT
from cau_hinh import (CAC_COT, FILE_DU_LIEU_THAT, FILE_LOP_MAU, FILE_MO_HINH, FILE_MO_HINH_THAT,
                      FILE_SINH_VIEN, FILE_SO_SANH, FILE_UCI, GIAI_DOAN, GIAI_DOAN_THAT, HANG_LOP,
                      KHOANG, MAC_DINH, TEN_COT, TEN_COT_THAT, TEN_GIAI_DOAN, TEN_LOP, THU_TU_LOP)

st.set_page_config(page_title="Dự đoán kết quả học tập", page_icon="🎓", layout="wide")

VI_DU = {
    "Sinh viên khá": {"gio_hoc": 5.0, "chuyen_can": 80.0, "diem_bai_tap": 7.0,
                      "diem_giua_ky": 8.0, "diem_cuoi_ky": 7.0},
    "Sinh viên có nguy cơ": {"gio_hoc": 1.0, "chuyen_can": 55.0, "diem_bai_tap": 4.0,
                             "diem_giua_ky": 3.0, "diem_cuoi_ky": 3.0},
}
MUC_TIEU = {"Hết nguy cơ rớt môn": "an_toan", "Đạt loại Khá": "Kha", "Đạt loại Giỏi": "Gioi"}


# ---------- Nạp mô hình và dữ liệu ----------
@st.cache_resource
def tai_mo_hinh():
    if os.path.exists(FILE_MO_HINH):
        try:
            mo = joblib.load(FILE_MO_HINH)
            if MH.dung_duoc(mo):
                return mo, "mo_hinh_v3.pkl"
        except Exception:
            pass
    if os.path.exists(FILE_SINH_VIEN):
        return MH.huan_luyen_tat_ca(), "tự huấn luyện từ sinh_vien_v3.csv"
    return None, ""


@st.cache_data
def tai_du_lieu():
    return MH.doc_du_lieu()


@st.cache_data
def tai_danh_gia(_mo, nguon):
    return MH.danh_gia_tat_ca(_mo, tai_du_lieu())


@st.cache_resource
def tai_mo_hinh_that():
    if not os.path.exists(FILE_UCI):
        return None
    try:
        mo = joblib.load(FILE_MO_HINH_THAT)
        mo["co_g2"]["mo_hinh"].predict_proba(pd.DataFrame([[2, 0, 4, 10, 10]], columns=DT.GIAI_DOAN_THAT[2]["cot"]))
        return mo
    except Exception:
        return DT.huan_luyen_that()


@st.cache_data
def tai_danh_gia_that(_mo, khoa):
    return DT.danh_gia_that(_mo)


@st.cache_data
def tai_so_sanh_that(khoa):
    f = FILE_DU_LIEU_THAT.replace(".csv", "_so_sanh.csv")
    return pd.read_csv(f) if os.path.exists(f) else DT.so_sanh_that()


mo, nguon_mo = tai_mo_hinh()
if mo is None:
    st.error("Không có dữ liệu để huấn luyện. Hãy chạy `python tao_du_lieu_v3.py` trước.")
    st.stop()
df_goc = tai_du_lieu()
bang_dg = tai_danh_gia(mo, nguon_mo)

# ---------- Trạng thái nhập liệu ----------
for c in CAC_COT:
    st.session_state.setdefault(f"in_{c}", MAC_DINH[c])


def dat_gia_tri(gia_tri):
    for c, v in gia_tri.items():
        st.session_state[f"in_{c}"] = float(v)


# ---------- Thanh bên ----------
with st.sidebar:
    st.header("Giai đoạn dự đoán")
    ten_gd = st.radio("Đang ở giai đoạn nào của học kỳ?", TEN_GIAI_DOAN, index=2, label_visibility="collapsed")
    idx = TEN_GIAI_DOAN.index(ten_gd)
    g = GIAI_DOAN[idx]
    m = mo[g["ma"]]
    st.caption(g["mo_ta"])
    dg = bang_dg.iloc[idx]
    st.caption(f"Độ chính xác giai đoạn này: **{dg['accuracy_cv_5x3'] * 100:.1f}%** (kiểm định chéo 5×3)")
    st.divider()
    st.header("Cảnh báo nguy cơ rớt môn")
    nguong = st.slider("Cảnh báo khi xác suất xếp loại Yếu từ", 0.05, 0.95, float(m["nguong"]), 0.05,
                       key=f"nguong_{g['ma']}",
                       help="Mặc định là ngưỡng cho F1 cao nhất của lớp Yếu, chọn bằng kiểm định chéo trên tập học. "
                            "Hạ ngưỡng thì báo nhiều hơn, ít bỏ sót hơn nhưng báo nhầm nhiều hơn.")
    st.divider()
    st.header("Điền nhanh ví dụ")
    for ten, gia_tri in VI_DU.items():
        st.button(ten, on_click=dat_gia_tri, args=(gia_tri,))
    st.button("Đặt lại", on_click=dat_gia_tri, args=(MAC_DINH,))

st.title("🎓 Hệ thống dự đoán kết quả học tập")
st.caption("Môn Học máy và ứng dụng · Dữ liệu mô phỏng 1000 sinh viên và dữ liệu thật UCI Student Performance")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dự đoán", "Dashboard lớp", "Lịch sử", "Mô hình và đánh giá", "Dữ liệu thật (UCI)"])

# =====================================================================
# Tab 1: dự đoán một sinh viên
# =====================================================================
with tab1:
    st.subheader(f"Nhập thông tin sinh viên · giai đoạn {g['ten']}")
    ma_sv = st.text_input("Mã sinh viên (tuỳ chọn, để theo dõi tiến bộ ở tab Lịch sử)", key="ma_sv")
    c1, c2 = st.columns(2)
    for i, c in enumerate(CAC_COT):
        lo, hi, buoc = KHOANG[c]
        co = c in g["cot"]
        nhan = TEN_COT[c] + ("" if co else " (chưa có ở giai đoạn này)")
        with (c1 if i % 2 == 0 else c2):
            st.number_input(nhan, lo, hi, step=buoc, key=f"in_{c}", disabled=not co,
                            format="%.0f" if c == "chuyen_can" else "%.1f")

    gia_tri = {c: float(st.session_state[f"in_{c}"]) for c in CAC_COT}
    hang = PT.tao_hang(gia_tri, m["cot"])
    kq = PT.du_doan(m, hang)

    st.divider()
    a, b, c3 = st.columns(3)
    a.metric("Xếp loại dự đoán", TEN_LOP[kq["lop"]])
    b.metric("Xác suất xếp loại Yếu", f"{kq['p_yeu'] * 100:.1f}%")
    c3.metric("Điểm tổng dự kiến", f"{kq['diem']:.2f}")
    if idx < 3:
        st.caption(f"Điểm tổng dự kiến nằm trong khoảng {kq['diem_thap']:.2f} đến {kq['diem_cao']:.2f} "
                   "(khoảng sai số 90%, tính từ kiểm định chéo).")
    if kq["p_yeu"] >= nguong:
        st.error(f"⚠️ Có nguy cơ rớt môn (xác suất Yếu {kq['p_yeu'] * 100:.1f}%, ngưỡng cảnh báo {nguong * 100:.0f}%)")
    else:
        st.success(f"✅ Có khả năng qua môn (xác suất Yếu {kq['p_yeu'] * 100:.1f}%, ngưỡng cảnh báo {nguong * 100:.0f}%)")
    if idx < 3:
        st.info(f"Đây là dự báo sớm. Độ chính xác của giai đoạn này khoảng {dg['accuracy_cv_5x3'] * 100:.0f}%, "
                "kết quả cuối cùng còn phụ thuộc phần điểm chưa có.")

    ngoai = [TEN_COT[c] for c in m["cot"]
             if not (df_goc[c].min() <= gia_tri[c] <= df_goc[c].max())]
    if ngoai:
        st.warning("Các giá trị sau nằm ngoài khoảng dữ liệu huấn luyện, kết quả có thể kém tin cậy: "
                   + ", ".join(ngoai))

    xs = pd.DataFrame({"Xác suất (%)": [round(kq["xac_suat"].get(k, 0) * 100, 1) for k in THU_TU_LOP]},
                      index=[TEN_LOP[k] for k in THU_TU_LOP])
    st.write("**Xác suất theo từng xếp loại**")
    st.bar_chart(xs)

    # ----- Giải thích -----
    with st.expander("Vì sao có dự đoán này?", expanded=True):
        dong_gop = PT.dong_gop(m, hang)
        tong = dong_gop.abs().sum()
        ty_le = (dong_gop / tong * 100) if tong > 0 else dong_gop
        st.bar_chart(ty_le.rename(index=TEN_COT).sort_values().rename("Đóng góp vào nguy cơ Yếu (%)"))
        tang = ty_le[ty_le > 0].sort_values(ascending=False)
        giam = ty_le[ty_le < 0].sort_values()
        if len(tang):
            st.write("**Làm tăng nguy cơ:** " + "; ".join(f"{TEN_COT[c]} (+{v:.0f}%)" for c, v in tang.items()))
        if len(giam):
            st.write("**Làm giảm nguy cơ:** " + "; ".join(f"{TEN_COT[c]} ({v:.0f}%)" for c, v in giam.items()))
        st.caption("Đóng góp được tính so với một sinh viên trung bình: cột dương đẩy sinh viên về phía xếp loại Yếu, "
                   "cột âm kéo ra xa. Phần trăm là tỷ trọng trong tổng mức đóng góp của các thuộc tính.")

    # ----- Gợi ý cải thiện -----
    with st.expander("Cần làm gì để cải thiện?", expanded=True):
        if idx == len(GIAI_DOAN) - 1:
            st.write("Đã có đủ điểm nên xếp loại này là chính thức, không còn gợi ý hành động.")
        else:
            ten_mt = st.radio("Mục tiêu", list(MUC_TIEU), horizontal=True, key="muc_tieu")
            mt = MUC_TIEU[ten_mt]
            da_dat = (kq["p_yeu"] < nguong) if mt == "an_toan" else (HANG_LOP[kq["lop"]] >= HANG_LOP[mt])
            if da_dat:
                st.success("Với thông tin hiện có, mô hình đã dự đoán đạt mục tiêu này.")
            cac_goi_y = PT.goi_y(mo, idx, gia_tri, mt, nguong)
            dong = []
            for r in cac_goi_y:
                if r["loai"] == "diem_tiep":
                    if r["trang_thai"] == "da_dat":
                        dong.append(f"**{r['ten']}**: đạt mục tiêu dù điểm này thấp, với các thông tin hiện có.")
                    elif r["trang_thai"] == "can":
                        dong.append(f"**{r['ten']}** cần đạt từ **{r['can_dat']:.1f}** trở lên.")
                    else:
                        dong.append(f"**{r['ten']}**: không đạt được mục tiêu dù điểm này tối đa.")
                else:
                    dv = "" if r["ten"].startswith("Điểm") else " giờ"
                    if r["trang_thai"] == "da_dat":
                        dong.append(f"**{r['ten']}**: mức hiện tại ({r['hien_tai']:.1f}) đã đủ cho mục tiêu.")
                    elif r["trang_thai"] == "can":
                        dong.append(f"Tăng **{r['ten']}** từ {r['hien_tai']:.1f} lên **{r['can_dat']:.1f}**{dv}.")
                    else:
                        dong.append(f"**{r['ten']}**: chỉ thay đổi riêng mục này là chưa đủ.")
            for d in dong:
                st.markdown("- " + d)
            st.caption("Mỗi gợi ý được tính riêng, giữ nguyên các thông tin còn lại. Càng về sau trong học kỳ, "
                       "thói quen học càng ít ảnh hưởng tới dự đoán vì điểm thành phần đã quyết định phần lớn kết quả.")

    # ----- Lưu lịch sử -----
    if st.button("Lưu kết quả vào lịch sử"):
        LS.them({"ma_sinh_vien": ma_sv.strip() or "(không có mã)", "giai_doan": g["ten"],
                 **{c: (gia_tri[c] if c in m["cot"] else None) for c in CAC_COT},
                 "xep_loai": TEN_LOP[kq["lop"]], "xac_suat_yeu": round(kq["p_yeu"], 4),
                 "diem_du_kien": round(kq["diem"], 2),
                 "nguy_co": "Có" if kq["p_yeu"] >= nguong else "Không"})
        st.toast("Đã lưu vào lịch sử")

# =====================================================================
# Tab 2: dashboard cả lớp
# =====================================================================
with tab2:
    st.subheader(f"Phân tích cả lớp · giai đoạn {g['ten']}")
    st.write("File CSV cần có các cột: " + ", ".join(f"`{c}`" for c in m["cot"])
             + ". Các cột `ma_sinh_vien` và `ho_ten` là tuỳ chọn.")
    nguon = st.radio("Nguồn dữ liệu", ["Dùng lớp mẫu (120 sinh viên)", "Tải file CSV lên"], horizontal=True)
    df_lop = None
    if nguon.startswith("Dùng"):
        if os.path.exists(FILE_LOP_MAU):
            df_lop = pd.read_csv(FILE_LOP_MAU)
    else:
        file_tai = st.file_uploader("Chọn file CSV", type="csv")
        if file_tai is not None:
            try:
                df_lop = pd.read_csv(file_tai)
            except Exception as loi:
                st.error(f"Không đọc được file: {loi}")
    if df_lop is not None:
        thieu = [c for c in m["cot"] if c not in df_lop.columns]
        if thieu:
            st.error("File thiếu cột: " + ", ".join(thieu))
        else:
            truoc = len(df_lop)
            df_lop = df_lop.dropna(subset=m["cot"]).reset_index(drop=True)
            if len(df_lop) < truoc:
                st.warning(f"Đã bỏ {truoc - len(df_lop)} dòng thiếu dữ liệu.")
            kq_lop = PT.phan_tich_lop(df_lop, m, nguong)
            so_nc = int((kq_lop["nguy_co_rot_mon"] == "Có").sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Số sinh viên", len(kq_lop))
            k2.metric("Có nguy cơ rớt môn", so_nc)
            k3.metric("Tỷ lệ nguy cơ", f"{so_nc / max(len(kq_lop), 1) * 100:.1f}%")
            k4.metric("Điểm tổng dự kiến TB", f"{kq_lop['diem_tong_du_kien'].mean():.2f}")
            t_hop = PT.tong_hop_lop(kq_lop)
            cl, cr = st.columns(2)
            with cl:
                st.write("**Phân bố xếp loại dự đoán**")
                st.bar_chart(t_hop.set_index("Xếp loại dự đoán")["Số sinh viên"])
            with cr:
                st.write("**Trung bình từng thuộc tính theo xếp loại dự đoán**")
                tb = (kq_lop.groupby("xep_loai_du_doan")[m["cot"]].mean().round(2)
                      .reindex([TEN_LOP[k] for k in THU_TU_LOP]).dropna(how="all").rename(columns=TEN_COT))
                st.dataframe(tb)
            st.write("**Danh sách sinh viên nguy cơ cao nhất** (xếp theo xác suất Yếu)")
            st.dataframe(kq_lop[kq_lop["nguy_co_rot_mon"] == "Có"].head(20), hide_index=True)
            with st.expander("Xem toàn bộ kết quả"):
                st.dataframe(kq_lop, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("Tải báo cáo Excel", PT.xuat_excel(kq_lop, g["ten"], nguong),
                               "bao_cao_lop.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            d2.download_button("Tải kết quả CSV", kq_lop.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_lop.csv", "text/csv")

# =====================================================================
# Tab 3: lịch sử
# =====================================================================
with tab3:
    st.subheader("Lịch sử dự đoán")
    ls = LS.doc()
    if ls.empty:
        st.info("Chưa có bản ghi nào. Ở tab Dự đoán, bấm “Lưu kết quả vào lịch sử”.")
    else:
        loc = st.text_input("Lọc theo mã sinh viên")
        ls_hien = ls[ls["ma_sinh_vien"].fillna("").str.contains(loc, case=False)] if loc else ls
        st.dataframe(ls_hien.rename(columns={**TEN_COT, "thoi_gian": "Thời gian", "ma_sinh_vien": "Mã SV",
                                             "giai_doan": "Giai đoạn", "xep_loai": "Xếp loại",
                                             "xac_suat_yeu": "Xác suất Yếu", "diem_du_kien": "Điểm dự kiến",
                                             "nguy_co": "Nguy cơ"}), hide_index=True)
        nhieu = ls["ma_sinh_vien"].value_counts()
        nhieu = nhieu[(nhieu >= 2) & (nhieu.index != "(không có mã)")]
        if len(nhieu):
            ma = st.selectbox("Theo dõi tiến bộ của sinh viên (có từ 2 lần dự đoán trở lên)", list(nhieu.index))
            ct = ls[ls["ma_sinh_vien"] == ma].sort_values("thoi_gian")
            nhan = [f"{t[5:16]} · {gd}" for t, gd in zip(ct["thoi_gian"], ct["giai_doan"])]
            ca, cb = st.columns(2)
            ca.write("**Xác suất xếp loại Yếu (%)**")
            ca.line_chart(pd.Series((ct["xac_suat_yeu"] * 100).to_numpy(), index=nhan))
            cb.write("**Điểm tổng dự kiến**")
            cb.line_chart(pd.Series(ct["diem_du_kien"].to_numpy(), index=nhan))
        else:
            st.caption("Nhập mã sinh viên và lưu từ 2 lần dự đoán trở lên (ví dụ ở các giai đoạn khác nhau) "
                       "để xem biểu đồ tiến bộ.")
        st.download_button("Tải lịch sử CSV", ls.to_csv(index=False).encode("utf-8-sig"), "lich_su.csv", "text/csv")
        if st.checkbox("Tôi muốn xoá toàn bộ lịch sử"):
            if st.button("Xoá lịch sử"):
                LS.xoa_het()
                st.toast("Đã xoá lịch sử")

# =====================================================================
# Tab 4: mô hình và đánh giá
# =====================================================================
with tab4:
    st.subheader("Mô hình và đánh giá")
    st.write(f"Nguồn mô hình đang dùng: **{nguon_mo}**. Mô hình chính ở mọi giai đoạn là hồi quy logistic có chuẩn hoá, "
             "kèm hồi quy tuyến tính dự đoán điểm tổng.")
    bang = bang_dg.copy()
    bang_hien = pd.DataFrame({
        "Giai đoạn": bang["giai_doan"],
        "Accuracy kiểm tra": bang["accuracy_kiem_tra"].map("{:.3f}".format),
        "Accuracy CV 5×3": [f"{a:.3f} ± {s:.3f}" for a, s in zip(bang["accuracy_cv_5x3"], bang["do_lech_cv"])],
        "Macro F1": bang["macro_f1"].map("{:.3f}".format),
        "Mức sàn": bang["muc_san"].map("{:.3f}".format),
        "Ngưỡng cảnh báo": bang["nguong_canh_bao"].map("{:.2f}".format),
        "Recall Yếu": bang["recall_yeu"].map("{:.3f}".format),
        "Precision Yếu": bang["precision_yeu"].map("{:.3f}".format),
        "AUC Yếu": bang["auc_yeu"].map("{:.3f}".format),
        "MAE điểm tổng": bang["mae_diem_tong"].map("{:.2f}".format),
    })
    st.dataframe(bang_hien, hide_index=True)
    st.write("**Độ chính xác tăng dần theo giai đoạn** (kiểm định chéo 5×3, so với mức đoán bừa)")
    st.bar_chart(pd.DataFrame({"Accuracy CV 5×3": bang["accuracy_cv_5x3"].to_numpy(),
                               "Mức sàn (đoán bừa)": bang["muc_san"].to_numpy()}, index=bang["giai_doan"]))

    st.write(f"**Ma trận nhầm lẫn · giai đoạn {g['ten']}** (hàng là xếp loại thật, cột là xếp loại đoán, tập kiểm tra)")
    X_tr, X_te, y_tr, y_te, t_tr, t_te = MH.chia_du_lieu(df_goc)
    du = m["phan_loai"].predict(X_te[m["cot"]])
    cm = pd.DataFrame(confusion_matrix(y_te, du, labels=THU_TU_LOP),
                      index=[f"Thật: {TEN_LOP[k]}" for k in THU_TU_LOP],
                      columns=[f"Đoán: {TEN_LOP[k]}" for k in THU_TU_LOP])
    st.dataframe(cm)

    st.write(f"**Hệ số của hồi quy logistic · giai đoạn {g['ten']}** (trên dữ liệu đã chuẩn hoá, dương là đẩy về lớp đó)")
    hs = pd.DataFrame(m["phan_loai"][-1].coef_, index=[TEN_LOP[k] for k in m["phan_loai"].classes_],
                      columns=[TEN_COT[c] for c in m["cot"]]).round(2)
    st.dataframe(hs)

    if os.path.exists(FILE_SO_SANH):
        ss = pd.read_csv(FILE_SO_SANH)
        pv = ss.pivot(index="mo_hinh", columns="giai_doan", values="accuracy_kiem_tra")[TEN_GIAI_DOAN]
        st.write("**So sánh các mô hình theo giai đoạn** (accuracy trên tập kiểm tra)")
        st.dataframe(pv.sort_values(TEN_GIAI_DOAN[-1], ascending=False).round(3))
    else:
        st.caption("Chạy `python so_sanh_mo_hinh.py` để tạo bảng so sánh các mô hình.")

    with st.expander("Giải thích các chỉ số"):
        st.markdown("""
- **Accuracy kiểm tra**: tỷ lệ sinh viên trong tập kiểm tra (200 người) được đoán đúng xếp loại.
- **Accuracy CV 5×3**: kiểm định chéo 5 phần lặp 3 lần trên cả 1000 sinh viên, ổn định hơn một lần chia.
- **Macro F1**: trung bình F1 của bốn xếp loại, mỗi loại tính ngang nhau dù đông hay ít người.
- **Mức sàn**: accuracy nếu luôn đoán nhóm đông nhất. Mô hình phải vượt mức này mới có ý nghĩa.
- **Ngưỡng cảnh báo**: xác suất Yếu từ mức này trở lên thì báo nguy cơ rớt môn, chọn cho F1 của lớp Yếu cao nhất.
- **Recall Yếu**: trong các sinh viên thật sự Yếu, tỷ lệ được báo nguy cơ. **Precision Yếu**: trong các sinh viên được báo nguy cơ, tỷ lệ thật sự Yếu.
- **AUC Yếu**: khả năng xếp sinh viên Yếu có điểm nguy cơ cao hơn sinh viên không Yếu, không phụ thuộc ngưỡng (1 là hoàn hảo, 0,5 là ngẫu nhiên).
- **MAE điểm tổng**: sai số tuyệt đối trung bình của điểm tổng dự kiến, tính trên thang 10.
""")
    st.info("Dữ liệu mô phỏng: mỗi sinh viên có năng lực và thói quen học ẩn, cộng nhiễu ngẫu nhiên, nhãn tính theo công thức điểm tổng. "
            "Ở giai đoạn Sau cuối kỳ, mô hình thực chất tính lại công thức nên gần như tuyệt đối. "
            "Các giai đoạn trước mới là dự báo thật sự. Kết quả không phản ánh dữ liệu thực tế của trường, "
            "xem tab Dữ liệu thật để có con số trên dữ liệu thật.")

# =====================================================================
# Tab 5: dữ liệu thật UCI
# =====================================================================
with tab5:
    st.subheader("Dữ liệu thật: UCI Student Performance (môn Toán)")
    mo_that = tai_mo_hinh_that()
    if mo_that is None:
        st.info("Chưa có file `du_lieu/student-mat.csv`. Tải bộ dữ liệu từ UCI "
                "(https://archive.ics.uci.edu/dataset/320/student+performance) rồi đặt vào thư mục du_lieu.")
    else:
        df_that = DT.doc_uci()
        st.write(f"395 học sinh hai trường trung học ở Bồ Đào Nha, tỷ lệ đậu {df_that['dau'].mean() * 100:.1f}% "
                 "(đậu là điểm cuối kỳ G3 từ 10 trở lên, thang 20). "
                 "Nguồn: Cortez, P. (2008), UCI Machine Learning Repository, giấy phép CC BY 4.0. "
                 "Bài toán: đoán đậu hay rớt theo ba giai đoạn tuỳ thông tin đã biết.")
        dg_that = tai_danh_gia_that(mo_that, "that")
        st.dataframe(pd.DataFrame({
            "Giai đoạn": dg_that["giai_doan"],
            "Accuracy kiểm tra": dg_that["accuracy_kiem_tra"].map("{:.3f}".format),
            "Accuracy CV 5×3": [f"{a:.3f} ± {s:.3f}" for a, s in zip(dg_that["accuracy_cv_5x3"], dg_that["do_lech_cv"])],
            "Macro F1": dg_that["macro_f1"].map("{:.3f}".format),
            "Recall rớt": dg_that["recall_rot"].map("{:.3f}".format),
            "AUC rớt": dg_that["auc_rot"].map("{:.3f}".format),
            "Mức sàn": dg_that["muc_san"].map("{:.3f}".format),
        }), hide_index=True)
        st.bar_chart(pd.DataFrame({"Accuracy CV 5×3": dg_that["accuracy_cv_5x3"].to_numpy(),
                                   "Mức sàn (đoán bừa)": dg_that["muc_san"].to_numpy()},
                                  index=dg_that["giai_doan"]))
        st.caption("Khi chưa có điểm, mô hình chỉ ngang mức đoán bừa: thời gian tự học, số lần rớt và số buổi vắng "
                   "chưa đủ để dự báo. Điểm kỳ 1 và kỳ 2 là yếu tố quyết định, đúng như kết luận của nghiên cứu gốc.")
        st.write("**So sánh các mô hình** (accuracy trên tập kiểm tra)")
        st.dataframe(tai_so_sanh_that("that"), hide_index=True)

        st.divider()
        st.write("**Thử dự đoán đậu hay rớt**")
        ten_gd_that = st.selectbox("Giai đoạn", [x["ten"] for x in GIAI_DOAN_THAT], index=2)
        gd_that = next(x for x in GIAI_DOAN_THAT if x["ten"] == ten_gd_that)
        mac_dinh_that = {"studytime": 2, "failures": 0, "absences": 4, "G1": 10, "G2": 10}
        gioi_han = {"studytime": (1, 4), "failures": (0, 3), "absences": (0, 75), "G1": (0, 20), "G2": (0, 20)}
        cc = st.columns(5)
        gt_that = {}
        for i, c in enumerate(mac_dinh_that):
            co = c in gd_that["cot"]
            with cc[i]:
                gt_that[c] = st.number_input(c + ("" if co else " (chưa có)"), gioi_han[c][0], gioi_han[c][1],
                                             mac_dinh_that[c], step=1, key=f"that_{c}", disabled=not co,
                                             help=TEN_COT_THAT[c])
        mt = mo_that[gd_that["ma"]]["mo_hinh"]
        h = pd.DataFrame([[gt_that[c] for c in gd_that["cot"]]], columns=gd_that["cot"])
        p_dau = float(mt.predict_proba(h)[0][1])
        e1, e2 = st.columns(2)
        e1.metric("Xác suất đậu", f"{p_dau * 100:.1f}%")
        e2.metric("Xác suất rớt", f"{(1 - p_dau) * 100:.1f}%")
        if 1 - p_dau >= 0.5:
            st.error("⚠️ Có nguy cơ rớt môn (xác suất rớt từ 50% trở lên)")
        else:
            st.success("✅ Có khả năng qua môn")
        z = mt[0].transform(h)[0]
        dong_gop_that = pd.Series(-mt[-1].coef_[0] * z, index=gd_that["cot"])   # dương là tăng nguy cơ rớt
        tong_that = dong_gop_that.abs().sum()
        if tong_that > 0:
            st.write("**Đóng góp vào nguy cơ rớt (%)** (dương là làm tăng nguy cơ)")
            st.bar_chart((dong_gop_that / tong_that * 100).sort_values().rename("Đóng góp (%)"))
        st.caption("Ý nghĩa các cột: " + "; ".join(f"{k}: {v}" for k, v in TEN_COT_THAT.items()))
