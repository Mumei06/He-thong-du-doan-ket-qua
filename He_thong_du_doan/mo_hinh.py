"""
Huấn luyện và đánh giá mô hình cho từng giai đoạn của học kỳ (dùng chung cho script và app).

Mỗi giai đoạn có:
- phan_loai: pipeline StandardScaler + LogisticRegression (nhiều lớp), chọn C bằng GridSearchCV (Lab 2, 4, 7)
- hoi_quy: pipeline StandardScaler + LinearRegression dự đoán điểm tổng (Lab 1)
- nguong: ngưỡng cảnh báo nguy cơ Yếu cho F1 cao nhất, chọn bằng cross_val_predict trên tập học (Lab 7)
- khoang_sai_so: khoảng sai số 90% của điểm tổng dự kiến, tính từ phần dư kiểm định chéo
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cau_hinh import CAC_COT, FILE_SINH_VIEN, GIAI_DOAN

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def doc_du_lieu():
    return pd.read_csv(FILE_SINH_VIEN)


def chia_du_lieu(df):
    """Chia 80/20, giữ tỷ lệ các xếp loại. Trả về X_tr, X_te, y_tr, y_te, t_tr, t_te (t là điểm tổng)."""
    return train_test_split(df[CAC_COT], df["xep_loai"], df["diem_tong"],
                            test_size=0.2, random_state=42, stratify=df["xep_loai"])


def _nguong_toi_uu(pc, X, y):
    """Ngưỡng xác suất Yếu cho F1 cao nhất, chọn trên xác suất kiểm định chéo của tập học."""
    xs = cross_val_predict(pc, X, y, cv=CV, method="predict_proba")
    p_yeu = xs[:, list(pc.classes_).index("Yeu")]
    that = (y == "Yeu").to_numpy()
    cac = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1 = [f1_score(that, p_yeu >= t, zero_division=0) for t in cac]
    return float(cac[int(np.argmax(f1))])


def huan_luyen_giai_doan(X_tr, y_tr, t_tr, cot):
    mo_hinh = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    luoi = GridSearchCV(mo_hinh, {"logisticregression__C": [0.1, 1, 10, 100, 1000]}, cv=CV)
    luoi.fit(X_tr[cot], y_tr)
    pc = luoi.best_estimator_
    nguong = _nguong_toi_uu(pc, X_tr[cot], y_tr)

    hq = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr[cot], t_tr)
    phan_du = t_tr - cross_val_predict(hq, X_tr[cot], t_tr, cv=5)
    return {
        "cot": list(cot),
        "phan_loai": pc,
        "hoi_quy": hq,
        "nguong": nguong,
        "khoang_sai_so": (float(np.quantile(phan_du, 0.05)), float(np.quantile(phan_du, 0.95))),
        "C": luoi.best_params_["logisticregression__C"],
        "cv_tap_hoc": float(luoi.best_score_),
    }


def huan_luyen_tat_ca(df=None):
    """Huấn luyện mô hình cho cả bốn giai đoạn. Trả về dict {ma_giai_doan: thông tin mô hình}."""
    df = doc_du_lieu() if df is None else df
    X_tr, _, y_tr, _, t_tr, _ = chia_du_lieu(df)
    return {g["ma"]: huan_luyen_giai_doan(X_tr, y_tr, t_tr, g["cot"]) for g in GIAI_DOAN}


def dung_duoc(cac_mo_hinh):
    """Thử dự đoán một dòng để chắc mô hình dùng được trên máy này."""
    try:
        for g in GIAI_DOAN:
            m = cac_mo_hinh[g["ma"]]
            hang = pd.DataFrame([[3, 80, 7, 6.5, 6.5]], columns=CAC_COT)[m["cot"]]
            m["phan_loai"].predict_proba(hang)
            m["hoi_quy"].predict(hang)
        return True
    except Exception:
        return False


def danh_gia_giai_doan(m, df):
    """Đánh giá một giai đoạn trên tập kiểm tra và bằng kiểm định chéo 5x3 trên toàn bộ dữ liệu."""
    X_tr, X_te, y_tr, y_te, t_tr, t_te = chia_du_lieu(df)
    cot, pc, hq = m["cot"], m["phan_loai"], m["hoi_quy"]
    du = pc.predict(X_te[cot])
    p_yeu = pc.predict_proba(X_te[cot])[:, list(pc.classes_).index("Yeu")]
    yeu_that = (y_te == "Yeu").to_numpy()
    nguong = m["nguong"]
    bao = p_yeu >= nguong
    luoi = DummyClassifier(strategy="most_frequent").fit(X_tr[cot], y_tr)
    cv = cross_val_score(pc, df[cot], df["xep_loai"],
                         cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0))
    pred_t = hq.predict(X_te[cot])
    return {
        "accuracy_kiem_tra": accuracy_score(y_te, du),
        "macro_f1": f1_score(y_te, du, average="macro"),
        "accuracy_cv_5x3": float(cv.mean()),
        "do_lech_cv": float(cv.std()),
        "muc_san": accuracy_score(y_te, luoi.predict(X_te[cot])),
        "nguong_canh_bao": nguong,
        "recall_yeu": recall_score(yeu_that, bao, zero_division=0),
        "precision_yeu": precision_score(yeu_that, bao, zero_division=0),
        "auc_yeu": roc_auc_score(yeu_that, p_yeu),
        "mae_diem_tong": mean_absolute_error(t_te, pred_t),
        "r2_diem_tong": r2_score(t_te, pred_t),
        "C": m["C"],
    }


def danh_gia_tat_ca(cac_mo_hinh, df=None):
    df = doc_du_lieu() if df is None else df
    dong = []
    for g in GIAI_DOAN:
        d = danh_gia_giai_doan(cac_mo_hinh[g["ma"]], df)
        d = {"giai_doan": g["ten"], **d}
        dong.append(d)
    return pd.DataFrame(dong)
