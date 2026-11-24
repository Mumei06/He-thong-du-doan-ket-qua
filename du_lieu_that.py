"""
Đánh giá trên dữ liệu thật: UCI Student Performance (môn Toán, 395 học sinh hai trường trung học ở Bồ Đào Nha).
Nguồn: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
Giấy phép CC BY 4.0. Đặt file student-mat.csv vào thư mục du_lieu/.

Bài toán: đoán học sinh đậu (điểm cuối kỳ G3 từ 10 trở lên, thang 20) hay rớt, theo ba giai đoạn:
chưa có điểm, có điểm kỳ 1 (G1), có điểm kỳ 1 và kỳ 2 (G2). Chạy: python du_lieu_that.py
"""
import warnings

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cau_hinh import FILE_DU_LIEU_THAT, FILE_MO_HINH_THAT, FILE_UCI, GIAI_DOAN_THAT

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def doc_uci():
    df = pd.read_csv(FILE_UCI, sep=";")
    df["dau"] = (df["G3"] >= 10).astype(int)   # 1: đậu, 0: rớt
    return df


def chia(df):
    return train_test_split(df, df["dau"], test_size=0.2, random_state=42, stratify=df["dau"])


def huan_luyen_that(df=None):
    df = doc_uci() if df is None else df
    tr, _, y_tr, _ = chia(df)
    ket_qua = {}
    for g in GIAI_DOAN_THAT:
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}, cv=CV).fit(tr[g["cot"]], y_tr)
        ket_qua[g["ma"]] = {"cot": g["cot"], "mo_hinh": gs.best_estimator_, "C": gs.best_params_["logisticregression__C"]}
    return ket_qua


def danh_gia_that(cac_mo_hinh=None, df=None):
    df = doc_uci() if df is None else df
    cac_mo_hinh = huan_luyen_that(df) if cac_mo_hinh is None else cac_mo_hinh
    tr, te, y_tr, y_te = chia(df)
    dong = []
    for g in GIAI_DOAN_THAT:
        m = cac_mo_hinh[g["ma"]]["mo_hinh"]
        du = m.predict(te[g["cot"]])
        p_rot = m.predict_proba(te[g["cot"]])[:, 0]
        cv = cross_val_score(m, df[g["cot"]], df["dau"],
                             cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0))
        luoi = DummyClassifier(strategy="most_frequent").fit(tr[g["cot"]], y_tr)
        dong.append({"giai_doan": g["ten"], "mo_hinh": "Hồi quy logistic",
                     "accuracy_kiem_tra": accuracy_score(y_te, du),
                     "macro_f1": f1_score(y_te, du, average="macro"),
                     "recall_rot": recall_score(1 - y_te, 1 - du),
                     "auc_rot": roc_auc_score(1 - y_te, p_rot),
                     "accuracy_cv_5x3": float(cv.mean()), "do_lech_cv": float(cv.std()),
                     "muc_san": accuracy_score(y_te, luoi.predict(te[g["cot"]]))})
    return pd.DataFrame(dong)


def so_sanh_that(df=None):
    """Accuracy trên tập kiểm tra của vài mô hình ở từng giai đoạn."""
    df = doc_uci() if df is None else df
    tr, te, y_tr, y_te = chia(df)
    mo_hinh = {
        "Cây quyết định": (DecisionTreeClassifier(random_state=42), {"max_depth": [2, 3, 4, 5]}),
        "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [5, 9, 15]}),
        "Rừng ngẫu nhiên": (RandomForestClassifier(200, random_state=0), {"max_depth": [3, 5, None]}),
        "SVM RBF": (make_pipeline(StandardScaler(), SVC()), {"svc__C": [1, 10], "svc__gamma": [0.01, 0.1]}),
        "Hồi quy logistic": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                             {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
    }
    dong = []
    for g in GIAI_DOAN_THAT:
        d = {"giai_doan": g["ten"]}
        for ten, (mh, luoi) in mo_hinh.items():
            gs = GridSearchCV(mh, luoi, cv=CV).fit(tr[g["cot"]], y_tr)
            d[ten] = round(accuracy_score(y_te, gs.predict(te[g["cot"]])), 4)
        dong.append(d)
    return pd.DataFrame(dong)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df = doc_uci()
    print("Du lieu that:", df.shape, "ty le dau:", round(df["dau"].mean(), 4))
    mh = huan_luyen_that(df)
    kq = danh_gia_that(mh, df)
    pd.set_option("display.width", 200)
    print(kq.round(4).to_string(index=False))
    ss = so_sanh_that(df)
    print(ss.to_string(index=False))
    kq.to_csv(FILE_DU_LIEU_THAT, index=False, encoding="utf-8-sig")
    ss.to_csv(FILE_DU_LIEU_THAT.replace(".csv", "_so_sanh.csv"), index=False, encoding="utf-8-sig")
    joblib.dump(mh, FILE_MO_HINH_THAT)
    print("Da luu ket qua")
