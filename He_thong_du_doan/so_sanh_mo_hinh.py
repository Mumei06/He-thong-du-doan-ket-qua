"""
So sánh các mô hình trong môn Học máy và ứng dụng ở từng giai đoạn của học kỳ.
Cùng dữ liệu, cùng cách chia 80/20. Tham số dò bằng GridSearchCV trên tập học, chấm một lần trên tập kiểm tra.
Chạy: python so_sanh_mo_hinh.py   (lưu ket_qua_so_sanh.csv)
"""
import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cau_hinh import FILE_SO_SANH, GIAI_DOAN
from mo_hinh import chia_du_lieu, doc_du_lieu

warnings.filterwarnings("ignore")
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def chuan(mh):
    return make_pipeline(StandardScaler(), mh)


CAC_MO_HINH = [
    ("Đoán lớp đông nhất (mức sàn)", lambda: (DummyClassifier(strategy="most_frequent"), {})),
    ("Cây quyết định", lambda: (DecisionTreeClassifier(random_state=42), {"max_depth": [2, 3, 4, 5, 6, 8]})),
    ("Naive Bayes", lambda: (GaussianNB(), {})),
    ("KNN", lambda: (chuan(KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [5, 9, 15, 25]})),
    ("Rừng ngẫu nhiên", lambda: (RandomForestClassifier(200, random_state=0), {"max_depth": [3, 5, None]})),
    ("SVM RBF", lambda: (chuan(SVC()), {"svc__C": [1, 10, 100], "svc__gamma": [0.01, 0.1, 1]})),
    ("MLP", lambda: (chuan(MLPClassifier(solver="lbfgs", max_iter=3000, random_state=0)),
                     {"mlpclassifier__hidden_layer_sizes": [(10,), (50,)], "mlpclassifier__alpha": [1e-2, 1]})),
    ("Hồi quy logistic (mô hình chính)", lambda: (chuan(LogisticRegression(max_iter=5000)),
                                                   {"logisticregression__C": [0.1, 1, 10, 100, 1000]})),
]

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _, _ = chia_du_lieu(doc_du_lieu())
    dong = []
    for g in GIAI_DOAN:
        for ten, tao in CAC_MO_HINH:
            mh, luoi = tao()
            if luoi:
                gs = GridSearchCV(mh, luoi, cv=cv).fit(X_tr[g["cot"]], y_tr)
                tot, cv_hoc, ts = gs.best_estimator_, gs.best_score_, str(gs.best_params_)
            else:
                tot, cv_hoc, ts = mh.fit(X_tr[g["cot"]], y_tr), float("nan"), ""
            du = tot.predict(X_te[g["cot"]])
            dong.append({"giai_doan": g["ten"], "mo_hinh": ten,
                         "accuracy_kiem_tra": round(accuracy_score(y_te, du), 4),
                         "macro_f1": round(f1_score(y_te, du, average="macro"), 4),
                         "cv_tap_hoc": round(cv_hoc, 4) if cv_hoc == cv_hoc else None,
                         "tham_so_tot": ts})
        print("xong", g["ten"])
    kq = pd.DataFrame(dong)
    kq.to_csv(FILE_SO_SANH, index=False, encoding="utf-8-sig")
    print(kq.pivot(index="mo_hinh", columns="giai_doan", values="accuracy_kiem_tra")
          [[g["ten"] for g in GIAI_DOAN]].to_string())
