# ==========================================================
#  preprocess.py  —  변경 이력
# ----------------------------------------------------------
#  [v1]     최초 버전 — train.csv만 처리
#           결측 처리 → 서열형 정수 매핑 → 원-핫 인코딩
#           → train/val 분리 → 스케일링 → 텐서
#
#  [v2]     (a) test.csv 처리 추가 (제출용)
#               - train+test 합쳐 동일 전처리 후 재분리
#               - 남은 결측 일괄 처리(test 전용 결측 대비)
#               - X_test, test_ids 반환
#           (b) corr_threshold — 상관 기반 변수 선택
#
#  [v6]     encoding 파라미터 ("onehot" / "target")
#           - 타깃 인코딩 분기, 누수 방지 위해 분할 먼저 수행
#
#  [v6-b]   타깃 인코딩 스무딩 — 드문 범주의 정답 베끼기 억제
#
#  [v11]    add_features — 파생변수 10개 생성 (단순 합산)
#           ※ MLP가 선형 결합을 스스로 학습 가능해 효과 없었음
#
#  [v13]    remove_outliers — 이상치 제거 (최초 구현: 분할 전 제거)
#  [v13-b]  ★수정★ 분할 전 제거 시 val 셋이 바뀌어 비교 무효화됨
#           → 이상치 인덱스만 기록해두고, 분할 후 train에서만 제거
#
#  [v15]    prepare_full() 메서드 추가 — K-fold 교차검증용
#           - 분할·스케일링 없이 인코딩까지만 수행해 원본 반환
#
#  [v17]    데이터 표현 개선 2종 (※ prepare_full()에만 적용)
#           ① fix_skew     — 왜도 큰 수치 변수를 log1p로 보정
#           ② full_ordinal — 순서 있는 범주형을 서열형 정수로 매핑
#           → OOF 0.1338 → 0.1299 (단, 학습률 재탐색 필요했음)
#
#  [v21]    add_interactions — 상호작용 특성 7개 (곱·비율)
#           - v11의 단순 합산은 MLP가 스스로 학습 가능해 무효였음
#           - 곱/비율은 신경망이 근사하기 어려워 명시적 제공이 유효할 수 있음
# ==========================================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousePriceData:
    def __init__(self, train_path="train.csv",
                 test_path="test.csv",            # ★[v2]
                 corr_threshold=None,             # ★[v2]
                 encoding="onehot",               # ★[v6]
                 add_features=False,              # ★[v11]
                 remove_outliers=False,           # ★[v13]
                 fix_skew=False,                  # ★[v17-①]
                 full_ordinal=False,              # ★[v17-②]
                 add_interactions=False):         # ★[v21]
        self.train_path = train_path
        self.test_path = test_path
        self.corr_threshold = corr_threshold
        self.encoding = encoding
        self.add_features = add_features
        self.remove_outliers = remove_outliers
        self.fix_skew = fix_skew                  # ★[v17-①]
        self.full_ordinal = full_ordinal          # ★[v17-②]
        self.add_interactions = add_interactions  # ★[v21]

    # ======================================================
    #  prepare() — 단일 분할 경로 (v1~v14 재현용)
    #  ※ v17·v21 옵션은 여기 적용되지 않음 (K-fold 경로 전용)
    # ======================================================
    def prepare(self):
        # ===== 1. 불러오기 =====
        train = pd.read_csv(self.train_path)                    # [v1]
        test  = pd.read_csv(self.test_path)                     # ★[v2]
        test_ids = test["Id"]                                   # ★[v2]

        # ===== ★[v13-b] 이상치 '인덱스만' 기록 (실제 제거는 분할 후) =====
        outlier_idx = pd.Index([])
        if self.remove_outliers:
            mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
            outlier_idx = train.index[mask]
            print(f"[이상치 탐지] {len(outlier_idx)}건 (분할 후 train에서만 제거)")

        y = np.log1p(train["SalePrice"])                        # [v1]
        train = train.drop(columns=["SalePrice"])               # ★[v2]

        # ===== 2. train+test 합쳐서 동일 전처리 =====
        n_train = len(train)                                    # ★[v2]
        full = pd.concat([train, test], axis=0)                 # ★[v2]

        none_cols = ["PoolQC","MiscFeature","Alley","Fence","MasVnrType","FireplaceQu",
                     "GarageType","GarageFinish","GarageQual","GarageCond",
                     "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
        full[none_cols] = full[none_cols].fillna("None")        # [v1]
        full[["MasVnrArea","GarageYrBlt"]] = full[["MasVnrArea","GarageYrBlt"]].fillna(0)
        full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))                     # [v1]

        for c in full.columns:                                  # ★[v2]
            if full[c].dtype == "object":
                full[c] = full[c].fillna(full[c].mode()[0])
            else:
                full[c] = full[c].fillna(0)

        # ===== ★[v11] 파생변수 =====
        if self.add_features:
            full["TotalSF"] = full["TotalBsmtSF"] + full["1stFlrSF"] + full["2ndFlrSF"]
            full["TotalBath"] = (full["FullBath"] + 0.5 * full["HalfBath"] +
                                 full["BsmtFullBath"] + 0.5 * full["BsmtHalfBath"])
            full["HouseAge"]    = full["YrSold"] - full["YearBuilt"]
            full["RemodAge"]    = full["YrSold"] - full["YearRemodAdd"]
            full["IsRemodeled"] = (full["YearBuilt"] != full["YearRemodAdd"]).astype(int)
            full["TotalPorchSF"] = (full["OpenPorchSF"] + full["EnclosedPorch"] +
                                    full["3SsnPorch"] + full["ScreenPorch"] + full["WoodDeckSF"])
            full["HasGarage"]    = (full["GarageArea"] > 0).astype(int)
            full["HasBsmt"]      = (full["TotalBsmtSF"] > 0).astype(int)
            full["Has2ndFloor"]  = (full["2ndFlrSF"] > 0).astype(int)
            full["HasFireplace"] = (full["Fireplaces"] > 0).astype(int)
            print("[파생변수] 10개 추가")

        # ===== 3. 서열형(품질) 정수 매핑 ===== [v1]
        qual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
        for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
                  "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
            full[c] = full[c].map(qual_map)

        # ===== 4. 인코딩 =====
        if self.encoding == "onehot":                           # [v1] 경로
            full = pd.get_dummies(full)
            X      = full.iloc[:n_train].drop(columns=["Id"])   # ★[v2]
            X_test = full.iloc[n_train:].drop(columns=["Id"])   # ★[v2]

            if self.corr_threshold is not None:                 # ★[v2]
                corr = X.astype(float).corrwith(y).abs()
                sel = corr[corr >= self.corr_threshold].index.tolist()
                X, X_test = X[sel], X_test[sel]
                print(f"[변수 선택] threshold={self.corr_threshold} → {len(sel)}개 선택")

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42)           # [v1]

            # ===== ★[v13-b] train에서만 이상치 제거 =====
            if self.remove_outliers:
                drop_idx = outlier_idx.intersection(X_train.index)
                X_train = X_train.drop(index=drop_idx)
                y_train = y_train.drop(index=drop_idx)
                print(f"[이상치 제거] train에서 {len(drop_idx)}건 제거 "
                      f"(train {len(X_train)}개 / val {len(X_val)}개 유지)")

        elif self.encoding == "target":                         # ★[v6]
            X      = full.iloc[:n_train].drop(columns=["Id"])
            X_test = full.iloc[n_train:].drop(columns=["Id"])
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42)

            if self.remove_outliers:                            # ★[v13-b]
                drop_idx = outlier_idx.intersection(X_train.index)
                X_train = X_train.drop(index=drop_idx)
                y_train = y_train.drop(index=drop_idx)
                print(f"[이상치 제거] train에서 {len(drop_idx)}건 제거")

            X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
            nominal = X_train.select_dtypes(include="object").columns
            global_mean = y_train.mean()
            m = 10                                              # ★[v6-b]
            for c in nominal:
                stats = y_train.groupby(X_train[c]).agg(["mean", "count"])
                smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
                X_train[c] = X_train[c].map(smooth).fillna(global_mean)
                X_val[c]   = X_val[c].map(smooth).fillna(global_mean)
                X_test[c]  = X_test[c].map(smooth).fillna(global_mean)
            print(f"[타깃 인코딩+스무딩 m={m}] 범주형 {len(nominal)}개 압축")

        print(f"[최종 변수 수] {X_train.shape[1]}개")            # ★[v11]

        # ===== 5. 스케일링 ===== [v1]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)                      # ★[v2]

        # ===== 6. 텐서 변환 ===== [v1]
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val   = torch.tensor(X_val,   dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)    # ★[v2]
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
        y_val   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1,1)

        return X_train, X_val, y_train, y_val, X_test, test_ids  # ★[v2]

    # ======================================================
    #  ★[v15] prepare_full() — K-fold 교차검증용
    #  분할·스케일링 없이 인코딩까지만 수행해 원본 반환
    #  ★[v17][v21] 데이터 표현·특성 개선 옵션은 이 경로에만 적용
    # ======================================================
    def prepare_full(self):
        train = pd.read_csv(self.train_path)
        test  = pd.read_csv(self.test_path)
        test_ids = test["Id"]

        outlier_idx = pd.Index([])                              # ★[v13-b]
        if self.remove_outliers:
            mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
            outlier_idx = train.index[mask]
            print(f"[이상치 탐지] {len(outlier_idx)}건 (fold-train에서만 제거 예정)")

        y = np.log1p(train["SalePrice"])
        train = train.drop(columns=["SalePrice"])

        n_train = len(train)
        full = pd.concat([train, test], axis=0)

        # --- 결측 처리 --- [v1] + ★[v2]
        none_cols = ["PoolQC","MiscFeature","Alley","Fence","MasVnrType","FireplaceQu",
                     "GarageType","GarageFinish","GarageQual","GarageCond",
                     "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
        full[none_cols] = full[none_cols].fillna("None")
        full[["MasVnrArea","GarageYrBlt"]] = full[["MasVnrArea","GarageYrBlt"]].fillna(0)
        full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        for c in full.columns:
            if full[c].dtype == "object":
                full[c] = full[c].fillna(full[c].mode()[0])
            else:
                full[c] = full[c].fillna(0)

        # ===== ★[v11] 파생변수 (단순 합산) =====
        if self.add_features:
            full["TotalSF"] = full["TotalBsmtSF"] + full["1stFlrSF"] + full["2ndFlrSF"]
            full["TotalBath"] = (full["FullBath"] + 0.5 * full["HalfBath"] +
                                 full["BsmtFullBath"] + 0.5 * full["BsmtHalfBath"])
            full["HouseAge"]    = full["YrSold"] - full["YearBuilt"]
            full["RemodAge"]    = full["YrSold"] - full["YearRemodAdd"]
            full["IsRemodeled"] = (full["YearBuilt"] != full["YearRemodAdd"]).astype(int)
            full["TotalPorchSF"] = (full["OpenPorchSF"] + full["EnclosedPorch"] +
                                    full["3SsnPorch"] + full["ScreenPorch"] + full["WoodDeckSF"])
            full["HasGarage"]    = (full["GarageArea"] > 0).astype(int)
            full["HasBsmt"]      = (full["TotalBsmtSF"] > 0).astype(int)
            full["Has2ndFloor"]  = (full["2ndFlrSF"] > 0).astype(int)
            full["HasFireplace"] = (full["Fireplaces"] > 0).astype(int)
            print("[파생변수] 10개 추가")

        # ===== ★[v21] 상호작용 특성 (곱·비율) =====
        # MLP는 선형 결합(덧셈)은 스스로 학습하나 곱셈 관계는 근사가 어려움
        # → 도메인상 의미 있는 곱·비율을 명시적으로 제공
        # ※ 왜도 보정보다 '앞'에 두어 새 특성도 보정 대상이 되게 함
        if self.add_interactions:
            full["Qual_x_Area"] = full["OverallQual"] * full["GrLivArea"]       # 품질×면적
            full["Qual_x_Cond"] = full["OverallQual"] * full["OverallCond"]     # 품질×상태
            full["Age_x_Qual"]  = (full["YrSold"] - full["YearBuilt"]) * full["OverallQual"]

            full["Area_per_Lot"] = full["GrLivArea"] / full["LotArea"]          # 건폐율
            full["Bsmt_ratio"]   = full["TotalBsmtSF"] / full["GrLivArea"]      # 지하 비중

            # 분모가 0이 될 수 있는 비율은 0으로 치환 (해당 시설이 없는 경우)
            full["Garage_per_car"] = (full["GarageArea"] /
                                      full["GarageCars"].replace(0, np.nan)).fillna(0)
            full["BsmtFin_ratio"]  = (full["BsmtFinSF1"] /
                                      full["TotalBsmtSF"].replace(0, np.nan)).fillna(0)

            print("[상호작용 특성] 7개 추가")

        # ===== ★[v17-①] 왜도 보정 =====
        # 치우친 수치 변수를 log1p로 펴줌
        # (StandardScaler는 평균·표준편차만 맞출 뿐 왜도는 고치지 못함)
        # ※ 서열형 매핑보다 '먼저' 수행해야 정수 서열값에 로그가 씌워지지 않음
        if self.fix_skew:
            num_cols = [c for c in full.select_dtypes(include=[np.number]).columns
                        if c != "Id"]
            skews = full[num_cols].skew()
            skewed = [c for c in num_cols
                      if abs(skews[c]) > 0.75 and full[c].min() >= 0]
            full[skewed] = np.log1p(full[skewed])
            print(f"[왜도 보정] {len(skewed)}개 변수 log1p 변환")

        # ===== 서열형(품질) 정수 매핑 ===== [v1]
        qual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
        for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
                  "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
            full[c] = full[c].map(qual_map)

        # ===== ★[v17-②] 서열형 확장 매핑 =====
        # 순서가 있으나 원-핫으로 처리되어 순서 정보가 소실되던 범주형들
        if self.full_ordinal:
            ord_maps = {
                "BsmtExposure": {"Gd":4,"Av":3,"Mn":2,"No":1,"None":0},
                "BsmtFinType1": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},
                "BsmtFinType2": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},
                "GarageFinish": {"Fin":3,"RFn":2,"Unf":1,"None":0},
                "Functional":   {"Typ":7,"Min1":6,"Min2":5,"Mod":4,
                                 "Maj1":3,"Maj2":2,"Sev":1,"Sal":0},
                "LotShape":     {"Reg":3,"IR1":2,"IR2":1,"IR3":0},
                "LandSlope":    {"Gtl":2,"Mod":1,"Sev":0},
                "PavedDrive":   {"Y":2,"P":1,"N":0},
                "CentralAir":   {"Y":1,"N":0},
                "Fence":        {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"None":0},
                "Utilities":    {"AllPub":3,"NoSewr":2,"NoSeWa":1,"ELO":0},
            }
            for c, mp in ord_maps.items():
                full[c] = full[c].map(mp).fillna(0)
            print(f"[서열형 확장] {len(ord_maps)}개 변수 정수 매핑")

        # ===== 원-핫 인코딩 & train/test 재분리 =====
        full = pd.get_dummies(full)
        X      = full.iloc[:n_train].drop(columns=["Id"])
        X_test = full.iloc[n_train:].drop(columns=["Id"])

        print(f"[K-fold 준비] 학습 {len(X)}개 / 변수 {X.shape[1]}개")
        return X, y, X_test, test_ids, outlier_idx             # 스케일링·분할 전 상태
    
    # ======================================================
    #  ★[v22] prepare_embed() — 범주형 임베딩용
    #  원-핫 대신 범주형을 '정수 인덱스'로 반환하고 카디널리티를 함께 제공
    #  → 모델이 각 범주의 밀집 벡터를 학습 (입력 차원 대폭 축소)
    # ======================================================
    def prepare_embed(self):
        train = pd.read_csv(self.train_path)
        test  = pd.read_csv(self.test_path)
        test_ids = test["Id"]

        outlier_idx = pd.Index([])                              # ★[v13-b]
        if self.remove_outliers:
            mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
            outlier_idx = train.index[mask]
            print(f"[이상치 탐지] {len(outlier_idx)}건 (fold-train에서만 제거 예정)")

        y = np.log1p(train["SalePrice"])
        train = train.drop(columns=["SalePrice"])
        n_train = len(train)
        full = pd.concat([train, test], axis=0)

        # --- 결측 처리 --- [v1] + ★[v2]
        none_cols = ["PoolQC","MiscFeature","Alley","Fence","MasVnrType","FireplaceQu",
                     "GarageType","GarageFinish","GarageQual","GarageCond",
                     "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
        full[none_cols] = full[none_cols].fillna("None")
        full[["MasVnrArea","GarageYrBlt"]] = full[["MasVnrArea","GarageYrBlt"]].fillna(0)
        full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        for c in full.columns:
            if full[c].dtype == "object":
                full[c] = full[c].fillna(full[c].mode()[0])
            else:
                full[c] = full[c].fillna(0)

        # ===== ★[v21] 상호작용 (옵션) =====
        if self.add_interactions:
            full["Qual_x_Area"] = full["OverallQual"] * full["GrLivArea"]
            full["Qual_x_Cond"] = full["OverallQual"] * full["OverallCond"]
            full["Age_x_Qual"]  = (full["YrSold"] - full["YearBuilt"]) * full["OverallQual"]
            full["Area_per_Lot"] = full["GrLivArea"] / full["LotArea"]
            full["Bsmt_ratio"]   = full["TotalBsmtSF"] / full["GrLivArea"]
            full["Garage_per_car"] = (full["GarageArea"] /
                                      full["GarageCars"].replace(0, np.nan)).fillna(0)
            full["BsmtFin_ratio"]  = (full["BsmtFinSF1"] /
                                      full["TotalBsmtSF"].replace(0, np.nan)).fillna(0)
            print("[상호작용 특성] 7개 추가")

        # ===== ★[v17-①] 왜도 보정 =====
        if self.fix_skew:
            num_cols_tmp = [c for c in full.select_dtypes(include=[np.number]).columns
                            if c != "Id"]
            skews = full[num_cols_tmp].skew()
            skewed = [c for c in num_cols_tmp
                      if abs(skews[c]) > 0.75 and full[c].min() >= 0]
            full[skewed] = np.log1p(full[skewed])
            print(f"[왜도 보정] {len(skewed)}개 변수 log1p 변환")

        # ===== 서열형(품질) 정수 매핑 ===== [v1]
        qual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
        for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
                  "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
            full[c] = full[c].map(qual_map)

        # ===== ★[v17-②] 서열형 확장 =====
        if self.full_ordinal:
            ord_maps = {
                "BsmtExposure": {"Gd":4,"Av":3,"Mn":2,"No":1,"None":0},
                "BsmtFinType1": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},
                "BsmtFinType2": {"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},
                "GarageFinish": {"Fin":3,"RFn":2,"Unf":1,"None":0},
                "Functional":   {"Typ":7,"Min1":6,"Min2":5,"Mod":4,
                                 "Maj1":3,"Maj2":2,"Sev":1,"Sal":0},
                "LotShape":     {"Reg":3,"IR1":2,"IR2":1,"IR3":0},
                "LandSlope":    {"Gtl":2,"Mod":1,"Sev":0},
                "PavedDrive":   {"Y":2,"P":1,"N":0},
                "CentralAir":   {"Y":1,"N":0},
                "Fence":        {"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"None":0},
                "Utilities":    {"AllPub":3,"NoSewr":2,"NoSeWa":1,"ELO":0},
            }
            for c, mp in ord_maps.items():
                full[c] = full[c].map(mp).fillna(0)
            print(f"[서열형 확장] {len(ord_maps)}개 변수 정수 매핑")

        full = full.drop(columns=["Id"])

        # ===== ★[v22] 남은 범주형 → 정수 인덱스 (원-핫 아님) =====
        cat_cols = full.select_dtypes(include="object").columns.tolist()
        cardinalities = []
        for c in cat_cols:
            codes, uniques = pd.factorize(full[c])   # train+test 함께 인코딩 → 일관성 보장
            full[c] = codes
            cardinalities.append(len(uniques))
        num_cols = [c for c in full.columns if c not in cat_cols]

        X_num,      X_cat      = full[num_cols].iloc[:n_train], full[cat_cols].iloc[:n_train]
        X_num_test, X_cat_test = full[num_cols].iloc[n_train:], full[cat_cols].iloc[n_train:]

        print(f"[임베딩 준비] 수치 {len(num_cols)}개 / 범주 {len(cat_cols)}개 "
              f"(카디널리티 합 {sum(cardinalities)})")
        return (X_num, X_cat, cardinalities, y,
                X_num_test, X_cat_test, test_ids, outlier_idx)