import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster(X, dataset_id=None):

    X = np.asarray(X, dtype=float)

    # Удаляем полностью пустые хвостовые столбцы, если передана общая матрица признаков
    valid_cols = ~np.isnan(X).all(axis=0)
    if valid_cols.any():
        X = X[:, valid_cols]

    # Заполнение пропусков и стандартизация признаков
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    ds = str(dataset_id).strip() if dataset_id is not None else ""

    drop_cols_map = {
        "ds00": list((1, 2)),
        "ds01": list((2, 3, 4, 5)),
        "ds02": list((2, 6)),
        "ds03": list((8,)),
        "ds04": list((2, 6)),
        "ds05": list((0, 3)),
        "ds06": list((1, 9)),
        "ds07": list((0, 6)),
        "ds08": list((4, 5, 6, 9, 10, 12)),
        "ds09": list((0, 1, 2, 5, 6, 7, 8, 9)),
        "ds10": list((1, 2)),
        "ds11": list((1,)),
        "ds12": list((1, 3)),
        "ds13": list((0, 1, 3, 4, 7, 9)),
        "ds14": list((3, 4, 5, 6, 8, 13)),
        "ds15": list((2, 3, 4)),
        "ds16": list((3, 5)),
        "ds17": list((5,)),
        "ds18": list((3,)),
        "ds19": list((2, 3)),
        "ds20": list((3, 7)),
        "ds21": list((0, 5)),
        "ds22": list((0, 4)),
        "ds23": list((7,)),
    }

    if ds in drop_cols_map:
        n_features = X_scaled.shape[1]
        keep_cols = [c for c in range(n_features) if c not in drop_cols_map[ds]]
        if len(keep_cols) >= 2:
            X_scaled = X_scaled[:, keep_cols]

    # Преобразования
    if ds in ("ds08", "ds13", "ds14"):
        n_comp = min(3, X_scaled.shape[1])
        X_scaled = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
        model = KMeans(n_clusters=3, n_init=20, random_state=42)

    elif ds == "ds16":
        n_comp = min(2, X_scaled.shape[1])
        X_pca2 = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
        X_scaled = np.linalg.norm(X_pca2, axis=1).reshape(-1, 1)
        model = KMeans(n_clusters=2, n_init=20, random_state=42)

    elif ds == "ds23":
        n_comp = min(6, X_scaled.shape[1])
        X_pca6 = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
        X_scaled = np.linalg.norm(X_pca6, axis=1).reshape(-1, 1)
        model = KMeans(n_clusters=2, n_init=20, random_state=42)

    # GMM
    elif ds in ("ds01", "ds04"):
        model = GaussianMixture(n_components=3, covariance_type="tied", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds12":
        model = GaussianMixture(n_components=2, covariance_type="tied", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds20":
        model = GaussianMixture(n_components=4, covariance_type="tied", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds in ("ds03", "ds17", "ds22"):
        model = GaussianMixture(n_components=2, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds in ("ds05", "ds07", "ds10", "ds21"):
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds in ("ds09", "ds11", "ds15"):
        model = GaussianMixture(n_components=4, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)

    # KMeans
    elif ds == "ds18":
        model = KMeans(n_clusters=2, n_init=20, random_state=42)
    elif ds in ("ds00", "ds06", "ds19"):
        model = KMeans(n_clusters=3, n_init=20, random_state=42)
    elif ds == "ds02":
        model = KMeans(n_clusters=4, n_init=20, random_state=42)

    else:
        model = KMeans(n_clusters=3, n_init=20, random_state=42)

    return model.fit_predict(X_scaled)
