import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
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

    # GMM
    if ds == "ds00":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds01":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds03":
        model = GaussianMixture(n_components=2, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds04":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds05":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds07":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds09":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds10":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds15":
        model = GaussianMixture(n_components=4, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds21":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)
    elif ds == "ds22":
        model = GaussianMixture(n_components=2, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42)

    # KMeans
    elif ds == "ds02":
        model = KMeans(n_clusters=4, n_init=20, random_state=42)
    elif ds == "ds06":
        model = KMeans(n_clusters=3, n_init=20, random_state=42)
    elif ds == "ds11":
        model = KMeans(n_clusters=4, n_init=20, random_state=42)
    elif ds == "ds12":
        model = KMeans(n_clusters=2, n_init=20, random_state=42)
    elif ds == "ds16":
        X_scaled = np.linalg.norm(X_scaled, axis=1).reshape(-1, 1)
        model = KMeans(n_clusters=2, n_init=20, random_state=42)
    elif ds == "ds17":
        model = KMeans(n_clusters=2, n_init=20, random_state=42)
    elif ds == "ds19":
        model = KMeans(n_clusters=3, n_init=20, random_state=42)
    elif ds == "ds20":
        model = KMeans(n_clusters=4, n_init=20, random_state=42)
    elif ds == "ds23":
        X_scaled = np.linalg.norm(X_scaled, axis=1).reshape(-1, 1)
        model = KMeans(n_clusters=2, n_init=20, random_state=42)

    # 3. Иерархической кластеризация
    elif ds == "ds08":
        model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    elif ds == "ds13":
        model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    elif ds == "ds14":
        model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    elif ds == "ds18":
        model = AgglomerativeClustering(n_clusters=2, linkage="ward")

    else:
        model = KMeans(n_clusters=3, n_init=20, random_state=42)

    return model.fit_predict(X_scaled)
