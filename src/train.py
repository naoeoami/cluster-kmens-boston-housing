"""Módulo de treinamento.

1. Carrega a base Boston Housing
2. Normaliza (MinMax nos quantitativos, One-Hot nos categóricos)
3. Determina o número ótimo de grupos pelo método do cotovelo (distâncias)
4. Treina o K-Means com o k ótimo e salva modelo + pré-processador

Uso:
    python src/train.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import (DATA_PATH, K_MAX, K_MIN, MODEL_PATH, MODELS_DIR,
                    PREPROCESSOR_PATH, RANDOM_STATE, REPORTS_DIR)
from preprocessing import Preprocessor


def elbow_inertias(X, k_range):
    """Soma das distâncias quadráticas de cada ponto ao seu centroide (inércia)."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inércia={km.inertia_:10.4f}")
    return np.array(inertias)


def optimal_k(k_range, inertias):
    """Cotovelo pelo método das distâncias.

    Traça-se a reta entre o primeiro ponto (k_min, inércia_max) e o último
    (k_max, inércia_min) da curva. O k ótimo é aquele cujo ponto da curva
    está à MAIOR distância perpendicular dessa reta.
    """
    x1, y1 = k_range[0], inertias[0]
    x2, y2 = k_range[-1], inertias[-1]
    distances = []
    for x0, y0 in zip(k_range, inertias):
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        distances.append(num / den)
    distances = np.array(distances)
    return int(k_range[int(np.argmax(distances))]), distances


def plot_elbow(k_range, inertias, distances, k_opt):
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "o-", label="Inércia (soma das distâncias²)")
    ax.plot([k_range[0], k_range[-1]], [inertias[0], inertias[-1]], "--",
            color="gray", label="Reta entre extremos")
    ax.axvline(k_opt, color="red", linestyle=":", label=f"k ótimo = {k_opt}")
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Inércia")
    ax.set_title("Método do cotovelo – Boston Housing")
    ax.set_xticks(list(k_range))
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "cotovelo.png", dpi=120)
    plt.close(fig)

    pd.DataFrame({"k": k_range, "inercia": inertias,
                  "distancia_a_reta": distances}).to_csv(
        REPORTS_DIR / "cotovelo.csv", index=False)


def main():
    print(f"Carregando base: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]} registros, {df.shape[1]} colunas")

    print("Normalizando dados...")
    prep = Preprocessor()
    X = prep.fit_transform(df)
    print(f"  matriz normalizada: {X.shape}  ({', '.join(prep.feature_names())})")

    print(f"Método do cotovelo (k = {K_MIN}..{K_MAX}):")
    k_range = np.arange(K_MIN, K_MAX + 1)
    inertias = elbow_inertias(X, k_range)
    k_opt, distances = optimal_k(k_range, inertias)
    plot_elbow(k_range, inertias, distances, k_opt)
    print(f"  -> número ótimo de grupos: k = {k_opt}  (gráfico em reports/cotovelo.png)")

    print(f"Treinando K-Means final com k = {k_opt}...")
    model = KMeans(n_clusters=k_opt, n_init=10, random_state=RANDOM_STATE).fit(X)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(prep, PREPROCESSOR_PATH)
    print(f"  modelo salvo em {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")
    print(f"  pré-processador salvo em {PREPROCESSOR_PATH.relative_to(PREPROCESSOR_PATH.parents[1])}")

    sizes = pd.Series(model.labels_).value_counts().sort_index()
    centroids = prep.inverse_transform(model.cluster_centers_)
    centroids.insert(0, "n_residencias", sizes.values)
    centroids.index.name = "grupo"
    centroids.to_csv(REPORTS_DIR / "centroides.csv")
    pd.set_option("display.width", 200)
    print("\nCentroides desnormalizados:")
    print(centroids.round(3).to_string())


if __name__ == "__main__":
    main()
