"""Normalização e desnormalização dos dados.

- Quantitativos: MinMaxScaler (cada coluna levada para o intervalo [0, 1]).
- Categóricos:   OneHotEncoder (uma coluna 0/1 por categoria), pois códigos
  como `rad` = 24 não têm significado de magnitude e não devem ser escalados.

Os dois transformadores são ajustados SOMENTE no treino e salvos juntos,
para que a inferência aplique exatamente a mesma transformação.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from config import CATEGORICAL, QUANTITATIVE


class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.medians = None
        self.modes = None

    # ------------------------------------------------------------------ treino
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = self._clean(df, fitting=True)
        self.scaler.fit(df[QUANTITATIVE])
        self.encoder.fit(df[CATEGORICAL].astype(int))
        return self

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._clean(df, fitting=False)
        quant = self.scaler.transform(df[QUANTITATIVE])
        cat = self.encoder.transform(df[CATEGORICAL].astype(int))
        return np.hstack([quant, cat])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------- desnormalização
    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Volta para a escala original.

        Quantitativos: inverso do MinMax.
        Categóricos: a categoria com maior peso no centroide (argmax do bloco
        one-hot). O peso indica a proporção de residências do grupo naquela
        categoria.
        """
        X = np.atleast_2d(X)
        n_q = len(QUANTITATIVE)
        quant = self.scaler.inverse_transform(X[:, :n_q])
        out = pd.DataFrame(quant, columns=QUANTITATIVE)

        start = n_q
        for col, cats in zip(CATEGORICAL, self.encoder.categories_):
            block = X[:, start:start + len(cats)]
            out[col] = cats[block.argmax(axis=1)]
            out[f"{col}_proporcao"] = block.max(axis=1)
            start += len(cats)
        return out

    def feature_names(self):
        return QUANTITATIVE + list(self.encoder.get_feature_names_out(CATEGORICAL))

    # ------------------------------------------------------------ auxiliares
    def _clean(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in QUANTITATIVE + CATEGORICAL if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}")
        for c in QUANTITATIVE + CATEGORICAL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Valores ausentes: mediana (quantitativos) / moda (categóricos)
        if fitting:
            self.medians = df[QUANTITATIVE].median()
            self.modes = df[CATEGORICAL].mode().iloc[0]
        df[QUANTITATIVE] = df[QUANTITATIVE].fillna(self.medians)
        df[CATEGORICAL] = df[CATEGORICAL].fillna(self.modes)
        return df
