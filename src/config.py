"""Configurações compartilhadas entre o módulo de treinamento e o de inferência."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BostonHousing.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Artefatos salvos pelo treinamento
MODEL_PATH = MODELS_DIR / "kmeans.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Atributos categóricos -> One-Hot Encoding
#   chas: 1 se o imóvel faz divisa com o rio Charles, 0 caso contrário (binário)
#   rad : índice de acessibilidade a rodovias radiais (códigos discretos 1..8 e 24)
CATEGORICAL = ["chas", "rad"]

# Atributos quantitativos -> MinMaxScaler (0..1)
QUANTITATIVE = [
    "crim",     # taxa de criminalidade per capita
    "zn",       # % de terrenos residenciais com lotes > 25.000 pés²
    "indus",    # % de área comercial não-varejista
    "nox",      # concentração de óxidos nítricos
    "rm",       # nº médio de cômodos por residência
    "age",      # % de imóveis construídos antes de 1940
    "dis",      # distância ponderada a 5 centros de emprego
    "tax",      # imposto predial por US$ 10.000
    "ptratio",  # razão aluno/professor
    "b",        # 1000(Bk - 0.63)^2
    "lstat",    # % da população de baixa renda
    "medv",     # valor mediano das residências (US$ 1000)
]

FEATURES = QUANTITATIVE + CATEGORICAL

# Faixa de k avaliada no método do cotovelo
K_MIN, K_MAX = 1, 10
RANDOM_STATE = 42
