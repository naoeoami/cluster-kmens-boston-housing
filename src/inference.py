"""Módulo de inferência.

1. Recebe os dados de uma nova residência
2. Normaliza com o MESMO pré-processador do treino
3. Infere o grupo com o modelo K-Means salvo
4. Exibe os dados desnormalizados do centroide inferido

Uso:
    python src/inference.py                      # modo interativo (pergunta cada atributo)
    python src/inference.py --exemplo            # usa uma residência de exemplo
    python src/inference.py --json casa.json     # lê os atributos de um arquivo JSON
    python src/inference.py --crim 0.03 --zn 0 ... --medv 22   # via argumentos
"""
import argparse
import json
import sys

import joblib
import pandas as pd

from config import CATEGORICAL, FEATURES, MODEL_PATH, PREPROCESSOR_PATH

DESCRICOES = {
    "crim": "Taxa de criminalidade per capita",
    "zn": "% terrenos residenciais com lotes > 25.000 pés²",
    "indus": "% área comercial não-varejista",
    "chas": "Faz divisa com o rio Charles? (1 = sim, 0 = não)",
    "nox": "Concentração de óxidos nítricos (ppm/10)",
    "rm": "Nº médio de cômodos",
    "age": "% imóveis construídos antes de 1940",
    "dis": "Distância ponderada a centros de emprego",
    "rad": "Índice de acessibilidade a rodovias (1-8 ou 24)",
    "tax": "Imposto predial por US$ 10.000",
    "ptratio": "Razão aluno/professor",
    "b": "1000(Bk - 0.63)²",
    "lstat": "% população de baixa renda",
    "medv": "Valor mediano da residência (US$ 1000)",
}

EXEMPLO = {
    "crim": 0.03, "zn": 0.0, "indus": 7.07, "chas": 0, "nox": 0.469,
    "rm": 6.8, "age": 60.0, "dis": 4.9, "rad": 2, "tax": 242.0,
    "ptratio": 17.8, "b": 392.8, "lstat": 6.5, "medv": 27.0,
}


def carregar_artefatos():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        sys.exit("Modelo não encontrado. Rode primeiro: python src/train.py")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def ler_interativo():
    print("Informe os dados da nova residência:")
    dados = {}
    for f in FEATURES:
        while True:
            txt = input(f"  {f:8s} - {DESCRICOES[f]}: ").strip().replace(",", ".")
            try:
                dados[f] = int(txt) if f in CATEGORICAL else float(txt)
                break
            except ValueError:
                print("    valor inválido, tente novamente.")
    return dados


def inferir(dados: dict):
    model, prep = carregar_artefatos()
    nova = pd.DataFrame([dados])

    X_novo = prep.transform(nova)
    grupo = int(model.predict(X_novo)[0])
    centroide = prep.inverse_transform(model.cluster_centers_[[grupo]]).iloc[0]
    return grupo, X_novo[0], centroide, prep


def exibir(dados, grupo, x_norm, centroide, prep):
    print("\n=== Nova residência ===")
    tabela = pd.DataFrame({
        "valor informado": pd.Series(dados)[FEATURES],
    })
    print(tabela.to_string())

    print("\n=== Dados normalizados ===")
    print(pd.Series(x_norm, index=prep.feature_names()).round(4).to_string())

    print(f"\n>>> A residência pertence ao GRUPO {grupo}")

    print(f"\n=== Centroide do grupo {grupo} (desnormalizado) ===")
    linhas = []
    for f in FEATURES:
        val = centroide[f]
        if f in CATEGORICAL:
            val_txt = f"{int(val)}  ({centroide[f + '_proporcao']:.0%} do grupo)"
        else:
            val_txt = f"{val:.4f}"
        linhas.append((f, DESCRICOES[f], val_txt))
    print(pd.DataFrame(linhas, columns=["atributo", "descrição", "centroide"])
          .to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Inferência de grupo (K-Means) para nova residência")
    p.add_argument("--exemplo", action="store_true", help="usa uma residência de exemplo")
    p.add_argument("--json", help="arquivo JSON com os atributos da residência")
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float, help=DESCRICOES[f])
    args = p.parse_args()

    if args.exemplo:
        dados = EXEMPLO
    elif args.json:
        with open(args.json, encoding="utf-8") as fh:
            dados = json.load(fh)
    elif any(getattr(args, f) is not None for f in FEATURES):
        faltando = [f for f in FEATURES if getattr(args, f) is None]
        if faltando:
            p.error(f"faltam atributos: {', '.join(faltando)}")
        dados = {f: getattr(args, f) for f in FEATURES}
    else:
        dados = ler_interativo()

    dados = {k.lower(): v for k, v in dados.items()}
    grupo, x_norm, centroide, prep = inferir(dados)
    exibir(dados, grupo, x_norm, centroide, prep)


if __name__ == "__main__":
    main()
