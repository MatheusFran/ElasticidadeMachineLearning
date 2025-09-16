import pandas as pd
import numpy as np
import statsmodels.api as sm

"""
Elasticidade-preço (E): mede o quanto a demanda muda em resposta a variação de preço.

|E| < 1 → inelástico (demanda pouco sensível)

|E| ≈ 1 → unitário (variação proporcional)

|E| > 1 → elástico (demanda muito sensível)

p-value: indica se a elasticidade estimada é estatisticamente confiável.

p < 0.05 → significativa

p ≥ 0.05 → não significativa

R²: mostra quanto da variação da demanda é explicada pelo preço.

R² alto → preço explica boa parte da variação

R² baixo → outros fatores influenciam mais a demanda
"""


def calculate_elasticity(
        df: pd.DataFrame,
        group_by: list = None,
        price_col: str = None,
        quantity_col: str = None
):
    results = []
    if group_by is None:
        group_by = []
    if not group_by:
        group_by = [None]


    grouped = df.groupby(group_by) if group_by[0] is not None else [(None, df)]

    for group_name, group_df in grouped:
        group_df = group_df.copy()
        group_df["ln_q"] = np.log(group_df[quantity_col])
        group_df["ln_p"] = np.log(group_df[price_col])

        X = sm.add_constant(group_df["ln_p"])
        y = group_df["ln_q"]

        model = sm.OLS(y, X).fit()
        elasticity = model.params["ln_p"]
        p_value = model.pvalues["ln_p"]
        r2 = model.rsquared

        row = {
            "group": group_name,
            "elasticity": elasticity,
            "p_value": p_value,
            "r2": r2,
            "significant": p_value < 0.05
        }
        results.append(row)

    return pd.DataFrame(results)

"""
Essa função calculate_elasticity calcula a elasticidade-preço da demanda a partir de dados históricos de preço e quantidade vendida, usando uma regressão linear em escala logarítmica (modelo log-log).

👉 Em resumo, ela:

Permite rodar a análise por grupos (ex.: categoria, marca, produto) ou no dataset inteiro.

Para cada grupo:

Converte price e sales_volume em logaritmo.

Ajusta uma regressão OLS:


Extrai:

Elasticidade (β₁) → variação percentual na quantidade para 1% de variação no preço.

p-value do coeficiente → significância estatística.(h0=elasticidade = 0)

R² → quanto o preço explica da variação nas vendas.

Flag significant → se o efeito é estatisticamente relevante (p < 0.05).

Retorna um DataFrame resumindo esses indicadores para cada grupo.
"""