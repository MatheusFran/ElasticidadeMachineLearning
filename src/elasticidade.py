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
        price_col: str = "price",
        quantity_col: str = "sales_volume"
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
