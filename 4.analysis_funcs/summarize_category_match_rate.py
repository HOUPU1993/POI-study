def summarize_category_match_rate(
    df: pd.DataFrame,
    cat_col: str = "primary_cat",
    true_col: str = "is_true_match",
    true_values=(True, "1"),
    compare_col="ove_count",
    ref_col="google_count",
    cg_m_col: str = "ove_google_m",
    cat_per: str = "cat_per",
) -> pd.DataFrame:
    """
    Summarize Google POI counts, matched Overture counts,
    and match rate by category.
    """

    # ---- matched subset ----
    df_true = df[df[true_col].isin(true_values)]

    # ---- Google category counts (denominator) ----
    g_cat = (
        df[cat_col]
        .value_counts()
        .rename(ref_col)
        .to_frame()
    )
    g_cat[cat_per] = g_cat[ref_col] / len(df)

    # ---- Overture matched category counts (numerator) ----
    o_cat = (
        df_true[cat_col]
        .value_counts()
        .rename(compare_col)
        .to_frame()
    )

    # ---- merge & compute rate ----
    df_out = (
        o_cat
        .merge(g_cat, left_index=True, right_index=True, how="inner")
        .reset_index()
        .rename(columns={"index": cat_col})
        .sort_values(ref_col, ascending=False)
    )

    df_out[cg_m_col] = df_out[compare_col] / df_out[ref_col]

    return df_out