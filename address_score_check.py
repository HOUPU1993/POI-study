import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz
import re
import unicodedata


def clean_name(s):
    if not isinstance(s, str):
        return ""

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)) # 1. unicode normalize (remove accents)
    s = s.upper() # 2. uppercase
    s = re.sub(r"\([^)]*\)", "", s) 
    s = s.encode("ascii", errors="ignore").decode() # 4. remove emoji / non ascii
    s = re.sub(r"[^\w\s]", " ", s) # 5. replace special chars with space
    s = re.sub(r"\s+", " ", s) # 6. collapse spaces

    return s.strip()

def address_score_check(
    reference_gdf: gpd.GeoDataFrame,
    compared_gdf: gpd.GeoDataFrame,
    addr_col_ref: str = "addr_simple",
    addr_col_cmp: str = "address",
    matched_id_col: str = "matched_id",
    id_col: str = "id",
    out_col: str = "address_score",
    out_addr_col: str = "matched_address", 
):
    """
    Compute address similarity score (0-100) for already-matched rows.

    Only rows with non-null `matched_id_col` will be scored.
    Others will have NaN.

    Returns
    -------
    GeoDataFrame with a new column `out_col`.
    """

    # map: compared id -> compared address
    id_to_addr = compared_gdf.set_index(id_col)[addr_col_cmp].apply(clean_name).to_dict()

    scores = []
    matched_addrs = []

    for _, row in reference_gdf.iterrows():
        matched_id = row.get(matched_id_col)

        # no matched id -> no score
        if pd.isna(matched_id):
            scores.append(pd.NA)
            matched_addrs.append(pd.NA)
            continue

        addr_ref = clean_name(row.get(addr_col_ref))
        addr_cmp = id_to_addr.get(matched_id)

        if isinstance(addr_cmp, str) and addr_cmp.strip():
            matched_addrs.append(addr_cmp)
        else:
            matched_addrs.append(pd.NA)

        # missing address on either side -> no score
        if not isinstance(addr_ref, str) or not addr_ref.strip():
            scores.append(pd.NA)
            continue
        if not isinstance(addr_cmp, str) or not addr_cmp.strip():
            scores.append(pd.NA)
            continue

        wr = fuzz.WRatio(addr_ref, addr_cmp)
        pr = fuzz.partial_ratio(addr_ref, addr_cmp)
        ts = fuzz.token_sort_ratio(addr_ref, addr_cmp)
        tset = fuzz.token_set_ratio(addr_ref, addr_cmp)

        scores.append(int(max(wr, pr, ts, tset)))

    result = reference_gdf.copy()
    result[out_col] = scores
    result[out_addr_col] = matched_addrs
    return result