from rapidfuzz import process, fuzz
import pandas as pd
import re
import unicodedata


def clean_name(s):
    if not isinstance(s, str):
        return ""

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)) # 1. unicode normalize (remove accents)
    s = s.upper() # 2. uppercase
    s = re.sub(r"\([^)]*\)", "", s) 
    s = re.sub(r"\b'S\b", "", s) # new change
    s = re.sub(r"\bS\b", "", s) # new change
    s = s.encode("ascii", errors="ignore").decode() # 4. remove emoji / non ascii
    s = re.sub(r"[^\w\s]", " ", s) # 5. replace special chars with space
    s = re.sub(r"\s+", " ", s) # 6. collapse spaces

    return s.strip()

def extract_prinmary_str(name):

    tokens = name.split()
    core = [t for t in tokens if t not in NON_PRIMARY_TOKENS]
    if len(core) == 1 and len(core[0]) < 3:
        return name
    if core:
        return " ".join(core)
    return name

def match_by_name(
    reference_gdf: gpd.GeoDataFrame,
    compared_gdf: gpd.GeoDataFrame,
    re_name_col: str = "name",
    comp_name_col: str = "name",
    comp_id: str = "id",
    comp_id_col: str="cat_main",
    threshold: int = 80,
):
    """
    Perform WRatio name matching within spatial candidates.

    Returns
    -------
    GeoDataFrame with:
    - matched_id_name
    - name_score
    """

    # clean names for matching
    id_to_name_clean = compared_gdf.set_index(comp_id)[comp_name_col].apply(clean_name).apply(extract_prinmary_str).to_dict()
    # raw names for storage
    id_to_name_raw = compared_gdf.set_index(comp_id)[comp_name_col].to_dict()
    # raw compared df category
    id_to_cat = compared_gdf.set_index(comp_id)[comp_id_col].to_dict()

    matched_ids = []
    scores = []
    loc_dists = []
    matched_names = []
    matched_cats = []

    for _, row in reference_gdf.iterrows():
        query = extract_prinmary_str(clean_name(row.get(re_name_col)))

        if not isinstance(query, str) or not row["cand_ids"]:
            matched_ids.append(pd.NA)
            scores.append(pd.NA)
            loc_dists.append(pd.NA)
            matched_names.append(pd.NA)
            matched_cats.append(pd.NA)
            continue

        cand_names = [id_to_name_clean.get(cid, "") for cid in row["cand_ids"]]

        top_matches = process.extract(
            query,
            cand_names,
            scorer=fuzz.WRatio,
            limit=5
        )

        best_score = -1
        best_pos = None

        for name, wr, pos in top_matches:

            pr = fuzz.partial_ratio(query, name)
            ts = fuzz.token_sort_ratio(query, name)
            tset = fuzz.token_set_ratio(query, name)

            combined = max(wr, pr, ts, tset)

            if combined > best_score:
                best_score = combined
                best_pos = pos

        score = best_score

        if score >= threshold:
            matched_ids.append(row["cand_ids"][best_pos])
            scores.append(score)
            loc_dists.append(row["cand_dist_m"][best_pos])
            matched_names.append(id_to_name_raw.get(row["cand_ids"][best_pos]))
            matched_cats.append(id_to_cat.get(row["cand_ids"][best_pos]))
        else:
            matched_ids.append(pd.NA)
            scores.append(score)
            loc_dists.append(pd.NA)
            matched_names.append(pd.NA)
            matched_cats.append(pd.NA)


    result = reference_gdf.copy()
    result["matched_id"] = matched_ids
    result["name_score"] = scores
    result["location_distance"] = loc_dists
    result["matched_name"] = matched_names
    result["matched_cat_main"] = matched_cats

    return result