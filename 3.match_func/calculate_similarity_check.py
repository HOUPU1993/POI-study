import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def calculate_similarity_check(
    df, 
    cat_col_ref: str = "primary_type", 
    cat_col_cmp: str = "matched_cat_main", 
    id_col: str = "matched_id", 
    result_col: str =  "category_sim",
):

    # 1. Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # 2. Primary Gatekeeper: matched_id must be present
    mask = df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")
    
    # 3. Create a helper to handle potential Nulls in text columns
    temp_df = df[mask].copy()
    
    # Identify where text is actually missing within the masked rows
    text_missing_mask = temp_df[cat_col_ref].isna() | temp_df[cat_col_cmp].isna()
    
    # Fill NaNs with empty strings just for the encoding step
    texts_1 = temp_df[cat_col_ref].fillna("").astype(str).tolist()
    texts_2 = temp_df[cat_col_cmp].fillna("").astype(str).tolist()

    print(f"Encoding {len(temp_df)} rows...")

    # 4. Batch Encoding
    emb1 = model.encode(texts_1, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    emb2 = model.encode(texts_2, batch_size=256, show_progress_bar=True, convert_to_tensor=True)

    # 5. Calculate Similarity
    scores = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).cpu().numpy()
    
    # 6. Apply NaN to rows where text was missing
    # Even if we encoded an empty string, the result isn't "real" if data was missing
    scores[text_missing_mask.values] = np.nan

    # 7. Map back to original dataframe
    df[result_col] = np.nan
    df.loc[mask, result_col] = scores
    
    return df