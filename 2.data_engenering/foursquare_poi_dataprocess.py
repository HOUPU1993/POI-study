foursquare_places = query_foursquare(token_path="/Users/houpuli/Redlining Lab Dropbox/HOUPU LI/foursquare_token.json",
                       minLon=-118.951733, maxLon=-117.412999,
                       minLat=32.750045, maxLat=34.823307)

import ast

def safe_parse(x):
    if x is None or (isinstance(x, float)):
        return None
    try:
        result = ast.literal_eval(x)
        return result[0] if isinstance(result, list) and len(result) > 0 else None
    except:
        return x

foursquare_places["cat_str"] = foursquare_places["fsq_category_labels"].apply(safe_parse)
cats = foursquare_places["cat_str"].str.split(" > ", expand=True)
foursquare_places["cat_main"] = cats[0]
foursquare_places["cat_alt"] = cats[1]

foursquare_places["fsq_category_ids"] = foursquare_places["fsq_category_ids"].apply(lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) and x.startswith("[") else x)
foursquare_places["fsq_category_labels"] = foursquare_places["fsq_category_labels"].apply(lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) and x.startswith("[") else x)
foursquare_places