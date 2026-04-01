import json
import pandas as pd

def safe_json_load(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except:
        return None

def extract_categories(x):
    data = safe_json_load(x)
    if isinstance(data, dict):
        primary = data.get('primary')
        alternate_list = data.get('alternate', [])
        alternate_str = ", ".join(alternate_list) if alternate_list else None
        return primary, alternate_str
    return None, None

def extract_name(x):
    data = safe_json_load(x)
    if isinstance(data, dict):
        return data.get('primary')
    return None

def extract_address(x):
    data = safe_json_load(x)
    if isinstance(data, list) and len(data) > 0:
        return data[0].get('freeform')
    return None

def extract_sources(x):
    data = safe_json_load(x)
    if isinstance(data, list) and len(data) > 0:
        first = data[0] 
        return first.get('dataset'), first.get('record_id'), first.get('update_time')
    return None, None, None


ny_ove = pd.read_csv('/Users/houpuli/Downloads/ove_raw.csv')

ny_ove[['cat_main', 'cat_alt']] = ny_ove['categories'].apply(
    lambda x: pd.Series(extract_categories(x))
)

ny_ove['name'] = ny_ove['names'].apply(extract_name)
ny_ove['address'] = ny_ove['addresses'].apply(extract_address)
ny_ove[['src_dataset', 'src_record_id', 'src_update_time']] = ny_ove['sources'].apply(lambda x: pd.Series(extract_sources(x)))


ny_ove['WKB'] = ny_ove['geometry'].apply(lambda x: x[2:])
geometry = gpd.GeoSeries.from_wkb(ny_ove['WKB'], crs=4326)

ny_ove = gpd.GeoDataFrame(ny_ove, geometry=geometry)
ny_ove = ny_ove[['id','name','address','cat_main','cat_alt','confidence','operating_status','version','src_dataset','src_update_time','geometry']]
ny_ove