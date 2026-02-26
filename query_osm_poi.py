import pandas as pd
from pyrosm import OSM

def extract_comprehensive_pois(pbf):
    """
    Extracts POIs from an OSM PBF file and cleanses the data for spatial analysis.
    """
    # Initialize OSM parser
    osm = OSM(pbf)
    
    # Define broad filter to capture maximum POI density
    custom_filter = {
        'amenity': True, 'shop': True, 'tourism': True, 'leisure': True, 
        'office': True, 'healthcare': True, 'religion': True, 'emergency': True,
        'historic': True, 'government': True, 'craft': True, 'public_transport': True,
    }
    
    # Load data
    pois = osm.get_pois(custom_filter=custom_filter)
    if pois is None: return None

    # Convert Polygons to Points and defragment DataFrame to fix PerformanceWarnings
    pois = pois.to_crs(3857)
    pois['geometry'] = pois.geometry.centroid
    pois = pois.to_crs(4326)
    pois = pois.copy()

    # Unified Category Logic
    cat_columns = [
        'amenity', 'shop', 'tourism', 'leisure', 'office', 'healthcare', 
        'religion', 'emergency', 'historic', 'government', 'craft', 'public_transport'
    ]
    existing_cats = [c for c in cat_columns if c in pois.columns]
    pois['cat'] = pois[existing_cats].bfill(axis=1).iloc[:, 0]
    
    # Address Reconstruction Function
    def build_address(row):
        num = str(row.get('addr:housenumber', '')).strip()
        street = str(row.get('addr:street', '')).strip()
        num = '' if num.lower() in ['nan', 'none'] else num
        street = '' if street.lower() in ['nan', 'none'] else street
        
        if num and street:
            return f"{num} {street}"
        elif street: 
            return street
        
        # Fallback to building name
        housename = str(row.get('addr:housename', '')).strip()
        if housename and housename.lower() not in ['nan', 'none']:
            return housename
        return "N/A"

    pois['address'] = pois.apply(build_address, axis=1)
    
    # Filter final schema
    output_cols = ['id','timestamp','name', 'cat', 'address', 'tags', 'visible', 'version', 'geometry']
    return pois[output_cols]

# gdf_ny = extract_comprehensive_pois("/Users/houpuli/Downloads/new-york-260218.osm.pbf")
# gdf_pa = extract_comprehensive_pois("/Users/houpuli/Downloads/pennsylvania-260218.osm.pbf")
# gdf_ct = extract_comprehensive_pois("/Users/houpuli/Downloads/connecticut-260218.osm.pbf")
# gdf_nj = extract_comprehensive_pois("/Users/houpuli/Downloads/new-jersey-260218.osm.pbf")

gdf_posa = extract_comprehensive_pois("/Users/houpuli/Downloads/washington-260225.osm.pbf")

# ny_msa = us_msa[us_msa['CBSACODE'] == '35620']
# msa_ny_osm = pd.concat([gdf_ny, gdf_pa, gdf_ct, gdf_nj], ignore_index=True)
# msa_ny_osm = gpd.sjoin(msa_ny_osm, ny_msa, how='inner', predicate='within').reset_index(drop=True)
# msa_ny_osm = msa_ny_osm.drop(columns=['index_right','OBJECTID','CBSACODE','CBSANAME','CBSATYPE','ALAND','AWATER'])
# msa_ny_osm['id'] = msa_ny_osm['id'].astype(str)
# msa_ny_osm['timestamp'] = msa_ny_osm['timestamp'].astype(str)
# msa_ny_osm = msa_ny_osm.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

msa_posa_osm = gpd.sjoin(gdf_posa, msa_bspo, how='inner', predicate='within').reset_index(drop=True)
msa_posa_osm = msa_posa_osm.drop(columns=['index_right','OBJECTID','CBSACODE','CBSANAME','CBSATYPE','ALAND','AWATER'])
msa_posa_osm['id'] = msa_posa_osm['id'].astype(str)
msa_posa_osm['timestamp'] = msa_posa_osm['timestamp'].astype(str)
msa_posa_osm = msa_posa_osm.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

def normalize_nulls(df):
    NULL_LIKE = ["none", "null", "nan", "n/a", "na", "", " ", "   "]
    df = df.copy()

    geom = None
    if isinstance(df, gpd.GeoDataFrame):
        geom = df.geometry
        geom_name = df.geometry.name


    cols = df.columns if geom is None else df.columns.drop(geom_name)
    df[cols] = df[cols].replace(r"^\s*$", pd.NA, regex=True)
    df[cols] = df[cols].replace(
        [x.upper() for x in NULL_LIKE] +
        [x.lower() for x in NULL_LIKE] +
        [x.capitalize() for x in NULL_LIKE],
        pd.NA
    )

    df[cols] = df[cols].replace({None: pd.NA})
    df[cols] = df[cols].where(pd.notna(df[cols]), pd.NA)

    if geom is not None:
        df = gpd.GeoDataFrame(df, geometry=geom, crs=geom.crs)

    return df

msa_posa_osm = normalize_nulls(msa_posa_osm)

import json

def extract_from_tags(tag_str, priority_keys):
    # if not tag_str or tag_str == 'None':
    #     return None
    if pd.isna(tag_str):
        return None
    try:
        tags_dict = json.loads(tag_str)
        for key in priority_keys:
            if key in tags_dict:
                return tags_dict[key]
    except:
        return None
    return None

target_keys = [
    'amenity', 'shop', 'tourism', 'leisure', 'healthcare', 
    'office', 'religion', 'emergency', 'historic', 
    'government', 'craft', 'public_transport', 'building'
]

mask = msa_posa_osm['cat'].isna()
msa_posa_osm.loc[mask, 'cat'] = msa_posa_osm.loc[mask, 'tags'].apply(lambda x: extract_from_tags(x, target_keys))
# msa_posa_osm.to_file('/Users/houpuli/Redlining Lab Dropbox/HOUPU LI/POI research/posa_osm.geojson')