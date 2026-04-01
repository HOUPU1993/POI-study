import pandas as pd
from glob import glob

folder = "/Users/houpuli/Redlining Lab Dropbox/HOUPU LI/POI research/ny msa/safegraph"
files = glob(folder + "/*.csv.gz")

# print(len(files))  

dfs = []

for f in files:
    print("reading:", f)
    dfs.append(pd.read_csv(f))

ny_pa_nj_pt_sf = pd.concat(dfs, ignore_index=True)
ny_pa_nj_pt_sf = ny_pa_nj_pt_sf[['PLACEKEY','PARENT_PLACEKEY','LOCATION_NAME', 'TRACKING_CLOSED_SINCE', 'LATITUDE','LONGITUDE','TOP_CATEGORY','SUB_CATEGORY','CATEGORY_TAGS','NAICS_CODE','STREET_ADDRESS']]
ny_pa_nj_pt_sf = gpd.GeoDataFrame(ny_pa_nj_pt_sf, geometry=gpd.points_from_xy(ny_pa_nj_pt_sf["LONGITUDE"], ny_pa_nj_pt_sf["LATITUDE"]), crs="EPSG:4326")
ny_pa_nj_pt_sf = ny_pa_nj_pt_sf.drop(columns=['LATITUDE','LONGITUDE'])

msa_ny = msa_ny.to_crs(ny_pa_nj_pt_sf.crs)
ny_sf = gpd.sjoin(ny_pa_nj_pt_sf, msa_ny, how='inner',predicate="within").reset_index(drop=True)
ny_sf = ny_sf.drop(columns=['index_right','OBJECTID','CBSACODE','CBSANAME','CBSATYPE','ALAND','AWATER'])