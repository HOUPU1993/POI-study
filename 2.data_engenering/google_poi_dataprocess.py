datasets = {
    "automotive": google_automotive_5000,
    "business": google_business_5000,
    "culture": google_culture_5000,
    "education": google_education_5000,
    "entertainment": google_entertainment_5000,
    "facilities": google_facilities_5000,
    "finance": google_finance_5000,
    "food": google_food_5000,
    "government": google_government_5000,
    "health": google_health_5000,
    "nature": google_nature_5000,
    "worship": google_places_worship_5000,
    "services": google_services_5000,
    "shop": google_shop_5000,
    "sport": google_sport_5000,
    "transportation": google_transportation_5000,
}

# Add primary_cat to each GeoDataFrame
for cat, gdf in datasets.items():
    gdf["primary_cat"] = cat

google_placescat_5000 = gpd.GeoDataFrame(
    pd.concat(list(datasets.values()), ignore_index=True),
    crs=google_automotive_5000.crs
)
google_placescat_5000


table_blist = ["administrative_area_level_3", "administrative_area_level_4", "administrative_area_level_5", "administrative_area_level_6", "administrative_area_level_7", "archipelago", "colloquial_area", "continent", "establishment", "finance", "food", "general_contractor",
"geocode", "health", "intersection", "landmark", "natural_feature", "neighborhood", "place_of_worship", "plus_code"	, "point_of_interest", "political", "postal_code_prefix", "postal_code_suffix", "postal_town", "premise", "route", "street_address", "sublocality", "sublocality_level_1",
"sublocality_level_2", "sublocality_level_3", "sublocality_level_4", "sublocality_level_5", "subpremise", "town_square"]


CAT_TO_TYPES = {
    "shop": [
        "asian_grocery_store", "auto_parts_store", "bicycle_store", "book_store", "building_materials_store",
        "butcher_shop", "cell_phone_store", "clothing_store", "convenience_store", "cosmetics_store",
        "department_store", "discount_store", "discount_supermarket", "electronics_store", "farmers_market",
        "flea_market", "food_store", "furniture_store", "garden_center", "general_store", "gift_shop",
        "grocery_store", "hardware_store", "health_food_store", "home_goods_store", "home_improvement_store",
        "hypermarket", "jewelry_store", "liquor_store", "market", "pet_store", "shoe_store", "shopping_mall",
        "sporting_goods_store", "sportswear_store", "store", "supermarket", "tea_store", "thrift_store",
        "toy_store", "warehouse_store", "wholesaler", "womens_clothing_store",
    ],
    "automotive": [
        "car_dealer", "car_rental", "car_repair", "car_wash", "ebike_charging_station",
        "electric_vehicle_charging_station", "gas_station", "parking", "parking_garage", "parking_lot",
        "rest_stop", "tire_shop", "truck_dealer",
    ],
    "business": [
        "business_center", "corporate_office", "coworking_space", "farm", "manufacturer", "ranch",
        "supplier", "television_studio",
    ],
    "culture": [
        "art_gallery", "art_museum", "art_studio", "auditorium", "castle", "cultural_landmark", "fountain",
        "historical_place", "history_museum", "monument", "museum", "performing_arts_theater", "sculpture",
    ],
    "education": [
        "academic_department", "educational_institution", "library", "preschool", "primary_school",
        "research_institute", "school", "secondary_school", "university",
    ],
    "entertainment": [
        "adventure_sports_center", "amphitheatre", "amusement_center", "amusement_park", "aquarium",
        "banquet_hall", "barbecue_area", "botanical_garden", "bowling_alley", "casino", "childrens_camp",
        "city_park", "comedy_club", "community_center", "concert_hall", "convention_center", "cultural_center",
        "cycling_park", "dance_hall", "dog_park", "event_venue", "ferris_wheel", "garden", "go_karting_venue",
        "hiking_area", "historical_landmark", "indoor_playground", "internet_cafe", "karaoke", "live_music_venue",
        "marina", "miniature_golf_course", "movie_rental", "movie_theater", "national_park", "night_club",
        "observation_deck", "off_roading_area", "opera_house", "paintball_center", "park", "philharmonic_hall",
        "picnic_ground", "planetarium", "plaza", "roller_coaster", "skateboard_park", "state_park",
        "tourist_attraction", "video_arcade", "vineyard", "visitor_center", "water_park", "wedding_venue",
        "wildlife_park", "wildlife_refuge", "zoo",
    ],
    "facilities": ["public_bath", "public_bathroom", "stable"],
    "finance": ["accounting", "atm", "bank"],
    "food": [
        "acai_shop", "afghani_restaurant", "african_restaurant", "american_restaurant", "argentinian_restaurant",
        "asian_fusion_restaurant", "asian_restaurant", "australian_restaurant", "austrian_restaurant", "bagel_shop",
        "bakery", "bangladeshi_restaurant", "bar", "bar_and_grill", "barbecue_restaurant", "basque_restaurant",
        "bavarian_restaurant", "beer_garden", "belgian_restaurant", "bistro", "brazilian_restaurant",
        "breakfast_restaurant", "brewery", "brewpub", "british_restaurant", "brunch_restaurant", "buffet_restaurant",
        "burmese_restaurant", "burrito_restaurant", "cafe", "cafeteria", "cajun_restaurant", "cake_shop",
        "californian_restaurant", "cambodian_restaurant", "candy_store", "cantonese_restaurant", "caribbean_restaurant",
        "cat_cafe", "chicken_restaurant", "chicken_wings_restaurant", "chilean_restaurant", "chinese_noodle_restaurant",
        "chinese_restaurant", "chocolate_factory", "chocolate_shop", "cocktail_bar", "coffee_roastery", "coffee_shop",
        "coffee_stand", "colombian_restaurant", "confectionery", "croatian_restaurant", "cuban_restaurant",
        "czech_restaurant", "danish_restaurant", "deli", "dessert_restaurant", "dessert_shop", "dim_sum_restaurant",
        "diner", "dog_cafe", "donut_shop", "dumpling_restaurant", "dutch_restaurant", "eastern_european_restaurant",
        "ethiopian_restaurant", "european_restaurant", "falafel_restaurant", "family_restaurant", "fast_food_restaurant",
        "filipino_restaurant", "fine_dining_restaurant", "fish_and_chips_restaurant", "fondue_restaurant", "food_court",
        "french_restaurant", "fusion_restaurant", "gastropub", "german_restaurant", "greek_restaurant",
        "gyro_restaurant", "halal_restaurant", "hamburger_restaurant", "hawaiian_restaurant", "hookah_bar",
        "hot_dog_restaurant", "hot_dog_stand", "hot_pot_restaurant", "hungarian_restaurant", "ice_cream_shop",
        "indian_restaurant", "indonesian_restaurant", "irish_pub", "irish_restaurant", "israeli_restaurant",
        "italian_restaurant", "japanese_curry_restaurant", "japanese_izakaya_restaurant", "japanese_restaurant",
        "juice_shop", "kebab_shop", "korean_barbecue_restaurant", "korean_restaurant", "latin_american_restaurant",
        "lebanese_restaurant", "lounge_bar", "malaysian_restaurant", "meal_delivery", "meal_takeaway",
        "mediterranean_restaurant", "mexican_restaurant", "middle_eastern_restaurant", "mongolian_barbecue_restaurant",
        "moroccan_restaurant", "noodle_shop", "north_indian_restaurant", "oyster_bar_restaurant", "pakistani_restaurant",
        "pastry_shop", "persian_restaurant", "peruvian_restaurant", "pizza_delivery", "pizza_restaurant",
        "polish_restaurant", "portuguese_restaurant", "pub", "ramen_restaurant", "restaurant", "romanian_restaurant",
        "russian_restaurant", "salad_shop", "sandwich_shop", "scandinavian_restaurant", "seafood_restaurant",
        "shawarma_restaurant", "snack_bar", "soul_food_restaurant", "soup_restaurant", "south_american_restaurant",
        "south_indian_restaurant", "southwestern_us_restaurant", "spanish_restaurant", "sports_bar",
        "sri_lankan_restaurant", "steak_house", "sushi_restaurant", "swiss_restaurant", "taco_restaurant",
        "taiwanese_restaurant", "tapas_restaurant", "tea_house", "tex_mex_restaurant", "thai_restaurant",
        "tibetan_restaurant", "tonkatsu_restaurant", "turkish_restaurant", "ukrainian_restaurant", "vegan_restaurant",
        "vegetarian_restaurant", "vietnamese_restaurant", "western_restaurant", "wine_bar", "winery",
        "yakiniku_restaurant", "yakitori_restaurant",
    ],
    "government": [
        "city_hall", "courthouse", "embassy", "fire_station", "government_office",
        "local_government_office", "neighborhood_police_station", "police", "post_office",
    ],
    "health": [
        "chiropractor", "dental_clinic", "dentist", "doctor", "drugstore", "general_hospital", "hospital",
        "massage", "massage_spa", "medical_center", "medical_clinic", "medical_lab", "pharmacy",
        "physiotherapist", "sauna", "skin_care_clinic", "spa", "tanning_studio", "wellness_center", "yoga_studio",
    ],
    "nature": ["beach", "island", "lake", "mountain_peak", "nature_preserve", "river", "scenic_spot", "woods"],
    "worship": ["buddhist_temple", "church", "hindu_temple", "mosque", "shinto_shrine", "synagogue"],
    "services": [
        "aircraft_rental_service", "association_or_organization", "astrologer", "barber_shop", "beautician",
        "beauty_salon", "body_art_service", "catering_service", "cemetery", "chauffeur_service", "child_care_agency",
        "consultant", "courier_service", "electrician", "employment_agency", "florist", "food_delivery", "foot_care",
        "funeral_home", "hair_care", "hair_salon", "insurance_agency", "laundry", "lawyer", "locksmith",
        "makeup_artist", "marketing_consultant", "moving_company", "nail_salon", "non_profit_organization", "painter",
        "pet_boarding_service", "pet_care", "plumber", "psychic", "real_estate_agency", "roofing_contractor",
        "service", "shipping_service", "storage", "summer_camp_organizer", "tailor",
        "telecommunications_service_provider", "tour_agency", "tourist_information_center", "travel_agency",
        "veterinary_care",
    ],
    "sport": [
        "arena", "athletic_field", "fishing_charter", "fishing_pier", "fishing_pond", "fitness_center",
        "golf_course", "gym", "ice_skating_rink", "indoor_golf_course", "playground", "race_course", "ski_resort",
        "sports_activity_location", "sports_club", "sports_coaching", "sports_complex", "sports_school",
        "stadium", "swimming_pool", "tennis_court",
    ],
    "transportation": [
        "airport", "airstrip", "bike_sharing_station", "bridge", "bus_station", "bus_stop", "ferry_service",
        "ferry_terminal", "heliport", "international_airport", "light_rail_station", "park_and_ride",
        "subway_station", "taxi_service", "taxi_stand", "toll_station", "train_station", "train_ticket_office",
        "tram_stop", "transit_depot", "transit_station", "transit_stop", "transportation_service", "truck_stop",
    ],
}

df = google_placescat_5000.copy()
dup_mask = df.duplicated(subset="id", keep=False)

def is_valid_type(row, cat_to_types):
    cat = row["primary_cat"]
    ptype = row["primary_type"]
    if cat not in cat_to_types:
        return False
    return ptype in cat_to_types[cat]

df["valid_type"] = True
df.loc[dup_mask, "valid_type"] = df.loc[dup_mask].apply(is_valid_type, axis=1, cat_to_types=CAT_TO_TYPES)
df_clean = df[(~dup_mask) | (df["valid_type"])].drop(columns="valid_type")

df_clean = df_clean.drop_duplicates(subset="id", keep="first")
df_clean = df_clean[~df_clean["primary_type"].isin(table_blist)].reset_index(drop=True)

# merge with google_naics_mapping to get naics_code and naics_definition
google_naics_mapping = pd.read_csv('/Users/houpuli/Redlining Lab Dropbox/HOUPU LI/POI research/mapping_google_naics.csv')
df_clean = df_clean.dropna(subset=['primary_type'])
df_clean = df_clean.merge(google_naics_mapping[['SubCategory','naics_code','naics_definition']], left_on = 'primary_type', right_on='SubCategory', how="left")
df_clean['addr_simple'] = df_clean['address'].str.split(',', n=1).str[0]
df_clean = df_clean.drop(columns=['SubCategory']).reset_index(drop=True)
df_clean