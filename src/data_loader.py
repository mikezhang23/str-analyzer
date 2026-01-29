import pandas as pd  # Import pandas library for working with tabular data
import json # Import json library for working with JSON data

def parse_price(price_str):  # Takes a string like "$150.00"
    if pd.isna(price_str):  # Check if it's empty/missing
        return 0.0
    if isinstance(price_str, float):  # Check if it's already a number
        return price_str
    cleaned = price_str.replace("$", "").replace(",", "")  # Remove $ and commas, save to "cleaned"
    number = float(cleaned)  # Convert string to decimal number
    return number  # Send back the number

# Define a function called load_listings
def load_listings():  
    """
    Load Airbnb listings from local CSV file.
    """
    filepath = "data/listings.csv"  # Path to our data file
    df = pd.read_csv(filepath)  # Read CSV into a DataFrame (like a spreadsheet)
    df['price'] = df['price'].apply(parse_price)  # Apply our function to every price
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
    df['bedrooms_int'] = df['bedrooms'].round().astype('Int64')
    df = add_location_cells(df)  # Adds location grid cells for spatial controls
    df = add_revenue_features(df)  # Adds revenue-related columns
    df = add_amenities_flags(df)  # Adds amenities flag columns
    df = remove_outliers_iqr(df, ['price', 'revpar'])  # Remove extreme outliers

# Filter out bad data
    df = df[df['price'] > 25] # Remove listings < $25 per night (likely not real)
    df = df[df['price'] < 2000] # Remove listings > $2000 per night (likely outliers)
    df = df[df['bedrooms'].notna()] # Remove listings with no bedroom info
    df = df[df['bedrooms'] <= 10] # Focused on typical STRs, remove listings with >10 bedrooms
    return df  # Return cleaned DataFrame

def add_revenue_features(df):
    df['estimated_occupancy_rate'] = (df['reviews_per_month'].fillna(0) * 2 * 3) / 30 # estimates occupancy by multiplying reviews by 3 days (avg stay est). and by 2 bc roughly 50% of guests leave reviews; fillna fills na with 0.
    df['revpar'] = df['price'] * df['estimated_occupancy_rate'].clip(0,0.95)  # revenue per available room, capped at 95% occupancy
    df['estimated_annual_revenue'] = df['revpar'] * 365
    return df

def add_location_cells(df, cell_size=0.02):
    """Create a grid-based location cell from latitude/longitude for spatial controls."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df
    df = df.copy()
    lat = pd.to_numeric(df['latitude'], errors='coerce')
    lon = pd.to_numeric(df['longitude'], errors='coerce')
    lat_bin = (lat / cell_size).round(0).astype('Int64')
    lon_bin = (lon / cell_size).round(0).astype('Int64')
    df['location_cell'] = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    df.loc[lat.isna() | lon.isna(), 'location_cell'] = pd.NA
    return df

def remove_outliers_iqr(df, columns, k=1.5, min_rows=50):
    """Remove outliers using the IQR rule for the specified columns."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        valid = series.dropna()
        if len(valid) < min_rows:
            continue
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        df = df[(series >= lower) & (series <= upper)]
    return df

def parse_amenities(amenities_str):
    """Convert amenities string to a Python list."""
    if pd.isna(amenities_str): # Check if it's empty/missing
        return []
    try:
        return json.loads(amenities_str) # Try to parse the string as JSON
    except:
        return []  # If parsing fails, return empty list

def add_amenities_flags(df):
    """Create binary columns for key amenities."""
    df['amenities_list'] = df['amenities'].apply(parse_amenities)  # Parse amenities into lists

    # define amenities we want to test
    key_amenities = ['Pool', 'Hot tub', 'Kitchen', 'Gym', 'Sauna', 'BBQ Grill', 'Washer', 'Dryer', 'View']

    # create binary columns for each key amenity
    for amenity in key_amenities:
        col_name = 'has_' + amenity.lower().replace(' ', '_') # e.g., 'has_pool'
        df[col_name] = df['amenities_list'].apply(lambda x: amenity in x) # True/False if amenity is in the list

    return df
