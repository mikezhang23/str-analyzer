import pandas as pd  # Add this line
import streamlit as st  # Import Streamlit, nickname it "st"
import sys  # Python's system module - lets us modify where Python looks for files
import plotly.express as px  # Import Plotly Express for easy plotting
import folium # Import Folium for map visualization
from streamlit_folium import st_folium # Import Streamlit-Folium for map display

sys.path.insert(0, "src")  # Add "src" folder to Python's search path

from data_loader import load_listings  # Import our function from data_loader.py

st.title("STR Investment Analyzer")  # Big title at top of page

with st.spinner("Loading data..."):  # Show spinning animation while code inside runs
    df = load_listings()  # Call our function, store result in "df"

st.success(f"Loaded {len(df)} listings!")  # Green success message with row count

# Let's inspect the price column
st.write("Price column sample:")
st.write(df['price'].head(10))  # Show first 10 prices
st.write(f"Data type: {df['price'].dtype}")  # Show what type of data it is
st.write(df[['price', 'reviews_per_month', 'estimated_occupancy_rate', 'revpar', 'estimated_annual_revenue']].head(10))

# Plot distribution chart
st.subheader("Price Distribution") # Smaller header for this section
fig = px.histogram(df, x="price", nbins=50) # Create histogram with 50 bars
st.plotly_chart(fig) # Display the plot in the app

# RevPar by bedrooms
st.subheader("RevPAR by Bedroom Count")
fig2 = px.box(df, x="bedrooms", y="revpar") # box plot to show distribution
st.plotly_chart(fig2)   

# Map of listings
st.subheader("Listings Map")
m = folium.Map(location=[36.1699, -115.1398], zoom_start=11)  # Center map on Vegas

# Add markets for each listing
for idx, row in df.head(500).iterrows():  # Loop through first 500 for performance
    if row['revpar'] > 150:  # Only plot if RevPAR > $150
        color = 'green' # high revpar
    elif row['revpar'] > 75: # medium revpar
        color = 'orange' # medium revpar
    else:
        color = 'red' # low revpar
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']], # position from data
        radius=5, # size of dot
        color=color, # outline set to color variable
        fill=True, # makes solid circles
        fillColor=color, # color based on revpar
        fillOpacity=0.7,
        popup=f"${row['price']}/night | RevPAR: ${row['revpar']:.0f}", # shows when you click and popup formats with no decimals
        weight=1, # makes outline thinner (default is 3)
    ).add_to(m) # add to map

st_folium(m, width=700, height=500)  # Display the map

# Check amenities column
st.subheader("Amenities Sample")
st.write(df[['has_pool', 'has_hot_tub', 'has_gym']].head(10))  # Show first 10 rows of amenities flags
st.write(f"Properties with pools: {df['has_pool'].sum()}")  # Count of listings with pool

# Naive comparison of pool vs. no pool
st.subheader("Pool vs No Pool: Simple Comparison")

pool_properties = df[df['has_pool'] == True] # Listings with pools
no_pool_properties = df[df['has_pool'] == False] # Listings without pools

col1, col2 = st.columns(2) # Create two columns for side-by-side display

with col1:
    st.metric("With Pool - Avg RevPAR", f"${pool_properties['revpar'].mean():.2f}") # Average revpar for listings with pools
    st.metric("Properties", len(pool_properties)) # Count of listings with pools

with col2:
    st.metric("Without Pool - Avg RevPAR", f"${no_pool_properties['revpar'].mean():.2f}") # Average revpar for listings without pools
    st.metric("Properties", len(no_pool_properties)) # Count of listings without pools

difference = pool_properties['revpar'].mean() - no_pool_properties['revpar'].mean() # Calculate difference in average revpar
st.write(f"**Difference: ${difference:.2f}") # Show the difference

# Check for confounding
st.subheader("Checking for Confounders")

st.write("Average bedrooms:")
st.write(f"With Pool: {pool_properties['bedrooms'].mean():.2f}")
st.write(f"Without Pool: {no_pool_properties['bedrooms'].mean():.2f}")

### Causal Analysis ###

import sys
sys.path.insert(0, "src")

from data_loader import load_listings
from causal_analysis import analyze_amenity_impact  # Add this import

# ... keep your existing code ...

# Check for zip code
st.write("Columns with 'zip' in name:")
st.write([col for col in df.columns if 'zip' in col.lower()])

st.write("All columns:")
st.write(df.columns.tolist())

# Check location columns
st.subheader("Location Data Check")
st.write("Columns with 'neigh' in name:")
st.write([col for col in df.columns if 'neigh' in col.lower()])
st.write("Sample neighbourhood data:")
if 'neighbourhood_cleansed' in df.columns:
    st.write(df['neighbourhood_cleansed'].value_counts().head(10))

# Causal Analysis
st.subheader("Causal Analysis: Pool Impact")

results = analyze_amenity_impact(df, 'has_pool')

st.write(f"**Naive difference:** ${results['naive_difference']:.2f}")
st.write(f"**Causal effect (after matching):** ${results['causal_effect']:.2f}")
st.write(f"**Matched pairs:** {results['n_matched_pairs']}")

st.subheader("Causal Analysis: Hot Tub Impact")

results_hottub = analyze_amenity_impact(df, 'has_hot_tub')

st.write(f"**Naive difference:** ${results_hottub['naive_difference']:.2f}")
st.write(f"**Causal effect (after matching):** ${results_hottub['causal_effect']:.2f}")
st.write(f"**Matched pairs:** {results_hottub['n_matched_pairs']}")

st.subheader("Causal Impact: All Amenities")

amenities_to_test = ['has_pool', 'has_hot_tub', 'has_gym', 'has_kitchen', 'has_washer', 'has_dryer']

results_list = []
for amenity in amenities_to_test:
    try:
        result = analyze_amenity_impact(df, amenity)
        results_list.append({
            'Amenity': amenity.replace('has_', '').replace('_', ' ').title(),
            'Naive Diff ($)': round(result['naive_difference'], 2),
            'Causal Effect ($)': round(result['causal_effect'], 2),
            'Matched Pairs': result['n_matched_pairs']
        })
    except Exception as e:
        st.write(f"Error with {amenity}: {e}")

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('Causal Effect ($)', ascending=False)  # Best amenities at top

st.dataframe(results_df)