import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def calculate_propensity_scores(df, treatment_col, confounders):
    X = df[confounders].fillna(0)
    y = df[treatment_col].astype(int)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    propensity_scores = model.predict_proba(X)[:, 1]
    return propensity_scores


def match_properties(df, treatment_col, propensity_scores):
    df = df.copy()
    df['propensity_score'] = propensity_scores
    
    treated = df[df[treatment_col] == True]
    control = df[df[treatment_col] == False]
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    matched_control_idx = control.iloc[indices.flatten()].index
    matched_treated_idx = treated.index
    
    return matched_treated_idx, matched_control_idx


def analyze_amenity_impact(df, amenity_col, outcome_col='revpar'):
    confounders = ['bedrooms', 'accommodates']
    
    df_clean = df.dropna(subset=confounders + [amenity_col, outcome_col])
    
    # Add location grid as dummy variables
    if 'location_cell' in df_clean.columns:
        cell_counts = df_clean['location_cell'].value_counts()
        valid_cells = cell_counts[cell_counts >= 10].index
        df_clean = df_clean[df_clean['location_cell'].isin(valid_cells)]
        
        location_dummies = pd.get_dummies(df_clean['location_cell'], prefix='loc')
        df_clean = pd.concat([df_clean, location_dummies], axis=1)
        confounders = confounders + list(location_dummies.columns)
    
    propensity_scores = calculate_propensity_scores(df_clean, amenity_col, confounders)
    treated_idx, control_idx = match_properties(df_clean, amenity_col, propensity_scores)
    
    treated_outcome = df_clean.loc[treated_idx, outcome_col].mean()
    control_outcome = df_clean.loc[control_idx, outcome_col].mean()
    causal_effect = treated_outcome - control_outcome
    
    naive_treated = df_clean[df_clean[amenity_col] == True][outcome_col].mean()
    naive_control = df_clean[df_clean[amenity_col] == False][outcome_col].mean()
    naive_difference = naive_treated - naive_control
    
    return {
        'amenity': amenity_col,
        'naive_difference': naive_difference,
        'causal_effect': causal_effect,
        'treated_mean': treated_outcome,
        'control_mean': control_outcome,
        'n_matched_pairs': len(treated_idx),
    }