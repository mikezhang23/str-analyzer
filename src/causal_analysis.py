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


def match_properties(df, treatment_col, propensity_scores, exact_match_cols=None):
    df = df.copy()
    df['propensity_score'] = propensity_scores
    
    treated = df[df[treatment_col] == True]
    control = df[df[treatment_col] == False]
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    matched_treated_idx = []
    matched_control_idx = []

    if exact_match_cols:
        for treated_idx, treated_row in treated.iterrows():
            subset = control.copy()
            for col in exact_match_cols:
                subset = subset[subset[col] == treated_row[col]]
            if subset.empty:
                continue
            nn.fit(subset[['propensity_score']])
            _, indices = nn.kneighbors([[treated_row['propensity_score']]])
            matched_control_idx.append(subset.iloc[indices.flatten()[0]].name)
            matched_treated_idx.append(treated_idx)
    else:
        nn.fit(control[['propensity_score']])
        _, indices = nn.kneighbors(treated[['propensity_score']])
        matched_control_idx = control.iloc[indices.flatten()].index
        matched_treated_idx = treated.index
    
    return pd.Index(matched_treated_idx), pd.Index(matched_control_idx)


def estimate_ipw_effect(df, treatment_col, outcome_col, propensity_scores):
    treated_mask = df[treatment_col] == True
    control_mask = df[treatment_col] == False

    treated_weights = 1 / propensity_scores[treated_mask]
    control_weights = 1 / (1 - propensity_scores[control_mask])

    treated_mean = np.average(df.loc[treated_mask, outcome_col], weights=treated_weights)
    control_mean = np.average(df.loc[control_mask, outcome_col], weights=control_weights)
    return treated_mean - control_mean, treated_mean, control_mean


def add_location_confounders(df, confounders, min_cell_count=10):
    if 'location_cell' in df.columns:
        cell_counts = df['location_cell'].value_counts()
        valid_cells = cell_counts[cell_counts >= min_cell_count].index
        df = df[df['location_cell'].isin(valid_cells)].copy()
        location_dummies = pd.get_dummies(df['location_cell'], prefix='loc')
        df = pd.concat([df, location_dummies], axis=1)
        confounders = confounders + list(location_dummies.columns)
        return df, confounders

    if 'neighbourhood_cleansed' in df.columns:
        neighbourhood_counts = df['neighbourhood_cleansed'].value_counts()
        valid_neighbourhoods = neighbourhood_counts[neighbourhood_counts >= min_cell_count].index
        df = df[df['neighbourhood_cleansed'].isin(valid_neighbourhoods)].copy()
        location_dummies = pd.get_dummies(df['neighbourhood_cleansed'], prefix='neigh')
        df = pd.concat([df, location_dummies], axis=1)
        confounders = confounders + list(location_dummies.columns)
        return df, confounders

    return df, confounders


def analyze_amenity_impact(
    df,
    amenity_col,
    outcome_col='revpar',
    base_confounders=None,
    min_cell_count=10,
    exact_match_cols=None,
    ps_trim=0.05,
):
    if base_confounders is None:
        base_confounders = ['bedrooms', 'accommodates']
    
    df_clean = df.dropna(subset=base_confounders + [amenity_col, outcome_col])
    
    df_clean, confounders = add_location_confounders(
        df_clean,
        base_confounders,
        min_cell_count=min_cell_count,
    )
    
    propensity_scores = calculate_propensity_scores(df_clean, amenity_col, confounders)
    if ps_trim is not None:
        keep_mask = (propensity_scores >= ps_trim) & (propensity_scores <= 1 - ps_trim)
        df_clean = df_clean.loc[keep_mask].copy()
        propensity_scores = propensity_scores[keep_mask]

    treated_idx, control_idx = match_properties(
        df_clean,
        amenity_col,
        propensity_scores,
        exact_match_cols=exact_match_cols,
    )

    treated_outcome = df_clean.loc[treated_idx, outcome_col].mean()
    control_outcome = df_clean.loc[control_idx, outcome_col].mean()
    causal_effect = treated_outcome - control_outcome
    
    naive_treated = df_clean[df_clean[amenity_col] == True][outcome_col].mean()
    naive_control = df_clean[df_clean[amenity_col] == False][outcome_col].mean()
    naive_difference = naive_treated - naive_control

    ipw_effect, ipw_treated, ipw_control = estimate_ipw_effect(
        df_clean,
        amenity_col,
        outcome_col,
        propensity_scores,
    )
    
    return {
        'amenity': amenity_col,
        'naive_difference': naive_difference,
        'causal_effect': causal_effect,
        'treated_mean': treated_outcome,
        'control_mean': control_outcome,
        'ipw_effect': ipw_effect,
        'ipw_treated_mean': ipw_treated,
        'ipw_control_mean': ipw_control,
        'n_matched_pairs': len(treated_idx),
    }
