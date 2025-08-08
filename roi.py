
import pandas as pd
import numpy as np
import plotly.express as px

def create_bar_plot_treshold(df, x_col, y_col, title, treshold, treshold_label='max', highlight_above=None):
    # Mapping raw column names to pretty labels
    pretty_labels = {
        'CostPerUsage': 'Cost per Usage',
        'UsageFrequency': 'Usage Frequency',
        'ROI_Score': 'ROI Score'
    }

    # Determine base threshold logic
    if treshold_label == 'max':
        df['ThresholdFlag'] = np.where(df[y_col] > treshold, 'Above Threshold', 'Normal')
    else:
        df['ThresholdFlag'] = np.where(df[y_col] < treshold, 'Below Threshold', 'Normal')

    # Extra highlight for very high values
    if highlight_above is not None:
        df.loc[df[y_col] > highlight_above, 'ThresholdFlag'] = 'Outstanding'

    # Color map with optional extra category
    color_map = {
        'Above Threshold': 'red',
        'Below Threshold': 'red',
        'Normal': 'lightgray',
        'Outstanding': 'darkgoldenrod'
    }

    # Plot
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color='ThresholdFlag',
        color_discrete_map=color_map,
        title=title
    )

    fig.update_yaxes(title_text=pretty_labels.get(y_col, y_col.replace('_', ' ')))
    fig.update_layout(xaxis_tickangle=-90, width=1000, height=500, template='plotly_white')
    fig.add_hline(y=treshold, line_dash="dash", line_color="darkred")

    if highlight_above is not None:
        fig.add_hline(y=highlight_above, line_dash="dash", line_color="darkgoldenrod")

    return fig  # <-- Return figure instead of showing it


def get_roi(df: pd.DataFrame):
    # group by employeeid and benefit id, take average satisfaction score, first comment, sum usagefrequency, take lastuseddate, take max benefit cost, other columns take the first value
    agg_dict = {
        'SatisfactionScore': 'mean',
        'UsageFrequency': 'sum',
        'BenefitCost': 'max',
    }
    # For all other columns, take the first value
    other_cols = [col for col in df.columns if col not in ['EmployeeID','BenefitID'] + list(agg_dict.keys())]
    for col in other_cols:
        agg_dict[col] = 'first'

    merged = df.groupby(['EmployeeID','BenefitID'], as_index=False).agg(agg_dict)


    # Calculate cost-per-usage by BenefitID/BenefitSubType
    merged['CostPerUsage'] = merged['BenefitCost'] / merged['UsageFrequency'].replace(0, np.nan)  # Avoid division by zero
    merged['CostPerUsage'].fillna(0, inplace=True)  # Fill NaN with 0 for no usage  

    # Develop an ROI score (normalize cost-per-usage and satisfaction score)

    # Normalize cost-per-usage
    max_cost = merged['CostPerUsage'].max()
    min_cost = merged['CostPerUsage'].min()
    merged['NormalizedCost'] = (merged['CostPerUsage'] - min_cost) / (max_cost - min_cost)

    # Normalize satisfaction score
    max_satisfaction = merged['SatisfactionScore'].max()
    min_satisfaction = merged['SatisfactionScore'].min()
    
    print(max_satisfaction, min_satisfaction)
    merged['NormalizedSatisfaction'] = (merged['SatisfactionScore'] - min_satisfaction) / (max_satisfaction - min_satisfaction)

    # Calculate ROI score
    # Those that have no cost or satisfaction will have NaN ROI, which we can handle later
    merged['ROI_Score'] = merged['NormalizedSatisfaction'] / merged['NormalizedCost'].replace(0, np.nan)  # Avoid division by zero
    
    ##

    df['CostPerUsage'].fillna(0, inplace=True)  # Fill NaN with 0 for no usage
    # Identify underutilized high-cost subcategories
    # Group by BenefitSubType and calculate mean cost-per-usage and usage frequency, satisfaction score, and rank ROI score
    subcategory_stats = df.groupby('BenefitSubType').agg({
        'CostPerUsage': 'mean',
        'UsageFrequency': 'sum',
        'SatisfactionScore': 'mean',
        'ROI_Score': 'mean'
    }).reset_index()

    # Define high-cost threshold as the 75th percentile of cost-per-usage
    high_cost_threshold = subcategory_stats['CostPerUsage'].quantile(0.75)

    # Define low-usage threshold as the 25th percentile of usage frequency
    low_usage_threshold = subcategory_stats['UsageFrequency'].quantile(0.25)
    roi_usage_treshold = subcategory_stats['ROI_Score'].quantile(0.25)

    # Rank benefits by cost efficiency
    subcategory_stats['CostEfficiencyRank'] = (subcategory_stats['ROI_Score'].rank(ascending=False)).astype(int)

    # Flag which benefits could be removed
    # Define conditions for removal - high cost and low usage or low satisfaction
    subcategory_stats['RemoveFlag'] = np.where(
        (subcategory_stats['CostPerUsage'] > high_cost_threshold) &
        ((subcategory_stats['UsageFrequency'] < low_usage_threshold) |
        (subcategory_stats['SatisfactionScore'] < subcategory_stats['SatisfactionScore'].quantile(0.25))),
        'Remove', 'Keep'
    )
    # Find BenefitType
    benefit_types = merged[['BenefitType','BenefitSubType']].drop_duplicates().reset_index(drop=True)

    # Merge subcategory stats with benefit types
    category_stats = subcategory_stats.merge(
        benefit_types,
        on='BenefitSubType',
        how='left'
    )

    return category_stats