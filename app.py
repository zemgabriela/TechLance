import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from recommender_system import load_model, obtain_all_predictions, get_top_n_benefits
from roi import get_roi


def plot_metrics_for_benefit_type(df, benefit_type):
    metrics = {
        'ROI_Score': 'ROI Score',
        'CostPerUsage': 'Cost per Usage',
        'UsageFrequency': 'Usage Frequency',
        'SatisfactionScore': 'Satisfaction Score'
    }
    
    # Filter to the chosen BenefitType
    subset = df[df['BenefitType'] == benefit_type]
    
    # Group by subtype
    grouped = subset.groupby('BenefitSubType').agg({
        'ROI_Score': 'mean',
        'CostPerUsage': 'mean',
        'UsageFrequency': 'mean',
        'SatisfactionScore': 'mean'
    }).reset_index()
    
    # Create subplot layout (2x2)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{label} by Benefit Subtype' for label in metrics.values()]
    )
    
    # Add each metric as a subplot
    row_col_positions = [(1,1), (1,2), (2,1), (2,2)]
    for (col, label), (row, col_pos) in zip(metrics.items(), row_col_positions):
        fig.add_trace(
            go.Bar(
                x=grouped['BenefitSubType'],
                y=grouped[col],
                name=label
            ),
            row=row, col=col_pos
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=False,
        template='plotly_white',
        title_text=f'Metrics by Benefit Subtype ({benefit_type})'
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=-45)
    
    st.plotly_chart(fig)

    return None

# Load your data
df = pd.read_csv('data/cleaned_data.csv')

############################ DATA PREPROCESSING ############################
# Combine Gender_Female, Gender_Male, Gender_Non-Binary into single column
df['Gender'] = df.apply(lambda row: 'Female' if row['Gender_Female'] else ('Male' if row['Gender_Male'] else ('Non-Binary' if row['Gender_Non-Binary'] else 'Unknown')),axis=1)

# Convert one-hot encoded generation columns to a single 'age_group' column
def get_age_group(row):
    if row['Age_Gen_Boomer']:
        return '58+'
    elif row['Age_Gen_Gen_X']:
        return '42-57'
    elif row['Age_Gen_Millenial']:
        return '26-41'
    elif row['Age_Gen_Gen_Z']:
        return '18-25'
    else:
        return 'Unknown'
    
df['Age Group'] = df.apply(get_age_group, axis=1)

# Convert one-hot encoded tenure group columns to a single 'tenure_group' column
def get_tenure_group(row):
    if row['TenureGroups_ >25_years']:
        return '>25'
    elif row['TenureGroups_16-25_years']:
        return '16-25'
    elif row['TenureGroups_5-15_years']:
        return '5-15'
    elif row['TenureGroups_<5_years']:
        return '<5'
    else:
        return 'Unknown'

df['Tenure Group'] = df.apply(get_tenure_group, axis=1)

# Convert one-hot encoded department columns to a single 'Department' column
def get_department(row):
    if row['Department_Finance']:
        return 'Finance'
    elif row['Department_HR']:
        return 'HR'
    elif row['Department_IT']:
        return 'IT'
    elif row['Department_Marketing']:
        return 'Marketing'
    elif row['Department_Sales']:
        return 'Sales'
    else:
        return 'Unknown'

df['Department'] = df.apply(get_department, axis=1)
df['CostPerUsage'] = df['BenefitCost'] / df['UsageFrequency'].replace(0, np.nan)  # Avoid division by zero
df['CostPerUsage'].fillna(0, inplace=True)  # Fill NaN with 0 for no usage  
df['LastUsedDate'] = pd.to_datetime(df['LastUsedDate'])

# Develop an ROI score (normalize cost-per-usage and satisfaction score)

# Normalize cost-per-usage
max_cost = df['CostPerUsage'].max()
min_cost = df['CostPerUsage'].min()
df['NormalizedCost'] = (df['CostPerUsage'] - min_cost) / (max_cost - min_cost)

# Normalize satisfaction score
max_satisfaction = df['SatisfactionScore'].max()
min_satisfaction = df['SatisfactionScore'].min()
df['NormalizedSatisfaction'] = (df['SatisfactionScore'] - min_satisfaction) / (max_satisfaction - min_satisfaction)

# Calculate ROI score
# Those that have no cost or satisfaction will have NaN ROI, which we can handle later
df['ROI_Score'] = df['NormalizedSatisfaction'] / df['NormalizedCost'].replace(0, np.nan)  # Avoid division by zero


############################ STREAMLIT APP ############################

st.title("Benefits Dashboard")

# Sidebar filters

# --- Demographic Filters ---
st.sidebar.header("Demographic Filters")
age_group = st.sidebar.multiselect("Age Group", options=df['Age Group'].unique())
gender = st.sidebar.multiselect("Gender", options=df['Gender'].unique())
department = st.sidebar.multiselect("Department", options=df['Department'].unique())
tenure_group = st.sidebar.multiselect("Tenure Group", options=df['Tenure Group'].unique())

# --- Benefit Filters ---
st.sidebar.header("Benefit Filters")
benefit_type = st.sidebar.multiselect("Benefit Type", options=df['BenefitType'].unique())
if benefit_type:
    subtype_options = df[df['BenefitType'].isin(benefit_type)]['BenefitSubType'].unique()
else:
    subtype_options = df['BenefitSubType'].unique()
benefit_subtype = st.sidebar.multiselect("Benefit Subtype", options=subtype_options)


# Filter data
filtered_df = df.copy()
if age_group:
    filtered_df = filtered_df[filtered_df['Age Group'].isin(age_group)]
if gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
if department:
    filtered_df = filtered_df[filtered_df['Department'].isin(department)]
if tenure_group:
    filtered_df = filtered_df[filtered_df['Tenure Group'].isin(tenure_group)]
if benefit_subtype:
    filtered_df = filtered_df[filtered_df['BenefitSubType'].isin(benefit_subtype)]
if benefit_type:
    filtered_df = filtered_df[filtered_df['BenefitType'].isin(benefit_type)]

# Metrics side by side
col1, col2, col3 = st.columns(3)
col1.metric("Average Utilization", round(filtered_df['UsageFrequency'].mean(), 2))
col2.metric("Average Satisfaction", round(filtered_df['SatisfactionScore'].mean(), 2))
col3.metric("Average ROI", round(filtered_df['ROI_Score'].mean(), 2) if 'ROI_Score' in filtered_df else "N/A")

# Segment breakdown
st.write("Segment Breakdown")
segment_df = filtered_df.groupby(['Age Group', 'Gender', 'Department', 'Tenure Group'])[['UsageFrequency', 'SatisfactionScore']].mean().reset_index()
segment_df['UsageFrequency'] = segment_df['UsageFrequency'].round(2)
segment_df['SatisfactionScore'] = segment_df['SatisfactionScore'].round(2)
st.dataframe(segment_df)

# Add more charts/plots as needed (e.g., bar charts, scatter plots)

# Bar Plot
st.header("View Distributions")
options_x = df.select_dtypes(exclude=["number","bool_", "datetime"]).columns.to_list()
options_y = df.select_dtypes(include=["number"]).columns.to_list()

remove_cols = ['Comments','EmployeeID', 'BenefitID']
options_x = [x for x in options_x if x not in remove_cols]
options_y = [y for y in options_y if y not in remove_cols]

col1, col2, col3 = st.columns(3)
x_axis_opt = col1.segmented_control("Select X-Axis Value: ", options=options_x, selection_mode="single")
y_axis_opt = col2.segmented_control("Select Y-Axis Value: ", options=options_y, selection_mode="single")
options_hue = [x for x in options_x if x != x_axis_opt]
hue_color = col3.segmented_control("Select Hue: ", options=options_hue, selection_mode="single")

if x_axis_opt and y_axis_opt:
    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)

    if hue_color:
        data = filtered_df[[x_axis_opt, y_axis_opt, hue_color]].groupby(by=[x_axis_opt, hue_color]).mean()
        #st.bar_chart(data=data)
        fig, ax = plt.subplots()
        fig.set_size_inches(20,10)
        sns.barplot(data=data, x=x_axis_opt, y=y_axis_opt, hue=hue_color, ax=ax, palette='rocket')

    else: 
        data = filtered_df[[x_axis_opt, y_axis_opt]].groupby(by=x_axis_opt).mean()
        sns.barplot(data=data, x=x_axis_opt, y=y_axis_opt, hue=hue_color, ax=ax, palette='rocket')

    ax.set_xticklabels(filtered_df[x_axis_opt].unique(), rotation=45)
    st.pyplot(fig)

# 
st.header("Cost Efficiency")

roi_df = get_roi(df=filtered_df)

option = st.selectbox(
    "Select Benefit Type:",
    roi_df['BenefitType'].unique().tolist(),
)
plot_metrics_for_benefit_type(df=roi_df, benefit_type=option)


# Get Predictions
st.header("Best Recommendations per Employee")

svd = load_model(path="model/svd_model.pkl")
print(svd)

#cols = ['EmployeeID', 'BenefitID', 'CombinedScore']
#data = list(df[cols].itertuples(index=False, name=None))

df['CombinedScore'] = df['SatisfactionScore'] * df['UsageFrequency'].replace(0.0,1)

user_item_matrix = df.pivot_table(index='EmployeeID', 
                                  columns='BenefitID', 
                                  values='CombinedScore',
                                  fill_value=0)

predictions = obtain_all_predictions(svd=svd, df=user_item_matrix)


n = st.number_input("Select Top N Benefits", step=1, min_value=1, max_value=len(df.columns), value=10)
top_n = get_top_n_benefits(predictions=predictions, n=n)

flattened_data = [(uid, iid) for uid, iids in top_n.items() for iid in iids]
top_n_df = pd.DataFrame(flattened_data, columns=['EmployeeID', 'BenefitID'])

top_n_df = top_n_df.merge(right=df[['BenefitID', 'BenefitType', 'BenefitSubType', 'BenefitCost']].drop_duplicates(), on='BenefitID')
#st.dataframe(top_n_df[['BenefitID', 'BenefitType', 'BenefitSubType']])
#mean_use_freq = df[['BenefitID', 'UsageFrequency', 'SatisfactionScore']].groupby(by='BenefitID').mean()
#mean_use_freq['UsageFrequency'] = mean_use_freq['UsageFrequency'].apply(lambda x: round(x))
#mean_use_freq['SatisfactionScore'] = mean_use_freq['SatisfactionScore'].apply(lambda x: round(x))
#top_n_df = top_n_df.merge(right=mean_use_freq, on='BenefitID')

st.dataframe(top_n_df)



