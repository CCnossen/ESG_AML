# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:31:42 2024

@author: CCnossen
"""

import os
import re
import pandas as pd
import numpy as np
import warnings

#------------------------------------------------------------------------------
# read in world production & mining of minerals
#------------------------------------------------------------------------------

def combine_files_in_directory(directory_path):
    # List to hold individual dataframes
    dataframes = []

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, filename)
        
        # Check if the path is a file (and optionally if it ends with .csv)
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df['filename'] = filename
            # Append the DataFrame to the list
            dataframes.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

# Readin and combine :
directory_path = 'C:/Users/CCnossen/Desktop/Open data/World_commodities/'
combined_df = combine_files_in_directory(directory_path)


# Identify first and second value column from data
def first_non_null_value(df):
    # Apply a function to each row that finds the first non-null value
    first_values = df.apply(lambda row: next((val for val in row if not pd.isnull(val)), None), axis=1)
    return first_values

def find_second_non_null(row):
    # Extract non-null values from the row
    non_null_values = [val for val in row if not pd.isnull(val)]
    # Return the second non-null value if it exists
    if len(non_null_values) >= 2:
        return non_null_values[1]
    else:
        return None

def second_non_null_value(df):
    # Apply a function to each row that finds the second non-null value
    second_values = df.apply(lambda row: find_second_non_null(row), axis=1)
    return second_values

# values only
df_vals = combined_df[['Prod_kt_2022','Prod_kt_est_2023','Prod_t_2022','Prod_t_est_2023','Prod_kt_est_2022','Prod_t_est_2022','Prod_mct_2022','Prod_mct_est_2023','Cap_t_est_2022','Cap_t_est_2023','Cap_kt_2022','Cap_kt_est_2023']]

# extract values from the list, there are only 2 possible values for each commodity
df_vals['first_values'] = first_non_null_value(df_vals)
df_vals['second_values'] = second_non_null_value(df_vals)
df_vals = df_vals[['first_values', 'second_values']]

#merge the values back to the original dataset
combined_df = pd.merge(combined_df, df_vals, left_index=True, right_index=True)
commodity_data = combined_df[['Source','Country','Type','Prod_notes','filename','first_values','second_values']]
commodity_data = commodity_data[commodity_data['Country'].notna()]
commodity_data = commodity_data[commodity_data.Country != 'World total (rounded)']


# cleanup
del df_vals


#------------------------------------------------------------------------------
# readin agricultural data from UN FAO
#------------------------------------------------------------------------------
fao_agri = pd.read_csv('C:/Users/CCnossen/Desktop/Open data/FAOSTAT_data_en_1-17-2025.csv') #https://www.fao.org/faostat/en/#data/QV

# cleanup

#------------------------------------------------------------------------------
# readin TPI data
#------------------------------------------------------------------------------

# TPI - corruption perception index
tpi = pd.read_excel('C:/Users/CCnossen/Desktop/Open data/CPI2023_Global_Results__Trends.xlsx', sheet_name='CPI 2023')
tpi.columns = tpi.iloc[2] # use row 3 for colnames
tpi = tpi.iloc[3:] # drop first 3 rows
tpi['CPI score 2023'] = pd.to_numeric(tpi['CPI score 2023'])

#------------------------------------------------------------------------------
# readin ILO data
#------------------------------------------------------------------------------

# ILO - child labour statistics
#'SDG indicator 8.7.1 - Proportion of children engaged in economic activity (%)'
ilo = pd.read_excel('C:/Users/CCnossen/Desktop/Open data/SDG indicator 8.7.1 - Proportion of children engaged in economic activity (%).xlsx', sheet_name='Sheet1')
ilo_subset = ilo[ilo['sex.label'] == 'Sex: Total']
ilo_subset = ilo_subset[ilo_subset['classif1.label'] == "Age (Child labour bands): '5-17"]
ilo_subset = ilo_subset.sort_values(by=['ref_area.label'] + ['time'], ascending=[True] * len(['ref_area.label']) + [False])
ilo_subset = ilo_subset.drop_duplicates(subset=['ref_area.label'], keep='first')
ilo_childlabour = ilo_subset

# cleanup
del ilo, ilo_subset

#------------------------------------------------------------------------------
# readin Tree coverage data
#------------------------------------------------------------------------------

# UN FAO - tree coverage
un_fao = pd.read_csv('C:/Users/CCnossen/Desktop/Open data/FAOSTAT_data_en_7-23-2024.csv')
un_fao_subset = un_fao[un_fao['Element'] == 'Area from CCI_LC'] # keep only the dataset consistent since 1992

# sum the mangroves + tree coverage
un_fao_subset_grouped = un_fao_subset.groupby(['Area','Year'])['Value'].agg(['sum']).reset_index()

# keep only the minimum and maximum years in range
un_fao_subset_minmax = un_fao_subset_grouped.groupby('Area')['Year'].agg(['min', 'max']).reset_index()
    
# merge back to un_fao_subset_grouped, and keep only the min/max years
merged = pd.merge(un_fao_subset_grouped, un_fao_subset_minmax, left_on = ['Area','Year'], right_on = ['Area','max'], how = 'left')
merged = pd.merge(merged, un_fao_subset_minmax, left_on = ['Area','Year'], right_on = ['Area','min'], how = 'left')
un_fao_merged = merged[merged['max_x'].notna() | merged['min_y'].notna()]

# Function to calculate the percentage difference for each group
un_fao_min = un_fao_merged[un_fao_merged['Year'] == un_fao_merged['min_y']]
un_fao_max = un_fao_merged[un_fao_merged['Year'] == un_fao_merged['max_x']]

# Add interesting columns (absolute and relative differences)
un_fao_merged = pd.merge(un_fao_max[['Area', 'sum']], un_fao_min[['Area', 'sum']], left_on = 'Area', right_on = 'Area', how = 'left')
un_fao_merged.columns = ['Area', 'max_year_val', 'min_year_val']
un_fao_merged['pct_diff'] = (un_fao_merged['max_year_val'] -  un_fao_merged['min_year_val'] ) / un_fao_merged['min_year_val'] 
un_fao_merged['abs_diff'] = (un_fao_merged['max_year_val'] -  un_fao_merged['min_year_val'] )

fao_tree = un_fao_merged

# cleanup
del un_fao_subset, un_fao_subset_grouped, un_fao_subset_minmax, merged, un_fao_min, un_fao_max, un_fao_merged, un_fao

#------------------------------------------------------------------------------
# readin supporting data
#------------------------------------------------------------------------------

# get reference list of country ISO codes
c_iso = pd.read_excel('C:/Users/CCnossen/Desktop/Open data/ISO Country list.xlsx', sheet_name='Sheet1')

#------------------------------------------------------------------------------
# combine datasets
#------------------------------------------------------------------------------

# functions

# whitespace cleaning function
def clean_whitespaces(s):
    s = s.strip()
    s = re.sub(r'\\n', ' ', s) # remove \n (whitespaces)
    s = re.sub(' \t\n\r\x0b\x0c', ' ', s) # all kinds of white spaces
    s = re.sub(r"\s+", " ", s).strip() # clean leftover and repeated white spaces
    return s

# cleaning of special characters & stuff
def clean_special_stuff(s):
    # insert whitespaces between numbers and letters
    s = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', s)
    s = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', s)
    # clean non-unicode characters
    s = re.sub(r'[^\x00-\x7F]+', '', s) 
    s = re.sub("&nsbp", " ", s)
    s = re.sub(r'&[a-zA-Z]+;', '', s)
    s = re.sub(r'[^\x00-\x7F]+', '', s) 
    s = re.sub("\\\\xa 0", " ", s)
    return(s)

# Sorting strings function
def sort_string_alphabetically(s):
    return ''.join(sorted(s))

# remove all non alphanumeric characters
def remove_nonalfanumerics(s):
    s = re.sub(r'[^A-Za-z0-9]+', '', s)
    return s

def find_partial_string_in_dataframe(df1, df2, column1, column2):
    """
    Looks for partial string matches from column1 of df1 in column2 of df2,
    and returns matching rows from df2 with a reference column from df1.
    Omits rows with multiple matches and gives a warning.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame containing strings to search for.
    df2 (pd.DataFrame): The second DataFrame to search in.
    column1 (str): The column in df1 containing strings to search for.
    column2 (str): The column in df2 to search within.

    Returns:
    pd.DataFrame: A DataFrame with rows from df2 where column2 partially matches 
                  any string in column1 of df1, along with a reference column from df1.
    """
    # Initialize a list to store the matching reference from df1 for each row in df2
    references = []
    rows_to_keep = []
    
    # Loop through each row in df2
    for idx, item in enumerate(df2[column2]):
        # Find all matches from df1[column1]
        matches = [partial for partial in df1[column1] if partial in item]
        
        if len(matches) > 1:
            # Warn if multiple matches are found and skip the row
            warnings.warn(f"Row '{item}' has multiple matches in df1: {matches}. It will be omitted.")
            references.append(None)  # Placeholder for clarity
        elif len(matches) == 1:
            # Keep the row if exactly one match is found
            references.append(matches[0])
            rows_to_keep.append(idx)
        else:
            # No match found
            references.append(None)

    # Add the reference column to df2
    df2['Reference'] = references

    # Keep only rows with a valid single match
    matching_rows = df2.iloc[rows_to_keep]
    
    return matching_rows


# merge country iso to combined_df
commodity_data['Country'] = commodity_data['Country'].apply(remove_nonalfanumerics)
c_iso['Country'] = c_iso['Country'].apply(remove_nonalfanumerics)
temp = pd.merge(commodity_data, c_iso, left_on = ['Country'], right_on = ['Country'], how = 'left')

# Find rows in df2 where 'Item' partially matches any string in df1['Strings']
temp2 = temp[temp['Iso2'].isnull()]
result = find_partial_string_in_dataframe(c_iso, temp2, 'Country', 'Country')
result = result.drop(['Iso2', 'Iso3','Numeric'], axis = 1)
temp3 = pd.merge(result, c_iso, left_on = ['Reference'], right_on = ['Country'], how = 'left')
temp3 = temp3.drop(['Reference', 'Country_y'], axis = 1)
temp3 = temp3.rename(columns={'Country_x': 'Country'})

# Merge into complete set
temp4 = temp[temp.Iso2.notnull()]
commodity_data_country = pd.concat([temp4,temp3])

#cleanup names of Type of commodity
commodity_data_country['Type_refined'] = commodity_data_country['Type'].str.replace('Plant capacity, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Plant production, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Smelter production, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Mine production, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Mine production: ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('mine production, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Mne poduction, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace('Refinery production, ','')
commodity_data_country['Type_refined'] = commodity_data_country['Type_refined'].str.replace(', estimated, ','')


del temp, temp2, temp3, temp4, result

#------------------------------------------------------------------------------
# combine datasets to summarized view
#------------------------------------------------------------------------------

commodities = pd.Series(commodity_data_country.Type_refined.unique())
agris = pd.Series(fao_agri.Item.unique())

all_raw = pd.concat([commodities, agris]).to_list()
all_raw = [x for x in all_raw if str(x) != 'nan']






