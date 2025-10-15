import pandas as pd
import numpy as np
from pathlib import Path


def preprocess(df):

    # Handling missing values
    # drop rows
    cleaned_df = df[df['DYRK1A_N'].notna()]
    # drop cols
    drop_cols = cleaned_df.notna().sum() < 1000
    cleaned_df = cleaned_df.loc[:, ~drop_cols]

    modified_df = cleaned_df.copy()

    # imputing with median
    for col in cleaned_df.columns[1:73]:

        medians = cleaned_df.groupby('class')[col].transform('median')

        # Impute missing values with the median of the same category

        cleaned_df.loc[:,col] = cleaned_df [col].fillna(medians)


        cleaned_df.isna().sum().sort_values(ascending=False)

    # Handling outliers

    for col in cleaned_df.columns[1:73]:
        data = cleaned_df[col]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)

        IQR = Q3 - Q1

        # Calculate the modified lower and upper bounds
        lower_bound = Q1 -  1.5 * IQR
        upper_bound = Q3 + 2.0 * IQR

        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Handle outliers

        # Replace outliers with the median
        cleaned_data = data.where((data >= lower_bound), lower_bound)
        cleaned_data = data.where((data <= upper_bound), upper_bound)

        modified_df.loc[:,col] = cleaned_data

    return modified_df


def load_and_clean():

    SCRIPT_DIR = Path(__file__).resolve().parent

    data_path = SCRIPT_DIR.parent/'data'/'Data_Cortex_Nuclear.xls'
    
    output_path = SCRIPT_DIR.parent /'data'/'processed_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_excel(data_path)
        print(f"Data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: One or more files were not found. Please check paths.")


    processed_df = preprocess(df)

    print(processed_df.info())

    try:
        processed_df.to_csv(output_path, index=False) 
        print(f"\nSuccessfully exported processed data to: {output_path}")

    except Exception as e:
        print(f"\nError exporting data: {e}")

if __name__ == "__main__":
    load_and_clean() 



















    