import json
import pandas as pd
import numpy as np
import os
import glob
from functools import reduce

def xlsx_to_csv(read_path, write_path):
    xlsx_files = glob.glob(read_path + "*.xlsx")
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file, sheet_name = 'Complete responses')
        df = clean_data(df)
        df = add_region_column(df)
        df = add_score_columns(df)
        df.to_csv(write_path + xlsx_file.split('/')[-1].split('.')[0] + '.csv', index = False)

def clean_data(df):
    df.dropna(subset=['Gender', 'Age', 'Geography'])
    df = df[df.Geography != 'Unknown']
    return df

def add_region_column(df):
    df["Region"] = ""
    for index,row in df.iterrows():
        df.at[index, "Region"] = df.at[index, "Geography"].split('-')[1];
    return df

def add_score_columns(df):
    for index,row in df.iterrows():
        # Question 1
        # Click on all those make you worried about the future.
        chosen_item = 0
        for column in ['Question #1 Answer 1: Genetic engineering',
                        'Question #1 Answer 2: Brain implants',
                        'Question #1 Answer 3: Synthetic materials',
                        'Question #1 Answer 4: Artificial intelligence',
                        'Question #1 Answer 5: Quantum computing',
                        'Question #1 Answer 6: Nuclear power']:
            if not pd.isna(df.at[index, column]):
                chosen_item += 1
        q1_score = 10 - chosen_item * 10 / 6
        df.at[index, "Q1_Score"] = q1_score

        # Question 2
        # Click on all those make you worried about the future.
        chosen_item = 0
        for column in ['Question #2 Answer 1: New vaccine development',
                    'Question #2 Answer 2: Military drones',
                    'Question #2 Answer 3: Self-driving cars',
                    'Question #2 Answer 4: Satellite internet service',
                    'Question #2 Answer 5: Robotics',
                    'Question #2 Answer 6: Facial recognition algorithms']:
            if not pd.isna(df.at[index, column]):
                chosen_item += 1
        q2_score = 10 - chosen_item * 10 / 6
        df.at[index, "Q2_Score"] = q2_score

        # Question 3
        # Click all of the following technologies you consider yourself informed about.
        chosen_item = 0
        for column in ['Question #3 Answer 1: Genetic engineering',
                        'Question #3 Answer 2: Brain implants',
                        'Question #3 Answer 3: Synthetic materials',
                        'Question #3 Answer 4: Artificial intelligence',
                        'Question #3 Answer 5: Quantum computing',
                        'Question #3 Answer 6: Nuclear power']:
            if not pd.isna(df.at[index, column]):
                chosen_item += 1
        q3_score = chosen_item * 10 / 6
        df.at[index, "Q3_Score"] = q3_score

        # Question 4
        # Click all of the following technologies you consider yourself informed about.
        chosen_item = 0
        for column in ['Question #4 Answer 1: New vaccine development',
                    'Question #4 Answer 2: Military drones',
                    'Question #4 Answer 3: Self-driving cars',
                    'Question #4 Answer 4: Satellite internet service',
                    'Question #4 Answer 5: Robotics',
                    'Question #4 Answer 6: Facial recognition algorithms']:
            if not pd.isna(df.at[index, column]):
                chosen_item += 1
        q4_score = chosen_item * 10 / 6
        df.at[index, "Q4_Score"] = q4_score

        # Question 5
        # How worried are you about artificial intelligence?
        q5_score = (7 - df.at[index, 'Question #5 Answer']) * 10 / 6
        df.at[index, "Q5_Score"] = q5_score

        # Question 6
        # How worried are you about self-driving cars?
        q6_score = (7 - df.at[index, 'Question #6 Answer']) * 10 / 6
        df.at[index, "Q6_Score"] = q6_score

        # Question 7
        # How worried are you about recognition algorithms?
        q7_score = (7 - df.at[index, 'Question #7 Answer']) * 10 / 6
        df.at[index, "Q7_Score"] = q7_score

        # Question 8
        # All in all, most technology provides more benefits than drawbacks
        q8_score = (df.at[index, 'Question #8 Answer'] - 1) * 10 / 6
        df.at[index, "Q8_Score"] = q8_score

        # Question 9: Developing new technologies creates more danger to society
        q9_score = (7 - df.at[index, 'Question #9 Answer']) * 10 / 6
        df.at[index, "Q9_Score"] = q9_score

        # Question 10: name one other technology that you are concerned about
        q10_score = 0
        q10_answer = df.at[index, 'Question #10 Synonym Group']
        if pd. isna(q10_answer):
            q10_score = 10
        else:
            matches = ['None', 'Na', 'No', 'Not really anymore', 'Nothing', "isn't", "not really"]
            if any(x in q10_answer for x in matches):
                q10_score = 10
        df.at[index, "Q10_Score"] = q10_score

        total_score = q1_score + q2_score + q3_score + q4_score + q5_score + q6_score + q7_score + q8_score + q9_score + q10_score
        df.at[index, "Total_Score"] = total_score;
    return df

def get_df_list(path):
    df_list = []
    for df_path in glob.glob(path + '*.csv'):
        df = pd.read_csv(df_path)
        df_list.append(df)
    df_list.sort(key=lambda df: sorted(df['Time (UTC)'].tolist())[len(df['Time (UTC)'].tolist())//2])
    return df_list

def create_trend_dataframe(df_list):
    trend_df = pd.DataFrame()
    for df in df_list:
        curr_df = pd.DataFrame()
        curr_df['Time'] = [pd.to_datetime(df['Time (UTC)']).dt.normalize().tolist()[len(df['Time (UTC)'].tolist())//2]]
        curr_df['Time'] = curr_df['Time'].dt.date.astype(str)
        curr_df['Q1_Score'] = [df['Q1_Score'].mean()]
        curr_df['Q2_Score'] = [df['Q2_Score'].mean()]
        curr_df['Q3_Score'] = [df['Q3_Score'].mean()]
        curr_df['Q4_Score'] = [df['Q4_Score'].mean()]
        curr_df['Q5_Score'] = [df['Q5_Score'].mean()]
        curr_df['Q6_Score'] = [df['Q6_Score'].mean()]
        curr_df['Q7_Score'] = [df['Q7_Score'].mean()]
        curr_df['Q8_Score'] = [df['Q8_Score'].mean()]
        curr_df['Q9_Score'] = [df['Q9_Score'].mean()]
        curr_df['Q10_Score'] = [df['Q10_Score'].mean()]
        curr_df['Total_Score'] = [df['Total_Score'].mean()]
        trend_df = trend_df.append(curr_df)
    return trend_df

def create_trend_dataframe_breakdown_gender(df_list):
    df_trend_by_gender = pd.DataFrame()
    for df in df_list:
        curr_df = pd.DataFrame()
        curr_df['Time'] = [pd.to_datetime(df['Time (UTC)']).dt.normalize().tolist()[len(df['Time (UTC)'].tolist())//2]]
        curr_df['Time'] = curr_df['Time'].dt.date.astype(str)
        curr_df['Male'] = [df.loc[df.Gender =='Male']['Total_Score'].mean()]
        curr_df['Female'] = [df.loc[df.Gender =='Female']['Total_Score'].mean()]
        df_trend_by_gender = df_trend_by_gender.append(curr_df)
    return df_trend_by_gender

def create_trend_dataframe_breakdown_age(df_list):
    df_trend_by_age = pd.DataFrame()
    for df in df_list:
        curr_df = pd.DataFrame()
        curr_df['Time'] = [pd.to_datetime(df['Time (UTC)']).dt.normalize().tolist()[len(df['Time (UTC)'].tolist())//2]]
        curr_df['Time'] = curr_df['Time'].dt.date.astype(str)
        curr_df['age_18_24'] = [df.loc[df.Age =='18-24']['Total_Score'].mean()]
        curr_df['age_25_34'] = [df.loc[df.Age =='25-34']['Total_Score'].mean()]
        curr_df['age_35_44'] = [df.loc[df.Age =='35-44']['Total_Score'].mean()]
        curr_df['age_45_54'] = [df.loc[df.Age =='45-54']['Total_Score'].mean()]
        curr_df['age_55_64'] = [df.loc[df.Age =='55-64']['Total_Score'].mean()]
        curr_df['age_65_plus'] = [df.loc[df.Age =='65+']['Total_Score'].mean()]
        df_trend_by_age = df_trend_by_age.append(curr_df)
    return df_trend_by_age

def create_trend_dataframe_breakdown_region(df_list):
    df_trend_by_region = pd.DataFrame()
    for df in df_list:
        curr_df = pd.DataFrame()
        curr_df['Time'] = [pd.to_datetime(df['Time (UTC)']).dt.normalize().tolist()[len(df['Time (UTC)'].tolist())//2]]
        curr_df['Time'] = curr_df['Time'].dt.date.astype(str)
        curr_df['ne'] = [df.loc[df.Region =='NORTHEAST']['Total_Score'].mean()]
        curr_df['mw'] = [df.loc[df.Region =='MIDWEST']['Total_Score'].mean()]
        curr_df['west'] = [df.loc[df.Region =='WEST']['Total_Score'].mean()]
        curr_df['south'] = [df.loc[df.Region =='SOUTH']['Total_Score'].mean()]
        df_trend_by_region = df_trend_by_region.append(curr_df)
    return df_trend_by_region

def merge_into_single_file(dfs):
    return pd.concat(dfs, ignore_index=True)

def get_average_scores_for_each_age_group(df):
    dfs = []
    Questions = ['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score', 'Q5_Score',
                'Q6_Score', 'Q7_Score', 'Q8_Score', 'Q9_Score', 'Q10_Score', 'Total_Score']
    for question in Questions:
        df_curr = df.groupby('Age', as_index = False)[question].mean()
        dfs.append(df_curr)

    return reduce(lambda left,right: pd.merge(left,right, on = 'Age'), dfs)

def get_average_scores_for_each_region_group(df):
    dfs = []
    Questions = ['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score', 'Q5_Score',
                'Q6_Score', 'Q7_Score', 'Q8_Score', 'Q9_Score', 'Q10_Score', 'Total_Score']
    for question in Questions:
        df_curr = df.groupby('Region', as_index = False)[question].mean()
        dfs.append(df_curr)

    return reduce(lambda left,right: pd.merge(left,right, on = 'Region'), dfs)
