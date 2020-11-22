import json
from sklearn import metrics, decomposition, manifold
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist

from flask import Flask, render_template, request, redirect, Response, jsonify


import pandas as pd
import numpy as np
import os
import glob
from functools import reduce


#First of all you have to import it from the flask module:
app = Flask(__name__)
#By default, a route only answers to GET requests. You can use the methods argument of the route() decorator to handle different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    global  moible_trend_df, mobile_avg_scores_by_age_df, mobile_avg_scores_by_region_df, df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    #The current request method is available by using the method attribute
    if request.method == 'POST':
        if request.form['data'] == 'ran_df':
            data = y_ran
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'raw_df':
            data = y_raw
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'y_str':
            data = y_str
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

    # trend_data = moible_trend_df
    main_chart_data = moible_trend_df.to_dict(orient='records')
    main_chart_data = json.dumps(main_chart_data, indent=2)
    mobile_trend_data = {'chart_data': main_chart_data}

    chart_data_sub1_age = mobile_avg_scores_by_age_df.to_dict(orient='records')
    chart_data_sub1_age = json.dumps(chart_data_sub1_age, indent=2)
    trend_by_age_data = {'chart_data': chart_data_sub1_age}

    chart_data_sub1_region = mobile_avg_scores_by_region_df.to_dict(orient='records')
    chart_data_sub1_region = json.dumps(chart_data_sub1_region, indent=2)
    trend_by_region_data = {'chart_data': chart_data_sub1_region}

    return render_template("index.html",
                            mobile_trend_data = mobile_trend_data,
                            trend_by_age_data = trend_by_age_data,
                            trend_by_region_data = trend_by_region_data)

@app.route("/test", methods = ['POST', 'GET'])
def index1():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    #The current request method is available by using the method attribute
    if request.method == 'POST':
        if request.form['data'] == 'ran_df':
            data = y_ran
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'raw_df':
            data = y_raw
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'y_str':
            data = y_str
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

    data = y_str
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)

#Two PCA vectors ploting
@app.route("/2dplot", methods = ['POST', 'GET'])
def _2dplot():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_str)
    principal2_df = pd.DataFrame(data=principalComponents,columns=['PC1','PC2'])
    principal2_df['price'] = target

    data = principal2_df
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("2dplot.html", data=data)


#MDS 2D Plots using Euclidean and correlation distances
@app.route("/mds", methods = ['POST', 'GET'])
def mds():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df_e,mds_df_c
    if request.method == 'POST':
        if request.form['data'] == 'mds_df_c':
            data = mds_df_c
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)
        if request.form['data'] == 'mds_df_e':
            data = mds_df_e
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

    #(1)Euclidean distance
    #return a dataframe
    mds_df_e = pd.DataFrame()
    mds_e = MDS(n_components=2, dissimilarity='euclidean')
    mds_e = mds_e.fit_transform(x_str)
    mds_e = pd.DataFrame(mds_e)
    mds_df_e['x'] = mds_e[0]
    mds_df_e['y'] = mds_e[1]
    mds_df_e['price'] = target

    #(2)Correlation distance
    #return a dataframe
    x_str_df = pd.DataFrame(x_str)
    x_str_df = x_str_df.transpose()
    cor_matrix = x_str_df.corr()
    for col in cor_matrix.columns:
        cor_matrix[col].values[:] = 1 - cor_matrix[col].values[:]
    mds_c = MDS(n_components=2, dissimilarity='precomputed')
    mds_df_c = mds_c.fit_transform(cor_matrix)
    mds_df_c = pd.DataFrame(mds_df_c)
    mds_df_c['x'] = mds_df_c[0]
    mds_df_c['y'] = mds_df_c[1]
    mds_df_c['price'] = target

    data = mds_df_e
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("mds.html", data=data)


#Three highest loading attributes plot matrix
@app.route("/attri_matrix", methods = ['POST', 'GET'])
def attri_matrix():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    mainAttri_df = strati_df[['MedianListingPrice_SingleFamilyResidence','MedianListingPrice_2Bedroom','MedianListingPrice_3Bedroom','price']]

    data = mainAttri_df
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("attri_matrix.html", data=data)





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
        # curr_df.timestamp.dt.strftime('%Y-%m-%d')
        # curr_df['Time'] = [df['Time (UTC)'].tolist()[len(df['Time (UTC)'].tolist())//2]]
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

def create_trend_dataframe_breakdown_age(df_list):
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

if __name__ == "__main__":
    xlsx_path_mobile = 'data/mobile-xlsx/'
    csv_path_mobile = 'data/mobile-csv/'

    xlsx_path_pub = 'data/pubNetwork-xlsx/'
    csv_path_pub = 'data/pubNetwork-csv/'

    xlsx_to_csv(xlsx_path_mobile, csv_path_mobile)
    xlsx_to_csv(xlsx_path_pub, csv_path_pub)

    mobile_dfs = get_df_list(csv_path_mobile)
    pubNetwork_dfs = get_df_list(csv_path_pub)

    moible_trend_df = create_trend_dataframe(mobile_dfs)
    print(moible_trend_df)
    mobile_all_sample_df = merge_into_single_file(mobile_dfs)
    mobile_avg_scores_by_age_df = get_average_scores_for_each_age_group(mobile_all_sample_df)
    mobile_avg_scores_by_region_df = get_average_scores_for_each_region_group(mobile_all_sample_df)
    # print(mobile_avg_scores_by_region_df.head())


    df = pd.read_csv('AmericanHousingPrice.csv')
    dataClean = df.fillna(df.mean())
    no_ch_df = dataClean.drop(['Date','RegionName'], axis=1)

    #Random sampling
    random_df = dataClean.sample(frac=0.5)

    #Kmeans clustering, Elbow has found the optimal k is 5
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(no_ch_df)
        kmeanModel.fit(no_ch_df)
        distortions.append(sum(np.min(cdist(no_ch_df,kmeanModel.cluster_centers_,
        'euclidean'),axis=1))/no_ch_df.shape[0])

    kmeans=KMeans(n_clusters=5, max_iter=500)
    kmeans.fit(no_ch_df)

    strati_df = pd.DataFrame()
    n_clusters = 5
    persnt = 0.5
    for i in range(n_clusters):
        Clstr_i = np.where(kmeans.labels_ == i)[0].tolist()
        num_i = len(Clstr_i)
        sample_i = np.random.choice(Clstr_i, int(persnt*num_i))
        i_cluster_df = no_ch_df.loc[sample_i]
        strati_df = pd.concat([strati_df,i_cluster_df],axis = 0)

    #PCA
    #devide the midean housing price into 5 groups according to the price level
    def function(a):
        if a < 180000: return 'very low-price housing'
        if a>=180000 and a< 250000: return 'low-price housing'
        if a>=250000 and a< 380000: return 'medium-price housing'
        if a>=380000 and a< 520000: return 'high-price housing'
        else: return 'very high-price housing'
    #add the target column to the stritified data frame
    strati_df['price'] = strati_df.apply(lambda x: function(x.MedianListingPrice_AllHomes), axis = 1)
    target = strati_df.loc[:,['price']].values

    rawdat_df2 = dataClean.drop(['Date','RegionName','MedianListingPrice_AllHomes'],axis=1)
    strati_df2 = strati_df.drop(['MedianListingPrice_AllHomes','price'],axis=1)
    random_df2 = random_df.drop(['Date','RegionName','MedianListingPrice_AllHomes'],axis=1)
    x_raw = StandardScaler().fit_transform(rawdat_df2)
    x_str = StandardScaler().fit_transform(strati_df2)
    x_ran = StandardScaler().fit_transform(random_df2)
    rawdat_pca = decomposition.PCA()
    strati_pca = decomposition.PCA()
    random_pca = decomposition.PCA()
    rawdat_pca.fit(x_raw)
    strati_pca.fit(x_str)
    random_pca.fit(x_ran)
    y_raw = pd.DataFrame()
    y_str = pd.DataFrame()
    y_ran = pd.DataFrame()
    y_raw['variance'] = rawdat_pca.explained_variance_
    y_str['variance'] = strati_pca.explained_variance_
    y_ran['variance'] = random_pca.explained_variance_

    #Obtain 3 top-loading attributes
    #My data has 4 intrinsic dimensions
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(x_str)
    loading_df=pca.components_.T * np.sqrt(pca.explained_variance_)
    attributeName=pd.DataFrame()
    attributeName['VariableName']=strati_df2.columns.values
    significanceValues=pd.DataFrame(data=loading_df,columns=['PC1','PC2','PC3','PC4'])
    significance_df=pd.concat([attributeName,significanceValues],sort=True,axis=1)
    significance = significance_df.drop(['VariableName'],axis=1)
    significance_df['SumOfSquaredLoadings']=significance\
    .apply(lambda x:np.sqrt(np.square(x['PC1'])+np.square(x['PC2'])+np.square(x['PC3'])+np.square(x['PC4'])),axis=1)
    s_df_sort = significance_df.sort_values(by=['SumOfSquaredLoadings'],ascending=False)



    app.run(debug=True)
