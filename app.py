import json
from flask import Flask, render_template, request, redirect, Response, jsonify
import glob
import data_transformer as transformer

# Import it from the flask module:
app = Flask(__name__)

#a route handles different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    global  mobile_df, mobile_gender_df, mobile_age_df, mobile_region_df

    def process_df_to_data(df):
        data = df
        chart_data = data.to_dict(orient='records')
        chart_data = json.dumps(chart_data, indent=2)
        data = {'chart_data': chart_data}
        return data

    #The current request method is available by using the method attribute
    if request.method == 'POST':
        if request.form['data'] == 'mobile_gender_df':
            return jsonify(process_df_to_data(mobile_gender_df))

        if request.form['data'] == 'mobile_age_df':
            return jsonify(process_df_to_data(mobile_age_df))

        if request.form['data'] == 'mobile_region_df':
            return jsonify(process_df_to_data(mobile_region_df))

        if request.form['data'] == 'mobile_df':
            return jsonify(process_df_to_data(mobile_df))

    return render_template("index.html", mobile_data = process_df_to_data(mobile_df))

# Main function
if __name__ == "__main__":
    xlsx_path_mobile = 'data/mobile-xlsx/'
    csv_path_mobile = 'data/mobile-csv/'

    xlsx_path_pub = 'data/pubNetwork-xlsx/'
    csv_path_pub = 'data/pubNetwork-csv/'

    # Convert xlsx file to csv
    transformer.xlsx_to_csv(xlsx_path_mobile, csv_path_mobile)
    transformer.xlsx_to_csv(xlsx_path_pub, csv_path_pub)

    mobile_dfs = transformer.get_df_list(csv_path_mobile)
    pubNetwork_dfs = transformer.get_df_list(csv_path_pub)

    mobile_df = transformer.create_trend_dataframe(mobile_dfs)
    mobile_gender_df = transformer.create_trend_dataframe_breakdown_gender(mobile_dfs)
    mobile_age_df = transformer.create_trend_dataframe_breakdown_age(mobile_dfs)
    print(mobile_age_df.head())
    mobile_region_df = transformer.create_trend_dataframe_breakdown_region(mobile_dfs)
    print(mobile_region_df.head())
    # mobile_all_sample_df = merge_into_single_file(mobile_dfs)
    # mobile_avg_scores_by_age_df = get_average_scores_for_each_age_group(mobile_all_sample_df)
    # mobile_avg_scores_by_region_df = get_average_scores_for_each_region_group(mobile_all_sample_df)
    # print(mobile_avg_scores_by_region_df.head())
    app.run(debug=True)
