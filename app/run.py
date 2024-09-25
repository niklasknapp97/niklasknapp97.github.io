import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.express as px
import plotly.graph_objects as graph
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Load the data
    data = df

    # Devide the columns
    help_columns = ["medical_help", "medical_products", "search_and_rescue", "security", "military", "child_alone", "water", "food", "shelter", "clothing", "money", "missing_people", "refugees", "death", "other_aid"]
    infrastructure_columns = ["transport", "buildings", "electricity", "tools", "hospitals", "shops", "aid_centers", "other_infrastructure"]
    weather_columns = ['floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather']
    
    # 1. Distribution of aid requests (Barchart)
    help_counts = data[help_columns].sum().reset_index()
    help_counts.columns = ['Aid type', 'Number of requests']
    help_counts = help_counts.sort_values('Number of requests', ascending=False)
    
    bar_fig_aid = px.bar(help_counts, x='Aid type', y='Number of requests', 
                     title='Distribution of aid requests', 
                     color='Number of requests', 
                     labels={'Number of requests': 'Number', 'Aid type': 'Aid type'})
    
    # 2. Distribution of infrastructure impact (Barchart)
    infrastructure_counts = data[infrastructure_columns].sum().reset_index()
    infrastructure_counts.columns = ['Infrastructure impact', 'Number of requests']
    infrastructure_counts = infrastructure_counts.sort_values('Number of requests', ascending=False)
    
    bar_fig_infra = px.bar(infrastructure_counts, x='Infrastructure impact', y='Number of requests', 
                     title='Distribution of infrastructure impact', 
                     color='Number of requests', 
                     labels={'Number of requests': 'Number', 'Infrastructure impact': 'Infrastructure impact'})
    
    # 3. Distribution of weather (Barchart)
    weather_counts = data[weather_columns].sum().reset_index()
    weather_counts.columns = ['Weather event', 'Number of requests']
    weather_counts = weather_counts.sort_values('Number of requests', ascending=False)
    
    bar_fig_weather = px.bar(weather_counts, x='Weather event', y='Number of requests', 
                     title='Distribution of weather events', 
                     color='Number of requests', 
                     labels={'Number of requests': 'Number', 'Weather event': 'Weather event'})
    
    # 4. Frequency of aid requests by message type (Pie chart)
    message_type_counts = data['genre'].value_counts().reset_index()
    message_type_counts.columns = ['Message type', 'Count']
    
    pie_fig = px.pie(message_type_counts, values='Count', names='Message type', 
                     title='Frequency of aid requests by message type',
                     color='Message type')

    # 5. Correlation matrix of aid requests (Heatmap)
    help_columns = ["medical_help", "medical_products", "search_and_rescue", "security", "military", "water", "food", "shelter", "clothing", "money", "missing_people", "refugees", "death", "other_aid"]
    correlation_matrix = data[help_columns].corr()
    
    # Create the heatmap
    help_heatmap = graph.Figure(data=graph.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'))

    help_heatmap.update_layout(title='Correlation matrix of aid requests')

    # 6. Heatmap of aid types vs. weather events
    # Create an aggregated table of sums for each combination of weather and aid type
    weather_help = data[weather_columns].T.dot(data[help_columns])

    # Create a heatmap for the correlations between weather events and aid types
    help_weather_heatmap = graph.Figure(data=graph.Heatmap(
        z=weather_help.values,
        x=weather_help.columns,
        y=weather_help.index,
        colorscale='Viridis'
    ))

    help_weather_heatmap.update_layout(
        title='Correlation between weather events and aid types',
        xaxis_title='Aid types',
        yaxis_title='Weather events'
    )

    # 7. Heatmap of aid types vs. infrastructure impact
    # Create an aggregated table of sums for each combination of infrastructure and aid type
    infrastructure_help = data[infrastructure_columns].T.dot(data[help_columns])

    # Create a heatmap for the correlations between infrastructure and aid types
    help_infrastructure_heatmap = graph.Figure(data=graph.Heatmap(
        z=infrastructure_help.values,
        x=infrastructure_help.columns,
        y=infrastructure_help.index,
        colorscale='Viridis'
    ))

    help_infrastructure_heatmap.update_layout(
        title='Correlation between infrastructure impact and aid types',
        xaxis_title='Aid types',
        yaxis_title='Infrastructure impact'
    )

    # Create the visualizations
    graphs = [bar_fig_aid, bar_fig_infra, bar_fig_weather, pie_fig, help_heatmap, help_weather_heatmap, help_infrastructure_heatmap]
    
    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render the webpage with Plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()