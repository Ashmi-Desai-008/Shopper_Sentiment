import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('TeePublic_review.csv', encoding='ISO-8859-1')  # Replace 'your_dataset.csv' with the actual filename
    return data

# Main function
def main():
    st.title('Review Analysis')
    
    # Load data
    data = load_data()

    if data is not None:
        # Display the dataset
        st.write('### Dataset')
        st.write(data)

        # Sidebar filters
        st.sidebar.title('Filters')
        selected_month = st.sidebar.selectbox('Select Month', ['All'] + data['month'].unique().tolist())
        selected_year = st.sidebar.selectbox('Select Year', ['All'] + data['year'].unique().tolist())
        selected_rating = st.sidebar.selectbox('Select Rating', ['All', 1, 2, 3, 4, 5])

        # Apply filters
        filtered_data = data.copy()
        if selected_month != 'All':
            filtered_data = filtered_data[filtered_data['month'] == selected_month]
        if selected_year != 'All':
            filtered_data = filtered_data[filtered_data['year'] == selected_year]
        if selected_rating != 'All':
            filtered_data = filtered_data[filtered_data['review-label'] == selected_rating]

        # Display filtered dataset
        st.write('### Filtered Dataset')
        st.write(filtered_data)

        # Data Visualization
        st.write('### Data Visualization')

        # Distribution of ratings
        st.write('#### Distribution of Ratings')
        rating_counts = filtered_data['review-label'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Ratings')
        st.pyplot(fig)

        # Interactive map of store locations
        st.write('#### Store Locations')
        st.map(filtered_data[['latitude', 'longitude']])

        # Word cloud of review text
        st.write('#### Word Cloud of Reviews')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_data['review'].astype(str)))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Machine Learning Model
        st.write('### Machine Learning Model')

        # Select features and target variable
        features = ['latitude', 'longitude']  # Add other relevant features
        target = 'review-label'

        # Prepare data
        X = filtered_data[features]
        y = filtered_data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy:.2f}')

        # Display classification report
        st.write('#### Classification Report')
        st.write(classification_report(y_test, y_pred))

        # Feature importance
        st.write('#### Feature Importance')
        feature_importance = pd.Series(model.feature_importances_, index=features)
        st.bar_chart(feature_importance)

        # Cross-validation
        st.write('#### Cross-validation')
        cv_scores = cross_val_score(model, X, y, cv=5)
        st.write(f'Mean CV accuracy: {np.mean(cv_scores):.2f}')

    else:
        st.error("Failed to load dataset.")

if __name__ == '__main__':
    main()
