# Sentiment-analysis with BERT-uncased
 Our project revolves around sentiment analysis of text, so for our datasets we relied mostly on sentiment  of language rather than a specific topic or study. This meant we could use a wide range of entries for our  dataset, so we used 2 datasets, one with generic statements with the sentient, and another of movie  reviews.

 Main functions of our project were to analyse a dataset and modify it to have sentiments for each entry,
 and to provide a recommendation system for a dataset which has already been analysed for its sentiments.
 Our model would analyse each entry to determine its sentiment and provide a given confidence rating to
 it, the confidence rating alongside the sentiment itself (if it was positive, negative, or neutral) would be
 weighted depending on the topic of the dataset.
 For example if the given dataset contained social media posts regarding people’s opinions on movies it
 would also require to have the Movie’s name in the dataset, then our recommendation system would
 determine the highest weighted score based on which movie received the most amount of positive entries,
 along with the highest confidence rating.

Datasets used: 
- https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
- https://www.kaggle.com/datasets/krystalliu152/imbd-movie-reviewnpl
