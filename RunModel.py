from transformers import pipeline, BertForSequenceClassification, BertTokenizer
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import display
import torch

model_path = #you're model path
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Our Model
sentiment_model = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Function to analyze a single message
def analyze_single_message(message):
    result = sentiment_model(message)
    sentiment = result[0]['label']
    score = result[0]['score']

    if sentiment == "LABEL_2":
        sentiment = "POSITIVE"

    elif sentiment == "LABEL_0":
        sentiment = "NEGATIVE"

    elif sentiment == "LABEL_1":
        sentiment = "NEUTRAL"

    return sentiment, score

# Function to analyze a dataset
def analyze_dataset(data_path, text_column):
    sentiment_counts = {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}  # Adjust labels as per your training
    sentiments = []
    confidences = []  # To store confidence scores

    # Load dataset
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty. Please provide a valid dataset.")
        return None, None

    if text_column not in data.columns:
        print(f"Error: Column '{text_column}' not found in the dataset.")
        return None, None
    
    for text in tqdm(data[text_column], desc="Analyzing Sentiments"):
        result = sentiment_model(text)
        label = result[0]['label']
        score = result[0]['score']
        if label == "LABEL_2":
            label = "POSITIVE"

        elif label == "LABEL_0":
            label = "NEGATIVE"

        elif label == "LABEL_1":
            label = "NEUTRAL"
        sentiments.append(label)
        confidences.append(score)
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
    
    data['Sentiment'] = sentiments
    data['Confidence'] = confidences  # Add confidence scores to the dataset
    stats = pd.DataFrame(sentiment_counts.items(), columns=['Sentiment', 'Count'])
    return data, stats

# New function to provide recommendations
def provide_recommendations(data, text_column, recommendation_column, sentiment_column="Sentiment", top_n=5):
    # Check if the sentiment column exists
    if sentiment_column not in data.columns:
        print(f"Sentiment column '{sentiment_column}' not found. Running sentiment analysis...")
        
        # Perform sentiment analysis on the text column
        sentiments = []
        scores = []  # To store confidence scores
        for text in tqdm(data[text_column], desc="Analyzing Sentiments"):
            result = sentiment_model(text)
            label = result[0]['label']
            score = result[0]['score']
            if label == "LABEL_2":
                label = "POSITIVE"
            elif label == "LABEL_0":
                label = "NEGATIVE"
            elif label == "LABEL_1":
                label = "NEUTRAL"
            sentiments.append(label)
            scores.append(score)
        
        data[sentiment_column] = sentiments
        data['Confidence'] = scores  # Add confidence scores to the dataset
    
    # Filter data for positive sentiments
    positive_data = data[data[sentiment_column] == "POSITIVE"]

    # Calculate the weighted score for each recommendation (frequency * average confidence)
    recommendation_stats = (
        positive_data.groupby(recommendation_column)
        .agg(Frequency=('Confidence', 'size'), AvgConfidence=('Confidence', 'mean'))
    )
    recommendation_stats['WeightedScore'] = recommendation_stats['Frequency'] * recommendation_stats['AvgConfidence']

    # Sort recommendations by weighted score
    sorted_recommendations = recommendation_stats.sort_values(by='WeightedScore', ascending=False).head(top_n)

    print("Top Recommendations:")
    for i, (recommendation, stats) in enumerate(sorted_recommendations.iterrows(), 1):
        print(f"{i}. {recommendation} (Positive mentions: {stats['Frequency']}, Avg Confidence: {stats['AvgConfidence']:.2f}, Weighted Score: {stats['WeightedScore']:.2f})")

# Main function for user interaction
def main():
    End = False
    print("Welcome to the Sentiment Analysis Bot!")

    while End == False:
        try:
            choice = input("Choose an option: \n1. Analyze Single Message\n2. Analyze Dataset\n3. Get Recommendations\n4. Exit\n")
            if choice == '1':
                message = input("Enter the message: ")
                sentiment, score = analyze_single_message(message)
                print(f"Sentiment: {sentiment}, Confidence: {score:.2f}")
            
            elif choice == '2':
                file_path = input("Enter the path to the dataset (CSV format): ")
                text_column = input("Enter the column name containing the text: ")
                data, stats = analyze_dataset(file_path, text_column)
                if data is not None and stats is not None:
                    print("\nSentiment Statistics:")
                    print(stats)
                    save = input("Do you want to save the results? (yes/no): ").lower()
                    if save == 'yes':
                        save_path = "sentiment_results.csv"
                        data.to_csv(save_path, index=False)
                        print(f"Results saved to {save_path}")
            elif choice == '3':
                file_path = input("Enter the path to the dataset (CSV format): ")
                text_column = input("Enter the column name containing the text: ")
                recommendation_column = input("Enter the column name for recommendations (e.g., Product or Movie name): ")
                sentiment_column = input("Enter the column name for sentiment (or leave blank to analyze it now): ")
                top_n = int(input("How many recommendations would you like?: "))

                # Load dataset
                try:
                    data = pd.read_csv(file_path)
                except FileNotFoundError:
                    print("Error: File not found. Please check the file path and try again.")
                    continue
                except pd.errors.EmptyDataError:
                    print("Error: The file is empty. Please provide a valid dataset.")
                    continue

                if not sentiment_column:
                    sentiment_column = "Sentiment"  # Default column for calculated sentiments

                try:
                    provide_recommendations(data, text_column, recommendation_column, sentiment_column, top_n)
                except KeyError as e:
                    print(f"Error: {e}. Please ensure the columns exist in the dataset.")
                    continue
            
            elif choice == '4':
                End = True
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.")

# Run the main function
if __name__ == "__main__":
    main()
