# Importing pandas to read in csv dataset
import pandas as pd

# Importing SpaCy and TextBlob as NLP tools
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Applying the SpaCy pipeline to a variable
nlp = spacy.load('en_core_web_sm') 

# Adding TextBlob to the pipeline to help with sentiment analysis
nlp.add_pipe("spacytextblob")

# Data set read in and applied to a variable
df = pd.read_csv(r'C:\Users\reece\OneDrive\Documents\myCoursespace\Data Science (Fundamentals)\T26 - Natural Language Processing with SpaCy\1429_1.csv', low_memory=False)

# One column selected from the dataframe, I have applied text 
# cleaning to the column text and dropped any empty values
reviews_data = df['reviews.text'].str.lower().str.strip().dropna()

# NLP is applied to sample review data chosen from the reviews variable
# The proccessed data is applied to a variable
sample_review_one = nlp(reviews_data[21634])
sample_review_two = nlp(reviews_data[3129])
sample_review_three = nlp(reviews_data[34029])
sample_review_four = nlp(reviews_data[21681])
sample_review_five = nlp(reviews_data[23176])

# Function created to return a review sentiment score for the chosen arguement variable
def sentiment(review):
    return print(review._.blob.sentiment)

# List created to hold the sample reviews
# Counter for review list iterations
reviews_list = [sample_review_one, sample_review_two, sample_review_three, sample_review_four, sample_review_five]
reviews_count = 0

# Title header for Sentiment Analysis report
print('Sentiment Analysis On 5 Sample Amazon Product Reviews\n')

# For loop to iterate over review list and print the review text with it's sentiment score
for i in reviews_list:
    reviews_count += 1
    print(f'- Sample Review {reviews_count}:\n{i}\n\nSentiment Score:')
    sentiment(i)
    print(f'')

# Title header for Similarity report
print(f'Similarity Score For The 5 Sample Reviews against sample_review_three\n\nSample Review 3: {sample_review_three}\n')

model_sentence = nlp(reviews_data[34029])
for i in reviews_list:
    similarity = nlp(i).similarity(model_sentence)
    print(f"{i}\nSimilarity Score: {similarity}")