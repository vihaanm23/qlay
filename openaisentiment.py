from openai import OpenAI
import pandas as pd

client = OpenAI(api_key='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')

df = pd.read_csv('reviews-data.csv', delimiter=',')
reviews = df[['en_content']]
ages = df[['reviewerAge']]
ages = ages.values.tolist()
reviews = reviews.values.tolist()

""" def analyze_sentiment(review, age):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=(f"Please analyze the sentiment in a few sentences of the following amazon review based on the content of the review and taking into account the age of the reviewer:{review}, age={age}"),
        temperature=0.5,
            max_tokens=1024,
            n = 1,
            stop=None
    )
    sentiment = response.choices[0].text.strip()
    return sentiment """

def analyze_sentiment(review, age):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"User: Please analyze the sentiment in a few sentences of the following amazon review based on the content of the review and taking into account the age of the reviewer:\n{review}, age={age}"}],
        max_tokens=1024, 
        temperature=0.5, 
        n = 1
    )
    sentiment = response.choices[0].message.content
    return sentiment



def analyze_amazon_reviews(reviews, ages):
    sentiments = []
    for x in range(10):
        review1 = reviews[x][0]
        age = ages[x][0]
        review1 = review1.replace("39", " ")
        sentiment = analyze_sentiment(review1, age)
        sentiments.append(sentiment)
    return sentiments

# Analyze sentiments
sentiments = analyze_amazon_reviews(reviews, ages)

# Print sentiments
for i, sentiment in enumerate(sentiments, 1):
    print(f"Review {i}: {sentiment}")

