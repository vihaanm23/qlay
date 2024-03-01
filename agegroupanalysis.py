from openai import OpenAI
import pandas as pd

numReviews = 1000 #Change to however many reviews you want to analyze 

client = OpenAI(api_key='') 


age_groups = [10,20,30,40,50,60]

age_group_likes = {age_group: set() for age_group in age_groups}

age_group_dislikes = {age_group: set() for age_group in age_groups}

df = pd.read_csv('reviews-data.csv', delimiter=',')
reviews = df[['en_content']]
ages = df[['reviewerAge']]
ages = ages.values.tolist()
reviews = reviews.values.tolist()

def analyze_sentiment(review, age):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": 
        f"User: In the following amazon review of a shampoo product, Point out one feature of the product that the user likes, and one feature of the product that the user dislikes, if there is none say 'None'. Give the response in relevant keywords, not sentences. Don't use conjunctions or propositions, Shorten as much as possible :\n{review}, age={age}"}],
        max_tokens=1024, 
        temperature=0.5, 
        n = 1
    )
    sentiment = response.choices[0].message.content
    age_group = age//10 * 10

    #Splitting response by likes and dislikes and parsing 
    split = sentiment.split('\n')
    likes = split[0]
    likes = likes.replace('Likes: ', '')
    likes = likes.replace('.', '')
    likes = likes.lower()
    dislikes = split[1] if len(split) > 1 else ''
    dislikes = dislikes.replace('Dislikes: ', '')
    dislikes = dislikes.replace('.', '')
    dislikes = dislikes.lower()

    #Parsing Likes
    likes_keywords = likes.split(',')
    cleaned_likes = [s.strip() for s in likes_keywords]
    for x in cleaned_likes: 
        age_group_likes[age_group].add(x)

    #Parsing Dislikes 
    dislikes_keywords = dislikes.split(',')
    cleaned_dislikes = [s.strip() for s in dislikes_keywords]
    for y in cleaned_dislikes:
        age_group_dislikes[age_group].add(y)
    return sentiment



def analyze_amazon_reviews(reviews, ages):
    sentiments = []
    for x in range(numReviews):
        review1 = reviews[x][0]
        age = ages[x][0]
        review1 = review1.replace("39", " ")
        sentiment = analyze_sentiment(review1, age)
        sentiments.append(sentiment)
    return sentiments

def age_group_sentiment(age, likes, dislikes):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": 
        f"User: Write a report about the overall sentiment about the shampoo product in the age group: ({age}, {age+10}), based on the following list of keywords from a large set of amazon reviews about the product. The first list is keywords that the age group likes: {likes} and the second is keywords the age group dislikes: {dislikes}."}],
        max_tokens=4096, 
        temperature=0.5, 
        n = 1
    )
    report = response.choices[0].message.content
    return report 

# Analyze sentiments
sentiments = analyze_amazon_reviews(reviews, ages)

# Print sentiments
#for i, sentiment in enumerate(sentiments, 1):
#    print(f"Review {i}: {sentiment}")


#Print keywords for each age group and their likes 
#for key, value in age_group_likes.items():
#    print(f"Key: {key}, Value: {value}")

#ages20_30 = age_group_sentiment(20, age_group_likes[20], age_group_dislikes[20])

#print(ages20_30)

f = open("age_group_analysis.txt", "w")
for x in range(len(age_groups)):
    age_group = age_groups[x]
    f.write(f"Ages {age_group} - {age_group+10}: ")
    f.write(age_group_sentiment(age_group, age_group_likes[age_group], age_group_dislikes[age_group]))
    f.write("\n\n\n") 
f.close()


