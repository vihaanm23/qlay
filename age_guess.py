#Name: Vihaan Mathur
#3/18/24

#The following code assesses GPT's accuracy in guessing the age of reviewers based on the review content. 
#It compares the guessed values to the actual values from the dataset 
#Found an accuracy of ~ 37%

from openai import OpenAI
import pandas as pd
import re

numReviews = 1000 #Change to however many reviews you want to analyze 

client = OpenAI() #Enter API KEY here


age_groups = [10,20,30,40,50,60]

age_group_likes = {age_group: set() for age_group in age_groups}

age_group_dislikes = {age_group: set() for age_group in age_groups}

df = pd.read_csv('reviews-data.csv', delimiter=',')
reviews = df[['en_content']]
ages = df[['reviewerAge']]
ages = ages.values.tolist()
reviews = reviews.values.tolist()

#In the case that the API returns a string instead of just the number, this method extracts the number from a string. 
def get_age_from_string(text):
    two_digit_numbers = re.findall(r'\b\d{2}\b', text)
    if two_digit_numbers:
        return int(two_digit_numbers[0])  
    else:
        return None 

def guess_age(review):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": 
        f"User: In the following amazon review of a shampoo product, guess the age of the reviewer based on the content, \n{review}. Only say the guessed number and put them in the following bins [10,20,30,40,50,60] where 10 means 10 - 19, 20 means between 20 and 29, 30 means 20-29, 40 means 40-49, 50 means 50-59, and 60 means 60+, only include one age group"}],
        max_tokens=1024, 
        temperature=0.5, 
        n = 1
    )
    guessed_age = response.choices[0].message.content
    guessed_age = get_age_from_string(guessed_age)
    return guessed_age



def analyze_amazon_reviews(reviews):
    guessed_ages = []
    for x in range(numReviews):
        review1 = reviews[x][0]
        review1 = review1.replace("39", " ")
        guess = guess_age(review1)
        guessed_ages.append(guess)
    return guessed_ages


#Doing age comparison but based on the age groups, because exact age isn't really relevant 
def compare_ages(guessed, actual):
    num = 0
    for x in range(len(guessed)):
        currGuess = guessed[x]
        currVal = actual[x][0]
        currVal = currVal//10 * 10
        if(currVal == currGuess):
            num += 1
    accuracy = (num/len(guessed))
    return accuracy 

guessed_ages = analyze_amazon_reviews(reviews)
#print(guessed_ages)
print(compare_ages(guessed_ages, ages))

#ACCURACY 
#10 Reviews - 70% 
#50 Reviews - 46% 
#100 Reviews - 48%
#300 Reviews - 40% 
#All Reviews (1000) - 37.4%





