import pandas as pd
from openai import OpenAI

#~10 seconds per 128 reviews 
#~23 minutes for 30,000 reviews

#numReviews = 128

batch_size = 128

client = OpenAI(api_key='') 

df = pd.read_csv('test.csv')

product_ratings = df.groupby('product_id')['stars'].agg(['mean', list])

#product_ratings = product_ratings.head(numReviews)


#DOING IT ONE AT A TIME
#f = open("stars_report.txt", "w")
# for product_id, ratings_data in product_ratings.iterrows():
#     avg_rating = ratings_data['mean']
#     ratings = ratings_data['list']
#     ratings_str = ', '.join(map(str, ratings))
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": 
#         f"User: Generate a one sentence max report for the product with ID {product_id} based on the variation of the following star ratings: {ratings_str} and the ooverall average rating for this product which is {avg_rating:.2f}."}],
#         max_tokens=1024, 
#         temperature=0.5, 
#         n = 1
#     )
#     sentiment = response.choices[0].message.content
#     f.write(f"Product ID: {product_id}")
#     f.write('\n')
#     f.write(f"Average Rating: {avg_rating:.2f}")
#     f.write('\n')
#     f.write("Report:")
#     f.write('\n')
#     f.write(sentiment)
#     f.write('\n')
#     f.write('\n')
#     f.write('\n')
#f.close()



#SEEING WHAT THE MAXIMUM BATCH SIZE IS ~ 128
# prompts = []
# for product_id, ratings_data in product_ratings.iterrows():
#     avg_rating = ratings_data['mean']
#     ratings = ratings_data['list']
#     ratings_str = ', '.join(map(str, ratings))
    
#     prompt = f"User: Generate a one sentence max report for the product with ID {product_id} based on the variation of the following star ratings: {ratings_str} and the overall average rating for this product which is {avg_rating:.2f}."
#     prompts.append(prompt)

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt} for prompt in prompts],
#     max_tokens=1024, 
#     temperature=0.5, 
#     n = numReviews
# )

# f = open("stars_report.txt", "w")
# for i, (product_id, ratings_data) in enumerate(product_ratings.iterrows()):
#     avg_rating = ratings_data['mean']
#     ratings = ratings_data['list']  
#     sentiment = response.choices[i].message.content
#     f.write(f"Product ID: {product_id}")
#     f.write('\n')
#     f.write(f"Average Rating: {avg_rating:.2f}")
#     f.write('\n')
#     f.write("Report:")
#     f.write('\n')
#     f.write(sentiment)
#     f.write('\n')
#     f.write('\n')
#     f.write('\n')
# f.close()

#USING BATCHES OF REVIEWS FOR API CALLS
num_batches = (len(product_ratings) + batch_size - 1) // batch_size

f = open("stars_report.txt", "w")

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(product_ratings))
    batch_ratings = product_ratings.iloc[start_idx:end_idx]

    prompts = []
    for product_id, ratings_data in batch_ratings.iterrows():
        avg_rating = ratings_data['mean']
        ratings = ratings_data['list']
        ratings_str = ', '.join(map(str, ratings))
        prompt = f"User: Generate a one sentence max report for the product with ID {product_id} based on the variation of the following star ratings: {ratings_str} and the overall average rating for this product which is {avg_rating:.2f}."
        prompts.append(prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt} for prompt in prompts],
        max_tokens=3896,
        temperature=0.5,
        n=batch_size
    )

    for i, (product_id, ratings_data) in enumerate(batch_ratings.iterrows()):
        avg_rating = ratings_data['mean']
        ratings = ratings_data['list']
        sentiment = response.choices[i].message.content
        f.write(f"Product ID: {product_id}")
        f.write('\n')
        f.write(f"Average Rating: {avg_rating:.2f}")
        f.write('\n')
        f.write("Report:")
        f.write('\n')
        f.write(sentiment)
        f.write('\n')
        f.write('\n')
        f.write('\n')

f.close()