#Name: Vihaan Mathur
#4/4/24
#https://www.datacamp.com/tutorial/how-to-fine-tune-gpt3-5
#https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates
#

#The following code fine tunes GPT 3.5 to better guess the ages of reviewers. 
#Still about 30% 

from openai import OpenAI
import numpy as np
import pandas as pd
import tiktoken 
import json
import re
from collections import defaultdict

numReviews = 1000 #Change to however many reviews you want to analyze 

client = OpenAI()

age_groups = [10,20,30,40,50,60]

df = pd.read_csv('reviews-data.csv', delimiter=',')
reviews = df['en_content']
ages = df['reviewerAge']
df['reviewerAge'] = (df['reviewerAge'] // 10) * 10
validating_set = df['reviewerAge'].iloc[800:]
validating_set = validating_set.values.tolist()
df['reviewerAge'] = df['reviewerAge'].astype(str)
df['en_content'] = df['en_content'].str.replace("39", "", regex=False)


pre_data = pd.DataFrame({
    'Question': df['en_content'].iloc[:800],
    'Answer': df['reviewerAge'].iloc[:800]
})

validation = pd.DataFrame({
    'Question': df['en_content'].iloc[800:],
    'Answer': df['reviewerAge'].iloc[800:]
})

DEFAULT_SYSTEM_PROMPT = 'You are a model trained to predict the age of an Amazon reviewer based on their review content of a shampoo product. You only say the guessed number and put them in the following bins [10,20,30,40,50,60] where 10 means 10 - 19, 20 means between 20 and 29, 30 means 30-39, 40 means 40-49, 50 means 50-59, and 60 means 60+. You only include one age group'

def create_dataset(question, answer):
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

with open("train.jsonl", "w") as f:
    for _, row in pre_data.iterrows():
        example_str = json.dumps(create_dataset(row["Question"], row["Answer"]))
        f.write(example_str + "\n")

with open("validate.jsonl", "w") as f:
    for _, row in validation.iterrows():
        example_str = json.dumps(create_dataset(row["Question"], row["Answer"]))
        f.write(example_str + "\n")


data_path = "train.jsonl"

with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# Warnings and tokens counts
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

training_file_name = 'train.jsonl'
validation_file_name = 'validate.jsonl'

training = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)

file_id = training.id

valid = client.files.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)

valid_id = valid.id

print("FILE ID", file_id)
print("Validation ID", valid_id)

# new_model = client.fine_tuning.jobs.create(
#   training_file=file_id, 
#   #validation_file=valid_id, 
#   hyperparameters={"n_epochs": 3},  #made n_epochs 3 due to cost reasons, can change later. 
#   model="gpt-3.5-turbo",
#   suffix="age-guess" 
# )


#______________
#VALIDATION
#______________
#Validating the model, because using a validation set in the fine tuning was taking a long time
#print(validating_set)
validating_reviews = df['en_content'].iloc[800:].tolist()
#print(validating_reviews)

#In the case that the API returns a string instead of just the number, this method extracts the number from a string. 
def get_age_from_string(text):
    two_digit_numbers = re.findall(r'\b\d{2}\b', text)
    if two_digit_numbers:
        return int(two_digit_numbers[0])  
    else:
        return None 

def guess_age(review):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:canaan-advisors::7v5erT71",
        messages=[{"role": "system", "content": 
        f"User: In the following amazon review of a shampoo product, guess the age of the reviewer based on the content, \n{review}. Only say the guessed number and put them in the following bins [10,20,30,40,50,60] where 10 means 10 - 19, 20 means between 20 and 29, 30 means 30-39, 40 means 40-49, 50 means 50-59, and 60 means 60+, only include one age group"}],
        max_tokens=1024, 
        temperature=0.5, 
        n = 1
    )
    guessed_age = response.choices[0].message.content
    guessed_age = get_age_from_string(guessed_age)
    return guessed_age



def analyze_amazon_reviews(reviews):
    guessed_ages = []
    for x in range(len(validating_set)):
        review1 = validating_reviews[x]
        guess = guess_age(review1)
        guessed_ages.append(guess)
    return guessed_ages


#Doing age comparison but based on the age groups, because exact age isn't really relevant 
def compare_ages(guessed, actual):
    num = 0
    for x in range(len(guessed)):
        currGuess = guessed[x]
        currVal = actual[x]
        currVal = currVal//10 * 10
        if(currVal == currGuess):
            num += 1
    accuracy = (num/len(guessed))
    return accuracy 

guessed_ages = analyze_amazon_reviews(validating_reviews)

print(compare_ages(guessed_ages, validating_set))



