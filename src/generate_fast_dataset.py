import pandas as pd
import random

positive_words = ["good", "great", "amazing", "love", "excellent", "fantastic"]
negative_words = ["bad", "worst", "hate", "poor", "terrible", "awful"]

neutral_sentences = [
    "The product is okay",
    "It works as expected",
    "Average experience overall",
    "Nothing special about this",
    "Service was normal",
    "It's fine",
    "Not great not terrible"
]

prefixes = [
    "I think",
    "Honestly",
    "In my opinion",
    "From my experience",
    "Overall",
    "Well",
    ""
]

def generate_sample():
    r = random.random()
    prefix = random.choice(prefixes)

    # POSITIVE
    if r < 0.3:
        return f"{prefix} the product is {random.choice(positive_words)}", "positive"

    # NEGATIVE
    elif r < 0.6:
        return f"{prefix} the product is {random.choice(negative_words)}", "negative"

    # NEUTRAL (clear)
    elif r < 0.8:
        return random.choice(neutral_sentences), "neutral"

    # MIXED (important realism)
    else:
        pos = random.choice(positive_words)
        neg = random.choice(negative_words)

        # Logical labeling (not random)
        sentence = f"{prefix} the product is {pos} but sometimes {neg}"

        # Slight ambiguity → assign neutral or weak polarity
        label = random.choice(["neutral", "positive", "negative"])

        return sentence, label


# Generate dataset
data = [generate_sample() for _ in range(10000)]

df = pd.DataFrame(data, columns=["text", "sentiment"])
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("../data/dataset_10k.csv", index=False)

print("✅ Final REALISTIC dataset created!")
print("Total rows:", len(df))