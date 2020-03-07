import pandas as pd

local_raw_data = "./politifact_data.csv"
local_training_data = "./training_data.csv"

data = pd.read_csv(local_raw_data, sep='|')


def preprocess_data(raw_data):
    # Convert SHA256 value from hex string to integer.
    raw_data.sha256 = raw_data.sha256.apply(int, base=16)
    politifact_data = raw_data[["quote", "rating", "sha256"]]
    # Rows with Flip-related labels are not relevant to the model.
    politifact_data = politifact_data[~politifact_data.rating.isin(["full-flop", "half-flip", "no-flip"])]
    # Cast ratings as strings to avoid errors when changing case
    politifact_data.rating = politifact_data.rating.astype(str)
    # Remove case from the rating to merge False/false and True/true ratings
    politifact_data.rating = politifact_data.rating.str.lower()
    # If the last digit in the converted SHA value is 0-7, label it for training data
    # If the last digit is 8 or 9, label it for the test set
    politifact_data["is_test"] = politifact_data.sha256 % 10 >= 8
    # Once we have the split, we don't need the SHA value anymore
    politifact_data.drop(columns="sha256", inplace=True)
    # Some duplicate quotes remain, remove them.
    politifact_data.drop_duplicates(subset="quote", inplace=True)
    # Remove connecting phrases from quotes unlikely to appear when model is in use.
    connecting_phrases = ["Says ", "Say ", "Tweeted ", "Quoted ", "Quotes ", "Says of "]
    for phrase in connecting_phrases:
        politifact_data.quote = politifact_data.quote.str.replace(phrase, "")
    return politifact_data


politifact_data = preprocess_data(data)
politifact_data.to_csv(local_training_data, header=True, index=False, sep='|')
