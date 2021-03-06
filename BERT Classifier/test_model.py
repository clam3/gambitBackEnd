import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from transformers import (
	BertForSequenceClassification,
	BertTokenizer,
	BertConfig
)

archive = os.path.join(os.getcwd(), "they_actually_said_that.txt")

def load_model(model_path="models_saved/bert-base-multilingual-uncased_English_translated_baseline_32/"):
	config = BertConfig.from_pretrained(model_path)
	tokenizer = BertTokenizer.from_pretrained(model_path)
	model = BertForSequenceClassification.from_pretrained(model_path)
	return config, tokenizer, model

def tokenize_inputs(tokenizer, inputs):
	return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

def run_model(model, tokenized_inputs):
	return model(**tokenized_inputs)["logits"]

def read_output(output):
	# logits = torch.mean(output[0][0], 0) # .detach().cpu().numpy()
	probs = torch.softmax(output, dim=1)
	output = probs.detach().numpy()[0]
	idx = np.argmax(output)
	return int(idx), int(100*output[idx])


def get_prediction(config, tokenizer, model, inputs):
	write_text(inputs)
	tokenized_inputs = tokenize_inputs(tokenizer, inputs)
	prob = read_output(run_model(model, tokenized_inputs))
	return prob

def write_text(inputs):
    with open(archive, "a") as outfile:
        outfile.write(inputs[:1000] + "\n\n")

if __name__ == "__main__":
	test_df = pd.read_csv("../data/no_chess_test.csv")
	test_df_hate = test_df.loc[test_df["hate_speech"] == 1, "message"].tolist()[45]
	# print(test_df_hate)
	
	config, tokenizer, model = load_model("chess/bert-base-multilingual-uncased_English_translated_baseline_32/")

	tokenized_inputs = tokenize_inputs(tokenizer, "I love you")
	output = run_model(model, tokenized_inputs)

	new_tokenized_input = tokenize_inputs(tokenizer, test_df_hate)
	new_output = run_model(model, new_tokenized_input)
	new_prob = read_output(new_output)
	# test_inputs = tokenize_inputs(tokenizer, test_df["message"].tolist())
	# test_labels = test_df["hate_speech"]

	# test_output = run_model(model, test_inputs)
	prob = read_output(output)

	print(f"not hate speech? {prob}")
	print(f"hate speech? {new_prob}")