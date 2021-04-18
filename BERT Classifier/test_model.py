from transformers import (
	BertModel,
	BertTokenizer,
	BertConfig
)
import pandas as pd

def load_model(model_path="models_saved/bert-base-multilingual-uncased_English_translated_baseline_32/"):
	config = BertConfig.from_pretrained(model_path)
	tokenizer = BertTokenizer.from_pretrained(model_path)
	model = BertModel.from_pretrained(model_path)
	return config, tokenizer, model

def tokenize_inputs(tokenizer, inputs):
	return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

def run_model(model, tokenized_inputs):
	return model(**tokenized_inputs)

if __name__ == "__main__":
	config, tokenizer, model = load_model("no_chess/bert-base-multilingual-uncased_English_translated_baseline_32/")

	tokenized_inputs = tokenize_inputs(tokenizer, "hello my name is bob")
	output = run_model(model, tokenized_inputs)

	test_df = pd.read_csv("../data/no_chess_test.csv")
	test_inputs = tokenize_inputs(tokenizer, test_df["message"].tolist())
	test_labels = test_df["hate_speech"]

	test_output = run_model(model, test_inputs)