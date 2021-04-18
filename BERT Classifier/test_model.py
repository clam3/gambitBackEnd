from transformers import (
	BertModel,
	BertTokenizer,
	BertConfig
)


def load_model(model_path="models_saved/bert-base-multilingual-uncased_English_translated_baseline_32/"):
	config = BertConfig.from_pretrained(model_path)
	tokenizer = BertTokenizer.from_pretrained(model_path)
	model = BertModel.from_pretrained(model_path)
	return config, tokenizer, model

def tokenize_inputs(tokenizer, inputs):
	return tokenizer(inputs, return_tensors="pt")

def run_model(model, tokenized_inputs):
	return model(**tokenized_inputs)

if __name__ == "__main__":
	config, tokenizer, model = load_model("no_chess/bert-base-multilingual-uncased_English_translated_baseline_32/")

	tokenized_inputs = tokenize_inputs(tokenizer, "hello my name is bob")
	output = run_model(model, tokenized_inputs)