import pandas as pd
import json
from clean_message import clean_text

with open("sample_comments.json", "rb") as infile:
	chess_json = json.load(infile)

chess_df = pd.DataFrame()
chess_df["message"] = pd.Series(chess_json.values())
chess_df["hate_speech"] = 0
chess_df["chess"] = 1

# chess_df[""] = pd.Series(chess_json.values())
hate_df = pd.read_csv("labeled_data.csv", usecols=["hate_speech", "tweet"])
hate_df.loc[hate_df["hate_speech"] != 0, "hate_speech"] = 1
hate_df = hate_df.rename(columns={"tweet": "message"})
hate_df["chess"] = 0

hate_df_2b = pd.read_csv("dataset-2b/dataset_2b.csv")
hate_df_2b["chess"] = 0

total_df = pd.concat([chess_df, hate_df, hate_df_2b])
total_df["message"] = total_df["message"].apply(clean_text)
total_df.to_csv("aggregated_data.csv", index=False)