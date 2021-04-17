import pandas as pd
import json

with open("sample_comments.json", "rb") as infile:
	chess_json = json.load(infile)

chess_df = pd.DataFrame()
chess_df["message"] = pd.Series(chess_json.values())
chess_df["hate_speech"] = 0

# chess_df[""] = pd.Series(chess_json.values())
hate_df = pd.read_csv("labeled_data.csv", usecols=["hate_speech", "tweet"])
hate_df.loc[hate_df["hate_speech"] != 0, "hate_speech"] = 1
hate_df = hate_df.rename(columns={"tweet": "message"})

total_df = pd.concat([chess_df, hate_df])
total_df.to_csv("aggregated_data.csv", index=False)