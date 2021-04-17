import pandas as pd
import glob

textfile_list = list()
textfile_df = pd.DataFrame()

for file in glob.glob("all_files/*.txt"):
	text_id = file.split("/")[-1].split(".")[0]

	with open(file) as infile:
		message = infile.read()
	textfile_list.append([text_id, message])

textfile_df = pd.DataFrame(textfile_list, columns=["file_id", "message"])
textfile_df.reset_index()

labeled_df = pd.read_csv("annotations_metadata.csv")

combined_df = labeled_df.merge(textfile_df, on="file_id")
combined_df["hate_speech"] = combined_df["label"].eq("hate").mul(1)

del combined_df["file_id"]
del combined_df["user_id"]
del combined_df["subforum_id"]
del combined_df["num_contexts"]
del combined_df["label"]

print(combined_df)

combined_df.to_csv("dataset_2b.csv", index=False)