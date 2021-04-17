import pandas as pd

total_df = pd.read_csv("aggregated_data.csv")

no_chess_df = total_df.loc[total_df["chess"] == 0, :]

chess_df = total_df.loc[total_df["chess"] == 1, :]

no_chess_df_train = no_chess_df.sample(frac=0.8)
no_chess_df_test = no_chess_df.loc[no_chess_df.index.difference(no_chess_df_train.index)]

chess_only_df_train = chess_df.sample(frac=0.8)
chess_only_df_test = chess_df.loc[chess_df.index.difference(chess_only_df_train.index)]

no_chess_df_train.to_csv("no_chess_train.csv")
no_chess_df_test.to_csv("no_chess_test.csv")

chess_df_train = pd.concat([chess_only_df_train, no_chess_df_train])
chess_df_train.to_csv("chess_train.csv")

chess_df_test = pd.concat([chess_only_df_test, no_chess_df_test])
chess_df_test.to_csv("chess_test.csv")