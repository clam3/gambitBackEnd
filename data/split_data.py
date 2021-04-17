import pandas as pd
import numpy as np

total_df = pd.read_csv("aggregated_data.csv")
# total_df = total_df.loc[total_df["message"].str.split().len() < 100]

no_chess_df = total_df.loc[total_df["chess"] == 0]

chess_df = total_df.loc[total_df["chess"] == 1]

no_chess_df_train, no_chess_df_validate, no_chess_df_test = \
              np.split(no_chess_df.sample(frac=1, random_state=42), 
                       [int(.6*len(no_chess_df)), int(.8*len(no_chess_df))])

chess_only_df_train, chess_only_df_validate, chess_only_df_test = \
              np.split(chess_df.sample(frac=1, random_state=42), 
                       [int(.6*len(chess_df)), int(.8*len(chess_df))])

# no_chess_df_train = no_chess_df.sample(frac=0.8)
# no_chess_df_test = no_chess_df.loc[no_chess_df.index.difference(no_chess_df_train.index)]

# chess_only_df_train = chess_df.sample(frac=0.8)
# chess_only_df_test = chess_df.loc[chess_df.index.difference(chess_only_df_train.index)]

no_chess_df_train.to_csv("no_chess_train.csv", index=False)
no_chess_df_test.to_csv("no_chess_test.csv", index=False)
no_chess_df_validate.to_csv("no_chess_validate.csv", index=False)

chess_df_train = pd.concat([chess_only_df_train, no_chess_df_train])
chess_df_train.to_csv("chess_train.csv", index=False)

chess_df_test = pd.concat([chess_only_df_test, no_chess_df_test])
chess_df_test.to_csv("chess_test.csv", index=False)

chess_df_validate = pd.concat([chess_only_df_validate, no_chess_df_validate])
chess_df_validate.to_csv("chess_validate.csv", index=False)


mini_df_train = total_df.sample(n=50)
mini_df_test = total_df.sample(n=50)
mini_df_validate = total_df.sample(n=50)

mini_df_train.to_csv("mini_train.csv", index=False)
mini_df_test.to_csv("mini_test.csv", index=False)
mini_df_validate.to_csv("mini_validate.csv", index=False)