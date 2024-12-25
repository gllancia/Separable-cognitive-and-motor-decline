import pandas as pd
import tqdm
from copy import deepcopy


if __name__ == "__main__":
    levels = [ 1,  2,  6,  7, 11, 12, 13, 16, 17, 21, 26, 27, 31, 32, 42, 43, 46]
    
    final_df = []
    for level in tqdm.tqdm(levels):
        df = pd.read_csv(f"./synth/synth_ll_level_{level}.csv")
        df = df[df["length"] < 1000]
        
        if level == 1:
            final_df = deepcopy(df)
            continue
        else:
            final_df = pd.concat([final_df, df])
    df.drop(['length_interpolated'], inplace=True, axis=1)
        
    final_df.columns = ["user_id", "level_id", "beta", "n_steps", "path"]
            
    final_df.to_csv("./synth/synth_all.csv", index=False)