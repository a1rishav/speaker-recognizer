import pandas as pd
from sklearn.preprocessing import StandardScaler

# create a dataframe
df = pd.DataFrame({
    'Name': ['a', 'b', 'c'],
    'classical_features': [[1, 2, 3], [2, 0, 1], [3, 2, 0]]
})


def split_list_col_to_multiple_cols(df, list_col_name):
    out_cols = [f'{list_col_name}_{col}' for col in range(len(df[list_col_name][0]))]

    split_df = pd.DataFrame(df[list_col_name].tolist(), columns=out_cols)
    df = pd.concat([df, split_df], axis=1)
    df = df.drop(columns=[list_col_name])
    return df

df_1 = split_list_col_to_multiple_cols(df, list_col_name='classical_features')
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
print()


