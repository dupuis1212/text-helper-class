# text-helper-class

#Loading data
`df=pd.read_csv('./data//train.csv')`

`df.head()`

`th = TextHelper()`

`th.missing_value_of_data(df)`

`df=df.dropna()`

`th.count_values_in_column(df,'sentiment')`

`th.unique_values_in_column(df,'sentiment')`

`th.duplicated_values_data(df)`

`df.describe()`

# mettre les mots

`df = th.add_all(df)`

`df['sad_or_sorrow'] = df['text'].apply(lambda x: th.or_cond(x, 'sad', 'sorrow'))`

`df['do_and_die'] = df['text'].apply(lambda x: th.and_cond(x))`

`df['bound'] = df['text'].apply(lambda x: th.boundary(x))`

`df['pick_senence'] = df['text'].apply(lambda x: th.pick_only_key_sentence(x, 'covid'))`

`df['search_day'] = df['text'].apply(lambda x: th.search_string(x, 'day'))`
