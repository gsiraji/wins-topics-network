import pandas as pd



def make_topic_word_df(topics_array):
    df = pd.DataFrame()
    data_dict = {}
    for i,tup in enumerate(topics_array):
        row_dict = {}
        for word_prob in tup[1].split(" + "):
            prob, word = word_prob.split("*")
            df.loc[word,i] = prob
            # print(prob,word)
            row_dict[word.strip('"')] = float(prob)
        data_dict[tup[0]] = row_dict

    # df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = df.fillna(0)
    return df
    # print(np.unique(df.transpose().columns))
    # df2 = df2.reindex(sorted(df2.columns), axis=1)