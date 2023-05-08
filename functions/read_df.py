import os
import pandas as pd
import get_file_name

def read_df(dataFrameName):
    path = os.getcwd()
    filePath = get_file_name(dataFrameName,path)
    df = pd.read_parquet(filePath,engine='auto')
    return df