import pandas as pd

def get_subset(df,year: int=2010):
    '''
    This function subsets the dataframe df
    by year

    inputs: 
    df: pandas dataframe
    year: int

    output:
    subset_df: pandas dataframe 
    with entries with date 'year-M-D'
    '''

    df['date'] = pd.to_datetime(df['update_date'], format='%Y-%m-%d')

    start_date = '{}-01-01'.format(str(year))
    end_date = '{}-01-01'.format(str(year+1))
    
    subset_df = df.loc[(df['date'] >= start_date)
                    & (df['date'] < end_date)]
    
    if subset_df.empty:
        print('DataFrame is empty for year {}!'.format(str(year)))
    
    return subset_df