import read_df
import get_subset

def get_subset_from_file_name(dataFrameName: str='arxiv_climate_change',year: int=2010):
    '''
    This function subsets the dataframe named
    dataFrameName by year

    inputs: 
    dataFrameName: name of the dataframe, string 
    year: int

    output:
    subset_df: pandas dataframe 
    with entries with date 'year-M-D'
    '''
    df = read_df(dataFrameName)

    subset_df = get_subset(df,year)
    
    return subset_df