def get_file_name(dataFrameName, path):
    filePath = '%s/project_data/%s.parquet' % (path,dataFrameName)
    
    return filePath