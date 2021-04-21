import pandas as pd
import os

col_needed = ['program_id', 'code', 'error_id_1', 'error_id_2', 'error_class_id']
DataSet_deepfix = pd.read_excel('./DataSet/DataSet_deepfix_ori.xlsx')
DataSet_deepfix = DataSet_deepfix[col_needed]
DataSet_deepfix.to_csv(r'./DataSet/DataSet_deepfix.csv', index=False)

DataSet_tegcer = pd.read_excel('./DataSet/DataSet_tegcer_ori.xlsx')
DataSet_tegcer = DataSet_tegcer[col_needed]
DataSet_tegcer.to_csv(r'./DataSet/DataSet_tegcer.csv', index=False)

DataSet = pd.concat([DataSet_deepfix, DataSet_tegcer])
DataSet.to_csv(r'./DataSet/DataSet.csv', index=False)
path = './result_repository/result_CrossValidation'
data1,data2,data3,data4 = [],[],[],[]
for dir1 in os.listdir(path):
    for dir2 in os.listdir(os.path.join(path, dir1)):
        data = pd.read_excel(os.path.join(path, dir1, dir2, 'analysis_of_test.xlsx'),usecols='G:K')
        if 'tegcer_TextCNN' in dir2:
            data1.append(list(data.iloc[13].values))

print(data1)
dataframe1 = pd.DataFrame(data1)
dataframe1.to_excel('C:/Users/湘/Desktop/compare.xlsx',sheet_name='32')
# dataframe2 = pd.DataFrame(data2)
# dataframe2.to_excel('C:/Users/Administrator/Desktop/compare.xlsx',sheet_name='64')
# dataframe3 = pd.DataFrame(data3)
# dataframe3.to_excel('C:/Users/Administrator/Desktop/compare.xlsx',sheet_name='128')
# dataframe4 = pd.DataFrame(data4)
# dataframe4.to_excel('C:/Users/Administrator/Desktop/compare.xlsx',sheet_name='256')
