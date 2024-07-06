import pandas as pd
import numpy as np

#读取整个Excel文件
data_file = pd.ExcelFile('dataset/deeponet.xlsx')
data_file1 = pd.ExcelFile('dataset/deeponet1.xlsx')

#获取文件中的sheet的名称
sheet_names = data_file.sheet_names
sheet_names1 = data_file1.sheet_names

data_input_list = []
data_output_list = []

for sheet_name in sheet_names:
    df = pd.read_excel('dataset/deeponet.xlsx',sheet_name=sheet_name)
    if '输入' in df.columns:
        input_data = df['输入'].tolist()
        data_input_list.append(input_data)
    if '输出' in df.columns:
        output_data = df['输出'].tolist()
        data_output_list.append(output_data)

for sheet_name1 in sheet_names1:
    df1 = pd.read_excel('dataset/deeponet1.xlsx',sheet_name=sheet_name1)
    if '输入' in df1.columns:
        input_data = df1['输入'].tolist()
        data_input_list.append(input_data)
    if '输出' in df1.columns:
        output_data = df1['输出'].tolist()
        data_output_list.append(output_data)

t = np.arange(0,100.001,0.001)

data_input_train = data_input_list[:25]
data_output_train = data_output_list[:25]

data_input_test = data_input_list[25:32]
data_output_test = data_output_list[25:32]


np.savez('dataset/data_train_25.npz', X = data_input_train, t = t, Y=data_output_train)
np.savez('dataset/data_test_7.npz', X = data_input_test, t = t, Y=data_output_test)





