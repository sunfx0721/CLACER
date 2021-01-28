import DataProcessor
import pandas as pd
import result_analysis

#----------------------------------------------------------------------------------------------------------------#
#                              产生fasttext的输入数据,便于导入aistudio中进行训练和预测
#----------------------------------------------------------------------------------------------------------------#
# sentence_repository = DataProcessor.SentenceRepository()
# sentence_repository.get()
#
# label_repository = DataProcessor.LabelRepository()
# label_repository.repository_get()
#
# train_frame = pd.read_excel('./DataSet/DataSet/TrainSet.xlsx')
# valid_frame = pd.read_excel('./DataSet/DataSet/ValidSet.xlsx')
# test_frame = pd.read_excel('./DataSet/DataSet/TestSet.xlsx')
#
# opt_train = open('./DataSet/FastText_data/code_train.txt', mode='w')
# opt_train_id = open('./DataSet/FastText_data/code_train_id.txt', mode='w')
# for key in list(train_frame["program_id"]):
#     code_line = sentence_repository.repository[key][0].strip()
#     error_message = sentence_repository.repository[key][1].strip()
#     label = label_repository.repository[key][2]
#
#     itstr = label + '\t' + code_line + ' ' + error_message + '\n'
#     opt_train.write(itstr)
#     opt_train_id.write(key + '\n')
# opt_train.close()
# opt_train_id.close()
#
# opt_valid = open('./DataSet/FastText_data/code_valid.txt', mode='w')
# opt_valid_id = open('./DataSet/FastText_data/code_valid_id.txt', mode='w')
# for key in list(valid_frame["program_id"]):
#     code_line = sentence_repository.repository[key][0].strip()
#     error_message = sentence_repository.repository[key][1].strip()
#     label = label_repository.repository[key][2]
#
#     itstr = label + '\t' + code_line + ' ' + error_message + '\n'
#     opt_valid.write(itstr)
#     opt_valid_id.write(key + '\n')
# opt_valid.close()
# opt_valid_id.close()
#
# opt_test = open('./DataSet/FastText_data/code_test.txt', mode='w')
# opt_test_id = open('./DataSet/FastText_data/code_test_id.txt', mode='w')
# for key in list(test_frame["program_id"]):
#     code_line = sentence_repository.repository[key][0].strip()
#     error_message = sentence_repository.repository[key][1].strip()
#     label = label_repository.repository[key][2]
#
#     itstr = label + '\t' + code_line + ' ' + error_message + '\n'
#     opt_test.write(itstr)
#     opt_test_id.write(key + '\n')
# opt_test.close()
# opt_test_id.close()
#
# # 预期生成文件,生成后导入aistudio
# # code_test.txt
# # code_test_id.txt
# # code_train.txt
# # code_train_id.txt
# # code_valid.txt
# # code_valid_id.txt

#----------------------------------------------------------------------------------------------------------------#
#                    从aistudio中导出结果文件相同格式的表,便于之后将fasttext结果分析
#----------------------------------------------------------------------------------------------------------------#
# 需要导出文件result.txt到result_DataSet_FastText下
results = eval(open('./result_repository/result_repository/result_DataSet_deepfix_FastText/result.txt', mode='r').read())

data_path = r'./DataSet/DataSet/TestSet.xlsx'
data = pd.read_excel(data_path)
# 存储预测结果的前几
top_N = 3
data = data.reindex(columns=list(data.columns) + ['pre_top_' + str(i + 1) for i in range(top_N)], fill_value=-1)
data = data.reindex(columns=list(data.columns) + ['in_predict_top1', 'in_predict_top3'], fill_value='')

label_repository = DataProcessor.LabelRepository()
label_repository.repository_get()
label_repository.map_get()
for result in results:
    data.loc[
        data['program_id'] == result[0], ['true_label'] + ['pre_top_' + str(i+1) for i in range(top_N)]
    ] = [label_repository.label_map[i] for i in result[1:]]

    if result[1] == result[2]:
        data.loc[
            data['program_id'] == result[0], 'in_predict_top1'
        ] = 'Yes'
    else:
        data.loc[
            data['program_id'] == result[0], 'in_predict_top1'
        ] = 'No'

    if result[1] in result[2:]:
        data.loc[
            data['program_id'] == result[0], 'in_predict_top3'
        ] = 'Yes'
    else:
        data.loc[
            data['program_id'] == result[0], 'in_predict_top3'
        ] = 'No'
data.to_excel('./result_repository/result_repository/result_DataSet_deepfix_FastText/test_predict.xlsx', index=False)

#----------------------------------------------------------------------------------------------------------------#
#                                             结果分析
#----------------------------------------------------------------------------------------------------------------#
# 需要从相应数据集的某一个训练结果下导出相应的训练规模数据
test = data
label_list = list(set(list(test['error_class_id'])))
statistics_frame = result_analysis.statistic_analysis(test, 'test', do_save=False)
statistics_frame.to_excel('./result_repository/result_repository/result_DataSet_deepfix_FastText/analysis_of_test.xlsx', index=False)