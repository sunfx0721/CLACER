import os
import sys
import numpy as np
import pandas as pd
import paddle
import paddle.fluid as fluid

dataset_name = sys.argv[1] if len(sys.argv) >= 2 else "deepfix"
model_name = sys.argv[2] if len(sys.argv) >= 3 else "TextCNN"
result_name = '_'.join(sys.argv[1:]) if len(sys.argv) >= 3 else model_name
word_dim = int(sys.argv[3]) if len(sys.argv) >= 4 else 512
num_filters = int(sys.argv[4]) if len(sys.argv) >= 5 else 256
epochs = int(sys.argv[5]) if len(sys.argv) >= 6 else 1
lr_rate = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.001
up_sample_method = sys.argv[7] if len(sys.argv) >= 8 else "SVMSMOTE"
CrossValidation = sys.argv[8] if len(sys.argv) >= 9 else None
# 参数和结果保存路径
param_repository = './param_repository'
result_repository = './result_repository'
if CrossValidation is None or CrossValidation != "CrossValidation":
    param_repository = param_repository + os.sep + 'param_repository'
    result_repository = result_repository + os.sep + 'result_repository'
else:
    param_repository = param_repository + os.sep + 'param_CrossValidation'
    result_repository = result_repository + os.sep + 'result_CrossValidation'
print('模型为:' + result_name)


def get_top_N(np_array, N):
    """获取np数组的前几大的位置"""
    np_array = np_array.copy()
    imax_N = []
    for i in range(N):
        imax_N.append(np_array.argmax())
        np_array[0, np_array.argmax()] = np_array.min()
    return imax_N


def predict(data):
    # 读取模型
    paddle.enable_static()
    # 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    path = param_repository + '/param_' + result_name
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

    # 存储预测结果的前几
    top_N = 3
    data = data.reindex(columns=list(data.columns)+['pre_top_' + str(i+1) for i in range(top_N)], fill_value=-1)
    data = data.reindex(columns=list(data.columns)+['in_predict_top1', 'in_predict_top3'], fill_value='')
    for index in data.index:
        # 提取输入
        code_v = data.loc[index, 'code_vec']
        code_v = eval(code_v)
        v_len = len(code_v)
        code_v = np.array(code_v).astype(dtype='int64').reshape([v_len, 1])
        code_v = fluid.create_lod_tensor(code_v, [[v_len]], place=fluid.CPUPlace())

        results = exe.run(inference_program,
                          feed={feed_target_names[0]: code_v},
                          fetch_list=fetch_targets)

        top_N_pred = get_top_N(results[0], top_N)
        data.loc[index, ['pre_top_' + str(i+1) for i in range(top_N)]] = top_N_pred
        if data.loc[index, 'error_class_id'] == top_N_pred[0]:
            data.loc[index, 'in_predict_top1'] = 'Yes'
        else:
            data.loc[index, 'in_predict_top1'] = 'No'

        if data.loc[index, 'error_class_id'] in top_N_pred:
            data.loc[index, 'in_predict_top3'] = 'Yes'
        else:
            data.loc[index, 'in_predict_top3'] = 'No'
        # print('{}已经预测完毕'.format(index))
    data.to_excel(result_repository + '/result_' + result_name + '/test_predict.xlsx', index=False)


if __name__ == '__main__':
    if CrossValidation is None or CrossValidation != "CrossValidation":
        # 读取数据
        data_path = r'./DataSet/DataSet/TestSet.xlsx'
        data = pd.read_excel(data_path)
        path = param_repository + '/param_' + result_name
        predict(data)
    else:
        param_repository_ori = param_repository
        result_repository_ori = result_repository
        DataSetParts = os.listdir('./DataSet/CrossValidationSets')
        for i in range(len(DataSetParts)):
            param_repository = param_repository_ori + os.sep + 'param_part_' + str(i)
            result_repository = result_repository_ori + os.sep + 'result_part_' + str(i)
            data = pd.read_excel('./DataSet/CrossValidationSets/DataSet_Part' + str(i) + '.xlsx')
            # data = pd.read_excel(r'./DataSet/DataSet/TestSet.xlsx')
            predict(data)


