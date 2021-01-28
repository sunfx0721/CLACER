import os
import sys
import numpy as np
import pandas as pd
import DataProcessor

print(sys.argv)
# 超参数
DataSetName = eval(sys.argv[1]) if len(sys.argv) >= 2 else "deepfix"
error_line_len = eval(sys.argv[2]) if len(sys.argv) >= 3 else 80
error_message_len = eval(sys.argv[3]) if len(sys.argv) >= 4 else 20
k_fold = eval(sys.argv[4]) if len(sys.argv) >= 5 else 5


# 读取数据
def get_DataSet(error_line_len, error_message_len):
    """将现有数据库中的含标签数据形成DataFrame"""
    sentence_repository = DataProcessor.SentenceRepository()
    sentence_repository.get()

    corpus = DataProcessor.Corpus()
    corpus.get()
    corpus.get_dict_map()

    vec_repository = DataProcessor.VecRepository(
        error_line_len=error_line_len,
        error_message_len=error_message_len
    )
    vec_repository.genr(sentence_repository=sentence_repository, corpus=corpus)

    label_repository = DataProcessor.LabelRepository()
    label_repository.repository_get()
    label_repository.map_get()

    data_frame = pd.DataFrame(label_repository.repository)
    data_frame = data_frame.T
    data_frame = data_frame.reset_index()
    data_frame.columns = ['program_id', 'error_id_1', 'error_id_2', 'error_id']
    data_frame = data_frame.reindex(
        columns=list(data_frame.columns) + ['code_vec', 'error_class_id']
    )

    for i in data_frame.index:
        program_id = data_frame.loc[i, 'program_id']
        program_label = data_frame.loc[i, 'error_id']
        data_frame.loc[i, 'code_vec'] = str(vec_repository.repository[program_id][0] + vec_repository.repository[program_id][1])
        data_frame.loc[i, 'error_class_id'] = label_repository.label_map[program_label]
    del data_frame['error_id_1']
    del data_frame['error_id_2']
    data_frame['error_class_id'] = data_frame['error_class_id'].astype('int64')
    return data_frame


def data_uniform_test(data, label_col):
    """
    检查数据是否为均匀分析
    :param data: DataFrame
    :param label_col: 类别标签名
    :return: 无
    """
    pass_flag = 0
    for label in range(13):
        i_mean = (0+data.shape[0])/2
        i_std = ((data.shape[0]-0)**2/12)**0.5
        index_labeled = list(data[data[label_col] == label].index)
        i_mean_labeled = np.mean(index_labeled)
        i_std_labeled = np.std(index_labeled)
        print('-'*20)
        print('label:          %d' % label)
        print('i_mean:         %.1f' % i_mean)
        print('i_var:          %.1f' % i_std)
        print('i_mean_labeled: %.1f' % i_mean_labeled)
        print('i_var_labeled:  %.1f' % i_std_labeled)
        if i_mean - i_std < i_mean_labeled < i_mean + i_std:
            print('***---Maybe OK---***')
            pass_flag += 1
        else:
            print('@@@----Not OK---@@@')
    if pass_flag == len(set(data[label_col])):
        result = '一共有' + str(pass_flag) + '个类别分布均匀' + '! Congratulations,所有类别均通过！！！！'
    else:
        result = '一共有' + str(pass_flag) + '个类别分布均匀'
    print(result)
    print('~'*20)
    return 1


def DataSetSpliter(data_frame, label_col_name='error_class_id', train_valid_test_pro=[8, 1, 1]):
    """"将数据划分为训练集,验证集和测试集"""
    data_uniform_test(data=data_frame, label_col=label_col_name)
    # 给每个标签下的数据打上训练集、验证集和测试集的标签
    label_set = set(list(data_frame[label_col_name]))
    data_frame.reindex(columns=list(data_frame.columns)+['train_valid_test'], fill_value='')
    total_pro = np.sum(train_valid_test_pro)
    for label in label_set:
        ixs = list(data_frame[data_frame[label_col_name] == label].index)
        counter = 0
        while ixs:
            ix = ixs.pop()
            if 0 <= counter < np.sum(train_valid_test_pro[:1]):
                data_frame.loc[ix, 'train_valid_test'] = 'train'
            elif np.sum(train_valid_test_pro[:1]) <= counter < np.sum(train_valid_test_pro[:2]):
                data_frame.loc[ix, 'train_valid_test'] = 'valid'
            elif np.sum(train_valid_test_pro[:2]) <= counter < np.sum(train_valid_test_pro[:3]):
                data_frame.loc[ix, 'train_valid_test'] = 'test'
            else:
                data_frame.loc[ix, 'train_valid_test'] = 'no-use'
            counter += 1
            if counter >= total_pro:
                counter = 0

    trainset = data_frame[data_frame['train_valid_test'] == 'train']
    validset = data_frame[data_frame['train_valid_test'] == 'valid']
    testset = data_frame[data_frame['train_valid_test'] == 'test']

    trainset.to_excel(r'./DataSet/DataSet/TrainSet.xlsx', index=False)
    validset.to_excel(r'./DataSet/DataSet/ValidSet.xlsx', index=False)
    testset.to_excel(r'./DataSet/DataSet/TestSet.xlsx', index=False)


def DataSetSpliter_K_fold(data_frame, label_col_name='error_class_id', K_fold=5):
    """"将数据划分为K份"""
    data_uniform_test(data=data_frame, label_col=label_col_name)
    part_col = 'part_id'

    min_class_count = min(list(data_frame.groupby(by="error_class_id").count().iloc[:, 0]))
    if min_class_count < K_fold:
        print("Waring:K-fold选取过大,当前最小类的数量为" + str(min_class_count) + '.')

    data_frame = data_frame.reindex(columns=list(data_frame.columns)+[part_col], fill_value=-1)
    data_frame[part_col] = data_frame[part_col].astype(dtype='int64')
    # 给每个标签下的数据打上训练集、验证集和测试集的标签
    label_set = set(list(data_frame[label_col_name]))
    data_frame.reindex(columns=list(data_frame.columns)+['train_valid_test'], fill_value='')
    total_pro = np.sum(K_fold)
    for label in label_set:
        ixs = list(data_frame[data_frame[label_col_name] == label].index)
        counter = 0
        while ixs:
            ix = ixs.pop()
            data_frame.loc[ix, part_col] = counter
            counter += 1
            if counter >= K_fold:
                counter = 0

    # 清空交叉验证数据集文件夹下的数据
    if os.listdir('./DataSet/CrossValidationSets'):
        for file_name in os.listdir('./DataSet/CrossValidationSets'):
            os.remove('./DataSet/CrossValidationSets' + os.sep + file_name)

    part_ids = set(list(data_frame[part_col]))
    for part_id in part_ids:
        part_set = data_frame[data_frame[part_col] == part_id]
        part_set.to_excel(r'./DataSet/CrossValidationSets/DataSet_Part' + str(part_id) + '.xlsx', index=False)


if __name__ == '__main__':
    data_frame = get_DataSet(error_line_len=error_line_len, error_message_len=error_message_len)
    DataSetSpliter(data_frame=data_frame, label_col_name='error_class_id', train_valid_test_pro=[8, 1, 1])
    train_frame = pd.read_excel('./DataSet/DataSet/TrainSet.xlsx')
    valid_frame = pd.read_excel('./DataSet/DataSet/ValidSet.xlsx')
    data_frame = pd.concat([train_frame, valid_frame])
    DataSetSpliter_K_fold(data_frame=data_frame, label_col_name='error_class_id', K_fold=k_fold)
