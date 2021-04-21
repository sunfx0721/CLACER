import os
import sys
import pandas as pd

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

true_label = 'error_class_id'
predict_label = 'pre_top_1'
# 计算准确率


def accuracy(result, label=None):
    if label is None:
        total = result.count().tolist()[0]
        right = result[result[true_label] == result[predict_label]].count().tolist()[0]
    else:
        result_label = result[result[true_label] == label]
        total = result_label.count().tolist()[0]
        right = result_label[result_label[true_label] == result_label[predict_label]].count().tolist()[0]
        if total == 0:
            breakpoint()
    return right / total


def accuracy_top3(result, label=None):
    if label is None:
        total = result.count().tolist()[0]
        right = result[result['in_predict_top3'] == 'Yes'].count().tolist()[0]
    else:
        result_label = result[result[true_label] == label]
        total = result_label.count().tolist()[0]
        right = result_label[result_label['in_predict_top3'] == 'Yes'].count().tolist()[0]
    return right / total


def get_TP(result, label):
    return result[(result[true_label] == label) & (result[predict_label] == label)].count().tolist()[0]


def get_FP(result, label):
    return result[(result[true_label] != label) & (result[predict_label] == label)].count().tolist()[0]


def get_FN(result, label):
    return result[(result[true_label] == label) & (result[predict_label] != label)].count().tolist()[0]


# 计算精确率
def precesion(result, label):
    TP = get_TP(result, label)
    FP = get_FP(result, label)
    try:
        return TP / (TP + FP)
    except ZeroDivisionError:
        return 0


# 计算召回率
def recall(result, label):
    TP = get_TP(result, label)
    FN = get_FN(result, label)
    try:
        return TP / (TP + FN)
    except ZeroDivisionError:
        return 0


# 计算F1-Score
def F1_Score(result, label):
    P = precesion(result, label)
    R = recall(result, label)
    try:
        return 2 * P * R / (P + R)
    except ZeroDivisionError:
        return 0


def macro_F1_Score(result):
    pass


def statistic_analysis(dataset, dataset_name, do_save=True):
    statistics_frame = pd.DataFrame(
        data=None,
        columns=[true_label, 'train_scale', 'sample_scale',
                 'TP', 'FN', 'FP',
                 'accuracy', 'accuracy_top3',
                 'precession', 'recall', 'F1-score']
    )

    # 为了便于FastText的结果分析加入一个分支
    if do_save:
        train_scales_path = result_repository + '/result_' + result_name + '/train_scale.txt'
        train_scales = eval(open(train_scales_path).read())
        train_frame = pd.read_excel('./DataSet/DataSet/TrainSet.xlsx')
        label_list = list(set(list(train_frame['error_class_id'])))
    else:
        train_frame = pd.read_excel('./DataSet/DataSet/TrainSet.xlsx')
        train_label = list(train_frame['error_class_id'])
        train_scales = {label_i: train_label.count(label_i) for label_i in train_label}
        label_list = list(set(list(train_frame['error_class_id'])))

    for label in label_list:
        data_labeled = dataset[dataset[true_label] == label]
        true_label_in = label
        train_scale = train_scales[label]
        sample_scale = dataset[dataset[true_label] == label].count().tolist()[0]
        TP = get_TP(dataset, label)
        FN = get_FN(dataset, label)
        FP = get_FP(dataset, label)
        Accuracy = accuracy(dataset, label)
        Accuracy_top3 = accuracy_top3(dataset, label)
        Precession = precesion(dataset, label)
        Recall = recall(dataset, label)
        f1_score = F1_Score(dataset, label)

        statistics_frame = statistics_frame.append(
            {
                true_label: true_label_in, 'train_scale': train_scale, 'sample_scale': sample_scale,
                'TP': TP, 'FN': FN, 'FP': FP,
                'accuracy': Accuracy, 'accuracy_top3': Accuracy_top3,
                'precession': Precession, 'recall': Recall, 'F1-score': f1_score
            },
            ignore_index=True
        )

    # 加入总计数据
    statistics_frame = statistics_frame.append(
        {
            true_label: 'total',
            'train_scale': statistics_frame['train_scale'].sum(),
            'sample_scale': statistics_frame['sample_scale'].sum(),
            'TP': statistics_frame['TP'].sum(),
            'FN': statistics_frame['FN'].sum(),
            'FP': statistics_frame['FP'].sum(),
            'accuracy': accuracy(dataset), 'accuracy_top3': accuracy_top3(dataset),
            'precession': statistics_frame['precession'].mean(),
            'recall': statistics_frame['recall'].mean(),
            'F1-score': statistics_frame['F1-score'].mean()
        },
        ignore_index=True
    )
    statistics_frame.iloc[:, -5:] = statistics_frame.iloc[:, -5:].round(decimals=3)
    if do_save:
        statistics_frame.to_excel(result_repository + '/result_' + result_name + '/analysis_of_' + dataset_name + '.xlsx', index=False)
        return 0
    else:
        return statistics_frame


# if __name__ == '__main__':
#     result_path = result_repository + '/result_' + result_name + '/test_predict.xlsx'
#     test_result = pd.read_excel(result_path)
#     label_list = list(set(list(test_result['error_class_id'])))
#     test = test_result[test_result['train_valid_test'] == 'test']
#     # statistic_analysis(train, 'train')
#     statistic_analysis(test, 'test')
#     # statistic_analysis(test_result, 'test_result')

if __name__ == '__main__':
    if CrossValidation is None or CrossValidation != "CrossValidation":
        result_path = result_repository + '/result_' + result_name + '/test_predict.xlsx'
        test = pd.read_excel(result_path)
        label_list = list(set(list(test['error_class_id'])))
        statistic_analysis(test, 'test')
    else:
        param_repository_ori = param_repository
        result_repository_ori = result_repository
        DataSetParts = os.listdir('./DataSet/CrossValidationSets')
        for i in range(len(DataSetParts)):
            param_repository = param_repository_ori + os.sep + 'param_part_' + str(i)
            result_repository = result_repository_ori + os.sep + 'result_part_' + str(i)
            result_path = result_repository + '/result_' + result_name + '/test_predict.xlsx'
            test = pd.read_excel(result_path)
            label_list = list(set(list(test['error_class_id'])))
            statistic_analysis(test, 'test')
