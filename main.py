import os

# sys.argv 文件名 数据集名称 神经网络类型 词嵌入维度 卷积核数量 训练世代 学习率 上采样方法(数据平衡方法) 是否进行交叉验证
dataset_name = ["deepfix", "tegcer", "all"]
model_name = ["TextCNN", "lstm"]
word_dims = [512, 256, 128, 64]
nums_filters = [256, 128, 64, 32]
epochs_list = [5, 7, 9, 11, 13, 15]
lr_rates = [0.005, 0.01, 0.05, 0.1]

up_sample_methods = [
    # 上采样方法
    "RandomOverSampler",
    "SMOTE",
    "BorderlineSMOTE",
    # "SVMSMOTE",
    # "KMeansSMOTE",
    # "SMOTENC",
    "ADASYN",
    # 组合方法
    "SMOTEENN",
    "SMOTETomek",
    # 欠采样方法
    # "CondensedNearestNeighbour",
    # "EditedNearestNeighbours",
    # "RepeatedEditedNearestNeighbours",
    # "AllKNN",
    # "InstanceHardnessThreshold"
]

CrossValidation = ['CrossValidation', 'None']

vec_proportion = [
    [80, 30],
    [60, 40],
    [60, 20]
]
k_folds = [10, 5, 9]
# 生成数据集(用于改变句子和错误信息截取长度的超参数)

# # 数据集 deepfix
# # os.system('python DataProcessor.py' + ' ' + dataset_name[0])
# # 对比实验：替换up_sample_methods
# for up_sample_method in up_sample_methods:
#     os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[1]))
#     result_name = 'deepfix TextCNN 128 256 5 0.001 ' + up_sample_method + ' None'
#     os.system('python model_train.py' + ' ' + result_name)
#     os.system('python model_predict.py' + ' ' + result_name)
#     os.system('python result_analysis.py' + ' ' + result_name)

# 数据集 tegcer
# os.system('python DataProcessor.py' + ' ' + dataset_name[1])
# 对比实验：替换up_sample_methods
# for up_sample_method in up_sample_methods:
#     os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[0]))
#     result_name = 'tegcer TextCNN 128 256 5 0.001 ' + up_sample_method + ' None'
#     os.system('python model_train.py' + ' ' + result_name)
#     os.system('python model_predict.py' + ' ' + result_name)
#     os.system('python result_analysis.py' + ' ' + result_name)


# 数据集 tegcer
# 对比实验：替换up_sample_methods
# for up_sample_method in up_sample_methods:
#     os.system('python DataSetGenerator.py' + ' ' + 'all//////////3' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[0]))
#     result_name = 'tegcer TextCNN 128 256 5 0.001 '+ up_sample_method +' CrossValidation'
#     os.system('python model_train.py' + ' ' + result_name)
#     os.system('python model_predict.py' + ' ' + result_name)
#     os.system('python result_analysis.py' + ' ' + result_name)
#
# # 数据集 deepfix
# os.system('python DataProcessor.py' + ' ' + dataset_name[0])
# # 对比实验：替换up_sample_methods
# for up_sample_method in up_sample_methods:
#     os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[1]))
#     result_name = 'deepfix TextCNN 128 256 5 0.001 '+ up_sample_method +' CrossValidation'
#     os.system('python model_train.py' + ' ' + result_name)
#     os.system('python model_predict.py' + ' ' + result_name)
#     os.system('python result_analysis.py' + ' ' + result_name)

os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[1]))
result_name = 'deepfix TextCNN 128 256 5 0.001 None CrossValidation'
os.system('python model_train.py' + ' ' + result_name)
os.system('python model_predict.py' + ' ' + result_name)
os.system('python result_analysis.py' + ' ' + result_name)

os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[1]))
result_name = 'deepfix lstm 128 256 5 0.001 None CrossValidation'
os.system('python model_train.py' + ' ' + result_name)
os.system('python model_predict.py' + ' ' + result_name)
os.system('python result_analysis.py' + ' ' + result_name)

os.system('python DataProcessor.py' + ' ' + dataset_name[1])
os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[0]))
result_name = 'tegcer TextCNN 128 256 5 0.001 None CrossValidation'
os.system('python model_train.py' + ' ' + result_name)
os.system('python model_predict.py' + ' ' + result_name)
os.system('python result_analysis.py' + ' ' + result_name)

os.system('python DataSetGenerator.py' + ' ' + 'all' + ' ' + ' '.join(list(map(str, vec_proportion[0]))) + " " + str(k_folds[0]))
result_name = 'tegcer lstm 128 256 5 0.001 None CrossValidation'
os.system('python model_train.py' + ' ' + result_name)
os.system('python model_predict.py' + ' ' + result_name)
os.system('python result_analysis.py' + ' ' + result_name)