import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
import DataSetGenerator

def dataset():
    """"将数据划分为训练集与测试集"""

    def dataset_get_useful(data_frame):
        """给定数据集,读取所需列,并以合适的数据结构返回"""
        prog_id = list(data_frame['program_id'])

        code = list(data_frame['code_vec'])
        code = list(map(eval, code))

        label = list(data_frame['error_class_id'])
        return prog_id, code, label

    # 训练集和验证集
    train_frame = pd.read_excel('./DataSet/TrainSet.xlsx')
    valid_frame = pd.read_excel('./DataSet/ValidSet.xlsx')

    train_data = dataset_get_useful(train_frame)
    valid_data = dataset_get_useful(valid_frame)
    return train_data[0], train_data[1], train_data[2], valid_data[0], valid_data[1], valid_data[2]


train_prog_id, train_code, train_label, valid_prog_id, valid_code, valid_label = dataset()
# X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(train_code, train_label)
X_resampled, y_resampled = SMOTE().fit_resample(train_code, train_label)
# X_resampled, y_resampled = BorderlineSMOTE().fit_resample(train_code, train_label)
# X_resampled, y_resampled = SVMSMOTE().fit_resample(train_code, train_label)
# X_resampled, y_resampled = KMeansSMOTE().fit_resample(train_code, train_label)
# X_resampled, y_resampled = SMOTENC().fit_resample(train_code, train_label)
train_resampled = pd.DataFrame({'vec': X_resampled, 'error_class_id': y_resampled})
DataSetGenerator.data_uniform_test(train_resampled, label_col='error_class_id')
train_resampled = train_resampled.sample(frac=1).reset_index(drop=True)
DataSetGenerator.data_uniform_test(train_resampled, label_col='error_class_id')
breakpoint()