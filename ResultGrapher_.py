import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import palettable


def get_ConfuseMatrix(data_frame, to_rate=True):
    """从预测结果的DataFrame中获取混淆矩阵DataFrame"""
    label_set = list(set(list(data_frame['error_class_id'])))
    label_set_pre = list(set(list(data_frame['pre_top_1'])))
    confuse_matrix = pd.DataFrame(columns=label_set, index=label_set_pre)
    for label in label_set:
        for label_pre in label_set_pre:
            confuse_matrix.loc[label_pre, label] = data_frame[
                (data_frame['error_class_id'] == label) & (data_frame['pre_top_1'] == label_pre)
            ].shape[0]
    if to_rate:
        columns = confuse_matrix.columns.tolist()
        for c in columns:
            confuse_matrix[c] = confuse_matrix[c] / confuse_matrix[c].sum()
        confuse_matrix = confuse_matrix.astype('float64')
    else:
        confuse_matrix = confuse_matrix.astype('int64')
    return confuse_matrix


def ClassHeatMap(confuse_matrix):
    """confuse_matrix为DataFrame"""
    plt.figure(dpi=120)
    sns.heatmap(data=confuse_matrix,
                annot=True,  # 默认为False，当为True时，在每个格子写入data中数据
                fmt=".2f",  # 设置每个格子中数据的格式，参考之前的文章，此处保留两位小数
                cmap=sns.light_palette("#2ecc71", as_cmap=True),
                cbar=True,  # 右侧图例(color bar)开关，默认为True显示
                )
    plt.title("ConfuseMatrix")
    # fig_save_path = './result' + os.sep + "ConfuseMatrix.jpg"
    # plt.savefig(fig_save_path)
    plt.show()


predict_result_path = r'E:\PYprojects\SyntacticErrorHelper\result_repository\result_repository\result_TextCNN_512_256_7_0.001_None_None\test_predict.xlsx'
predict_frame = pd.read_excel(predict_result_path)
confuse_matrix = get_ConfuseMatrix(predict_frame)
ClassHeatMap(confuse_matrix=confuse_matrix)



