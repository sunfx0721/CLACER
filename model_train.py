import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import random
import paddle
import paddle.fluid as fluid
import DataProcessor
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
random.seed(1000)

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
if not os.path.exists(param_repository):
    os.makedirs(param_repository)
if not os.path.exists(result_repository):
    os.makedirs(result_repository)

print('模型为:' + result_name)


def dataset(train_frame, valid_frame, up_sample_method=None):
    """"将数据划分为训练集与测试集"""

    def dataset_get_useful(data_frame):
        """给定数据集,读取所需列,并以合适的数据结构返回"""
        code = list(data_frame['code_vec'])
        if type(code[0]) == str:
            code = list(map(eval, code))

        label = list(data_frame['error_class_id'])
        return code, label

    train_data = dataset_get_useful(train_frame)
    valid_data = dataset_get_useful(valid_frame)

    # 对训练集上采样
    train_code = train_data[0]
    train_label = train_data[1]

    up_sample_methods = [
        "ADASYN",
        "RandomOverSampler",
        "SMOTE",
        "BorderlineSMOTE",
        "SVMSMOTE",
        "KMeansSMOTE",
        "SMOTENC",
        "EditedNearestNeighbours",
        "RepeatedEditedNearestNeighbours"
    ]

    if up_sample_method in up_sample_methods:
        if up_sample_method == "RandomOverSampler":
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(train_code, train_label)
        elif up_sample_method == "ADASYN":
            X_resampled, y_resampled = ADASYN().fit_resample(train_code, train_label)
        elif up_sample_method == "SMOTE":
            X_resampled, y_resampled = SMOTE().fit_resample(train_code, train_label)
        elif up_sample_method == "BorderlineSMOTE":
            X_resampled, y_resampled = BorderlineSMOTE().fit_resample(train_code, train_label)
        elif up_sample_method == "SVMSMOTE":
            X_resampled, y_resampled = SVMSMOTE().fit_resample(train_code, train_label)
        elif up_sample_method == "KMeansSMOTE":
            X_resampled, y_resampled = KMeansSMOTE().fit_resample(train_code, train_label)
        elif up_sample_method == "SMOTENC":
            X_resampled, y_resampled = SMOTENC().fit_resample(train_code, train_label)
        elif up_sample_method == "SMOTEENN":
            X_resampled, y_resampled = SMOTEENN().fit_resample(train_code, train_label)
        elif up_sample_method == "SMOTETomek":
            X_resampled, y_resampled = SMOTETomek().fit_resample(train_code, train_label)
        elif up_sample_method == "CondensedNearestNeighbour":
            X_resampled, y_resampled = CondensedNearestNeighbour().fit_resample(train_code, train_label)
        elif up_sample_method == "EditedNearestNeighbours":
            X_resampled, y_resampled = EditedNearestNeighbours().fit_resample(train_code, train_label)
        elif up_sample_method == "RepeatedEditedNearestNeighbours":
            X_resampled, y_resampled = RepeatedEditedNearestNeighbours().fit_resample(train_code, train_label)
        elif up_sample_method == "AllKNN":
            X_resampled, y_resampled = AllKNN().fit_resample(train_code, train_label)
        elif up_sample_method == "InstanceHardnessThreshold":
            X_resampled, y_resampled = InstanceHardnessThreshold().fit_resample(train_code, train_label)

        train_frame_resampled = pd.DataFrame({'code_vec': X_resampled, 'error_class_id': y_resampled})
        train_frame_resampled = train_frame_resampled.sample(frac=1).reset_index(drop=True)
        train_data = dataset_get_useful(train_frame_resampled)

    return train_data[0], train_data[1], valid_data[0], valid_data[1]


def get_top_N(np_array, N):
    """获取np数组的前几大的位置"""
    np_array = np_array.copy()
    imax_N = []
    for i in range(N):
        imax_N.append(np_array.argmax())
        np_array[np_array.argmax()] = np_array.min()
    return imax_N


def result_draw(train_cost, train_acc, valid_cost, valid_acc):
    fig = plt.figure(1)
    plt.title('Result Analysis')
    plt.plot(train_cost, color='green', label='training cost')
    plt.plot(valid_cost, color='skyblue', label='validing cost')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('cost')
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    fig_save_path = result_repository + '/result_' + result_name
    plt.savefig(fig_save_path + '/' + 'training_cost_iter.jpg', bbox_inches='tight')

    fig = plt.figure(2)
    plt.title('Result Analysis')
    plt.plot(train_acc, color='red', label='training accuracy')
    plt.plot(valid_acc, color='blue', label='validing accuracy')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    fig_save_path = result_repository + '/result_' + result_name
    plt.savefig(fig_save_path + '/' + 'training_acc_iter.jpg', bbox_inches='tight')


def textcnn_net(ipt, dict_dim, word_dim, num_filters):
    # 词嵌入层
    emb = fluid.layers.embedding(input=ipt, size=[dict_dim, word_dim], dtype='float32')
    # 卷积层
    convs = []
    kernel_sizes = [2, 2, 3, 3, 4, 4]
    for kernel_size in kernel_sizes:
        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=num_filters,
            filter_size=kernel_size,
            act='relu',
            pool_type='max'
        )
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)
    param_attr1 = fluid.ParamAttr(name='batch_norm_w1', initializer=fluid.initializer.Constant(value=1.0))
    bias_attr1 = fluid.ParamAttr(name='batch_norm_b1', initializer=fluid.initializer.Constant(value=0.0))
    fc3 = fluid.layers.batch_norm(input=convs_out, param_attr=param_attr1, bias_attr=bias_attr1)
    drop = fluid.layers.dropout(fc3, dropout_prob=0.5)
    fc_1 = fluid.layers.fc(input=drop, size=512, act='tanh')
    out = fluid.layers.fc(input=fc_1, size=13, act='softmax')
    return out


# 定义长短期记忆网络
def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
    # 第一个全连接层
    fc1 = fluid.layers.fc(input=emb, size=128)
    # 进行一个长短期记忆操作
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1,  # 返回：隐藏状态（hidden state），LSTM的神经元状态
                                         size=128)   # size=4*hidden_size
    # 第一个最大序列池操作
    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    # 第二个最大序列池操作
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')
    # 以softmax作为全连接的输出层，大小为2,也就是正负面
    out = fluid.layers.fc(input=[fc2, lstm2], size=13, act='softmax')
    return out



# 神经网络的输入
paddle.enable_static()
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

corpus = DataProcessor.Corpus()
corpus.get()
if model_name == "TextCNN":
    model = textcnn_net(words, dict_dim=corpus.scale, word_dim=word_dim, num_filters=num_filters)
else:
    model = lstm_net(words, input_dim=corpus.scale)
del corpus

# 使用交叉熵损失函数,描述真实样本标签和预测概率之间的差值
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)  # 平均误差
acc = fluid.layers.accuracy(input=model, label=label)  # 预测精度

# 获取预测程序
valid_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr_rate)  # 使用Adam算法进行优化
opts = optimizer.minimize(avg_cost)

# 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])


def train(train_code, train_label, valid_code, valid_label, epochs):
    def reader_train():  # paddle中生成batch的辅助函数。
        for x, y in zip(train_code, train_label):
            yield x, y

    def reader_valid():  # paddle中生成batch的辅助函数。
        for x, y in zip(valid_code, valid_label):
            yield x, y

    train_scale = {label_i: train_label.count(label_i) for label_i in train_label}

    train_acc_epoch = []
    train_cost_epoch = []
    valid_acc_epoch = []
    valid_cost_epoch = []

    train_acc_iter = []
    train_cost_iter = []
    valid_acc_iter = []
    valid_cost_iter = []
    # ----------------------------------------------------------------------#
    #                              训练分类网络                                #
    # ----------------------------------------------------------------------#
    for epoch in range(epochs):
        # 训练
        train_loader = paddle.batch(
            paddle.reader.shuffle(
                reader_train,  # 把输入数据打乱，把每10个数据组成一个batch。
                buf_size=1000),
            batch_size=200
        )

        train_accs = []
        train_costs = []

        for batch_id, data in enumerate(train_loader()):
            train_cost, train_acc, predict_v, true_v = exe.run(
                program=fluid.default_main_program(),  # 运行主程序
                feed=feeder.feed(list(map(lambda x: x[-2:], data))),  # 给模型喂入数据
                fetch_list=[avg_cost, acc, model.name, label.name]
            )  # fetch 误差、准确率

            train_accs.append(train_acc[0])  # 每个batch的准确率
            train_costs.append(train_cost[0])

            train_acc_iter.append(train_acc[0])
            train_cost_iter.append(train_cost[0])

            # 每5个batch打印一次信息  误差、准确率
            if batch_id % 5 == 0:
                print('Train:%d Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                      (epoch, batch_id, train_cost[0], train_acc[0]))

            # 每个batch进行一次测试评估,测试评估时不进行反向传播优化
            valid_loader = paddle.batch(paddle.reader.shuffle(reader_valid,
                                                              buf_size=200),
                                        batch_size=25)
            valid_accs = []
            valid_costs = []

            for batch_id, data in enumerate(valid_loader()):  # 遍历valid_reader
                valid_cost, valid_acc, predict_v, true_v = exe.run(
                    program=valid_program,  # 运行主程序
                    feed=feeder.feed(list(map(lambda x: x[-2:], data))),  # 给模型喂入数据
                    fetch_list=[avg_cost, acc, model.name, label.name]
                )  # fetch 误差、准确率
                valid_accs.append(valid_acc[0])  # 每个batch的准确率
                valid_costs.append(valid_cost[0])

            valid_cost_mean_batch = (sum(valid_costs) / len(valid_costs))  # 每轮的平均误差
            valid_acc_mean_batch = (sum(valid_accs) / len(valid_accs))  # 每轮的平均准确率
            valid_cost_iter.append(valid_cost_mean_batch)
            valid_acc_iter.append(valid_acc_mean_batch)

        train_cost_mean = (sum(train_costs) / len(train_costs))  # 每轮的平均误差
        train_acc_mean = (sum(train_accs) / len(train_accs))  # 每轮的平均准确率
        train_cost_epoch.append(train_cost_mean)
        train_acc_epoch.append(train_acc_mean)

        # 每个epoch进行一次测试评估,测试评估时不进行反向传播优化
        valid_loader = paddle.batch(paddle.reader.shuffle(reader_valid,
                                                          buf_size=200),
                                    batch_size=25)
        valid_accs = []
        valid_costs = []

        for batch_id, data in enumerate(valid_loader()):  # 遍历valid_reader
            valid_cost, valid_acc, predict_v, true_v = exe.run(
                program=valid_program,  # 运行主程序
                feed=feeder.feed(list(map(lambda x: x[-2:], data))),  # 给模型喂入数据
                fetch_list=[avg_cost, acc, model.name, label.name]
            )  # fetch 误差、准确率
            valid_accs.append(valid_acc[0])  # 每个batch的准确率
            valid_costs.append(valid_cost[0])

        valid_cost_mean = (sum(valid_costs) / len(valid_costs))  # 每轮的平均误差
        valid_acc_mean = (sum(valid_accs) / len(valid_accs))  # 每轮的平均准确率
        valid_cost_epoch.append(valid_cost_mean)
        valid_acc_epoch.append(valid_acc_mean)
        print('valid: %d Cost:%0.5f, Accuracy:%0.5f' % (epoch, valid_cost_mean, valid_acc_mean))

    # 保存模型
    model_save_dir = param_repository + '/param_' + result_name
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('save models to %s' % model_save_dir)
    fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                                  ['words'],  # 推理（inference）需要 feed 的数据
                                  [model],  # 保存推理（inference）结果的 Variables
                                  exe)  # executor 保存 inference model
    # fluid.io.save_params(
    #     executor=exe, dirname=model_save_dir, main_program=fluid.default_main_program()
    # )
    # fluid.io.save_persistables(
    #     executor=exe, dirname=model_save_dir, main_program=fluid.default_main_program()
    # )

    cost_and_acc = {
        'train_cost_iter': train_cost_iter,
        'train_acc_iter': train_acc_iter,
        'valid_cost_iter': valid_cost_iter,
        'valid_acc_iter': valid_acc_iter,
    }

    # 保存结果
    result_save_dir = result_repository + '/result_' + result_name
    # 如果保存路径不存在就创建
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    fp = open(result_save_dir + '/cost_and_acc.txt', mode='w')
    fp.write(str(cost_and_acc))
    fp.close()

    fp = open(result_save_dir + '/train_scale.txt', mode='w')
    fp.write(str(train_scale))
    fp.close()

    # 绘图
    result_draw(
        train_cost=train_cost_iter,
        train_acc=train_acc_iter,
        valid_cost=valid_cost_iter,
        valid_acc=valid_acc_iter,
    )


if __name__ == "__main__":
    # ----------------------------------------------------------------------#
    #                   获取训练集和验证集,并进行训练                             #
    # ----------------------------------------------------------------------#
    # 训练集和验证集
    if CrossValidation is None or CrossValidation != "CrossValidation":
        train_frame = pd.read_excel('./DataSet/DataSet/TrainSet.xlsx')
        valid_frame = pd.read_excel('./DataSet/DataSet/ValidSet.xlsx')
        train_code, train_label, valid_code, valid_label = dataset(
            train_frame=train_frame,
            valid_frame=valid_frame,
            up_sample_method=up_sample_method)

        train(train_code, train_label, valid_code, valid_label, epochs=epochs)
    else:
        param_repository_ori = param_repository
        result_repository_ori = result_repository
        DataSetParts = os.listdir('./DataSet/CrossValidationSets')

        for i in range(len(DataSetParts)):
            train_frame = pd.DataFrame()
            for j in range(len(DataSetParts)):
                if j != i:
                    train_frame_i = pd.read_excel('./DataSet/CrossValidationSets/DataSet_Part' + str(j) + '.xlsx')
                    train_frame = train_frame.append(train_frame_i, ignore_index=True)
            valid_frame = pd.read_excel('./DataSet/CrossValidationSets/DataSet_Part' + str(i) + '.xlsx')

            train_code, train_label, valid_code, valid_label = dataset(
                train_frame=train_frame,
                valid_frame=valid_frame,
                up_sample_method=up_sample_method)

            param_repository = param_repository_ori + os.sep + 'param_part_' + str(i)
            result_repository = result_repository_ori + os.sep + 'result_part_' + str(i)

            train(train_code, train_label, valid_code, valid_label, epochs=epochs)
