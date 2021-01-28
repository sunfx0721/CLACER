import os
import sys
import numpy as np
import pandas as pd
import paddle.fluid as fluid
from DataProcessor import *


def SyntacticErrorClassifier(code_str):
    # 修改工作路径
    path_ori = os.getcwd()
    path_target = os.path.dirname(sys.argv[0])
    if path_ori != path_target:
        os.chdir(path_target)

    def get_top_N(np_array, N):
        """获取np数组的前几大的位置"""
        np_array = np_array.copy()
        imax_N = []
        for i in range(N):
            imax_N.append(np_array.argmax())
            np_array[0, np_array.argmax()] = np_array.min()
        return imax_N

    # with open('lexer_file/test.c', 'r', encoding='UTF-8') as fp:
    #     codestr = fp.read()

    code = CodeData(code_str=code_str)
    code.code_annotation_strip()
    code.get_error_message()
    code.get_first_error_message()
    code.token_frame_genr()
    code.code_abstraction()
    code.error_message_process()
    # 需要的数据
    code_str = code.code_str  # "code"
    error_message = code.error_message  # "error_message"
    error_loc_true = code.error_loc_true  # "error_loc_true"
    error_line = code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str)  # "error_line"

    error_line_abs = code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str_abs)  # "error_line_abs"
    error_message_useful_abs = code.error_message_useful_abs  # "error_message_useful_abs"

    corpus = Corpus()
    corpus.get()
    corpus.get_dict_map()

    vecs = [0, 0]
    vecs[0] = corpus.text2vec(error_line_abs)
    vecs[1] = corpus.text2vec(error_message_useful_abs)

    vec_repository = VecRepository(error_line_len=80, error_message_len=20)
    vec_repository.get()
    vecs[0] = vec_repository.vec_process(vecs[0], len_max=vec_repository.error_line_len)
    vecs[1] = vec_repository.vec_process(vecs[1], len_max=vec_repository.error_message_len)

    code_v = vecs[0] + vecs[1]

    # 读取模型
    exe = fluid.Executor(fluid.CPUPlace())
    path = 'param'
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

    code_v = np.array(code_v).astype(dtype='int64').reshape([100, 1])
    code_v = fluid.create_lod_tensor(code_v, [[100]], place=fluid.CPUPlace())

    top_N = 3  # 预测概率前三
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: code_v},
                      fetch_list=fetch_targets)

    top_N_pred = get_top_N(results[0], top_N)

    print(top_N_pred[0])  # "error_class_id"
    print(top_N_pred)  # "error_class_id"

    result_dict = {
        "code": code_str,
        "error_message": error_message,
        "error_loc_true": error_loc_true,
        "error_line": error_line,
        "error_class_id": top_N_pred[0]
    }

    os.chdir(path_ori)
    return result_dict


if __name__ == "__main__":
    code_str = r"""  1 #include <stdio.h>
  2 int main(){
  3     int l[1002],k,n,i,f=0;
  4     int a[500];
  5     //Array l[1000] is for storing the numbers
  6          //Array a[500] is for recording the occurence of each number.
  7     for(i=0;i<500;i++) //Initialising each value of a[500] to 0.
  8     {
  9        a[i]=0;
 10     }
 11     scanf("%d",&k);
 12     scanf("%d",&n);
 13     for(i=0;i<n;i++)
 14     {
 15       scanf("%d",&l[i]);
 16     }
 17     for(i=0;i<n;i++) //Recording the occurence of each number
 18     {
 19         a[l[i]]++;
 20     }
 21     for(i=0;i<k;i++)
 22     { 
 23         if(i<500&&(k-i)<500) 
 24         {
 25             if(a[i]!=0&&a[k-i]!=0) //Both a[i] and a[k-i] are non zero 
 26             {                      //implies that i and k-i occur for
 27                 f=1;                       .
 28                 break;             //at least one i.
 29             }
 30         }
 31     }
 32     if(f)
 33     printf("lucky");
 34     else
 35     printf("unlucky");
 36     return 0;
 37 }"""
    code = CodeData(code_str=code_str)
    code.code_row_remove()
    code_str = code.code_str
    code.get_error_message()
    print(code.error_message)
    result_dict = SyntacticErrorClassifier(code_str)
