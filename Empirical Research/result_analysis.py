"""修复结果分析"""

import pandas as pd
from util.DataProcessor import LabelRepository


def fix_state_statistic(result_df, fix_state_colname):
    """给定修复结果框,计算修复结果占比"""
    result_df_grouped = result_df.groupby(fix_state_colname)

    print("{:^20}|{:^20}".format("fix_state", "fix_count"))
    print("-"*41)
    for index, row in result_df_grouped.count().iterrows():
        print("{:<20}|{:^20}".format(index, row[0]))


def error_class_fix_state_statistic(result_df, label_colname, fix_state_colname):
    labels = LabelRepository()
    labels.map_get()

    result_df = result_df[[label_colname, fix_state_colname]]
    result_df_grouped = result_df.groupby(label_colname)

    statistic_df = pd.DataFrame(columns=["error_class", "CompletelyFixed", "PartiallyFixed", "Unfixed"])
    for error_class_id, error_class_fix_state in result_df_grouped:
        temp_dict={}
        temp_dict["error_class"] = labels.label_map_reverse[error_class_id]

        for fix_state, fix_state_count in error_class_fix_state.groupby(fix_state_colname).count().iterrows():
            temp_dict[fix_state] = fix_state_count[0]
        statistic_df = statistic_df.append(temp_dict,ignore_index=True)

    statistic_df[statistic_df.isnull()] = 0
    return statistic_df


def error_class_statistic(result_df, label_colname):
    labels = LabelRepository()
    labels.map_get()

    result_df = result_df[[label_colname]]
    result_df_grouped = result_df.groupby(label_colname)

    statistic_df = pd.DataFrame(columns=["error_class", "count"])
    for error_class_id, error_class_fix_state in result_df_grouped:
        temp_dict={}
        temp_dict["error_class"] = labels.label_map_reverse[error_class_id]

        temp_dict["count"] = error_class_fix_state.count()[0]
        statistic_df = statistic_df.append(temp_dict,ignore_index=True)

    statistic_df[statistic_df.isnull()] = 0
    return statistic_df


def draw_venn3(subsets):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3, venn3_circles

    my_dpi = 150
    plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)  # 控制图尺寸的同时，使图高分辨率（高清）显示
    g = venn3(subsets=subsets,  # 传入三组数据
              set_labels=('deepfix', 'rlassist', 'macer'),  # 设置组名
              set_colors=("#01a2d9", "#31A354", "#c72e29"),  # 设置圈的颜色，中间颜色不能修改
              alpha=0.8,  # 透明度
              normalize_to=1.0,  # venn图占据figure的比例，1.0为占满
              )
    g = venn3_circles(subsets=subsets,
                      linestyle='--', linewidth=0.8, color="black"  # 外框线型、线宽、颜色
                      )
    plt.show()

# # 读取数据
# result_deepfixS = pd.read_excel(r"data/ALL_fix_state.xlsx", sheet_name="deepfixS_fix_state")
# result_deepfix = pd.read_excel(r"data/ALL_fix_state4Check.xlsx", sheet_name="deepfix_fix_state")
# result_ittk = pd.read_excel(r"data/ALL_fix_state.xlsx", sheet_name="ittk_fix_state")

# result_all = pd.read_excel(r"data/ALL_fix_state.xlsx", sheet_name="ALL_fix_state")
#
# result_all_DmR = pd.read_excel(r"result/statistics.xlsx", sheet_name="deepfix-rlassist")
# result_all_RmD = pd.read_excel(r"result/statistics.xlsx", sheet_name="rlassist-deepfix")
# result_all_Mm_RaD = pd.read_excel(r"result/statistics.xlsx", sheet_name="macer-(d∪r)")
# result_all_RaD_mM = pd.read_excel(r"result/statistics.xlsx", sheet_name="(d∪r)-macer")

# # 统计修复结果
# fix_state_colnames = ["deepfix", "rlassist", "macer"]
#
# for fix_state_colname in fix_state_colnames:
#     print(fix_state_colname)
#     fix_state_statistic(result_deepfix, fix_state_colname)

# #用kolmogorov-Smirnov检验deepfix和iitk是否同分布
# from scipy import stats
# print(stats.kstest(result_deepfixS["error_class_id"], result_ittk["error_class_id"]))
# print(stats.ks_2samp(result_deepfixS["error_class_id"], result_ittk["error_class_id"]))

# # 统计各语法错误类别的修复结果
# label_colname = "error_class_id"
# fix_state_colnames = ["deepfix", "rlassist", "macer"]
# for fix_state_colname in fix_state_colnames:
#     excel_fp = pd.ExcelWriter("result/" + fix_state_colname + "_fix_state_by_error_class.xlsx")
#
#     statistic_df = error_class_fix_state_statistic(
#         result_deepfixS, label_colname=label_colname, fix_state_colname=fix_state_colname
#     )
#
#     statistic_df.to_excel(
#         excel_fp, index=False, sheet_name="deepfix"
#     )
#
#     statistic_df = error_class_fix_state_statistic(
#         result_ittk, label_colname=label_colname, fix_state_colname=fix_state_colname
#     )
#
#     statistic_df.to_excel(
#         excel_fp, index=False, sheet_name="ittk"
#     )
#
#     excel_fp.save()
#     excel_fp.close()

# # 统计各修复工具的异同
# excel_fp = pd.ExcelWriter("result/toolsDiff_fix_state_by_error_class.xlsx")
# label_colname = "error_class_id"
# # deepfix-rlassist
# statistic_df = error_class_statistic(result_all_DmR, label_colname=label_colname)
# statistic_df.to_excel(excel_fp, index=False, sheet_name="deepfix-rlassist")
# # rlassist-deepfix
# statistic_df = error_class_statistic(result_all_RmD, label_colname=label_colname)
# statistic_df.to_excel(excel_fp, index=False, sheet_name="rlassist-deepfix")
# # macer-(d∪r)
# statistic_df = error_class_statistic(result_all_Mm_RaD, label_colname=label_colname)
# statistic_df.to_excel(excel_fp, index=False, sheet_name="macer-(d∪r)")
# # (d∪r)-macer
# statistic_df = error_class_statistic(result_all_RaD_mM, label_colname=label_colname)
# statistic_df.to_excel(excel_fp, index=False, sheet_name="(d∪r)-macer")
#
# excel_fp.save()
# excel_fp.close()

# # 绘制venn图
# deepfix_set = set(result_all.groupby("deepfix").get_group("CompletelyFixed")["program_id"])
# rlassist_set = set(result_all.groupby("rlassist").get_group("CompletelyFixed")["program_id"])
# macer_set = set(result_all.groupby("macer").get_group("CompletelyFixed")["program_id"])
#
# draw_venn3([deepfix_set, rlassist_set, macer_set])




