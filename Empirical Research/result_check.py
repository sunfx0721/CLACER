import os
import pandas as pd

from util.helpers import my_txt_reader

code_deepfix_path = r"../data/Codes/Code_deepfix.txt"
code_ittk_path = r"../data/Codes/Code_ittk.txt"

result_deepfix_path = r"data/deepfix.txt"
result_rlassist_path = r"data/RLAssist.txt"
result_macer_path = r"data/macer.txt"

result_ALL_path = r"data/ALL_fix_state4Check.xlsx"

program_id = []
with open(code_deepfix_path, mode='r') as fp_codes:
    program_id.extend(eval(fp_codes.read()).keys())
with open(code_ittk_path, mode='r') as fp_codes:
    program_id.extend(eval(fp_codes.read()).keys())

result_deepfix = my_txt_reader(result_deepfix_path)
result_rlassist = my_txt_reader(result_rlassist_path)
result_macer = my_txt_reader(result_macer_path)

result_col = ['program_id', 'deepfix', 'rlassist', 'macer']
result_df = pd.DataFrame(columns=result_col)

result_df[result_col[0]] = program_id

for i in result_df.index:
    program_id = result_df.loc[i, "program_id"]

    fix_state_d = result_deepfix[program_id] if program_id in result_deepfix.keys() else "Unfixed"
    fix_state_r = result_rlassist[program_id] if program_id in result_rlassist.keys() else "Unfixed"
    fix_state_m = result_macer[program_id] if program_id in result_macer.keys() else "Unfixed"

    result_df.loc[i, result_col[1:]] = [fix_state_d, fix_state_r, fix_state_m]

result_df_deepfix = result_df[result_df["program_id"].map(lambda x:"prog" in x)]
result_df_ittk = result_df[result_df["program_id"].map(lambda x:"tegcer" in x)]

with pd.ExcelWriter(result_ALL_path) as excel_fp:
    result_df.to_excel(excel_fp, index=False, sheet_name="ALL_fix_state")
    result_df_deepfix.to_excel(excel_fp, index=False, sheet_name="deepfixS_fix_state")
    result_df_ittk.to_excel(excel_fp, index=False, sheet_name="ittk_fix_state")



