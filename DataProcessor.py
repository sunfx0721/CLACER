import os
import sys
import re
import copy
import numpy as np
import pandas as pd
import lex_analysis


class CodeData:
    """
    定义了一些用于处理c代码作为字符串的函数
    """

    def __init__(self, code_str):
        self.code_str = code_str  # 用来存储代码原本的字符串

        self.code_list = None  # 当调用self.code_str_split方法后生成

        self.code_str_abs = None  # 当调用self.code_abstraction方法后生成

        self.token_frame = None  # 当调用self.token_frame_genr方法后生成
        self.token_frame_abs = None  # 当调用self.tokens_abstraction方法后生成

        self.error_message = None  # 当调用self.get_error_message生成
        self.error_message_useful = None  # 当调用self.error_message_process生成
        self.error_message_useful_abs = None  # 当调用self.error_message_process生成

        self.error_loc_expc = None  # 当调用self.get_error_message生成
        self.error_loc_true = None  # 当调用self.get_error_message生成
        self.error_loc_dis = None  # 当调用self.get_error_message生成

        self.compile_path = 'lexer_file/test.c'
        self.error_message_path = 'lexer_file/error_message.txt'
        self.path = os.getcwd()

    def code_annotation_strip(self):
        """给代码去除注释行"""

        def annotation_interval(code_str):
            """给出注释的区间,annotation_strip的辅助函数"""

            def annotation_dict(code_str):
                """annotation_interval的辅助函数"""
                # 给出注释符号的所有出现位置
                annotation_labels = ['//', '/*', '*/']
                anno_dict = {}
                for annotation_label in annotation_labels:
                    anno_dict[annotation_label] = []
                    flag = -2
                    while flag != -1:
                        if flag == -2:
                            flag = code_str.find(annotation_label)
                        else:
                            flag = code_str.find(annotation_label, flag + 2)
                        anno_dict[annotation_label].append(flag)
                    del anno_dict[annotation_label][-1]
                return anno_dict

            anno_dict = annotation_dict(code_str)
            # 确定多行注释
            anno_intervals = []
            for i in range(len(anno_dict['/*'])):
                for j in range(len(anno_dict['*/'])):
                    if anno_dict['/*'][i] <= anno_dict['*/'][j]:
                        anno_intervals.append([anno_dict['/*'][i], anno_dict['*/'][j] + 2])
                        break
            # 确定单行注释
            for i in range(len(anno_dict['//'])):
                for j in range(anno_dict['//'][i], len(code_str)):
                    if code_str[j] == '\n':
                        anno_intervals.append([anno_dict['//'][i], j])
                        break
            # 对注释区间进行排序
            anno_intervals = sorted(anno_intervals, key=lambda x: x[0])
            # 对注释区间去冗余
            index = 0
            while index < len(anno_intervals) - 1:
                if anno_intervals[index][0] <= anno_intervals[index + 1][0] \
                        and anno_intervals[index][1] >= anno_intervals[index + 1][1]:
                    del anno_intervals[index + 1]
                else:
                    index += 1
            return anno_intervals

        anno_interval = annotation_interval(code_str=self.code_str)
        code_str = list(self.code_str)
        for interval in anno_interval:
            for i in range(interval[0], interval[1]):
                if code_str[i] != '\n':
                    code_str[i] = ' '
        code_str = ''.join(code_str)
        self.code_str = code_str

    def code_row_embedding(self):
        """给代码加上行标"""
        code = copy.deepcopy(self.code_str)
        code = code.split('\n')

        for line_no in range(len(code)):
            code[line_no] = '%3d' % (line_no + 1) + ' ' + code[line_no]

        self.code_str = '\n'.join(code)

    def code_row_remove(self):
        code = copy.deepcopy(self.code_str)
        code = code.split('\n')

        for line_no in range(len(code)):
            code[line_no] = code[line_no][4:]

        self.code_str = '\n'.join(code)

    def get_error_message(self, strip=True):
        """获取代码的编译错误信息"""
        if strip:
            self.code_annotation_strip()

        c_file = self.path + '/' + self.compile_path
        error_message_file = self.path + '/' + self.error_message_path
        command = 'gcc -c ' + c_file + ' 2> ' + error_message_file

        with open(c_file, 'w') as f:
            temp = self.code_str
            temp = list(temp)
            while '\r' in temp:
                temp.remove('\r')
            f.write(''.join(temp))
        os.system(command)
        with open(error_message_file, 'r', encoding='UTF-8') as f:
            self.error_message = f.read()

        error_message_list = self.error_message.split('\n')
        path = self.path + '/' + self.compile_path
        remove_len = len(path)
        error_message_list = [error_message_line[remove_len + 1:] for error_message_line in error_message_list]
        self.error_message = '\n'.join(error_message_list)

    def get_first_error_message(self):
        """
        将数据集中错误信息分为各个单错误
        :param error_message: str,错误信息字符串
        """
        # 错误信息转化为列表
        error_message = self.error_message.split('\n')
        errors = []  # 存储分割后的错误
        # 函数索引与错误信息索引(用于定位)
        func_index_list = []
        error_index_list = []

        # 函数匹配与错误信息匹配标准
        func_str = 'In function'
        error_str = 'error:'

        # 求的函数索引与错误信息索引
        for i in range(len(error_message)):
            if func_str in error_message[i]:
                func_index_list.append(i)
            if error_str in error_message[i]:
                error_index_list.append(i)

        error_index_list.append(len(error_message))  # 使得分割完整
        func_index_list.append(len(error_message))  # 便于判断该段错误信息所在函数

        # 生成单条错误信息列表
        for i in range(len(error_index_list) - 1):
            message_start_index = error_index_list[i]
            message_end_index = error_index_list[i + 1]
            # 摘出错误信息并组合
            error = '\n'.join(error_message[message_start_index:message_end_index])
            # 加入所在函数
            for j in range(len(func_index_list) - 1):
                if message_start_index >= func_index_list[j] and message_end_index <= func_index_list[j + 1]:
                    error = error_message[func_index_list[j]] + '\n' + error
                    break
            # 添加错误
            errors.append(error)
        self.error_message = errors[0]

    def code_str_split(self, seq='\n', code_str=None, drop=False):
        """将代码字符串根据分隔符分解为代码列表"""
        if code_str is None:
            code_str = self.code_str
        code_list = code_str.split(seq)
        if drop:
            self.code_list = code_list
        else:
            return code_list

    def code_lines_pickup(self, index, code_str=None):
        """摘出指定行代码"""
        if code_str is None:
            code_str = self.code_str
        code_list = self.code_str_split(code_str=code_str)
        return code_list[index - 1]

    def token_frame_genr(self):
        """调用分词器将c代码进行分词,并将分词结果以DataFrame的数据结构返回"""

        def get_and_label_clibs(token_frame):
            """在分词数据生成后方可调用,用于生成clibs,并将分词信息中的头文件给出"""
            if token_frame is None:
                print("Warning:分词数据未生成！")

            try:
                tokens = token_frame
                include_flag = False
                clibs = []  # 用于存储c库文件的库名
                drop_indexs = []
                for index in list(tokens.index):
                    if tokens.loc[index, 'value'] == 'include':
                        include_flag = True
                        continue

                    if tokens.loc[index, 'type'] == 'ID' and include_flag:
                        # 把当前c库名加入到clibs中
                        clibs.append(tokens.loc[index, 'value'])
                        include_flag = False
                        if tokens.loc[index + 1, 'type'] == 'PERIOD':
                            # 表示该库文件带有后缀,处理后缀,并将分词类型改为LIB
                            tokens.loc[index, 'value'] = tokens.loc[index, 'value'] + \
                                                         tokens.loc[index + 1, 'value'] + \
                                                         tokens.loc[index + 2, 'value']
                            tokens.loc[index, 'type'] = "LIB"
                            drop_indexs.extend([index + 1, index + 2])
            except TypeError:
                print("请调用self.token_frame_genr()生成分词信息")
            else:
                tokens.drop(index=drop_indexs, inplace=True)
                token_frame = tokens.reset_index(drop=True)
                return clibs, token_frame

        def clibs_funtion_modify(path, clibs, token_frame):
            """将clibs的库函数进行替换"""
            lib_func_dir = path + '/' + 'lexer_file' + '/' + 'lib_func_dict.txt'
            lib = open(lib_func_dir, mode='r')
            lib_func_dict = eval(lib.read())
            lib.close()

            # 将包含的库函数加入到搜索列表
            lib_funcs_included = []
            for key in lib_func_dict:
                if key in clibs:
                    lib_funcs_included.extend(lib_func_dict[key])

            for index in list(token_frame[token_frame['type'] == 'ID'].index):
                if token_frame.loc[index, 'value'] in lib_funcs_included:
                    token_frame.loc[index, 'type'] = 'APIcall'
            return token_frame

        def macro_modify(token_frame):
            """将宏命令类型进行补充"""
            macro_list = ['define', 'include']
            for index in list(token_frame.index):
                if token_frame.loc[index, 'value'] in macro_list:
                    token_frame.loc[index, 'type'] = 'MACRO'
            return token_frame

        def main_modify(token_frame):
            """修改分词中的主函数类型"""
            token_frame.loc[token_frame['value'] == 'main', 'type'] = 'MAIN'
            return token_frame

        # 使用自己编写的分词器
        lexer = lex_analysis.lexer_self()

        # 向lexer输入数据
        lexer.input(self.code_str)

        # 分词结果列表
        toke_list = []
        toke_list_type = []
        toke_list_value = []
        toke_list_lineno = []
        toke_list_lexpos = []

        # 获取分词数据
        while True:
            tok = lexer.token()
            if not tok:
                break
            toke_list.append(tok)
            toke_list_type.append(tok.type)
            toke_list_value.append(tok.value)
            toke_list_lineno.append(tok.lineno)
            toke_list_lexpos.append(tok.lexpos)

        # 将分词结果存储到DataFame中
        self.token_frame = pd.DataFrame({'type': toke_list_type,
                                         'value': toke_list_value,
                                         'lineno': toke_list_lineno,
                                         'lexpos': toke_list_lexpos
                                         })
        clibs, self.token_frame = get_and_label_clibs(token_frame=self.token_frame)
        self.token_frame = clibs_funtion_modify(self.path, clibs, self.token_frame)
        self.token_frame = macro_modify(self.token_frame)
        self.token_frame = main_modify(self.token_frame)

    def code_abstraction(self):
        """代码字符串抽象化(当前仅将标识符抽象化为ID,需要把常量和字符串进行抽象化)"""
        self.code_str_abs = self.code_str
        code_str_list = list(self.code_str_abs)
        indexs = list(reversed(list(self.token_frame.index)))

        # 将标识符抽象化
        for index in indexs:
            token_type = self.token_frame.loc[index, 'type']
            token_value = self.token_frame.loc[index, 'value']
            start_pos = self.token_frame.loc[index, 'lexpos']  # 无论该词是否抽象化,都需要该分词的起始位置

            if ('CONST' in token_type and 'const' not in token_value) or 'STRING' in token_type or 'ID' == token_type or 'CHAR' in token_type:
                end_pos = start_pos + len(self.token_frame.loc[index, 'value'])
                if 'CONST' in token_type:
                    code_str_list[start_pos:end_pos] = list('CONST')
                if 'ID' in token_type:
                    code_str_list[start_pos:end_pos] = list('ID')
                if 'STRING' in token_type:
                    code_str_list[start_pos:end_pos] = list('STRING')
                if 'CHAR' in token_type and 'CHAR' != token_type:
                    code_str_list[start_pos:end_pos] = list('CHAR')

            # 在代码抽象化的过程中将代码进行格式化:如果一个分词前不包含空格,那个在该分词前添加空格,以便于代码向量化处理
            if code_str_list[start_pos - 1] not in [' ', '\r', '\t', '\n'] and start_pos != 0:
                code_str_list.insert(start_pos, ' ')

        # 生成为字符串
        self.code_str_abs = ''.join(code_str_list)

    def error_message_process(self):
        """输入error_message，返回编译报错行以及有效编译报错"""

        def get_error_loc_expc(error_message):
            """获取编译器提示的错误位置信息error_loc_expc"""
            # 错误行标匹配模式
            mode_loc = re.compile(r"(.*)error:", re.S)
            mode_lineno = re.compile(r"[0-9]+", re.S)

            try:
                error_message = error_message.split('\n')
                find_flag = False
                while not find_flag:
                    if 'error:' not in error_message[0]:
                        del error_message[0]
                    else:
                        find_flag = True
                error_message = '\n'.join(error_message)

                loc = re.findall(mode_loc, error_message)
                error_loc_expc = re.findall(mode_lineno, loc[0])
                error_loc_expc = int(error_loc_expc[0])
            except (TypeError, IndexError) as reason:
                print("-" * 50)
                print('错误原因为', reason)
            return error_loc_expc

        def get_error_loc_true(error_message, code_list):
            """若报错位置在报错行首个非空分词且报错信息含有before时,给出上个非注释非空行的行号"""

            # 若报错位置在报错行首个非空分词且报错信息含有before时,给出上个非注释非空行的行号
            def get_complie_loc(error_message):
                """
                获取编译信息的错误位置和错误列（单条错误信息）
                :param error_message:
                :return:
                """
                mode_loc = re.compile(r"(.*)error:", re.S)
                mode_lineno = re.compile(r"[0-9]+", re.S)

                loc = re.findall(mode_loc, error_message)
                lineno_colno = re.findall(mode_lineno, loc[0])
                lineno_error = int(lineno_colno[0])
                colno_error = int(lineno_colno[1])
                return lineno_error, colno_error

            def error_loc_isfirst(code_str, lineno_error, colno_error):
                # 辅助判断报错位置是否为该行的首个非空字符
                try:
                    return code_str[lineno_error - 1][0:colno_error - 1].strip() == ''
                except IndexError:
                    breakpoint()
                    return False

            def get_the_not_empty_lineno_before(code_str, lineno_error):
                """
                在判断错误应该出现在报错行的上面时,给出其向后搜索时的第一个非注释非空行的行号
                :param code_str:代码列表(按行分开)
                :param lineno: 当前报错行
                :return: lineno_before上一个非空的代码行标
                """
                find_flag = False
                lineno_before = lineno_error - 1
                while not find_flag:
                    lineno_before -= 1
                    if code_str[lineno_before].strip() == '':
                        pass
                    else:
                        find_flag = True
                return lineno_before + 1

            def error_loc_isbefore(error_message):
                if 'before' in error_message:
                    return True
                else:
                    return False

            # 去除非error信息
            error_message = error_message.split('\n')
            find_flag = False
            while not find_flag:
                if 'error:' not in error_message[0]:
                    del error_message[0]
                else:
                    find_flag = True
            error_message = '\n'.join(error_message)

            lineno_error, colno_error = get_complie_loc(error_message)
            lineno_before = lineno_error
            if error_loc_isfirst(code_list, lineno_error, colno_error) and error_loc_isbefore(error_message):
                lineno_before = get_the_not_empty_lineno_before(code_list, lineno_error)

            return lineno_before

        def get_error_message_useful(error_message):
            """将错误信息中的有效部分进行提取，返回error_message_useful和error_line_compiler"""
            error_message = error_message.split('\n')
            for i in range(len(error_message)):
                if 'error' in error_message[i]:
                    error_line_compiler = error_message[i + 1]
                    error_message = error_message[i]
                    break

            error_message = list(error_message)
            useful_start_pos = 0
            useful_end_pos = len(error_message)

            # 找到有效信息的起始位置
            times = 0
            for i in range(len(error_message)):
                if error_message[i] == ':':
                    times += 1
                if times == 3:
                    useful_start_pos = i + 1
                    break

            for i in list(reversed(range(len(error_message)))):
                if error_message[i] == '(' and error_message[i - 1] not in ["‘", "'"] and error_message[i + 1] not in [
                    "’", "'"]:
                    useful_end_pos = i

            return ''.join(error_message[useful_start_pos:useful_end_pos]), error_line_compiler

        def error_message_abstract(error_message_useful, token_frame, error_line_compiler=''):
            """
            对有效错误信息抽象化
            :param error_line_compiler:
            :param error_message_useful: 有效错误信息
            :param token_frame: 目标代码的token_frame
            :return: error_message_useful_abs
            """
            error_message_c = CodeData(error_message_useful)
            error_message_c.code_annotation_strip()
            error_message_useful = error_message_c.code_str
            error_toke_mode_new = re.compile(r"['](.*?)[']", re.S)  # 错误信息包含的程序分词识别模式

            error_tokens = re.findall(error_toke_mode_new, error_message_useful)  # 找到错误信息中的分词

            # error_tokens
            if error_tokens:
                index_curr = 0
                error_message_list = list(error_message_useful)

                for i in range(len(error_tokens)):
                    find_flag = False
                    while not find_flag:
                        start_index = error_message_useful.find(error_tokens[i], index_curr) - 1  # 找到该词的起始位置包括‘
                        if error_message_useful[start_index] == '‘' or error_message_useful[start_index] == "'":
                            find_flag = True
                        index_curr = start_index + 2

                    end_index = start_index + len(error_tokens[i]) + 2
                    error_tokens[i] = [error_tokens[i]] + [start_index, end_index]
                    index_curr = start_index

                for i in list(reversed(list(range(len(error_tokens))))):
                    # 查找分词信息里面的分词类型
                    token = error_tokens[i][0]
                    token_type = token_frame.loc[token_frame['value'] == token]

                    # 获取起始位置和终点位置
                    start_pos = error_tokens[i][1]
                    end_pos = error_tokens[i][2]

                    if not token_type.empty:
                        token_index = list(token_type.index)[0]  # 获取该分词第一次出现时的类型
                        token_type = token_type.loc[token_index, 'type']  # 获得分词的分词类型

                        # 错误信息抽象化(目前仅抽象标识符、常量、字符串)
                        if 'CONST' in token_type or 'STRING' in token_type or 'ID' in token_type or 'CHAR' in token_type:
                            if 'CONST' in token_type:
                                error_message_list[start_pos:end_pos] = list(' CONST ')
                            if 'ID' in token_type:
                                error_message_list[start_pos:end_pos] = list(' ID ')
                            if 'STRING' in token_type:
                                error_message_list[start_pos:end_pos] = list(' STRING ')
                            if 'CHAR' in token_type:
                                error_message_list[start_pos:end_pos] = list(' CHAR ')
                        else:
                            # 去除引号
                            error_message_list[end_pos - 1] = ' '
                            error_message_list[start_pos] = ' '
                    else:
                        # 去除引号
                        error_message_list[end_pos - 1] = ' '
                        error_message_list[start_pos] = ' '
                return ''.join(error_message_list)
            else:
                return error_message_useful

        self.code_str_split(drop=True)
        self.error_loc_expc = get_error_loc_expc(error_message=self.error_message)
        self.error_loc_true = get_error_loc_true(error_message=self.error_message, code_list=self.code_list)
        self.error_loc_dis = self.error_loc_true - self.error_loc_true
        self.error_message_useful, error_line_compiler = get_error_message_useful(self.error_message)
        self.error_message_useful_abs = error_message_abstract(
            self.error_message_useful, self.token_frame, error_line_compiler=error_line_compiler
        )

    def tokens_abstraction(self, drop=False):
        """代码抽象化,目前只将用户自定义的标识符抽象化为ID"""
        if drop:
            self.token_frame.loc[self.token_frame['type'] == 'ID', 'value'] = 'ID'
        else:
            self.token_frame_abs = self.token_frame.copy(deep=True)
            self.token_frame_abs.loc[self.token_frame_abs['type'] == 'ID', 'value'] = 'ID'

    def tokenframe2codestr(self, tokens=None):
        """将分词信息转化为程序代码(弃用)"""
        if tokens is None:
            tokens = self.token_frame
        linenos = list(tokens.lineno)
        lineno = linenos[0]
        code_str = ''
        for index in list(tokens.index):
            if tokens.loc[index, 'lineno'] == lineno:
                code_str = code_str + ' ' + tokens.loc[index, 'value']
            else:
                code_str = code_str + '\n' + tokens.loc[index, 'value']
                lineno = tokens.loc[index, 'lineno']
        return code_str


class CodeRepository:
    """代码存储库"""
    def __init__(self):
        self.path = os.getcwd()
        self.file_path = 'code_repository/code_repository.txt'  # 词的列表
        self.code_repository = None
        self.scale = None

    def code_repository_genr(self, codes, id_col_name=None, code_col_name=None):
        """根据数据库生成代码库"""
        # codes是Dataframe
        self.code_repository = {}
        if id_col_name is None or code_col_name is None:
            print("请指定代码标签列以及代码数据列")
        code_ids = list(codes[id_col_name])
        for code_id in code_ids:
            self.code_repository[code_id] = codes.loc[
                codes[id_col_name] == code_id, code_col_name
            ].item()
        self.scale = len(self.code_repository)

    def code_repository_update(self, codes_new):
        """根据新增数据库,更新代码库"""
        pass

    def get_code_repository(self):
        """根据语料库文件读取语料库corpus_list"""
        fp = open(self.path + '/' + self.file_path, mode='r', encoding='utf-8')
        self.code_repository = eval(fp.read())
        fp.close()
        self.scale = len(self.code_repository)

    def save(self):
        fp = open(self.path + '/' + self.file_path, mode='w', encoding='utf-8')
        fp.write(str(self.code_repository))
        fp.close()


class SentenceRepository:
    """语句存储库"""
    def __init__(self):
        self.path = os.getcwd()
        self.file_path = 'sentence_repository/sentence_repository.txt'  # 词的列表
        self.repository = None
        self.scale = None
        self.error_line_len_max = None
        self.error_message_len_max = None

    def genr(self, code_repository):
        """根据代码库code_repository生成语句库"""
        self.repository = {}
        for code_id, code_str in code_repository.code_repository.items():
            code = CodeData(code_str=code_str)
            code.code_annotation_strip()
            code.get_error_message()
            try:
                code.get_first_error_message()
            except IndexError:
                breakpoint()
            code.token_frame_genr()
            code.code_abstraction()
            code.error_message_process()
            self.repository[code_id] = [
                code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str_abs),
                code.error_message_useful_abs
            ]
        self.scale = len(self.repository)

    def get(self):
        """根据文件读取语句存储库"""
        """根据语料库文件读取语料库corpus_list"""
        fp = open(self.path + '/' + self.file_path, mode='r')
        self.repository = eval(fp.read())
        fp.close()
        self.scale = len(self.repository)

    def update(self):
        """根据更新文件进行更新"""
        pass

    def save(self):
        fp = open(self.path + '/' + self.file_path, mode='w', encoding='utf-8')
        fp.write(str(self.repository))
        fp.close()


class Corpus:
    """语料库"""
    def __init__(self):
        self.path = os.getcwd()
        self.file_path = 'corpus_file/Corpus.txt'  # 词的列表
        self.corpus_list = None  # 调用self.get_dict_list生成
        self.scale = None  # 调用self.get_dict_list生成
        self.dict_map = None  # 调用self.get_dict_map生成
        self.word_map = None  # 调用self.get_word_map生成

    def genr(self, sentence_repository):
        """根据文本库生成语料库corpus_list"""
        self.corpus_list = ['<blank>', '<unk>']  # 空白填充词,非语料库词
        for key, value in sentence_repository.repository.items():
            sentences = list(map(lambda sentence: sentence.strip().split(), value))
            for sentence in sentences:
                for word in sentence:
                    if word not in self.corpus_list:
                        self.corpus_list.append(word)
        self.scale = len(self.corpus_list)

    def updata(self, codes_new):
        """根据新增文本库,更新语料库corpus_list"""
        pass

    def get(self):
        """根据语料库文件读取语料库corpus_list"""
        fp = open(self.path + '/' + self.file_path, mode='r')
        self.corpus_list = eval(fp.read())
        fp.close()
        self.scale = len(self.corpus_list)

    def get_dict_map(self):
        """根据语料库corpus_list生成从词到数值的映射"""
        self.dict_map = {}
        for index in range(self.scale):
            self.dict_map[self.corpus_list[index]] = index

    def get_word_map(self):
        """获取dict_map的逆映射"""
        self.word_map = {}
        for key, value in self.dict_map.items():
            self.word_map[value] = key

    def text2vec(self, sentence):
        """将输入的文本映射为向量"""
        def get_token_id(token):
            if token in list(self.dict_map.keys()):
                return self.dict_map[token]
            else:
                return self.dict_map['<unk>']

        sentence = sentence.strip().split()
        sentence = list(map(get_token_id, sentence))
        return sentence

    def save(self):
        fp = open(self.path + '/' + self.file_path, mode='w', encoding='utf-8')
        fp.write(str(self.corpus_list))
        fp.close()


class LabelRepository:
    """记录了代码的对应标签以及标签的数值映射"""
    def __init__(self):
        self.path = os.getcwd()
        self.file_path = 'label_file/label_reporistory.txt'  # 标签列表
        self.label_path = 'label_file/label_dict.txt'
        self.label_map = None
        self.repository = None
        self.scale = None

    def map_get(self):
        """获取标签字典"""
        fp = open(self.path + '/' + self.label_path, mode='r')
        self.label_map = eval(fp.read())
        fp.close()
        self.scale = len(self.label_map)

    def map_genr(self, genr_type='appear'):
        """通过给定的顺序生成标签字典"""
        self.label_map = {}
        errorType = []
        if genr_type == 'appear':
            for v in self.repository.values():
                errorType.append(v[2])
        count = 0
        for error in errorType:
            if error not in self.label_map.keys():
                self.label_map[error] = count
                count += 1
        fp = open(self.path + '/' + self.label_path, mode='w')
        fp.write(str(self.label_map))
        fp.close()

    def repository_genr(self, codes, id_col_name=None, label_col_name=None):
        """根据数据库生成标签库"""
        # codes是Dataframe
        self.repository = {}
        if id_col_name is None or label_col_name is None:
            print("请指定代码标签列以及代码数据列")
        code_ids = list(codes[id_col_name])
        for code_id in code_ids:
            try:
                self.repository[code_id] = codes.loc[codes[id_col_name] == code_id, label_col_name].values[0].tolist()
                self.repository[code_id].append(
                    self.repository[code_id][0] + self.repository[code_id][1]
                )
            except TypeError:
                self.repository[code_id][1] = ''
                self.repository[code_id].append(
                    self.repository[code_id][0]
                )

        self.scale = len(self.repository)

    def repository_update(self):
        """标签库更新"""
        pass

    def repository_get(self):
        """根据标签库文件读取标签库"""
        fp = open(self.path + '/' + self.file_path, mode='r')
        self.repository = eval(fp.read())
        fp.close()
        self.scale = len(self.repository)

    def repository_save(self):
        """将标签库进行保存"""
        fp = open(self.path + '/' + self.file_path, mode='w', encoding='utf-8')
        fp.write(str(self.repository))
        fp.close()


class VecRepository:
    """神经网络输入向量库"""
    def __init__(self, error_line_len=None, error_message_len=None):
        self.path = os.getcwd()
        self.file_path = 'vector_repository/vector_repository.txt'  # 词的列表
        self.repository = None
        self.scale = None

        self.error_line_len_max = None
        self.error_message_len_max = None

        if error_line_len is not None:
            self.error_line_len = error_line_len
        else:
            self.error_line_len = None

        if error_message_len is not None:
            self.error_message_len = error_message_len
        else:
            self.error_message_len = None

    def genr(self, sentence_repository, corpus):
        """根据语句库和语料库生成向量库"""
        self.repository = {}
        for prog_id, sentence_list in sentence_repository.repository.items():
            vecs = [corpus.text2vec(sentence=sentence) for sentence in sentence_list]
            self.repository[prog_id] = vecs
        self.scale = len(self.repository)

        self.error_line_len_max = max(list(map(lambda x: len(x[0]), self.repository.values())))
        self.error_message_len_max = max(list(map(lambda x: len(x[1]), self.repository.values())))

        if self.error_line_len is None:
            self.error_line_len = (int(self.error_line_len_max / 10) + 1) * 10
        if self.error_message_len is None:
            self.error_message_len = (int(self.error_message_len_max / 10) + 1) * 10

        for prog_id, vecs in self.repository.items():
            self.repository[prog_id][0] = self.vec_process(
                self.repository[prog_id][0], len_max=self.error_line_len
            )
            self.repository[prog_id][1] = self.vec_process(
                self.repository[prog_id][1], len_max=self.error_message_len
            )

    def vec_process(self, vec_list, len_max, append_token=0):
        """将向量对齐处理"""
        if len(vec_list) > len_max:
            error_list = vec_list[:len_max]
        else:
            while len(vec_list) < len_max:
                vec_list.append(append_token)
        return vec_list

    def get(self):
        """根据语料库文件读取语料库corpus_list"""
        fp = open(self.path + '/' + self.file_path, mode='r')
        self.repository = eval(fp.read())
        fp.close()
        self.scale = len(self.repository)

        self.error_line_len_max = max(list(map(lambda x: len(x[0]), self.repository.values())))
        self.error_message_len_max = max(list(map(lambda x: len(x[1]), self.repository.values())))

        if self.error_line_len is None:
            self.error_line_len = self.error_line_len_max
        if self.error_message_len is None:
            self.error_message_len = self.error_message_len_max

    def save(self):
        fp = open(self.path + '/' + self.file_path, mode='w', encoding='utf-8')
        fp.write(str(self.repository))
        fp.close()


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) >= 2 else "deepfix"
    # dataset_name = 'DataSet'
    print("数据集名称为:", dataset_name)
    print("Yes, there is the main function!")
    print("*————准备测试CodeData类————*")
    if dataset_name == 'deepfix':
        DataSet = r'./DataSet/DataSet_deepfix.csv'
    elif dataset_name == 'tegcer':
        DataSet = r'./DataSet/DataSet_tegcer.csv'
    else:
        DataSet = r'./DataSet/DataSet.csv'

    # 测试CodeData
    with open('lexer_file/test.c', 'r', encoding='UTF-8') as fp:
        codestr = fp.read()
    print(codestr)
    print("-"*50)  # Pass

    code = CodeData(codestr)
    code.code_annotation_strip()
    print(code.code_str)
    print("-"*50)  # Pass

    code.code_row_embedding()
    # print(code.code_str)
    # print("-" * 50)  # Pass

    code.code_row_remove()
    # print(code.code_str)
    # print("-" * 50)  # Pass

    code.get_error_message()
    print(code.error_message)
    print("-" * 50)  # Pass

    code.get_first_error_message()
    print(code.error_message)
    print("-" * 50)  # Pass

    code.token_frame_genr()
    token_frame = code.token_frame
    # breakpoint()  # Pass

    code.code_abstraction()
    print(code.code_str_abs)
    print("-" * 50)  # Pass

    code.error_message_process()
    print(code.error_loc_expc)
    print(code.error_loc_true)
    print(code.error_message_useful)
    print(code.error_message_useful_abs)
    print("-" * 50)  # Pass

    print(code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str))
    print(code.code_lines_pickup(index=code.error_loc_true, code_str=code.code_str_abs))
    print("-" * 50)  # Pass

    print("*————准备测试CodeRepository类————*")
    if DataSet[-3:] == 'csv':
        data = pd.read_csv(DataSet)
    else:
        data = pd.read_excel(DataSet)

    code_repository = CodeRepository()
    code_repository.code_repository_genr(data, id_col_name='program_id', code_col_name='code')
    for key, value in code_repository.code_repository.items():
        code = CodeData(value)
        if '1' in code.code_str[:3]:
            code.code_row_remove()  # 如果代码有行标就加上
        code_repository.code_repository[key] = code.code_str
    code_repository.save()  # Pass
    print('生成成功')
    del code_repository
    code_repository = CodeRepository()
    code_repository.get_code_repository()  # Pass
    print('读取成功')

    print("*————准备测试SentenceRepository类————*")
    code_repository = CodeRepository()
    code_repository.get_code_repository()
    sentence_repository = SentenceRepository()
    sentence_repository.genr(code_repository=code_repository)
    sentence_repository.save()  # Pass

    print("*————准备测试CorpusRepository类————*")
    sentence_repository = SentenceRepository()
    sentence_repository.get()
    corpus = Corpus()
    corpus.genr(sentence_repository=sentence_repository)
    corpus.save()

    del corpus
    corpus = Corpus()
    corpus.get()
    corpus.get_dict_map()
    corpus.get_word_map()
    print(corpus.dict_map['<unk>'])
    print(corpus.dict_map['if'])
    print(corpus.word_map[20])

    print("*————准备测试LabelRepository类————*")
    if DataSet[-3:] == 'csv':
        data = pd.read_csv(DataSet)
    else:
        data = pd.read_excel(DataSet)

    label_repository = LabelRepository()
    label_repository.repository_genr(
        data, id_col_name='program_id', label_col_name=['error_id_1', 'error_id_2']
    )
    label_repository.repository_save()

    del label_repository

    label_repository = LabelRepository()
    label_repository.repository_get()
    label_repository.map_genr()

    print("*————准备测试VecRepository类————*")
    sentence_repository = SentenceRepository()
    sentence_repository.get()

    corpus = Corpus()
    corpus.get()
    corpus.get_dict_map()

    vec_repository = VecRepository()
    vec_repository.genr(sentence_repository=sentence_repository, corpus=corpus)
    print(vec_repository.error_line_len, vec_repository.error_message_len)
    vec_repository.save()

    del vec_repository
    vec_repository = VecRepository()
    vec_repository.get()
    print(vec_repository.error_line_len, vec_repository.error_message_len)

