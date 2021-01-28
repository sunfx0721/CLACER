import ply.lex as lex
from ply.lex import TOKEN

# 保留字 控制
reserved = {
    # 关键字
    'auto':     'AUTO',
    'break':    'BREAK',
    'case':     'CASE',
    'char':     'CHAR',
    'const':    'CONST',
    'continue': 'CONTINUE',
    'default':  'DEFAULT',
    'do':       'DO',
    'double':   'DOUBLE',
    'else':     'ELSE',
    'enum':     'ENUM',
    'extern':   'EXTERN',
    'float':    'FlOAT',
    'for':      'FOR',
    'goto':     'GOTO',
    'if':       'IF',
    'int':      'INT',
    'long':     'LONG',
    'register': 'REGISTER',
    'return':   'RETURN',
    'short':    'SHORT',
    'signed':   'SIGNED',
    'sizeof':   'SIZEOF',
    'static':   'STATIC',
    'struct':   'STRUCT',
    'switch':   'SWITCH',
    'type':     'TYPEDEF',
    'union':    'UNION',
    'unsigned': 'UNSIGNED',
    'void':     'VOID',
    'while':    'WHILE',

    # 预编译指令
    'include':  'INCLUDE',
    'define':   'DEFINE',

    # 主函数入口
    'main': 'MAIN',
}

# 分词表
tokens = list(reserved.values()) + [
    # 标识符
    'ID',

    # constants
    'INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN', 'INT_CONST_CHAR',
    'FLOAT_CONST', 'HEX_FLOAT_CONST',
    'CHAR_CONST',
    'WCHAR_CONST',

    # String literals
    'STRING_LITERAL',
    'WSTRING_LITERAL',
    'BAD_CHAR_CONST',
    'BAD_STRING_LITERAL',
    'UNMATCHED_QUOTE',


    # Operators
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
    'OR', 'AND', 'NOT', 'XOR', 'LSHIFT', 'RSHIFT',
    'LOR', 'LAND', 'LNOT',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',

    # Assignment
    'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL',
    'PLUSEQUAL', 'MINUSEQUAL',
    'LSHIFTEQUAL','RSHIFTEQUAL', 'ANDEQUAL', 'XOREQUAL',
    'OREQUAL',

    # Increment/decrement
    'PLUSPLUS', 'MINUSMINUS',

    # Structure dereference (->)
    'ARROW',

    # Conditional operator (?)
    'CONDOP',

    # Delimeters
    'LPAREN', 'RPAREN',         # ( )
    'LBRACKET', 'RBRACKET',     # [ ]
    'LBRACE', 'RBRACE',         # { }
    'COMMA', 'PERIOD',          # . ,
    'SEMI', 'COLON',            # ; :
    'QUOT', 'DQUOT',            # ' "

    # Ellipsis (...)
    'ELLIPSIS',

    # pre-processor
    'PPHASH',       # '#'
]

hex_prefix = '0[xX]'
hex_digits = '[0-9a-fA-F]+'
bin_prefix = '0[bB]'
bin_digits = '[01]+'

# 整型常量
integer_suffix_opt = r'(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?'
decimal_constant = '(0' + integer_suffix_opt + ')|([1-9][0-9]*' + integer_suffix_opt + ')'
octal_constant = '0[0-7]*' + integer_suffix_opt
hex_constant = hex_prefix + hex_digits + integer_suffix_opt
bin_constant = bin_prefix + bin_digits + integer_suffix_opt

# 浮点数常量
exponent_part = r"""([eE][-+]?[0-9]+)"""
fractional_constant = r"""([0-9]*\.[0-9]+)|([0-9]+\.)"""
floating_constant = '((((' + fractional_constant + ')' + exponent_part + '?)|([0-9]+' + exponent_part + '))[FfLl]?)'
binary_exponent_part = r'''([pP][+-]?[0-9]+)'''
hex_fractional_constant = '(((' + hex_digits + r""")?\.""" + hex_digits + ')|(' + hex_digits + r"""\.))"""
hex_floating_constant = '(' + hex_prefix + '(' + hex_digits + '|' + hex_fractional_constant + ')' + binary_exponent_part + '[FfLl]?)'

# 字符串
simple_escape = r"""([a-wyzA-Z._~!=&\^\-\\?'"]|x(?![0-9a-fA-F]))"""
decimal_escape = r"""(\d+)(?!\d)"""
hex_escape = r"""(x[0-9a-fA-F]+)(?![0-9a-fA-F])"""
bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-9])"""

escape_sequence = r"""(\\(""" + simple_escape + '|' + decimal_escape + '|' + hex_escape + '))'

escape_sequence_start_in_string = r"""(\\[0-9a-zA-Z._~!=&\^\-\\?'"])"""

cconst_char = r"""([^'\\\n]|""" + escape_sequence + ')'
char_const = "'" + cconst_char + "'"
wchar_const = 'L' + char_const
multicharacter_constant = "'" + cconst_char + "{2,4}'"
unmatched_quote = "('" + cconst_char + "*\\n)|('" + cconst_char + "*$)"
bad_char_const = r"""('""" + cconst_char + """[^'\n]+')|('')|('""" + bad_escape + r"""[^'\n]*')"""

string_char = r"""([^"\\\n]|""" + escape_sequence_start_in_string + ')'
string_literal = '"' + string_char + '*"'
wstring_literal = 'L' + string_literal
bad_string_literal = '"' + string_char + '*' + bad_escape + string_char + '*"'

# 运算符
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_MOD = r'%'
t_OR = r'\|'
t_AND = r'&'
t_NOT = r'~'
t_XOR = r'\^'
t_LSHIFT = r'<<'
t_RSHIFT = r'>>'
t_LOR = r'\|\|'
t_LAND = r'&&'
t_LNOT = r'!'
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_EQ = r'=='
t_NE = r'!='

# 赋值运算符
t_EQUALS = r'='
t_TIMESEQUAL = r'\*='
t_DIVEQUAL = r'/='
t_MODEQUAL = r'%='
t_PLUSEQUAL = r'\+='
t_MINUSEQUAL = r'-='
t_LSHIFTEQUAL = r'<<='
t_RSHIFTEQUAL = r'>>='
t_ANDEQUAL = r'&='
t_OREQUAL = r'\|='
t_XOREQUAL = r'\^='

# 自增/自减
t_PLUSPLUS = r'\+\+'
t_MINUSMINUS = r'--'

# ->
t_ARROW = r'->'

# ?
t_CONDOP = r'\?'

# 标点符号
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_PERIOD = r'\.'
t_SEMI = r';'
t_COLON = r':'
t_ELLIPSIS = r'\.\.\.'
t_PPHASH = r'\#'


@TOKEN(r'\{')
def t_LBRACE(t):
    return t


@TOKEN(r'\}')
def t_RBRACE(t):
    return t


@TOKEN(floating_constant)
def t_FLOAT_CONST(t):
    return t


@TOKEN(hex_floating_constant)
def t_HEX_FLOAT_CONST(t):
    return t


@TOKEN(hex_constant)
def t_INT_CONST_HEX(t):
    return t


@TOKEN(bin_constant)
def t_INT_CONST_BIN(t):
    return t


@TOKEN(octal_constant)
def t_INT_CONST_OCT(t):
    return t


@TOKEN(decimal_constant)
def t_INT_CONST_DEC(t):
    return t


@TOKEN(string_literal)
def t_STRING_LITERAL(t):
    return t


@TOKEN(multicharacter_constant)
def t_INT_CONST_CHAR(t):
    return t


@TOKEN(char_const)
def t_CHAR_CONST(t):
    return t


@TOKEN(wchar_const)
def t_WCHAR_CONST(t):
    return t


@TOKEN(unmatched_quote)
def t_UNMATCHED_QUOTE(t):
    return t


@TOKEN(bad_char_const)
def t_BAD_CHAR_CONST(t):
    return t


@TOKEN(wstring_literal)
def t_WSTRING_LITERAL(t):
    return t

# unmatched string literals are caught by the preprocessor


@TOKEN(bad_string_literal)
def t_BAD_STRING_LITERAL(t):
    return t


@TOKEN(r"\'")
def t_QUOT(t):
    return t


@TOKEN(r'\"')
def t_DQUOT(t):
    return t


# 复杂分词
def t_ID(t):
    r'[a-zA-Z_$][0-9a-zA-Z_$]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


t_ignore = ' \t\r'


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


def lexer_self():
    """构建分词器"""
    lexer = lex.lex()
    return lexer

