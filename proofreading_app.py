from all_wordpred import pred_word
from part_wordpred import pred_particle
import streamlit as st
import os
import re
import tensorflow as tf


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# tf.get_logger().setLevel("ERROR")

#--------------------
mode = "word"
pr = "all"


#----app----
st.title("学術論文 校正アプリ")


# サイドバーの設定
st.sidebar.title("オプション設定")

model = st.sidebar.radio("◆予測モードを選択", 
        ("全単語予測モード", "助詞予測モード"))

if model == "全単語予測モード":
    #st.sidebar.text("全単語モードで実行")
    mode = "word"
elif model == "助詞予測モード":
    #st.sidebar.text("助詞予測モードで実行")
    mode = "particle"

st.sidebar.text("")

print_pattern = st.sidebar.radio("◆出力モードを選択",
                ("全ての結果を出力", "誤り判定語のみを出力"))

if print_pattern == "全ての結果を出力":
    #st.sidebar.text("全ての単語の予測結果を出力")
    pr = "all"
elif print_pattern == "誤り判定語のみを出力":
    #st.sidebar.text("誤り判定になった語のみを出力")
    pr = "miss"

st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("-----------------------------")
st.sidebar.info('システムの概要・精度実験等について')
link = '[抄録](https://drive.google.com/file/d/17tO7KbtiEa6mnB5xPfBaDeaW8X3qoTvG/view?usp=sharing)'
st.sidebar.markdown(link, unsafe_allow_html=True)


# メイン
text = st.text_area("添削したい文章を入力", value="")
# 文末に 。 があるかどうか
pattern = re.compile(r'。$')
res = bool(re.search(pattern, text))
if res == False: text += "。"

play = st.button("予測開始")
st.text("")

if play:
    if mode == "word":
        result = pred_word(text, pr)
        result_ = result.split('\n')
        for r in result_:
            st.text(r)
    elif mode == "particle":
        result = pred_particle(text, pr)
        result_ = result.split('\n')
        for r in result_:
            st.text(r)
        
    play = False