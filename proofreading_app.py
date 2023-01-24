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
st.set_page_config(page_title="Proofreading App")
st.title("学術論文 校正アプリ")
st.info("学術論文で書くような文章に特化した校正システムです．  \nオプションの変更はサイドバーからできます．")


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
link1 = '[抄録](https://drive.google.com/file/d/17tO7KbtiEa6mnB5xPfBaDeaW8X3qoTvG/view?usp=sharing)'
st.sidebar.markdown(link1, unsafe_allow_html=True)
link2 = '[本論(準備中)]()'
st.sidebar.markdown(link2, unsafe_allow_html=True)
link3 = '[view app source](https://github.com/yamataro-46/proofreading_app)'
st.sidebar.markdown(link3, unsafe_allow_html=True)


# メイン
text = st.text_area("添削したい文章を入力", value="")
# 文末に 。 があるかどうか
pattern = re.compile(r'。$')
res = bool(re.search(pattern, text))
if res == False: text += "。"

play = st.button("校正開始", help="予測には時間がかかります")
st.text("")

if play:
    if mode == "word":
        result = pred_word(text, pr)
    elif mode == "particle":
        result = pred_particle(text, pr)
    
    result_ = result.split('\n')

    if pr == "all":
        st.success("結果を表示します")
        for r in result_:
            st.text(r)
    elif pr=="miss":
        for r in result_:
            if r == "誤りは検出されませんでした":
                st.success(r)
            elif r == "---誤字・誤用を検出---":
                st.error(r)
            else:
                st.text(r)
        
    play = False