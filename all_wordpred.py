import numpy as np
import re
import os
import copy
import pickle
import MeCab
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#---------------------------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#---------------------------------------------


#os.system('git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y -u -p $PWD')




# 必要な関数群
#--------------
# 任意の確率和をしきい値にして正解が含まれるかの率
def predsum_acc(y_true, y_pred):
    thres = 0.80
    y_true = tf.cast(y_true, tf.int32)
    pred_acc = tf.TensorArray(tf.float32, size=len(y_true), dynamic_size=True)
    pred_kmax_values = tf.math.top_k(y_pred, k=len(y_pred[0,0])).values # 各文各単語の確率を降順
    pred_kmax_index = tf.math.top_k(y_pred, k=len(y_pred[0,0])).indices # pred_kmax_valueのindex
    label_pos = tf.cast(tf.reshape(tf.where(tf.equal(y_true, pred_kmax_index)), shape=(len(y_true), len(y_true[0]), -1))[:,:,2], tf.int32) # [文][単語][2] -->確率降順にしたときの正解ラベルのindex番号
    n = 0
    for i in tf.range(len(y_true)):
        for j in tf.range(1,len(y_true[i])):
            if y_true[i,j] == 0: break # 最初のpadding0以降、paddingの0が来たときbreakする(labelは 0,1,5,3,...,2,0,0,0,...,0 だから)
            # paddingの0でないとき
            k = label_pos[i,j] # 該当する正解単語のindex
            pred_sum = tf.reduce_sum(pred_kmax_values[i,j,:k]) # 正解単語の1つ前までの確率和
            if pred_sum < thres or k == 0: # 正解単語の1つ前までの和が閾値を超えていない or 正解単語が1位
                pred_acc = pred_acc.write(n, 1.0)
            else:
                pred_acc = pred_acc.write(n, 0)
            n += 1
    acc = tf.reduce_mean(pred_acc.stack())
    return acc


# 誤字脱字検出メソッド
#--------------
# 文章の前処理関数
def regex_txt(text):
    text_shaped = text.replace(' ', '')
    text_shaped = text_shaped.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})) # 全角->半角
    text_shaped = text_shaped.replace('\n', '') # 改行文字の削除

    text_shaped = re.sub('[「」]', '', text_shaped) # 「」の削除
    text_shaped = re.sub('(\(.+?\))|(\[.+?\],*)', '', text_shaped) # ()[]内の文字を()[]ごと削除

    text_shaped = re.sub('[a-zA-Z0-9]+', 'n', text_shaped) # 英数字を n に置換
    text_shaped = re.sub('n[,.]', 'n', text_shaped) # n, や n. などを n に置換
    text_shaped = text_shaped.replace('、', ',').replace('。', '.').replace(',', '読点').replace('.', '句点') # , . を一旦文字で置き換える
    text_shaped = re.sub('\W', '', text_shaped) # 非単語文字(半角記号など)を''に置換
    text_shaped = re.sub('_', '', text_shaped)
    text_shaped = re.sub('n{2,}', 'n', text_shaped) # nn, nnn, ... などの連続したnをnに置換
    text_shaped = text_shaped.replace('読点', '、').replace('句点', '。')
    text_shaped = re.sub('、{2,}', '、', text_shaped)
    text_shaped = re.sub('(。{2,})|(、{1,}。)|(。{1,}、)', '。', text_shaped)
    return text_shaped

# 入力テキストをtoken化
def text_to_id(texts, vocab_dic):
    mecab = MeCab.Tagger('-r /dev/null -d /mecab-ipadic-neologd/')
    terms = []
    posList = []
    texts = regex_txt(texts).split('。')
    for text in texts:
        words = []
        poss = []
        node = mecab.parseToNode(text)
        while node:
            term = node.surface
            pos = node.feature.split(',')[0]
            if pos not in '補助記号' or term == '、' or term == '。':
                if term != '':
                    words.append(term)
                    poss.append(pos)
            node = node.next
        words.append('。')
        terms.append(words)
        posList.append(poss)
    terms.pop()
    posList.pop()
  
    text_id = []
    for i in range(len(terms)):
        term = terms[i]
        word_id = []
        for j in range(len(term)):
            word = term[j]
            try: # 語彙辞書に該当単語があったら
                word_id.append(vocab_dic[word])
            except: # 該当語が未知語の場合
                pos_ = posList[i][j]
                if pos_ == '名詞':
                    word = 'norn'
                elif pos_ == '動詞':
                    word = 'verb'
                elif pos_ == '形容詞':
                    word = 'adj'
                elif pos_ == '副詞':
                    word = 'adv'
                elif pos_ == '助詞':
                    word = 'post'
                else:
                    word = 'any'
                try:
                    word_id.append(vocab_dic[word])
                except:
                    word_id.append(vocab_dic['<UNK>'])
        text_id.append(word_id)
    return terms, posList, text_id

# modelに入力する用のデータを作成
def cre_InputData(texts, ids):
    fwInputs = []
    bwInputs = []
    label = []
    test_texts = []

    for i in range(len(texts)):
        input1 = copy.copy(ids[i])
        input1[0:0] = [0, 0]
        input2 = copy.copy(ids[i])
        input2.append(0)
        input2.append(0)
        output = copy.copy(ids[i])
        output[0:0] = [0]
        output.append(0)
        test_text = []

        fwInputs.append(input1)
        bwInputs.append(input2)
        label.append(output)

        for j in range(len(texts[i])):
            text_ = copy.copy(texts[i])
            text_[j] = '< >'
            test_text.append(text_)
        test_texts.append(test_text)

    fwInputs = pad_sequences(fwInputs, padding='post')
    bwInputs = pad_sequences(bwInputs, padding='post')
    label = pad_sequences(label, padding='post')
    fwInputs = np.reshape(fwInputs, (fwInputs.shape[0], fwInputs.shape[1], 1))
    bwInputs = np.reshape(bwInputs, (bwInputs.shape[0], bwInputs.shape[1], 1))
    label = np.reshape(label, (label.shape[0], label.shape[1], 1))
    return fwInputs, bwInputs, label, test_texts


# カスタム評価関数
#--------------
# selfのacc
def custom_acc_(y_true, y_pred): # 全文全単語の予測結果の平均を正解率とする
    pred_acc = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    false_indexes = [] # false(誤り判定)になった箇所のindexを格納(python配列にするのは妥協点)
    y_true = tf.cast(y_true, tf.int32)
    pred_argmax = tf.reshape(tf.cast(tf.math.argmax(y_pred, 2), tf.int32), shape=(len(y_true), len(y_true[0]), -1))# 予測確率が最も高いindex
    match = tf.cast(tf.math.equal(y_true, pred_argmax), tf.float32) # 予測確率が最も高い語とlabelが一致しているか
    n = 0
    for i in tf.range(len(y_true)):
        for j in tf.range(1, len(y_true[0])):
            if y_true[i,j] == 0: break
            pred_acc = pred_acc.write(n, match[i,j,0])
            n += 1
        false_index = tf.where(tf.equal(match[i,1:j], 0)).numpy()[:,0] # 誤り判定になった語のindexを取得
        false_indexes.append(false_index)
    # 正解率
    acc = tf.reduce_mean(pred_acc.stack())
    return acc, false_indexes

# 任意の確率和をしきい値にして正解が含まれるかの率
def predsum_acc_(y_true, y_pred):
    thres = 0.80
    y_true = tf.cast(y_true, tf.int32)
    false_indexes = []
    pred_acc = tf.TensorArray(tf.float32, size=len(y_true), dynamic_size=True)
    pred_kmax_values = tf.math.top_k(y_pred, k=len(y_pred[0][0])).values # 各文各単語の確率を降順
    pred_kmax_index = tf.math.top_k(y_pred, k=len(y_pred[0][0])).indices # pred_kmax_valueのindex
    label_pos = tf.reshape(tf.where(tf.equal(y_true, pred_kmax_index)), shape=(len(y_true), len(y_true[0]), -1))[:,:,2]
    n = 0
    for i in tf.range(len(y_true)):
        pred_acc_j = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        for j in tf.range(1,len(y_true[i])):
            if y_true[i,j] == 0: break # 最初のpadding0以降、paddingの0が来たときbreakする(labelは 0,1,5,3,...,2,0,0,0,...,0 だから)
            # paddingの0でないとき
            k = label_pos[i][j] # 該当する正解単語のindex
            pred_sum = tf.reduce_sum(pred_kmax_values[i][j][:k]) # 正解単語の1つ前までの確率和
            if pred_sum < thres or k == 0: # 正解単語の1つ前までの和が閾値を超えていない or 正解単語が1位
                pred_acc = pred_acc.write(n, 1.0)
                pred_acc_j = pred_acc_j.write(j-1, 1.0)
            else:
                pred_acc = pred_acc.write(n, 0)
                pred_acc_j = pred_acc_j.write(j-1, 0)
            n += 1
        false_index = tf.where(tf.equal(pred_acc_j.stack(), 0)).numpy()[:,0] # 誤り判定になった語のindexを取得
        false_indexes.append(false_index)
    acc = tf.reduce_mean(pred_acc.stack())
    return acc, false_indexes




#---------------------------------------------

# 誤字検出システムの根幹
#--------------
# 入力文の予測結果を出力する
def texts_pred(pred, label, test_texts, vocab_dic, dic_vocab):
    result = ''
    for i in range(len(test_texts)):
        for j in range(len(test_texts[i])):
            # 予測
            y_index = pred[i][j+1].argsort()[::-1][0:5]
            y_prob = np.sort(pred[i][j+1])[::-1][0:5]

            # 入力語
            label_id = label[i][j+1][0]
            word = dic_vocab[label_id]
            prob = pred[i][j+1][label_id]


            # print(test_texts[i][j])
            result += str(test_texts[i][j]) + '\n'

            # print('----入力----')
            # print(word, prob)
            # print()
            result += '----入力----\n'
            result += word + ' ' + str(prob) + '\n'

            # print('----予測----')
            result += '----予測----\n'
            for k in range(3):
                #print(dic_vocab[y_index[k]], y_prob[k])
                result += dic_vocab[y_index[k]] + ' ' + str(y_prob[k]) + '\n'
            result += '\n'
    return result
        

# 評価関数を改良して、不正解判定になった箇所を返すようにする
# 該当する文章と誤り箇所を出力 モデル予測による適切かも単語も出力する
def miss_check(method_acc, pred, label, test_texts, vocab_dic, dic_vocab):
    result = ''
    acc, falIndex = method_acc(label, pred)
    # print('正解率: ', acc.numpy())
    # print()
    result += '正解率: ' + str(acc.numpy()) + '\n'

    for i in range(len(falIndex)):
        if falIndex[i].size == 0:
            #print('誤字は検出されませんでした')
            result += '誤りは検出されませんでした\n'

        else:
            # print('---誤字を検出---')
            # print()
            result += '---誤字を検出---\n'
            try:
                for j in range(len(falIndex[i])):
                    fi = falIndex[i][j]
                    label_id = test_texts[i][fi].index('< >')

                    try:
                        word = dic_vocab[label[i][fi+1][0]]
                        prob = pred[i][fi+1][vocab_dic[word]]
                    except:
                        word = 'nan'
                        prob = 0.0

                    test_text = copy.copy(test_texts[i][fi])
                    test_text[label_id] = '<<' + word + '>>'

                    y_index = pred[i][fi+1].argsort()[::-1][0:5]
                    y_prob = np.sort(pred[i][fi+1])[::-1][0:5]


                    # print(test_text)
                    # print('------入力------')
                    # print(word, prob)
                    # print()
                    #print('----適切かもしれない語群----')
                    result += str(test_text) + '\n'
                    result += '------入力------\n'
                    result += word + ' ' + str(prob) + '\n'
                    result += '----適切かもしれない語----\n'
                    for j in range(len(y_index)):
                        # print(dic_vocab[y_index[j]], y_prob[j])
                        result += dic_vocab[y_index[j]] + ' ' + str(y_prob[j]) + '\n'
                    # print()
                    result += '\n'
            except:
                continue
    return result


#-------------------------
# メイン

def pred_word(text, mode='all'):

    # 保存した辞書のload
    with open('data/vocab_dic.pkl', 'rb') as f:
        vocab_dic = pickle.load(f)
    with open('data/dic_vocab.pkl', 'rb') as f:
        dic_vocab = pickle.load(f)
    with open('data/emb_matrix.pkl', 'rb') as f:
        emb_matrix = pickle.load(f)


    # model のロード
    path = 'data/all_word_model.h5'
    # model = load_model(path, custom_objects={'predsum_acc':predsum_acc})
    model = cre_model(emb_matrix, dic_vocab)
    model.load_weights(path)
    # print(model.summary())

    texts, pos, ids = text_to_id(text, vocab_dic)
    fwInputs, bwInputs, label, test_texts = cre_InputData(texts, ids)
    pred = model.predict([fwInputs, bwInputs])

    if mode == 'all':
        result = texts_pred(pred, label, test_texts, vocab_dic, dic_vocab)
    elif mode == 'miss':
        result = miss_check(predsum_acc_, pred, label, test_texts, vocab_dic, dic_vocab)
    
    # print(result)
    return result




# ---------------------------------------------------
# 2023/1/18
# load_modelでValueError: bad marshal data (unknown type code)が発生してしまう
# 原因不明
# モデルを新しく定義して、重みをloadする形で応急処置
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, TimeDistributed, concatenate, Lambda, Dropout
from tensorflow.keras.models import Model

def cre_model(emb_matrix, dic_vocab):
    # model構築
    # RNN model
    #----------
    # 順方向
    input1 = Input(shape=(None,), name='forward')
    # 逆方向
    input2 = Input(shape=(None,), name='backward')

    emb1 = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1],
                    weights=[emb_matrix], mask_zero=True, name='emb1')(input1)
    emb2 = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1],
                    weights=[emb_matrix], mask_zero=True, name='emb2')(input2)

    forlstm = LSTM(256, dropout=0, return_sequences=True, name='forlstm')(emb1)
    backlstm = LSTM(256, dropout=0, return_sequences=True, go_backwards=True, name='backlstm')(emb2)
    backlstm = Lambda(lambda x : x[:,::-1], name='rev_backlstm')(backlstm)

    bilstm = concatenate([forlstm, backlstm], name='bilstm')

    output = TimeDistributed(Dense(1024, activation='relu'), name='dense1')(bilstm)
    output = Dropout(0.2)(output)
    output = TimeDistributed(Dense(len(dic_vocab), activation='softmax'), name='softmax')(output)

    model = Model(inputs=[input1, input2], outputs=output)
    # model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', predsum_acc])

    return model