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

# 必要な関数群
#--------------
# 任意の確率和をしきい値にして正解が含まれるかの率
def predsum_acc(y_true, y_pred):
    thres = 0.80
    y_true = tf.reshape(tf.cast(y_true, tf.int32), shape=(-1, 1))
    pred_acc = tf.TensorArray(tf.float32, size=len(y_true))
    pred_kmax_values = tf.math.top_k(y_pred, k=len(y_pred[0])).values # 各文各単語の確率を降順
    pred_kmax_index = tf.math.top_k(y_pred, k=len(y_pred[0])).indices # pred_kmax_valueのindex
    label_pos =  tf.reshape(tf.where(tf.equal(y_true, pred_kmax_index)), shape=(len(y_true), -1))[:,1]
    for i in range(len(y_true)):
        k = tf.cast(label_pos[i], tf.int32) # 該当する正解単語のindex
        pred_sum = tf.reduce_sum(pred_kmax_values[i,:k]) # 正解単語の1つ前までの確率和
        if pred_sum < thres or k == 0: # 正解単語の1つ前までの和が閾値を超えていない or 正解単語が1位
            pred_acc = pred_acc.write(i, 1.0)
        else:
            pred_acc = pred_acc.write(i, 0)

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
    mecab = MeCab.Tagger('-r /dev/null -d /mecab-ipadic-neologd/dicrc')
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
                # print(word, ': 未知語')
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

# 品詞名を出力する関数
def get_pos(text):
    mecab = MeCab.Tagger(r'-d "/mecab-ipadic-neologd"')
    pos_list = []
    node = mecab.parseToNode(text)
    while node:
        pos = node.feature.split(',')[0]
        if pos != 'BOS/EOS':
            pos_list.append(pos)
        node = node.next
    return pos_list

# modelに入力する用のデータを作成
def cre_InputData(texts, poss, ids):
    fwInputs = []
    bwInputs = []
    label = []
    test_texts = []

    for i in range(len(texts)):
        for j in range(len(poss[i])):
            if poss[i][j] == '助詞':
                fwInputs.append(ids[i][:j])
                bwInputs.append(ids[i][j+1:])
                label.append(ids[i][j])

                text_ = copy.copy(texts[i])
                text_[j] = '< >'
                test_texts.append(text_)

    fwInputs = pad_sequences(fwInputs, padding='post')
    bwInputs = pad_sequences(bwInputs, padding='post')
    fwInputs = np.reshape(fwInputs, (fwInputs.shape[0], fwInputs.shape[1], 1))
    bwInputs = np.reshape(bwInputs, (bwInputs.shape[0], bwInputs.shape[1], 1))
    label = np.array(label)

    return fwInputs, bwInputs, label, test_texts


# カスタム評価関数
#--------------
# selfのacc
def custom_acc_(y_true, y_pred):
    y_true_ = tf.reshape(tf.cast(y_true, tf.int32), shape=(-1, 1))
    pred_argmax = tf.reshape(tf.cast(tf.math.argmax(y_pred, 1), tf.int32), shape=(-1, 1))
    matched = tf.cast(tf.math.equal(y_true_, pred_argmax), tf.float32)
    false_index = tf.where(tf.equal(matched, 0)).numpy()
    acc = tf.reduce_mean(matched)
    return acc, false_index, 1

# 任意の確率和をしきい値にして正解が含まれるかの率
def predsum_acc_(y_true, y_pred):
    thres = 0.80
    y_true = tf.reshape(tf.cast(y_true, tf.int32), shape=(-1, 1))
    pred_acc = tf.TensorArray(tf.float32, size=len(y_true))
    acc_count = tf.TensorArray(tf.int32, size=len(y_true))
    pred_kmax_values = tf.math.top_k(y_pred, k=len(y_pred[0])).values # 各文各単語の確率を降順
    pred_kmax_index = tf.math.top_k(y_pred, k=len(y_pred[0])).indices # pred_kmax_valueのindex
    label_pos =  tf.reshape(tf.where(tf.equal(y_true, pred_kmax_index)), shape=(len(y_true), -1))[:,1]
    for i in range(len(y_true)):
        k = tf.cast(label_pos[i], tf.int32) # 該当する正解単語のindex
        pred_sum = tf.reduce_sum(pred_kmax_values[i,:k]) # 正解単語の1つ前までの確率和
        if pred_sum < thres or k == 0: # 正解単語の1つ前までの和が閾値を超えていない or 正解単語が1位
            pred_acc = pred_acc.write(i, 1.0)
        else:
            pred_acc = pred_acc.write(i, 0)
        acc_count = acc_count.write(i, k)

    false_index = tf.where(tf.equal(pred_acc.stack(), 0)).numpy()
    acc = tf.reduce_mean(pred_acc.stack())
    return acc, false_index, acc_count.stack().numpy()





#---------------------------------------------

# 誤字検出システムの根幹
#--------------
# 入力文の予測結果を出力する
def texts_pred(pred, label, test_texts, vocab_dic, dic_vocab):
    result = ''
    for index, id in enumerate(test_texts):
        #print(id)
        result += str(id) + '\n'
        
        # 予測
        y_index = pred[index].argsort()[::-1][0:3]
        y_prob = np.sort(pred[index])[::-1][0:3]
    
        y_id = test_texts[index].index('< >')
        try: 
            word = dic_vocab[label[index]] #texts[0][y_id]
            prob = pred[index][vocab_dic[word]]
        except: 
            word = 'nan'
            prob = 0  
        
        #print('------入力------')
        #print(word, prob)
        #print()
        result += '------入力------\n'
        result += word + ' ' + str(prob) + '\n'
        result += '------予測------\n'
        for j in range(3): 
            #print(dic_vocab[y_index[j]], y_prob[j])
            result += dic_vocab[y_index[j]] + ' ' + str(y_prob[j]) + '\n'
    return result


# 評価関数を改良して、不正解判定になった箇所を返すようにする
# 該当する文章と誤り箇所を出力 モデル予測による適切かも単語も出力する
def miss_check(method_acc, pred, label, test_texts, vocab_dic, dic_vocab):
    result = ''
    acc, falIndex, aCount = method_acc(label, pred)
    falIndex = falIndex[:, 0]
    # print('正解率: ', acc.numpy())
    # print()
    result += '正解率: ' + str(acc.numpy()) + '\n'
    
    if falIndex.size == 0:
        # print('誤りは検出されませんでした')
        result += '誤りは検出されませんでした\n'
    else:
        #print('---誤字・誤用を検出---')
        result += '---誤字・誤用を検出---\n'

        for fi in falIndex:

            label_id = test_texts[fi].index('< >')

            try:
                word = dic_vocab[label[fi]]
                prob = pred[fi][vocab_dic[word]]
            except:
                word = 'nan'
                prob = 0.0

            test_text = copy.copy(test_texts[fi])
            test_text[label_id] = '<<' + word + '>>'

            y_index = pred[fi].argsort()[::-1][0:aCount[fi]]
            y_prob = np.sort(pred[fi])[::-1][0:aCount[fi]]


            # print(test_text)
            # print('------入力------')
            # print(word, prob)
            # print()
            # print('----適切かもしれない語----')
            result += str(test_text) + '\n'
            result += '------入力------\n'
            result += word + ' ' + str(prob) + '\n'
            result += '----適切かもしれない語----\n'
            for i in range(len(y_index)):
                #print(dic_vocab[y_index[i]], y_prob[i])
                result += dic_vocab[y_index[i]] + ' ' + str(y_prob[i]) + '\n'
            # print()
            result += '\n'
    return result



#-------------------------
# メインの関数

def pred_particle(text, mode='all'):
    # 保存した辞書のload
    with open('data/particle_n_vocab_dic.pkl', 'rb') as f:
        vocab_dic = pickle.load(f)
    with open('data/particle_n_dic_vocab.pkl', 'rb') as f:
        dic_vocab = pickle.load(f)


    # model のロード
    path = 'data/particle_word_model.h5'
    model = load_model(path, custom_objects={'predsum_acc':predsum_acc})
    # print(model.summary())

    texts, poss, ids = text_to_id(text, vocab_dic)
    fwInputs, bwInputs, label, test_texts = cre_InputData(texts, poss, ids)
    pred = model.predict([fwInputs, bwInputs])

    if mode == 'all':
        result = texts_pred(pred, label, test_texts, vocab_dic, dic_vocab)
    elif mode == 'miss':
        result = miss_check(predsum_acc_, pred, label, test_texts, vocab_dic, dic_vocab)
    
    # print(result)
    return result