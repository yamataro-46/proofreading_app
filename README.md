# proofreading_app

自然言語処理に基づく学術論文の校正システム

A proofreading system for academic papers based on natural language processing.

[Proofreading APP](https://yamataro-46-proofreading-app-proofreading-app-83bxdf.streamlit.app/)


## どんなシステム？

- 卒業研究の成果物です．
- 単純な誤字検出や、より適切な単語の提案を行うシステム

- 情報処理学会の論文の文章を訓練データに用いて、論文特有の文章構造を学習
- 学術論文的文章に特化したモデルの構築に成功


## Appの利用方法

1. サイドバーにて、モードを選択

  - 予測モード
    - 全単語予測モード：全ての単語に対して、予測を行う
  
    - 助詞予測モード：助詞に特化したモデルで、助詞の予測を行う
  
  - 出力モード
    - 全ての結果を出力：誤り判定の有無に関わらず、全ての予測結果を出力
      - 誤り検出がされない単語についても、予測率の高い単語が提案されるため、より的確な単語が見つかるかも
  
    - 誤り判定語のみを出力：予測の結果、誤りと判定された単語を出力する
      - 入力文に誤りがあるかどうかをすばやく判定するのにおすすめ


2. テキスト入力欄に、添削したい文章を入力
3. 校正開始ボタンを押す


## 予測の精度、実験について

以下の論文を参照してください

- [抄録](https://drive.google.com/file/d/17tO7KbtiEa6mnB5xPfBaDeaW8X3qoTvG/view?usp=share_link)

- [本論]()



## ChangeLog

- ver1.1: first release
