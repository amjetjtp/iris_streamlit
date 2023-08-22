# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# データセット読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値
df['target'] = iris.target

# 目標値を数字から花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

# 予測モデル構築　説明変数xに0列目と2列目を指定、目的変数yにtargetを指定
x = iris.data[:, [0, 2]] 
y = iris.target

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(x, y)

# streamlit run iris.py　実行の確認

# サイドバー（入力画面）
st.sidebar.header('Input Features')

sepalValue = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

# インプットデータ（1行のデータフレーム）
# 列をdata,sepalValue, petalValueの3列に指定
# data列をインデックスに指定、value_dfに直接適用
# →data列には全てdataが入っており、この操作は表示をきれいにさせるのが目的
value_df = pd.DataFrame({'data':'data', 'sepal length (cm)':sepalValue, 'petal length (cm)':petalValue}, index=[0])
value_df.set_index('data', inplace=True)

# 入力値の値
st.write(value_df)

# 予測値のデータフレーム
# probability はある事象が起こり得る確実性の度合い
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
# pred_df.idxmax(axis=1)で、各行における最大値を持つ列のインデックス(最も確率が高いアイリスの品種の名前）を取得
# tolist()を使用して、その結果をPythonのリストとして取得
# name[0]で最初の要素を取得して str(文字列)として表示させる
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')

