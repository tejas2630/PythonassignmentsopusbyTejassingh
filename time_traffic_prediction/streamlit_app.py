
import streamlit as st
import pandas as pd
import numpy as np
import os, glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time Traffic Prediction (G1traffic)", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path="data/G1traffic.csv"):
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ['datetime', 'date_time', 'timestamp']:
            rename[c] = 'DateTime'
        elif cl in ['junction', 'junction_id', 'junctionid']:
            rename[c] = 'Junction'
        elif cl in ['vehicles', 'vehicle_count', 'count', 'volume']:
            rename[c] = 'Vehicles'
    if rename:
        df = df.rename(columns=rename)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df.sort_values(['Junction', 'DateTime']).reset_index(drop=True)

def add_features(df):
    df = df.copy()
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.dayofweek
    out = []
    for j, g in df.groupby('Junction'):
        g = g.sort_values('DateTime')
        g['lag1'] = g['Vehicles'].shift(1)
        g['lag24'] = g['Vehicles'].shift(24)
        g['roll3'] = g['Vehicles'].shift(1).rolling(3).mean()
        out.append(g)
    return pd.concat(out).dropna()

def label_from_train(train_df):
    thr = {}
    for j, g in train_df.groupby('Junction'):
        q1 = g['Vehicles'].quantile(0.33)
        q2 = g['Vehicles'].quantile(0.66)
        thr[j] = (q1, q2)
    return thr

def apply_labels(df, thr):
    def f(r):
        q1, q2 = thr[r['Junction']]
        if r['Vehicles'] <= q1: return 0
        if r['Vehicles'] <= q2: return 1
        return 2
    df = df.copy()
    df['congestion'] = df.apply(f, axis=1)
    return df

st.sidebar.title('🚦 Time Traffic Prediction')
section = st.sidebar.radio('Go to', ['📁 Dataset', '🛠️ Train', '🔮 Predict', '📊 Analyze'])

if section == '📁 Dataset':
    st.title('📁 Dataset')
    df = load_data()
    st.metric('Rows', len(df))
    st.metric('Junctions', df['Junction'].nunique())
    st.dataframe(df.head(20))

elif section == '🛠️ Train':
    st.title('🛠️ Train Model')
    df = add_features(load_data())
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    thr = label_from_train(train)
    train = apply_labels(train, thr)
    test = apply_labels(test, thr)
    Xtr = pd.get_dummies(train[['hour','day','lag1','lag24','roll3','Junction']], columns=['Junction'])
    ytr = train['congestion']
    Xte = pd.get_dummies(test[['hour','day','lag1','lag24','roll3','Junction']], columns=['Junction'])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    yte = test['congestion']
    if st.button('Train'):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        st.code(classification_report(yte, yp), language='text')
        cm = confusion_matrix(yte, yp)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
        st.session_state['model'] = clf
        st.session_state['cols'] = Xtr.columns.tolist()
        st.session_state['thr'] = thr

elif section == '🔮 Predict':
    st.title('🔮 Live Prediction')
    if 'model' not in st.session_state:
        st.info('Train the model first')
    else:
        df = add_features(load_data())
        j = st.selectbox('Junction', sorted(df['Junction'].unique()))
        row = df[df['Junction']==j].iloc[-1]
        x = pd.DataFrame([{ 'hour': row['hour'], 'day': row['day'], 'lag1': row['lag1'], 'lag24': row['lag24'], 'roll3': row['roll3'], 'Junction': j }])
        x = pd.get_dummies(x, columns=['Junction'])
        x = x.reindex(columns=st.session_state['cols'], fill_value=0)
        y = st.session_state['model'].predict(x)[0]
        lbl = {0:'Low',1:'Medium',2:'High'}[int(y)]
        st.success(f'Predicted congestion: {lbl}')

else:
    st.title('📊 Analysis')
    df = load_data()
    df['hour'] = df['DateTime'].dt.hour
    st.line_chart(df.groupby(['hour','Junction'])['Vehicles'].mean().unstack())
