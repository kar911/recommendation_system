import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np

import pickle
# from sklearn.metrics.pairwise import cosine_similarity


file_path = "cosine_similarity_model.pkl"
# Load the model from the saved file using pickle
with open(file_path, 'rb') as f:
    cosine_sim = pickle.load(f)



df = pd.read_csv('cleaned.csv')
df=df.dropna(axis=0,how='any',subset=['clean_sorted'])
# count = CountVectorizer()
# count_matrix = count.fit_transform(df['clean_sorted'])
# cosine_sim = cosine_similarity(count_matrix, count_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(df.index, index=df['clean_sorted'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:6]
    movie_indices = [i[0] for i in sim_scores]
    return df[['course_title','url','price','num_subscribers','num_reviews','num_lectures','level','rating','subject','Free/Paid','vid','content_duration']].iloc[movie_indices]

def x(strr):
    text = strr.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = ' '.join([word for word in words if word.lower() not in stop_words])
    return re.sub(r'[^A-Za-z ]', '', filtered_words)
st.title('Course Recommendation System')

if 'key' not in st.session_state:
    st.session_state['key'] = False
    st.session_state['search'] = None
    st.session_state['term']=None
    st.session_state['info']=None

def send(div,recc):
    recommendations =get_recommendations(x(div))
    st.session_state['search'] = recommendations
    st.session_state['key']=True
    st.session_state['term']=div
    st.session_state['info']=recc[recc['course_title']==div]

def tab2():
    rec=st.session_state['search']
    h=rec['course_title'].values
    dur=rec['content_duration'].values
    rate=rec['rating'].values
    pric=rec['price'].values

    if st.session_state['term'] in h:
        indx=np.where(h == st.session_state['term'])
        dur=np.delete(dur, indx)
        rate=np.delete(rate, indx)
        h=np.delete(h, indx)
        pric=np.delete(pric, indx)
    st.header(st.session_state['info']["course_title"].values[0])
    video_file = open(f'{st.session_state["info"]["vid"].values[0]}.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.success(f"""
        Duration in Hour : {st.session_state["info"]["content_duration"].values[0]} \n
        Rating : {st.session_state["info"]["rating"].values[0]} \n
        Price : {st.session_state["info"]["price"].values[0]}
        """)
    st.code(st.session_state['info']["url"].values[0], language="txt")
    c3, c4 = st.columns(2)
    with c3:
        st.info(f"""
        Title : {h[1]} \n
        Duration in Hour : {dur[1]} \n
        Rating : {rate[1]} \n
        Price : {pric[1]}
        """)
        st.button('view',on_click=send,args=[h[1],rec],key='1')
        st.info(f"""
        Title : {h[2]}\n
        Duration in Hour : {dur[2]}\n
        Rating : {rate[2]}\n
        Price : {pric[2]}
        """)
        st.button('view',on_click=send,args=[h[2],rec],key='2')
    with c4:
        st.info(f"""
        Title : {h[3]}\n
        Duration in Hour : {dur[3]}\n
        Rating : {rate[3]}\n
        Price : {pric[3]}
        """)
        st.button('view',on_click=send,args=[h[3],rec],key='3')
        st.info(f"""
        Title : {h[4]}\n
        Duration in Hour : {dur[4]}\n
        Rating : {rate[4]}\n
        Price : {pric[4]}
        """)
        st.button('view',on_click=send,args=[h[4],rec],key='4')


def tab1():

    fee=st.radio('',['Free','Paid'],index=1,horizontal=True)

    c1, c2= st.columns(2)

    with c2:
        level=st.multiselect('Select Level of learning',[
            'select All'
            ,'Beginner Level'
            ,'Expert Level'
            ,'All Level'
            ,'Intermediate Level'
            ],default='select All')
    with c1:
        typee=st.multiselect('Select Stream',['All'
        ,'Web Development'
        ,'Musical Instruments'
        ,'Business Finance'
        ,'Graphic Design'],default='All')


    if "All" in typee:
        typee = ['Web Development'
    ,'Musical Instruments'
    ,'Business Finance'
    ,'Graphic Design']
    if "select All" in level:
        level = ['Beginner Level'
    ,'Expert Level'
    ,'All Level'
    ,'Intermediate Level']

    if fee == 'Paid':
        priceRange =st.slider(label='Price range', min_value = 0, max_value = 200,value=[5,10], step = 5)
        x1,x2=priceRange[0],priceRange[1]
        textt=" `Free/Paid`== @fee and `subject` in @typee and `level` in @level and @x1 < `price` & `price` < @x2 "
    else :
        textt=" `Free/Paid`== @fee and `subject` in @typee and `level` in @level"

    gp=df.query(textt)

    st.write(gp.shape)
    option = st.selectbox('Select a movie', gp['course_title'].values,label_visibility= "hidden")
    if st.button('Recommend') and  option != None:
        recommendations = get_recommendations(x(option))
        st.session_state['search'] = recommendations
        st.session_state['key']=True
        st.session_state['term']=option
        st.session_state['info']=gp[gp['course_title']==option]
        st.experimental_rerun()


if not st.session_state['key']:
    tab1()
else:
    tab2()


