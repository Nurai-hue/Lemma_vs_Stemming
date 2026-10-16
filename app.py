import streamlit as st
import pandas as pd
import re
import nltk
import pymorphy3
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

st.set_page_config(page_title="NLP Сайысы", page_icon="🏆", layout="wide")

@st.cache_resource
def setup():
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    return pymorphy3.MorphAnalyzer()

ru = setup()
df = pd.read_csv("forum_news_100.csv").fillna("")

def analyze(word, lang):
    word = word.lower()
    if lang == "ru":
        return SnowballStemmer("russian").stem(word), ru.parse(word)[0].normal_form
    if lang == "en":
        return PorterStemmer().stem(word), WordNetLemmatizer().lemmatize(word, pos="v")
    for s in ["лары","лері","дары","дері","тары","тері","лар","лер","дар","дер","тар","тер"]:
        if word.endswith(s) and len(word) > len(s)+2:
            word = word[:-len(s)]
            break
    return word, word

def tokens(text):
    return re.findall(r"[a-zа-яёәғқңөұүһі]+", text.lower())

def clean_words(text, lang):
    kk = {"және","мен","бен","пен","да","де","ол","бұл","үшін","бір","бар"}
    sw = kk if lang == "kk" else set(stopwords.words(
        "russian" if lang == "ru" else "english"))
    return [w for w in tokens(text) if w not in sw]

st.markdown("""
<style>
.stApp {background:#f0f7ff}
h1,h2,h3 {color:#124a91}
div[data-testid="stMetric"] {
 background:white; padding:15px; border-radius:14px;
 border:1px solid #cfe5ff;
}
.stButton button {background:#1976d2;color:white;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("🏆 Lemma vs Stemming — NLP Сайысы")
st.caption("ForumNews | Қазақша • Русский • English")

a,b,c = st.columns(3)
a.metric("📰 Мақалалар",len(df))
b.metric("🌍 Тілдер",df.language.nunique())
c.metric("🔑 Кілт сөздер",df.keywords.fillna("").str.split(",").map(len).sum())

page = st.sidebar.radio("📌 Мәзір",[
    "📰 Жаңалықтар","🧠 Мәтін талдау",
    "🔍 Сөз талдау","📊 Статистика","🏆 Сайыс"
])

langs = {"Қазақша":"kk","Русский":"ru","English":"en"}

if page == "📰 Жаңалықтар":
    st.header("Жаңалықтар корпусы")
    lang = st.selectbox("Тіл",list(langs))
    part = df[df.language == langs[lang]]
    if not part.empty:
        idx = st.selectbox("Мақала",part.index,format_func=lambda i: str(df.loc[i,"title"])[:90])
        row = df.loc[idx]
        st.subheader(row.title)
        st.write("👤 Автор:",row.author)
        st.write("📅 Жыл:",row.year)
        st.write("📂 Типі:",row.article_type)
        st.info("🔑 "+str(row.keywords))
        st.write(str(row.text)[:700])

elif page == "🧠 Мәтін талдау":
    st.header("Мәтінді толық талдау")
    lang = st.selectbox("Тіл",list(langs))
    text = st.text_area("Мәтін енгізіңіз",height=140)
    if st.button("Талдау"):
        words = tokens(text)
        clean = clean_words(text,langs[lang])
        stems = [analyze(w,langs[lang])[0] for w in clean]
        lemmas = [analyze(w,langs[lang])[1] for w in clean]
        st.write("**Токенизация:**",words)
        st.write("**Стоп-сөздерсіз:**",clean)
        st.write("**Стемминг:**",stems)
        st.write("**Лемматизация:**",lemmas)
        st.metric("Сөз саны",len(words))
        if clean:
            st.bar_chart(pd.Series(dict(Counter(clean).most_common(10))))

elif page == "🔍 Сөз талдау":
    st.header("Стемминг пен лемматизацияны салыстыру")
    lang = st.selectbox("Тіл",list(langs))
    word = st.text_input("Сөз енгізіңіз")
    if st.button("Салыстыру") and word:
        stem,lemma = analyze(word,langs[lang])
        a,b = st.columns(2)
        a.success("Стемминг: "+stem)
        b.info("Лемматизация: "+lemma)

elif page == "📊 Статистика":
    st.header("Корпус статистикасы")
    st.bar_chart(df.language.value_counts())
    st.dataframe(df[["title","language","keywords"]],use_container_width=True)

elif page == "🏆 Сайыс":
    st.header("Lemma vs Stemming сайысы")
    questions = [
        ("машины","ru",["машин","машина","машины"],"машина"),
        ("running","en",["running","run","runs"],"run"),
        ("жаңалықтар","kk",["жаңалық","жаңалықтар","жаңалықты"],"жаңалық"),
        ("студенты","ru",["студент","студенты","студ"],"студент"),
        ("studies","en",["study","studies","studi"],"study")
    ]
    if "q" not in st.session_state:
        st.session_state.q = 0
        st.session_state.score = 0
        st.session_state.answered = False

    q = st.session_state.q
    if q < len(questions):
        word,lang,choices,correct = questions[q]
        st.progress(q/len(questions))
        st.subheader(f"Сұрақ {q+1}/5 — {word}")
        answer = st.radio("Дұрыс лемманы таңдаңыз:",choices,key=f"answer_{q}")
        if not st.session_state.answered:
            if st.button("Тексеру"):
                st.session_state.answered = True
                if answer == correct:
                    st.session_state.score += 1
                st.rerun()
        else:
            if answer == correct:
                st.success("Дұрыс жауап!")
            else:
                st.error("Дұрыс жауап: "+correct)
            if st.button("Келесі сұрақ"):
                st.session_state.q += 1
                st.session_state.answered = False
                st.rerun()
    else:
        st.success(f"🏆 Нәтиже: {st.session_state.score}/5")
        if st.button("Қайта бастау"):
            st.session_state.q = 0
            st.session_state.score = 0
            st.session_state.answered = False
            st.rerun()
