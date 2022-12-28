import streamlit as st
import numpy as np
import json
import pandas as pd
import aspose.words as aw
import re
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from stop_words import get_stop_words
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# инициализируем элемент класса Mystem() для последующей лемматизации  текста на русском языке
m = Mystem() 

# ограничение количества слов в списке лемматизированных слов договора
MAX_WORD = 1000 # подбирается наилучший по метрике на кроссвалидации 

STOPWORDS_RU = get_stop_words('russian')

# словарь видов договоров
with open('data/kind_names.json', 'r', encoding='utf8') as f:
    kind_names = json.load(f)

with open('data/pkl_object/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('data/pkl_object/logit_grid_searcher.pkl', 'rb') as f:
    logit_grid_searcher = pickle.load(f)
         
@st.cache(allow_output_mutation=True)
# функция извлечения текста из договора
def GetTextContract(path):
    
    data = aw.Document(path)
    contract = data.get_text()
    contract = contract.replace('Evaluation Only. Created with Aspose.Words. Copyright 2003-2022 Aspose Pty Ltd.', '')
    contract = contract.replace('Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/', '')  
    
    return contract

# функция трансформации текста договора для предсказания
def ContractTransform(contract):

    contract = contract.lower()
    contract = re.sub('[^а-яА-ЯёЁ]', ' ', contract)
    lemm_text_list = [i for i in m.lemmatize(contract) if len(i.strip()) > 2]
    lemm_text_list = [word for word in lemm_text_list if word != 'договор']
    lemm_text_list = [word for word in lemm_text_list if 'сторон' not in word]
    lemm_text_list = lemm_text_list[:MAX_WORD]
    
    return pd.Series(' '.join(lemm_text_list))

# функция построения облака слов
def CreateWordCloud(text):
    
    wordcloud = WordCloud(width = 3000, 
                      height = 2000, 
                      random_state=42,
                      background_color='white',
                      repeat = True,
                      collocations=False,
                      stopwords = STOPWORDS_RU).generate(text) 
                        
    return wordcloud

# функция предсказания
def ModelPredictProba(contract, kind_names, vectorizer, logit_grid_searcher):
    
    X = vectorizer.transform(contract)
    kind_name_pred = kind_names[str(logit_grid_searcher.predict(X)[0])]
        
    return kind_name_pred
    
st.set_page_config(
    page_title="DocTypes", page_icon="❄️", initial_sidebar_state="expanded"
)

st.write(
    """
# 📃 Классификатор договоров

принимает на вход файлы расширения **.DOC**, **.RTF**, **.PDF**, **.DOCX**

"""
)

st.write('**Загрузка договора**')

uploaded_file = st.file_uploader(label='Выберите файл:',
                            type=["doc", "docx", "pdf", "rtf"], accept_multiple_files=False,
                            help='укажите путь к файлу или перетащите его в онко загрузки')
if uploaded_file:
    contract = GetTextContract(uploaded_file)
    contract_name = uploaded_file.name                    
    st.success("Файл успешно загружен", icon="✅")
    st.components.v1.html(contract, width=None, height=300, scrolling=True)
    result = st.button('Определить')
    if result:
        contract = ContractTransform(contract)
        kind_name_pred = ModelPredictProba(contract, kind_names, vectorizer, logit_grid_searcher)
        st.success("Вид договора успешно определён", icon="✅")
        
        st.write(f"""**Результаты:**
        
    📌 Наименование файла: {contract_name}
        
    ✔️ Предсказанный вид договора:  {kind_name_pred}
       
        """)
    # визуализация ключевой информации
        st.write('**Почему алгоритм так считает?**') 
        wordcloud = CreateWordCloud(contract[0])
    
        fig, ax = plt.subplots(figsize = (12, 8))
        ax.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(fig)
else:
    st.error("Вы ничего не вабрали", icon="❌")
    
st.markdown("<h5 style='text-align: center; color: blac;'> ©️ Команда 40+ </h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: blac;'> X-MAS HACK 2022 </h5>", unsafe_allow_html=True)