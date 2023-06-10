import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

model = joblib.load('model_joblib_test')

st.title("EMAIL SPAM DETECTION")

msg = st.text_input("enter a message")




if st.button("Predict"):
    data =[msg]
    data = cv.transform([data]).toarray()
   
   
    result = model.predict([[msg]])
    st.write("  " , result[0])

  