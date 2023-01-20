FROM python:3.8
WORKDIR /app
RUN pip install pandas scikit-learn streamlit joblib numpy 
ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]