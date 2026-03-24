import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. โหลดโมเดลที่เซฟไว้
model = joblib.load('bank_model.pkl')
# scaler = joblib.load('scaler.pkl') # โหลดถ้ามีการใช้งาน

st.title('🏦 ระบบทำนายการสมัครประกันธนาคาร')
st.write('กรุณากรอกข้อมูลลูกค้าเพื่อทำนายผล')

# 2. สร้างฟอร์มรับข้อมูล (ตัวอย่างตามหัวข้อใน dataset ของคุณ)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('อายุ (Age)', min_value=18, max_value=100, value=30)
    job = st.selectbox('อาชีพ', ['admin.', 'blue-collar', 'technician', 'services', 'management'])
    marital = st.selectbox('สถานภาพการสมรส', ['married', 'single', 'divorced'])

with col2:
    balance = st.number_input('ยอดเงินในบัญชี (Balance)', value=0)
    duration = st.number_input('ระยะเวลาที่คุยสายล่าสุด (วินาที)', value=0)
    campaign = st.number_input('จำนวนครั้งที่ติดต่อในแคมเปญนี้', value=1)

# 3. ส่วนการประมวลผลเมื่อกดปุ่ม
if st.button('เริ่มการทำนายผล'):
    # เตรียมข้อมูลให้อยู่ในรูปแบบ DataFrame เหมือนตอน Train
    input_data = pd.DataFrame([[age, job, marital, balance, duration, campaign]], 
                             columns=['age', 'job', 'marital', 'balance', 'duration', 'campaign'])
    
    # หมายเหตุ: คุณต้องทำการแปลงข้อมูล (Encoding) ให้เหมือนกับตอนที่ทำใน Notebook ก่อนนำเข้า model.predict
    # เช่น การเปลี่ยนข้อความเป็นตัวเลข หรือใช้ One-hot Encoding
    
    # สมมติว่าผ่านการแปลงข้อมูลแล้ว
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # 4. แสดงผลลัพธ์
    st.subheader('ผลการวิเคราะห์:')
    if prediction[0] == 1:
        st.success(f'✅ ลูกค้ามีแนวโน้มจะ "ตกลง" (โอกาส {prediction_proba[0][1]:.2%})')
    else:
        st.error(f'❌ ลูกค้ามีแนวโน้มจะ "ปฏิเสธ" (โอกาส {prediction_proba[0][0]:.2%})')
