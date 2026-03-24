# app.py — Streamlit application สำหรับทำนายการรับฝากเงินของลูกค้าธนาคาร

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="ระบบวิเคราะห์แนวโน้มการรับฝากเงิน",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== ฟังก์ชันโหลดโมเดลและตัวแปลงข้อมูล =====
@st.cache_resource
def load_bank_assets():
    """โหลด model, scaler และ encoders ที่สร้างจาก Bank.ipynb"""
    # ตรวจสอบว่ามีไฟล์ก่อนโหลด (ป้องกัน Error)
    try:
        model = joblib.load("bank_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # โหลด encoders สำหรับข้อมูลหมวดหมู่ (อ้างอิงจาก Bank.ipynb)
        categorical_cols = ['job', 'marital', 'education', 'housing']
        encoders = {}
        for col in categorical_cols:
            encoders[col] = joblib.load(f'le_{col}.pkl')
            
        return model, scaler, encoders
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
        return None, None, None

# โหลด Assets
with st.spinner("กำลังเตรียมระบบ..."):
    model, scaler, encoders = load_bank_assets()

# ===== Sidebar: ข้อมูลเบื้องต้น =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดล")
    st.write("**ประเภท:** Gradient Boosting Classifier")
    st.write("**เป้าหมาย:** ทำนายว่าลูกค้าจะตกลงฝากเงิน (Deposit) หรือไม่")
    st.divider()
    st.warning("⚠️ ผลลัพธ์นี้ใช้เพื่อการวิเคราะห์แนวโน้มทางการตลาดเบื้องต้นเท่านั้น")

# ===== ส่วนหลัก: Header =====
st.title("💰 ระบบวิเคราะห์โอกาสการรับฝากเงิน")
st.markdown("กรอกข้อมูลลูกค้าเพื่อประเมินโอกาสที่ลูกค้าจะตอบรับผลิตภัณฑ์เงินฝาก")
st.divider()

if model is not None:
    # ===== ส่วนรับ Input (อ้างอิง Features จาก Bank.ipynb) =====
    st.subheader("📋 ข้อมูลลูกค้า")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input("ระยะเวลาสนทนาล่าสุด (วินาที)", min_value=0, value=150)
        campaign = st.number_input("จำนวนครั้งที่ติดต่อในแคมเปญนี้", min_value=1, value=1)
        pdays = st.number_input("จำนวนวันที่ผ่านไปหลังติดต่อครั้งก่อน (-1 คือไม่เคย)", value=-1)
        previous = st.number_input("จำนวนครั้งที่ติดต่อก่อนแคมเปญนี้", min_value=0, value=0)

    with col2:
        job = st.selectbox("อาชีพ", encoders['job'].classes_)
        marital = st.selectbox("สถานภาพการสมรส", encoders['marital'].classes_)
        education = st.selectbox("ระดับการศึกษา", encoders['education'].classes_)
        housing = st.selectbox("มีสินเชื่อที่อยู่อาศัยหรือไม่", encoders['housing'].classes_)

    # ===== ส่วนการทำนาย =====
    if st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True):
        # 1. แปลงข้อมูลหมวดหมู่เป็นตัวเลขด้วย Encoders
        input_data = {
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'job': encoders['job'].transform([job])[0],
            'marital': encoders['marital'].transform([marital])[0],
            'education': encoders['education'].transform([education])[0],
            'housing': encoders['housing'].transform([housing])[0]
        }
        
        # 2. จัดเรียงข้อมูลตามลำดับ Features ใน Bank.ipynb
        features_list = ['duration', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'housing']
        X_input = np.array([[input_data[f] for f in features_list]])
        
        # 3. Scaling ข้อมูล
        X_scaled = scaler.transform(X_input)
        
        # 4. Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")
        
        if prediction == 1:
            st.success(f"### ✅ มีแนวโน้มสูงที่จะฝากเงิน\n**โอกาสสำเร็จ: {probabilities[1]*100:.1f}%**")
        else:
            st.error(f"### ❌ มีแนวโน้มที่จะไม่ฝากเงิน\n**โอกาสสำเร็จเพียง: {probabilities[1]*100:.1f}%**")
            
        st.progress(float(probabilities[1]), text=f"ความน่าจะเป็นในการตอบรับ: {probabilities[1]*100:.1f}%")

else:
    st.info("กรุณาตรวจสอบว่ามีไฟล์ bank_model.pkl, scaler.pkl และ le_*.pkl อยู่ใน Folder เดียวกับโปรแกรม")
