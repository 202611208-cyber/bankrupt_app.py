

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. 모델 및 스케일러 로드 ---
# .pkl 파일 경로 확인 (app.py와 같은 디렉토리에 있다고 가정)
model_path = 'bankrupt_model.pkl'
scaler_path = 'bankrupt_scaler.pkl'

# 파일이 존재하는지 확인
if not os.path.exists(model_path):
    st.error(f"오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop()
if not os.path.exists(scaler_path):
    st.error(f"오류: 스케일러 파일 '{scaler_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop()

knn_model_eng = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 모델 학습 시 사용된 컬럼명 (5개의 파생 변수)
training_columns = ['순자산_대_부채', 'ROA_x_유동비율', 'ROA_CFO_비율', '유동_부채_비율', 'ROA_부채_상호작용']

# --- 2. Streamlit UI 구성 ---
st.title("기업 파산 예측 서비스 🏢")
st.write("기업의 4가지 재무 비율을 입력하여 파산 여부를 예측해보세요.")

# 사용자 입력 받기
st.header("재무 비율 입력")
roa = st.number_input("세후_총자산이익률 (ROA(A) before interest and % after tax)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
debt_ratio = st.number_input("부채비율 (Debt ratio %)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
cfo_to_assets = st.number_input("현금흐름_대_총자산 (Cash Flow to Total Assets)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
current_ratio = st.number_input("유동비율 (Current Ratio)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

if st.button("파산 여부 예측"): # 예측 버튼 추가
    # --- 3. 사용자 입력으로 파생 변수 생성 ---
    temp_user_data = pd.DataFrame([{
        '세후_총자산이익률': roa,
        '부채비율': debt_ratio,
        '현금흐름_대_총자산': cfo_to_assets,
        '유동비율': current_ratio
    }])

    temp_user_data['순자산_대_부채'] = 1 / (temp_user_data['부채비율'] + 1e-6)
    temp_user_data['ROA_x_유동비율'] = temp_user_data['세후_총자산이익률'] * temp_user_data['유동비율']
    temp_user_data['ROA_CFO_비율'] = temp_user_data['세후_총자산이익률'] / (temp_user_data['현금흐름_대_총자산'] + 1e-6)
    temp_user_data['유동_부채_비율'] = temp_user_data['유동비율'] / (temp_user_data['부채비율'] + 1e-6)
    temp_user_data['ROA_부채_상호작용'] = temp_user_data['세후_총자산이익률'] * temp_user_data['부채비율']

    # 모델 학습 시 사용된 5개의 파생 변수만 선택
    user_input_df = temp_user_data[training_columns]

    # --- 4. 스케일러 적용 ---
    user_scaled = scaler.transform(user_input_df)

    # --- 5. 예측 수행 ---
    prediction = knn_model_eng.predict(user_scaled)
    probability = knn_model_eng.predict_proba(user_scaled)

    # --- 6. 결과 출력 ---
    st.header("예측 결과")
    if prediction[0] == 1:
        st.error(f"**파산 예측: 예 (파산 확률: {probability[0][1]*100:.2f}%)**")
        st.markdown("**분석:** 입력된 재무 비율을 바탕으로 해당 기업은 **파산할 가능성이 높다고 예측됩니다.** 추가적인 재무 상태 검토가 필요할 수 있습니다.")
    else:
        st.success(f"**파산 예측: 아니오 (파산 확률: {probability[0][1]*100:.2f}%)**")
        st.markdown("**분석:** 입력된 재무 비율을 바탕으로 해당 기업은 **파산하지 않을 것으로 예측됩니다.** 안정적인 재무 상태를 유지하고 있는 것으로 보입니다.")
