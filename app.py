import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os # 파일 경로 확인용
import random

# ------------------------------------------------------
# [설정] 여기에 결과로 보여줄 이미지 규칙을 정하세요!
# 왼쪽에는 'labels.txt'에서 숫자를 뺀 이름, 오른쪽에는 '보여줄 이미지 파일 경로'
# ------------------------------------------------------
'''
'''

# ------------------------------------------------------
# 1. 설정 및 클래스 정의 (기존 유지)
# ------------------------------------------------------
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

np.set_printoptions(suppress=True)

# ------------------------------------------------------
# 2. 모델 로드 함수
# ------------------------------------------------------
@st.cache_resource
def load_my_model():
    custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
    model = load_model("keras_model.h5", compile=False, custom_objects=custom_objects)
    
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = f.readlines()
    return model, class_names

# ------------------------------------------------------
# 3. 예측 함수
# ------------------------------------------------------
def predict_image(model, class_names, image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    # 라벨 파싱 ("0 오렌지" -> "오렌지")
    class_name_raw = class_names[index].strip()
    parts = class_name_raw.split(" ", 1)
    class_name = parts[1] if len(parts) > 1 else parts[0]
    confidence_score = float(prediction[0][index])
    
    return class_name, confidence_score

# ------------------------------------------------------
# 4. Streamlit 웹 인터페이스
# ------------------------------------------------------
st.title("🦊내가 동물이라면?!")
st.write("이미지를 올리면 당신과 닮은 동물을 사진으로 보여줍니다!")

try:
    model, class_names = load_my_model()
except Exception as e:
    st.error(f"모델 로드 중 오류 발생: {e}")
    st.stop()

file = st.file_uploader("이미지를 올려주세요", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    if st.button("분석 시작"):
        with st.spinner("AI가 분석 중입니다..."):
            class_name, score = predict_image(model, class_names, image)
            
        st.success(f"분석 결과: **[{class_name}]** (확신도: {score*100:.2f}%)")

        # --------------------------------------------------
        # ★ [자동화 버전] 딕셔너리 없이 파일 이름으로 찾기
        # --------------------------------------------------
        # 규칙: result_images 폴더 안에 "라벨이름.jpg"가 있어야 함
        
        # --------------------------------------------------
        # ★ [랜덤 뽑기 기능] 폴더 안에서 아무거나 하나 뽑아 보여주기
        # --------------------------------------------------
        
        # 1. 해당 라벨의 폴더 경로를 만듭니다. (예: result_images/사과)
        target_folder = f"result_images/{class_name}"
        
        # 2. 폴더가 실제로 있는지 확인
        if os.path.exists(target_folder):
            # 3. 폴더 안에 있는 모든 파일 목록을 가져옵니다.
            file_list = os.listdir(target_folder)
            
            # 4. 그 중에서 이미지 파일(.png, .jpg)만 골라냅니다. (이상한 시스템 파일 제외)
            # 대소문자 무시하고 png, jpg, jpeg 등을 찾습니다.
            image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(image_files) > 0:
                # 5. ★ 랜덤으로 하나 선택! (여기가 핵심)
                selected_image = random.choice(image_files)
                
                # 6. 전체 경로를 합쳐서 이미지를 보여줍니다.
                full_path = os.path.join(target_folder, selected_image)
                st.image(full_path, caption=f"랜덤으로 소환된 {class_name} 이미지!", use_column_width=True)
            else:
                st.warning(f"'{class_name}' 폴더는 있지만, 안에 이미지 파일이 하나도 없어요!")
        else:
            st.warning(f"'{class_name}' 이름의 폴더를 찾을 수 없습니다.")
            st.info(f"result_images 폴더 안에 '{class_name}' 폴더를 만들고 사진을 넣어주세요.")

        # --------------------------------------------------
        # ★ [추가한 멘트] 제일 마지막에 실행됩니다.
        # --------------------------------------------------
        st.header(f'당신은 "{class_name}"입니다!')
        st.balloons()  # (보너스) 풍선이 날아오르는 효과! 싫으면 지우세요.