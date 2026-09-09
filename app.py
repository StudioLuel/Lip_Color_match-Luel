import streamlit as st
import numpy as np
from skimage import color
from PIL import Image
import itertools

# 1. 웹 수집 기반 가상 색소 데이터베이스 (제조사, 색상명, HEX코드, LAB값)
# 실제 웹사이트 스와치에서 추출한 RGB/HEX 값을 LAB으로 변환해 입력해 둡니다.
PIGMENT_DB = {
    "Tina Davies": {
        "Flame (#E23D28)": [50.0, 55.0, 40.0],
        "Lust (#C62828)": [40.0, 48.0, 25.0],
        "Envy (#9B203B)": [35.0, 40.0, 10.0]
    },
    "Perma Blend": {
        "Sweet Melissa (#E08B9B)": [65.0, 30.0, 5.0],
        "Bazooka (#E65C7B)": [55.0, 45.0, 10.0],
        "Date Night (#B03A48)": [42.0, 40.0, 15.0]
    }
}

# 2. 색상 추출 함수
def get_average_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    return np.array([np.mean(img_lab[:, :, 0]), np.mean(img_lab[:, :, 1]), np.mean(img_lab[:, :, 2])])

# 3. 최적의 2색 혼합비 계산 알고리즘
def find_best_mix(target_lab):
    best_match = None
    min_delta_e = float('inf')
    
    # DB에 있는 모든 색소를 하나의 리스트로 평탄화
    all_pigments = []
    for manufacturer, pigments in PIGMENT_DB.items():
        for name, lab in pigments.items():
            all_pigments.append({"mfg": manufacturer, "name": name, "lab": np.array(lab)})
            
    # 단일 색상으로 해결되는 경우 먼저 확인
    for pig in all_pigments:
        delta_e = color.deltaE_ciede2000(target_lab, pig["lab"])
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = {"pigment1": pig, "pigment2": None, "ratio1": 100, "ratio2": 0, "delta_e": delta_e}

    # 2가지 색상의 조합 계산 (10% 단위 배합)
    for p1, p2 in itertools.combinations(all_pigments, 2):
        for ratio in range(10, 100, 10): # 10% ~ 90%
            w1 = ratio / 100.0
            w2 = 1.0 - w1
            
            # 두 색상의 LAB 값을 가중 평균하여 가상 혼합
            mixed_lab = (p1["lab"] * w1) + (p2["lab"] * w2)
            
            # 가상 혼합 색상과 목표 색상 간의 오차 계산
            delta_e = color.deltaE_ciede2000(target_lab, mixed_lab)
            
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match = {
                    "pigment1": p1,
                    "pigment2": p2,
                    "ratio1": ratio,
                    "ratio2": 100 - ratio,
                    "delta_e": delta_e
                }
                
    return best_match

# 4. Streamlit UI
st.title("💋 입술 반영구 색상 배합 계산기")
st.write("제조사 DB 기반으로 목표 색상 구현을 위한 최적의 배합비를 산출합니다.")

target_lip_file = st.file_uploader("목표 색상 사진을 올려주세요.", type=["jpg", "jpeg", "png"])

if target_lip_file:
    st.image(target_lip_file, caption="목표 색상", width=300)
    
    if st.button("배합비 계산하기"):
        target_lab = get_average_lab(target_lip_file)
        
        # 알고리즘 실행
        result = find_best_mix(target_lab)
        
        st.divider()
        st.subheader("🎯 추천 제조사 및 배합비")
        
        p1 = result["pigment1"]
        p2 = result["pigment2"]
        
        if p2 is None: # 단일 색상 추천 시
            st.success("단일 색상으로 목표 구현이 가능합니다.")
            st.markdown(f"**제조사:** {p1['mfg']}")
            st.markdown(f"**색상 코드:** {p1['name']}")
            st.markdown("**배합비:** 원액 100%")
        else: # 2색 혼합 추천 시
            st.markdown(f"**메인 색상:** {p1['mfg']} - {p1['name']} ({result['ratio1']}%)")
            st.markdown(f"**서브 색상:** {p2['mfg']} - {p2['name']} ({result['ratio2']}%)")
            
            # 표 형태로 결과 출력
            import pandas as pd
            df = pd.DataFrame({
                "제조사": [p1['mfg'], p2['mfg']],
                "색상 코드(명칭/HEX)": [p1['name'], p2['name']],
                "배합비 (%)": [result['ratio1'], result['ratio2']]
            })
            st.table(df)
            
        st.write(f"*오차율(Delta E): {result['delta_e']:.2f}* (3.0 이하일 경우 육안으로 매우 유사함)")
        st.caption("※ 본 산출식은 LAB 가중평균 기반의 수학적 시뮬레이션으로, 실제 시술 시 피부 톤에 따라 미세 조정이 필요할 수 있습니다.")