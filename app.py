import streamlit as st
import numpy as np
from skimage import color
from PIL import Image
import itertools
import pandas as pd
import re

# 1. 가상 색소 데이터베이스 (이름 안에 #HEX 코드를 넣어 도시화에 사용)
PIGMENT_DB = {
    "Tina Davies": {
        "Flame (#E23D28)": [50.0, 55.0, 40.0],
        "Lust (#C62828)": [40.0, 48.0, 25.0],
        "Envy (#9B203B)": [35.0, 40.0, 10.0],
        "Sweet Eau (#E5979C)": [68.0, 28.0, 8.0]
    },
    "Perma Blend": {
        "Sweet Melissa (#E08B9B)": [65.0, 30.0, 5.0],
        "Bazooka (#E65C7B)": [55.0, 45.0, 10.0],
        "Date Night (#B03A48)": [42.0, 40.0, 15.0],
        "Orange Crush (#F27930)": [60.0, 40.0, 50.0]
    }
}

# 2. 이미지에서 LAB 색상 추출 함수
def get_average_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    return np.array([np.mean(img_lab[:, :, 0]), np.mean(img_lab[:, :, 1]), np.mean(img_lab[:, :, 2])])

# 3. LAB 기반 색상 계열 및 톤 분석 함수
def analyze_lip_tone(lab):
    l, a, b = lab
    
    # 명도 분석
    if l < 45: lightness = "어두운 톤 (명도 보정 필요 가능성)"
    elif l > 62: lightness = "밝은 톤 (색소 발색이 잘 되는 베이스)"
    else: lightness = "중간 밝기 톤"
    
    # 색상(Hue) 분석
    if b < 5: hue = "쿨톤 (푸른기/보랏빛 베이스)"
    elif b > 18: hue = "웜톤 (오렌지/노란기 베이스)"
    else: hue = "뉴트럴톤 (자연스러운 붉은기)"
    
    return lightness, hue

# 4. 텍스트에서 HEX 코드 추출 및 컬러 박스(도시화) HTML 생성 함수
def get_color_box(name, size=25):
    match = re.search(r'#([A-Fa-f0-9]{6})', name)
    hex_code = f"#{match.group(1)}" if match else "#CCCCCC"
    return f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; border-radius: 4px; border: 1px solid #ccc; display: inline-block; vertical-align: middle;"></div>'

def get_hex_only(name):
    match = re.search(r'#([A-Fa-f0-9]{6})', name)
    return f"#{match.group(1)}" if match else "#CCCCCC"

# 5. 최적 배합비 산출 알고리즘
def find_best_mix(target_lab):
    best_match = None
    min_delta_e = float('inf')
    all_pigments = [{"mfg": m, "name": n, "lab": np.array(l)} for m, pigs in PIGMENT_DB.items() for n, l in pigs.items()]
            
    # 단일 색상 확인
    for pig in all_pigments:
        delta_e = color.deltaE_ciede2000(target_lab, pig["lab"])
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = {"p1": pig, "p2": None, "r1": 100, "r2": 0, "delta_e": delta_e}

    # 2색 혼합비 계산 (10% 단위)
    for p1, p2 in itertools.combinations(all_pigments, 2):
        for ratio in range(10, 100, 10):
            mixed_lab = (p1["lab"] * (ratio / 100.0)) + (p2["lab"] * ((100 - ratio) / 100.0))
            delta_e = color.deltaE_ciede2000(target_lab, mixed_lab)
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match = {"p1": p1, "p2": p2, "r1": ratio, "r2": 100 - ratio, "delta_e": delta_e, "mixed_lab": mixed_lab}
                
    return best_match

# ----------------- UI 구성 -----------------
st.set_page_config(page_title="PMU 컬러 매치 프로", page_icon="💋", layout="centered")
st.title("💋 PMU Lip Color Match Pro")
st.markdown("현재 입술 톤을 분석하고 목표 색상에 도달하기 위한 최적의 색소 배합을 제안합니다.")

# 단계별 사진 업로드
col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 입술 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("고객이 원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    # 데이터 추출
    curr_lab = get_average_lab(current_file)
    targ_lab = get_average_lab(target_file)
    lightness, hue = analyze_lip_tone(curr_lab)
    
    # 시각적 구분선 (CSS)
    st.markdown("""
        <style>
        .step-header { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    # 프로세스 1: 현재 입술 색상 분석
    st.markdown('<div class="step-header">🔍 분석 1. 현재 입술 색상 분포 및 계열 분석</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    c1.image(current_file, use_container_width=True)
    c2.markdown(f"**명도 분포:** {lightness}")
    c2.markdown(f"**색상 계열:** {hue}")
    c2.caption(f"*세부 수치(LAB): L({curr_lab[0]:.1f}) / a({curr_lab[1]:.1f}) / b({curr_lab[2]:.1f})*")

    # 프로세스 2: 목표와 현재 색상 대조 및 비교
    st.markdown('<div class="step-header">⚖️ 분석 2. 목표 색상 대조 및 방향성 설정</div>', unsafe_allow_html=True)
    diff_L = targ_lab[0] - curr_lab[0]
    
    direction_msg = []
    if diff_L > 3: direction_msg.append("현재보다 **톤업(밝게)** 필요")
    elif diff_L < -3: direction_msg.append("현재보다 **톤다운(어둡게)** 필요")
    else: direction_msg.append("현재 밝기 유지")
    
    if curr_lab[2] < 5 and targ_lab[2] > 10:
        direction_msg.append("푸른기 중화를 위한 **웜톤 코렉터(오렌지계열) 믹스 권장**")

    st.warning(" ➔ ".join(direction_msg))

    # 프로세스 3: 배합비 산출 및 컬러 도시화
    st.markdown('<div class="step-header">🎨 분석 3. 추천 색소 및 배합비 (컬러 도시화)</div>', unsafe_allow_html=True)
    
    with st.spinner('최적의 색소 배합을 계산 중입니다...'):
        result = find_best_mix(targ_lab)
    
    p1 = result["p1"]
    p2 = result["p2"]
    
    if p2 is None:
        # 단일 색상
        st.success("✨ 단일 색상 사용으로 목표 도달 가능")
        col_box = get_color_box(p1['name'], size=40)
        st.markdown(f"{col_box} &nbsp; **{p1['mfg']}** - {p1['name']} (원액 100%)", unsafe_allow_html=True)
    else:
        # 2색 혼합
        st.success("✨ 2가지 색상 믹스 추천")
        
        # 컬러 도시화 테이블
        html_table = f"""
        <table style="width:100%; text-align:center; border-collapse: collapse;">
            <tr style="background-color:#fafafa; border-bottom: 2px solid #ddd;">
                <th style="padding:10px;">컬러 도시화</th>
                <th style="padding:10px;">제조사</th>
                <th style="padding:10px;">색상 코드 및 명칭</th>
                <th style="padding:10px; font-size:1.1em;">배합비 (%)</th>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:10px;">{get_color_box(p1['name'], 30)}</td>
                <td>{p1['mfg']}</td>
                <td>{p1['name']}</td>
                <td style="font-weight:bold; color:#d63031;">{result['r1']} %</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:10px;">{get_color_box(p2['name'], 30)}</td>
                <td>{p2['mfg']}</td>
                <td>{p2['name']}</td>
                <td style="font-weight:bold; color:#0984e3;">{result['r2']} %</td>
            </tr>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)
        
        # 최종 혼합 결과 시각화
        from skimage.color import lab2rgb
        import matplotlib.colors as mcolors
        
        # 혼합된 LAB 값을 RGB로 변환하여 시각화 (예측 컬러)
        try:
            mix_rgb = lab2rgb(np.array([[result["mixed_lab"]]]))[0][0]
            mix_hex = mcolors.to_hex(mix_rgb)
            st.write("")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap: 15px; margin-top:10px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9;">
                <div style="background-color:{mix_hex}; width:60px; height:60px; border-radius:50%; border:2px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"></div>
                <div>
                    <h4 style="margin:0; color:#333;">예측 시술 컬러</h4>
                    <p style="margin:0; font-size:0.9em; color:#666;">계산된 오차율 (Delta E): {result['delta_e']:.2f} (3.0 이하면 매우 우수)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            pass

    st.caption("※ 본 가이드는 실험 데이터 기반의 참고용이며, 고객의 피부 두께 및 멜라닌 색소에 따라 발색 차이가 있을 수 있습니다.")
