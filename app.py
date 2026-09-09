import streamlit as st
import numpy as np
from skimage import color
from PIL import Image
import itertools
import pandas as pd
import re
import matplotlib.colors as mcolors

# 1. 가상 색소 데이터베이스 (Perma Blend 전용)
PIGMENT_DB = {
    "Perma Blend": {
        "Sweet Melissa (#E08B9B)": [65.0, 30.0, 5.0],
        "Bazooka (#E65C7B)": [55.0, 45.0, 10.0],
        "Date Night (#B03A48)": [42.0, 40.0, 15.0],
        "Orange Crush (#F27930)": [60.0, 40.0, 50.0],
        "Tres Pink (#E89EB3)": [70.0, 25.0, -2.0],
        "Pillow Talk (#D47883)": [62.0, 35.0, 8.0],
        "Passion Red (#BA2737)": [45.0, 50.0, 20.0],
        "French Fancy (#D66B85)": [58.0, 38.0, 2.0]
    }
}

# 2. 이미지에서 LAB 색상 추출 함수
def get_average_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    return np.array([np.mean(img_lab[:, :, 0]), np.mean(img_lab[:, :, 1]), np.mean(img_lab[:, :, 2])])

# 3. LAB 기반 현재 입술 톤 진단 함수
def analyze_lip_tone(lab):
    l, a, b = lab
    if l < 45: lightness = "어두운 톤"
    elif l > 62: lightness = "밝은 톤"
    else: lightness = "중간 밝기 톤"
    
    if b < 5: hue = "쿨톤 (푸른기/보랏빛)"
    elif b > 18: hue = "웜톤 (오렌지/노란기)"
    else: hue = "뉴트럴톤 (자연스러운 붉은기)"
    
    return lightness, hue

# 4. 사전 중화(Neutralizing) 진단 함수
def get_neutralizer_guide(lab):
    l, a, b = lab
    if l < 48 and b < 10:
        return {
            "needed": True,
            "type": "다크/보랏빛 입술 중화",
            "name": "브라이트 오렌지 코렉터 (예: Orange Crush 단독)",
            "hex": "#FF7F00",
            "desc": "명도를 높이고 푸른기를 강하게 잡아야 하는 입술입니다.",
            "guide": "어둡고 푸른 기가 도는 부위(주로 입술 테두리나 얼룩진 곳)에 오렌지 코렉터를 가볍게 터치합니다. 색소를 너무 깊게 찌르지 않고 얇은 픽셀 기법으로 깔아주어 베이스를 웜톤으로 끌어올리세요."
        }
    elif b < 8:
        return {
            "needed": True,
            "type": "창백/푸른빛 입술 중화",
            "name": "살몬 / 웜 코랄 코렉터",
            "hex": "#FF8C69",
            "desc": "전체적으로 차가운 톤을 띠어 발색이 탁해질 수 있습니다.",
            "guide": "본 컬러 주입 전, 입술 전체에 살몬이나 웜 코랄 코렉터로 1차 베이스 작업을 진행하여 입술 온도를 시각적으로 높여줍니다."
        }
    elif l < 42:
        return {
            "needed": True,
            "type": "브라운/어두운 입술 톤업",
            "name": "옐로우 오렌지 코렉터",
            "hex": "#FFA500",
            "desc": "입술 바탕이 어두워 본 컬러가 그대로 발색되지 않습니다.",
            "guide": "어두운 브라운 톤 부위에 옐로우 오렌지 계열을 사용하여 명도를 한 톤 밝혀주는 타겟 보정 작업이 필요합니다."
        }
    else:
        return {
            "needed": False,
            "guide": "현재 입술 베이스가 양호하여 사전 중화 작업 없이 즉시 본 컬러 시술이 가능합니다."
        }

# 5. 컬러 박스(도시화) HTML 생성 함수
def get_color_box(name, size=25):
    match = re.search(r'#([A-Fa-f0-9]{6})', name)
    hex_code = f"#{match.group(1)}" if match else "#CCCCCC"
    return f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; border-radius: 4px; border: 1px solid #ccc; display: inline-block; vertical-align: middle;"></div>'

def get_color_box_by_hex(hex_code, size=25):
    return f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; border-radius: 4px; border: 1px solid #ccc; display: inline-block; vertical-align: middle;"></div>'

# 6. 최적 배합비 산출 알고리즘
def find_best_mix(target_lab):
    best_match = None
    min_delta_e = float('inf')
    all_pigments = [{"mfg": m, "name": n, "lab": np.array(l)} for m, pigs in PIGMENT_DB.items() for n, l in pigs.items()]
            
    for pig in all_pigments:
        delta_e = color.deltaE_ciede2000(target_lab, pig["lab"])
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = {"p1": pig, "p2": None, "r1": 100, "r2": 0, "delta_e": delta_e}

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
st.title("💋 PMU Lip Color Match Pro (Perma Blend)")
st.markdown("입술 톤 진단부터 사전 중화, 퍼마블렌드 전용 색소 배합까지 안내하는 전문가용 가이드입니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 입술 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("고객이 원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    curr_lab = get_average_lab(current_file)
    targ_lab = get_average_lab(target_file)
    lightness, hue = analyze_lip_tone(curr_lab)
    
    st.markdown("""
        <style>
        .step-header { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
        .neutralize-box { background-color: #fff9e6; border-left: 4px solid #f39c12; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
        </style>
    """, unsafe_allow_html=True)
    
    # 프로세스 1: 현재 입술 색상 분석
    st.markdown('<div class="step-header">🔍 분석 1. 현재 입술 기본 톤 진단</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    c1.image(current_file, use_container_width=True)
    c2.markdown(f"**명도 상태:** {lightness}")
    c2.markdown(f"**색상 계열:** {hue}")
    c2.caption(f"*세부 수치(LAB): L({curr_lab[0]:.1f}) / a({curr_lab[1]:.1f}) / b({curr_lab[2]:.1f})*")

    # 프로세스 2: 사전 중화(Neutralizing) 가이드
    st.markdown('<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 처방</div>', unsafe_allow_html=True)
    neutralizer = get_neutralizer_guide(curr_lab)
    
    if neutralizer["needed"]:
        st.markdown(f"""
        <div class="neutralize-box">
            <h4 style="margin-top:0; color:#d35400;">⚠️ 중화 작업 필수 ({neutralizer['type']})</h4>
            <p style="margin-bottom:10px;">{neutralizer['desc']}</p>
            <div style="display:flex; align-items:center; gap: 10px; margin-bottom:10px;">
                {get_color_box_by_hex(neutralizer['hex'], 40)}
                <strong style="font-size:1.1em;">추천 컬러: {neutralizer['name']}</strong>
            </div>
            <hr style="margin: 10px 0; border: 0; border-top: 1px dashed #ccc;">
            <strong>💡 시술 팁:</strong><br>{neutralizer['guide']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✨ {neutralizer['guide']}")

    # 프로세스 3: 목표와 현재 색상 대조 및 방향성
    st.markdown('<div class="step-header">⚖️ 분석 3. 본 컬러 방향성 설정</div>', unsafe_allow_html=True)
    diff_L = targ_lab[0] - curr_lab[0]
    
    direction_msg = []
    if diff_L > 3: direction_msg.append("현재보다 **톤업(밝게)** 표현 필요")
    elif diff_L < -3: direction_msg.append("현재보다 **딥하게(어둡게)** 표현 필요")
    else: direction_msg.append("현재 밝기 수준 유지")
    
    st.info(" ➔ ".join(direction_msg))

    # 프로세스 4: 배합비 산출 및 컬러 도시화
    st.markdown('<div class="step-header">🎨 분석 4. 퍼마블렌드 전용 배합비 및 시뮬레이션</div>', unsafe_allow_html=True)
    
    with st.spinner('최적의 메인 색소 배합을 계산 중입니다...'):
        result = find_best_mix(targ_lab)
    
    p1 = result["p1"]
    p2 = result["p2"]
    
    if p2 is None:
        st.success("단일 색상으로 목표 컬러 구현")
        col_box = get_color_box(p1['name'], size=40)
        st.markdown(f"{col_box} &nbsp; **{p1['mfg']}** - {p1['name']} (원액 100%)", unsafe_allow_html=True)
    else:
        html_table = f"""
        <table style="width:100%; text-align:center; border-collapse: collapse;">
            <tr style="background-color:#fafafa; border-bottom: 2px solid #ddd;">
                <th style="padding:10px;">컬러 도시화</th>
                <th style="padding:10px;">브랜드</th>
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
        
        from skimage.color import lab2rgb
        
        try:
            mix_rgb = lab2rgb(np.array([[result["mixed_lab"]]]))[0][0]
            mix_hex = mcolors.to_hex(mix_rgb)
            st.write("")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap: 15px; margin-top:10px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9;">
                <div style="background-color:{mix_hex}; width:60px; height:60px; border-radius:50%; border:2px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"></div>
                <div>
                    <h4 style="margin:0; color:#333;">예측 시술 컬러</h4>
                    <p style="margin:0; font-size:0.9em; color:#666;">계산된 오차율 (Delta E): {result['delta_e']:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            pass
