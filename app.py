import streamlit as st
import numpy as np
from skimage import color
from PIL import Image
import itertools
import pandas as pd
import re
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

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

# 2. [업그레이드] 투톤 입술(테두리/메인) 정밀 분석 함수
@st.cache_data
def analyze_multi_tone_lip(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((150, 150))
    img_array = np.array(image)
    
    img_lab = color.rgb2lab(img_array)
    pixels = img_lab.reshape(-1, 3)
    
    # 4개의 그룹으로 나누어 피부, 치아, 입술(메인), 입술(테두리/어두운곳)을 분리
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    
    # a* 값(붉은기)이 가장 높은 2개의 그룹을 입술 영역으로 추출
    sorted_by_a = sorted(centers, key=lambda c: c[1], reverse=True)
    lip_1 = sorted_by_a[0]
    lip_2 = sorted_by_a[1]
    
    # 명도(L*)를 비교하여 밝은 쪽을 메인, 어두운 쪽을 테두리(착색)로 분류
    if lip_1[0] > lip_2[0]:
        main_lab = lip_1
        dark_lab = lip_2
    else:
        main_lab = lip_2
        dark_lab = lip_1
        
    # 메인과 어두운 부분의 명도(L) 차이가 6 이상이면 투톤/착색 입술로 판별
    is_two_tone = (main_lab[0] - dark_lab[0]) > 6.0
    
    return np.array(main_lab), np.array(dark_lab), is_two_tone

# 단일 목표 색상 추출 함수 (STEP 2 용)
@st.cache_data
def get_single_dominant_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((150, 150))
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    pixels = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    lip_cluster = max(centers, key=lambda c: c[1])
    return np.array(lip_cluster)

# 3. LAB 기반 입술 톤 진단 함수
def analyze_lip_tone(lab):
    l, a, b = lab
    if l < 43: lightness = "어두운 톤"
    elif l > 60: lightness = "밝은 톤"
    else: lightness = "중간 밝기 톤"
    
    if b < 5: hue = "쿨톤 (푸른기/보랏빛)"
    elif b > 18: hue = "웜톤 (오렌지/노란기)"
    else: hue = "뉴트럴톤 (자연스러운 붉은기)"
    
    return lightness, hue

# 4. 사전 중화(Neutralizing) 진단 함수 (투톤 입술 반영)
def get_neutralizer_guide(main_lab, dark_lab, is_two_tone):
    # 투톤 입술이면 문제의 원인인 '어두운 테두리(dark_lab)'를 기준으로 중화 처방
    target_lab = dark_lab if is_two_tone else main_lab
    l, a, b = target_lab
    
    if is_two_tone and l < 45:
        return {
            "needed": True,
            "type": "투톤 / 테두리 착색 톤업",
            "name": "브라이트 오렌지 코렉터 (예: Orange Crush 단독)",
            "hex": "#FF7F00",
            "desc": "입술 테두리나 특정 부위가 안쪽에 비해 유독 어둡게 착색된 투톤 입술입니다.",
            "guide": "전체 도포가 아닌 **'어두운 테두리 부분(타겟 부위)'에만** 오렌지 코렉터를 얇은 픽셀 기법으로 깔아주세요. 안쪽의 밝은 톤과 테두리의 명도를 균일하게 맞추는 '톤 맞춤(얼룩 제거)' 작업이 최우선입니다."
        }
    elif l < 48 and b < 10:
        return {
            "needed": True,
            "type": "전체 다크 / 보랏빛 입술 중화",
            "name": "브라이트 오렌지 코렉터",
            "hex": "#FF7F00",
            "desc": "전체적으로 명도가 낮고 푸른기를 강하게 띠는 입술입니다.",
            "guide": "입술 전체에 오렌지 코렉터를 얇게 깔아주어 베이스를 웜톤으로 끌어올리세요."
        }
    elif b < 8:
        return {
            "needed": True,
            "type": "창백 / 푸른빛 입술 중화",
            "name": "살몬 / 웜 코랄 코렉터",
            "hex": "#FF8C69",
            "desc": "명도는 나쁘지 않으나 전체적으로 차가운(푸른) 톤을 띠어 발색이 탁해질 수 있습니다.",
            "guide": "본 컬러 주입 전, 입술 전체에 살몬/코랄 코렉터로 1차 베이스 작업을 진행하여 온도를 높여줍니다."
        }
    else:
        return {
            "needed": False,
            "guide": "테두리 얼룩이나 심한 푸른기가 없어, 사전 중화 작업 없이 즉시 본 컬러 시술이 가능합니다."
        }

# 5. 컬러 박스 HTML 생성 함수
def get_color_box_by_hex(hex_code, size=25):
    return f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; border-radius: 4px; border: 1px solid #999; display: inline-block; vertical-align: middle;"></div>'

def get_color_box(name, size=25):
    match = re.search(r'#([A-Fa-f0-9]{6})', name)
    hex_code = f"#{match.group(1)}" if match else "#CCCCCC"
    return get_color_box_by_hex(hex_code, size)

def lab_to_hex(lab_array):
    try:
        rgb = color.lab2rgb(np.array([[lab_array]]))[0][0]
        return mcolors.to_hex(rgb)
    except:
        return "#CCCCCC"

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

# [수정] 다크모드에서도 글씨가 무조건 까맣게 보이도록 CSS color 속성에 !important 적용
st.markdown("""
    <style>
    .step-header { background-color: #e2e8f0; color: #111111 !important; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
    .neutralize-box { background-color: #fff9e6; color: #111111 !important; border-left: 4px solid #f39c12; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
    .extract-box { padding: 12px; background-color: #f8f9fa; color: #111111 !important; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px; font-size: 0.95em; line-height: 1.6; }
    .result-box { display:flex; align-items:center; gap: 15px; margin-top:10px; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; color: #111111 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("💋 PMU Lip Color Match Pro")
st.markdown("투톤(테두리 착색) 입술까지 정밀하게 감지하여 퍼마블렌드 전용 배합을 추천합니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    with st.spinner('AI가 입술의 투톤 여부와 색상 층을 정밀 분석 중입니다...'):
        curr_main_lab, curr_dark_lab, is_two_tone = analyze_multi_tone_lip(current_file)
        targ_lab = get_single_dominant_lab(target_file)
    
    # ----------------------------------------
    # 분석 1: 현재 입술 기본 톤 진단
    # ----------------------------------------
    st.markdown('<div class="step-header">🔍 분석 1. 현재 입술 정밀 톤 진단 (투톤 감지)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    c1.image(current_file, use_container_width=True)
    
    main_hex = lab_to_hex(curr_main_lab)
    dark_hex = lab_to_hex(curr_dark_lab)
    
    # 투톤 여부에 따른 화면 표시 변경
    if is_two_tone:
        c2.markdown('<div style="color: #e74c3c; font-weight: bold; margin-bottom: 5px;">🚨 짙은 테두리(착색) / 투톤 입술 감지됨</div>', unsafe_allow_html=True)
    
    # 추출 컬러 박스 표시 (가독성 개선)
    c2.markdown(f"""
    <div class="extract-box">
        <strong>✨ 메인 톤 (입술 안쪽):</strong> {get_color_box_by_hex(main_hex, 20)}<br>
        <strong>🌑 다크 톤 (테두리/착색):</strong> {get_color_box_by_hex(dark_hex, 20)}
    </div>
    """, unsafe_allow_html=True)
    
    # 진단은 메인 톤을 기준으로 출력
    lightness, hue = analyze_lip_tone(curr_main_lab)
    c2.markdown(f"**전체적 명도:** {lightness}")
    c2.markdown(f"**전체적 색상:** {hue}")

    # ----------------------------------------
    # 분석 2: 사전 중화(Neutralizer) 처방
    # ----------------------------------------
    st.markdown('<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 처방</div>', unsafe_allow_html=True)
    neutralizer = get_neutralizer_guide(curr_main_lab, curr_dark_lab, is_two_tone)
    
    if neutralizer["needed"]:
        st.markdown(f"""
        <div class="neutralize-box">
            <h4 style="margin-top:0; color:#d35400;">⚠️ 중화 작업 필수 ({neutralizer['type']})</h4>
            <p style="margin-bottom:10px;">{neutralizer['desc']}</p>
            <div style="display:flex; align-items:center; gap: 10px; margin-bottom:10px;">
                {get_color_box_by_hex(neutralizer['hex'], 40)}
                <strong style="font-size:1.1em; color:#111;">추천 컬러: {neutralizer['name']}</strong>
            </div>
            <hr style="margin: 10px 0; border: 0; border-top: 1px dashed #ccc;">
            <strong>💡 시술 팁:</strong><br>{neutralizer['guide']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✨ {neutralizer['guide']}")

    # ----------------------------------------
    # 분석 3: 목표 색상 대조 및 방향성
    # ----------------------------------------
    st.markdown('<div class="step-header">⚖️ 분석 3. 본 컬러 방향성 설정</div>', unsafe_allow_html=True)
    
    targ_hex = lab_to_hex(targ_lab)
    st.markdown(f"**목표 추출 컬러:** {get_color_box_by_hex(targ_hex, 25)}", unsafe_allow_html=True)
    
    diff_L = targ_lab[0] - curr_main_lab[0]
    direction_msg = []
    if diff_L > 3: direction_msg.append("현재보다 **톤업(밝게)** 표현 필요")
    elif diff_L < -3: direction_msg.append("현재보다 **딥하게(어둡게)** 표현 필요")
    else: direction_msg.append("현재 밝기 수준 유지")
    
    st.info(" ➔ ".join(direction_msg))

    # ----------------------------------------
    # 분석 4: 배합비 산출 및 시뮬레이션
    # ----------------------------------------
    st.markdown('<div class="step-header">🎨 분석 4. 퍼마블렌드 전용 배합비 시뮬레이션</div>', unsafe_allow_html=True)
    
    with st.spinner('목표 컬러 구현을 위한 최적 배합 계산 중...'):
        result = find_best_mix(targ_lab)
    
    p1 = result["p1"]
    p2 = result["p2"]
    
    if p2 is None:
        st.success("단일 색상으로 목표 컬러 구현 가능")
        col_box = get_color_box(p1['name'], size=40)
        st.markdown(f"{col_box} &nbsp; **{p1['mfg']}** - {p1['name']} (원액 100%)", unsafe_allow_html=True)
    else:
        html_table = f"""
        <table style="width:100%; text-align:center; border-collapse: collapse; background-color:#ffffff; color:#111111;">
            <tr style="background-color:#f8f9fa; border-bottom: 2px solid #ddd;">
                <th style="padding:10px;">컬러 도시화</th>
                <th style="padding:10px;">브랜드</th>
                <th style="padding:10px;">색상 코드 및 명칭</th>
                <th style="padding:10px; font-size:1.1em;">배합비 (%)</th>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:10px;">{get_color_box(p1['name'], 30)}</td>
                <td>{p1['mfg']}</td>
                <td>{p1['name']}</td>
                <td style="font-weight:bold; color:#c0392b;">{result['r1']} %</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:10px;">{get_color_box(p2['name'], 30)}</td>
                <td>{p2['mfg']}</td>
                <td>{p2['name']}</td>
                <td style="font-weight:bold; color:#2980b9;">{result['r2']} %</td>
            </tr>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)
        
        try:
            mix_rgb = color.lab2rgb(np.array([[result["mixed_lab"]]]))[0][0]
            mix_hex = mcolors.to_hex(mix_rgb)
            st.write("")
            st.markdown(f"""
            <div class="result-box">
                <div style="background-color:{mix_hex}; width:60px; height:60px; border-radius:50%; border:2px solid #999; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"></div>
                <div>
                    <h4 style="margin:0; color:#111;">예측 시술 컬러</h4>
                    <p style="margin:0; font-size:0.9em; color:#555;">계산된 오차율 (Delta E): {result['delta_e']:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            pass
