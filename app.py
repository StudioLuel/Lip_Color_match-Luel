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

# 2. 이미지 마스킹 및 분석 기능 (업그레이드)
@st.cache_data
def analyze_and_mask_lip(image_file):
    # 연산 속도와 안정성을 위해 해상도 최적화
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((400, 400))
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    img_lab = color.rgb2lab(img_array)
    pixels = img_lab.reshape(-1, 3)
    
    # 4개의 그룹으로 나누어 분석 (피부, 그림자, 입술 메인, 입술 테두리)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    # a* 값(붉은기)이 가장 높은 2개의 그룹을 입술 영역으로 추출
    sorted_by_a_idx = np.argsort(centers[:, 1])[::-1]
    lip_idx_1, lip_idx_2 = sorted_by_a_idx[0], sorted_by_a_idx[1]
    
    # 명도(L*)를 비교하여 밝은 쪽을 메인, 어두운 쪽을 테두리(착색)로 분류
    if centers[lip_idx_1][0] > centers[lip_idx_2][0]:
        main_idx, dark_idx = lip_idx_1, lip_idx_2
    else:
        main_idx, dark_idx = lip_idx_2, lip_idx_1
        
    main_lab = centers[main_idx]
    dark_lab = centers[dark_idx]
    is_two_tone = (main_lab[0] - dark_lab[0]) > 5.0
    
    # 마스크(영역) 생성
    labels_2d = labels.reshape(h, w)
    main_mask = (labels_2d == main_idx)
    dark_mask = (labels_2d == dark_idx)
    full_lip_mask = main_mask | dark_mask
    
    return img_array, main_mask, dark_mask, full_lip_mask, np.array(main_lab), np.array(dark_lab), is_two_tone

@st.cache_data
def get_single_dominant_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((150, 150))
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    pixels = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    lip_cluster = max(centers, key=lambda c: c[1])
    return np.array(lip_cluster)

# 오버레이 시각화 생성 함수
def apply_color_overlay(base_img, mask, hex_color, alpha=0.6):
    overlay = base_img.copy()
    rgb_color = mcolors.to_rgb(hex_color)
    color_uint8 = (np.array(rgb_color) * 255).astype(np.uint8)
    
    for c in range(3):
        overlay[mask, c] = (base_img[mask, c] * (1 - alpha) + color_uint8[c] * alpha).astype(np.uint8)
    return overlay

# 영역 분포도 시각화 (분석 1)
def generate_distribution_map(base_img, main_mask, dark_mask):
    overlay = base_img.copy()
    # 밝은 메인 영역은 핑크(#ff9ff3), 어두운 착색 영역은 블루(#54a0ff)로 맵핑
    c_main = np.array(mcolors.to_rgb('#ff9ff3')) * 255
    c_dark = np.array(mcolors.to_rgb('#54a0ff')) * 255
    
    for c in range(3):
        overlay[main_mask, c] = (base_img[main_mask, c] * 0.4 + c_main[c] * 0.6).astype(np.uint8)
        overlay[dark_mask, c] = (base_img[dark_mask, c] * 0.4 + c_dark[c] * 0.6).astype(np.uint8)
    return overlay

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

# 4. 사전 중화 진단 함수
def get_neutralizer_guide(main_lab, dark_lab, is_two_tone):
    target_lab = dark_lab if is_two_tone else main_lab
    l, a, b = target_lab
    
    if is_two_tone and l < 45:
        return {"needed": True, "type": "투톤 / 테두리 착색", "name": "브라이트 오렌지", "hex": "#FF7F00", "desc": "어두운 부위 타겟 톤업"}
    elif l < 48 and b < 10:
        return {"needed": True, "type": "전체 다크 / 보랏빛", "name": "브라이트 오렌지", "hex": "#FF7F00", "desc": "전체 명도 증가 및 웜톤화"}
    elif b < 8:
        return {"needed": True, "type": "창백 / 푸른빛", "name": "살몬 / 코랄", "hex": "#FF8C69", "desc": "온도감 상승"}
    else:
        return {"needed": False, "type": "양호", "name": "-", "hex": None, "desc": "사전 중화 불필요"}

# 5. UI용 컬러 박스 생성 함수
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

st.markdown("""
    <style>
    .step-header { background-color: #e2e8f0; color: #111111 !important; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
    .extract-box { padding: 12px; background-color: #f8f9fa; color: #111111 !important; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px; font-size: 0.95em; }
    </style>
""", unsafe_allow_html=True)

st.title("💋 PMU Lip Color Match Pro (시각화 분석)")
st.markdown("입술 영역을 스캔하여 중화 및 시술 부위를 시각적 맵핑으로 제공합니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    with st.spinner('AI가 입술의 영역을 추출하고 시각화 지도를 생성 중입니다...'):
        base_img, main_mask, dark_mask, full_lip_mask, curr_main_lab, curr_dark_lab, is_two_tone = analyze_and_mask_lip(current_file)
        targ_lab = get_single_dominant_lab(target_file)
        
        main_hex = lab_to_hex(curr_main_lab)
        dark_hex = lab_to_hex(curr_dark_lab)
        targ_hex = lab_to_hex(targ_lab)
        
        neutralizer = get_neutralizer_guide(curr_main_lab, curr_dark_lab, is_two_tone)
        best_mix = find_best_mix(targ_lab)
        
        # 혼합 결과 HEX 코드
        mix_hex = targ_hex
        if best_mix["p2"] is not None:
            mix_rgb = color.lab2rgb(np.array([[best_mix["mixed_lab"]]]))[0][0]
            mix_hex = mcolors.to_hex(mix_rgb)

    # ----------------------------------------
    # 분석 1: 입술 색상 분포도 시각화
    # ----------------------------------------
    st.markdown('<div class="step-header">🔍 분석 1. 시술 전 색상 분포도 (입술 영역 스캔)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    
    dist_img = generate_distribution_map(base_img, main_mask, dark_mask)
    c1.image(dist_img, caption="분홍색: 메인 밝은 톤 / 파란색: 테두리 착색 톤", use_container_width=True)
    
    c2.markdown(f"""
    <div class="extract-box">
        <strong>✨ 메인 톤:</strong> {get_color_box_by_hex(main_hex, 20)}<br>
        <strong>🌑 다크 톤:</strong> {get_color_box_by_hex(dark_hex, 20)}
    </div>
    """, unsafe_allow_html=True)
    
    if is_two_tone:
        c2.error("🚨 짙은 테두리(투톤) 감지됨")
    
    # ----------------------------------------
    # 분석 2: 중화 색상 주입 부위 시각화
    # ----------------------------------------
    st.markdown('<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 주입 타겟 시각화</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    
    if neutralizer["needed"]:
        # 투톤이면 테두리(dark_mask)에만, 아니면 전체(full_lip_mask)에 중화색 시각화
        target_mask = dark_mask if is_two_tone else full_lip_mask
        neu_img = apply_color_overlay(base_img, target_mask, neutralizer["hex"], alpha=0.7)
        c1.image(neu_img, caption="컬러 주입 타겟 부위", use_container_width=True)
        
        c2.warning(f"**필요 타입:** {neutralizer['type']}")
        c2.markdown(f"**추천 컬러:** {get_color_box_by_hex(neutralizer['hex'], 20)} {neutralizer['name']}", unsafe_allow_html=True)
        c2.info(f"**시술 가이드:** 위 사진에 표시된 색칠 영역에만 해당 코렉터를 적용하세요.")
    else:
        c1.image(base_img, caption="중화 불필요", use_container_width=True)
        c2.success("✨ 베이스가 양호하여 중화 단계 생략")

    # ----------------------------------------
    # 분석 3 & 4: 배합비 산출 (기존 로직 유지)
    # ----------------------------------------
    st.markdown('<div class="step-header">🎨 분석 3&4. 본 컬러 배합 가이드</div>', unsafe_allow_html=True)
    p1, p2 = best_mix["p1"], best_mix["p2"]
    
    if p2 is None:
        st.success(f"단일 색상: {p1['mfg']} - {p1['name']} (100%)")
    else:
        st.markdown(f"""
        **목표 컬러:** {get_color_box_by_hex(targ_hex, 25)} &nbsp; | &nbsp; **오차율(Delta E):** {best_mix['delta_e']:.2f}
        """, unsafe_allow_html=True)
        
        html_table = f"""
        <table style="width:100%; text-align:center; border-collapse: collapse; background-color:#ffffff; color:#111111;">
            <tr style="background-color:#f8f9fa; border-bottom: 2px solid #ddd;">
                <th style="padding:10px;">브랜드</th>
                <th style="padding:10px;">색상 명칭</th>
                <th style="padding:10px;">배합비 (%)</th>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td>{p1['mfg']}</td>
                <td>{get_color_box(p1['name'], 20)} {p1['name']}</td>
                <td style="font-weight:bold; color:#c0392b;">{best_mix['r1']} %</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td>{p2['mfg']}</td>
                <td>{get_color_box(p2['name'], 20)} {p2['name']}</td>
                <td style="font-weight:bold; color:#2980b9;">{best_mix['r2']} %</td>
            </tr>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)

    # ----------------------------------------
    # 분석 5 & 6: 배합 색상 주입 부위 및 예상 결과물 시각화
    # ----------------------------------------
    st.markdown('<div class="step-header">✨ 분석 5&6. 메인 컬러 주입 타겟 및 예상 결과 시각화</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    # 분석 5: 본 컬러를 주입해야 하는 전체 입술 영역 시각화
    target_area_img = apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.9)
    c1.image(target_area_img, caption="[분석 5] 메인 컬러 주입 타겟 부위", use_container_width=True)
    
    # 분석 6: 시술 후 피부톤과 어우러진 자연스러운 예상 결과물 시각화
    final_result_img = apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.45)
    c2.image(final_result_img, caption="[분석 6] 시술 직후 예상 맵핑 결과물", use_container_width=True)
    
    st.caption("※ 시각화된 예상 이미지는 픽셀 블렌딩을 통한 시뮬레이션이며, 실제 고객의 피부 조직 두께 및 탈각 과정에 따라 최종 발색은 다를 수 있습니다.")
