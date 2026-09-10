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

# 2. 음영 무시 및 초정밀 착색 감지 알고리즘
@st.cache_data
def analyze_and_mask_lip(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((400, 400))
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    img_lab = color.rgb2lab(img_array)
    
    features = img_lab.copy()
    features[:, :, 0] = features[:, :, 0] * 0.2  # 그림자 영향 최소화
    features[:, :, 1] = features[:, :, 1] * 1.5
    features[:, :, 2] = features[:, :, 2] * 1.2
    
    pixels_features = features.reshape(-1, 3)
    pixels_original = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels_features)
    
    actual_centers = []
    for i in range(4):
        cluster_pixels = pixels_original[labels == i]
        if len(cluster_pixels) > 0:
            actual_centers.append(np.mean(cluster_pixels, axis=0))
        else:
            actual_centers.append(np.array([0, 0, 0]))
    actual_centers = np.array(actual_centers)
    
    sorted_by_a_idx = np.argsort(actual_centers[:, 1])[::-1]
    lip_idx_1, lip_idx_2 = sorted_by_a_idx[0], sorted_by_a_idx[1]
    
    score_1 = actual_centers[lip_idx_1][0] + actual_centers[lip_idx_1][1]
    score_2 = actual_centers[lip_idx_2][0] + actual_centers[lip_idx_2][1]
    
    if score_1 > score_2:
        main_idx, dark_idx = lip_idx_1, lip_idx_2
    else:
        main_idx, dark_idx = lip_idx_2, lip_idx_1
        
    main_lab = actual_centers[main_idx]
    dark_lab = actual_centers[dark_idx]
    
    is_two_tone = (main_lab[0] - dark_lab[0]) > 1.5 or (main_lab[1] - dark_lab[1]) > 1.5
    
    labels_2d = labels.reshape(h, w)
    main_mask = (labels_2d == main_idx)
    dark_mask = (labels_2d == dark_idx)
    full_lip_mask = main_mask | dark_mask
    
    return img_array, main_mask, dark_mask, full_lip_mask, main_lab, dark_lab, is_two_tone

@st.cache_data
def get_single_dominant_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((150, 150))
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    
    features = img_lab.copy()
    features[:, :, 0] = features[:, :, 0] * 0.2
    features[:, :, 1] = features[:, :, 1] * 1.5
    features[:, :, 2] = features[:, :, 2] * 1.2
    
    pixels_features = features.reshape(-1, 3)
    pixels_original = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels_features)
    
    actual_centers = []
    for i in range(3):
        cluster_pixels = pixels_original[labels == i]
        if len(cluster_pixels) > 0:
            actual_centers.append(np.mean(cluster_pixels, axis=0))
        else:
            actual_centers.append(np.array([0, 0, 0]))
    actual_centers = np.array(actual_centers)
    
    lip_cluster = max(actual_centers, key=lambda c: c[1])
    return np.array(lip_cluster)

def apply_color_overlay(base_img, mask, hex_color, alpha=0.6):
    overlay = base_img.copy()
    rgb_color = mcolors.to_rgb(hex_color)
    color_uint8 = (np.array(rgb_color) * 255).astype(np.uint8)
    
    for c in range(3):
        overlay[mask, c] = (base_img[mask, c] * (1 - alpha) + color_uint8[c] * alpha).astype(np.uint8)
    return overlay

def generate_distribution_map(base_img, main_mask, dark_mask):
    overlay = base_img.copy()
    c_main = np.array(mcolors.to_rgb('#ff9ff3')) * 255
    c_dark = np.array(mcolors.to_rgb('#54a0ff')) * 255
    
    for c in range(3):
        overlay[main_mask, c] = (base_img[main_mask, c] * 0.4 + c_main[c] * 0.6).astype(np.uint8)
        overlay[dark_mask, c] = (base_img[dark_mask, c] * 0.4 + c_dark[c] * 0.6).astype(np.uint8)
    return overlay

def analyze_lip_tone(lab):
    l, a, b = lab
    if l < 43: lightness = "어두운 톤"
    elif l > 60: lightness = "밝은 톤"
    else: lightness = "중간 밝기 톤"
    
    if b < 5: hue = "쿨톤 (푸른기/보랏빛)"
    elif b > 18: hue = "웜톤 (오렌지/노란기)"
    else: hue = "뉴트럴톤 (자연스러운 붉은기)"
    
    return lightness, hue

def get_neutralizer_guide(main_lab, dark_lab, is_two_tone):
    target_lab = dark_lab if is_two_tone else main_lab
    l, a, b = target_lab
    
    if is_two_tone:
        if l < 45 or b < 8:
            return {"needed": True, "type": "투톤 / 짙은 테두리 착색", "name": "브라이트 오렌지", "hex": "#FF7F00", "desc": "명도가 낮고 짙은 착색 부위 강력한 톤업 필요"}
        else:
            return {"needed": True, "type": "투톤 / 옅은 테두리 착색", "name": "살몬 / 코랄", "hex": "#FF8C69", "desc": "미세한 착색 및 톤 불균형 교정 필요"}
    elif l < 48 and b < 10:
        return {"needed": True, "type": "전체 다크 / 보랏빛", "name": "브라이트 오렌지", "hex": "#FF7F00", "desc": "전체 명도 증가 및 웜톤화"}
    elif b < 8:
        return {"needed": True, "type": "창백 / 푸른빛", "name": "살몬 / 코랄", "hex": "#FF8C69", "desc": "온도감 상승 필요"}
    else:
        return {"needed": False, "type": "완벽한 균일 톤", "name": "-", "hex": None, "desc": "미세 착색도 없는 완벽한 베이스로 중화 전면 생략"}

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

st.title("💋 PMU Lip Color Match Pro")
st.markdown("그림자를 배제하고 테두리의 미세한 착색까지 잡아내어 필수 중화 영역을 시각화합니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    with st.spinner('미세 착색 영역을 분석 중입니다...'):
        base_img, main_mask, dark_mask, full_lip_mask, curr_main_lab, curr_dark_lab, is_two_tone = analyze_and_mask_lip(current_file)
        targ_lab = get_single_dominant_lab(target_file)
        
        main_hex = lab_to_hex(curr_main_lab)
        dark_hex = lab_to_hex(curr_dark_lab)
        targ_hex = lab_to_hex(targ_lab)
        
        neutralizer = get_neutralizer_guide(curr_main_lab, curr_dark_lab, is_two_tone)
        best_mix = find_best_mix(targ_lab)
        
        mix_hex = targ_hex
        if best_mix["p2"] is not None:
            mix_rgb = color.lab2rgb(np.array([[best_mix["mixed_lab"]]]))[0][0]
            mix_hex = mcolors.to_hex(mix_rgb)

    st.markdown('<div class="step-header">🔍 분석 1. 시술 전 색상 분포도 (그림자 무시 맵핑)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    
    dist_img = generate_distribution_map(base_img, main_mask, dark_mask)
    c1.image(dist_img, caption="분홍색: 메인 밝은 톤 / 파란색: 미세 착색 감지 부위", use_container_width=True)
    
    c2.markdown(f"""
    <div class="extract-box">
        <strong>✨ 메인 톤:</strong> {get_color_box_by_hex(main_hex, 20)}<br>
        <strong>🌑 착색 감지 톤:</strong> {get_color_box_by_hex(dark_hex, 20)}
    </div>
    """, unsafe_allow_html=True)
    
    if is_two_tone:
        c2.error("🚨 테두리 또는 일부 미세 착색 감지됨")
    else:
        c2.success("✨ 매우 균일한 톤 감지됨")
    
    st.markdown('<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 주입 타겟 시각화</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    
    if neutralizer["needed"]:
        target_mask = dark_mask if is_two_tone else full_lip_mask
        neu_img = apply_color_overlay(base_img, target_mask, neutralizer["hex"], alpha=0.7)
        c1.image(neu_img, caption="컬러 주입 타겟 부위", use_container_width=True)
        
        c2.warning(f"**진단 타입:** {neutralizer['type']}")
        c2.markdown(f"**추천 컬러:** {get_color_box_by_hex(neutralizer['hex'], 20)} {neutralizer['name']}", unsafe_allow_html=True)
        c2.info(f"**시술 가이드:** 사진에 파란색으로 표시되었던 착색 부위(색칠 영역)에만 중화제를 적용하여 톤을 균일하게 맞추세요.")
    else:
        c1.image(base_img, caption="중화 불필요", use_container_width=True)
        c2.success(f"✨ {neutralizer['desc']}")

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

    # 💡 3분할 화면으로 원본, 주입 타겟, 예상 결과를 나란히 배치
    st.markdown('<div class="step-header">✨ 분석 5&6. 초기 상태 ➔ 주입 타겟 ➔ 예상 결과 시뮬레이션</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    c1.image(base_img, caption="[원본] 처리 전 초기 입술", use_container_width=True)
    
    target_area_img = apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.9)
    c2.image(target_area_img, caption="[타겟] 메인 컬러 주입 부위", use_container_width=True)
    
    final_result_img = apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.45)
    c3.image(final_result_img, caption="[예상] 시술 후 예상 결과", use_container_width=True)
    
    st.caption("※ 시각화된 예상 이미지는 픽셀 블렌딩을 통한 시뮬레이션이며, 실제 고객의 피부 조직 두께 및 탈각 과정에 따라 최종 발색은 다를 수 있습니다.")
