import streamlit as st
import numpy as np
from skimage import color
from PIL import Image
import itertools
import pandas as pd
import re
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import cv2
import mediapipe as mp

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

# 💡 [업그레이드] 구글 안면 인식(MediaPipe) 기반 완벽한 입술 및 주변 피부 추출
def get_mediapipe_masks(img_array):
    mp_face_mesh = mp.solutions.face_mesh
    h, w, _ = img_array.shape
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_array)
        if not results.multi_face_landmarks:
            return None, None # 얼굴을 찾지 못한 경우 Fallback을 위해 None 반환

        landmarks = results.multi_face_landmarks[0].landmark

        # 입술 외곽선 및 안쪽(치아/입벌림) 좌표
        outer_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        inner_lip_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        outer_pts = np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in outer_lip_idx ], np.int32)
        inner_pts = np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in inner_lip_idx ], np.int32)

        # 다각형을 그려 입술은 1, 치아/입안은 0으로 완벽히 마스킹
        cv2.fillPoly(lip_mask, [outer_pts], 1)
        cv2.fillPoly(lip_mask, [inner_pts], 0)

        # 주변 피부 영역 추출 (입술 마스크를 팽창시킨 후 원래 입술을 뺌)
        kernel = np.ones((int(h*0.1), int(w*0.1)), np.uint8)
        dilated_lip = cv2.dilate(lip_mask, kernel, iterations=1)
        skin_mask = dilated_lip - lip_mask

        return lip_mask.astype(bool), skin_mask.astype(bool)

@st.cache_data
def analyze_and_mask_lip(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((500, 500))
    img_array = np.array(image)
    h, w, _ = img_array.shape
    img_lab = color.rgb2lab(img_array)
    
    lip_mask, skin_mask = get_mediapipe_masks(img_array)
    
    # 💡 퍼스널 컬러 (주변 피부톤) 분석
    skin_tone_result = "분석 불가 (얼굴 미인식)"
    if skin_mask is not None and np.any(skin_mask):
        skin_lab = np.mean(img_lab[skin_mask], axis=0)
        sl, sa, sb = skin_lab
        st_type = "웜톤(Warm)" if sb > 13 else "쿨톤(Cool)" if sb < 10 else "뉴트럴(Neutral)"
        st_light = "밝은 피부" if sl > 65 else "어두운 피부" if sl < 50 else "중간 피부"
        skin_tone_result = f"{st_light} / {st_type}"

    if lip_mask is not None and np.any(lip_mask):
        # AI가 입술을 완벽히 찾은 경우: 입술 내부에서만 착색 분석 진행
        lip_pixels_lab = img_lab[lip_mask]
        
        features = lip_pixels_lab.copy()
        features[:, 0] = features[:, 0] * 0.1
        features[:, 1] = features[:, 1] * 2.0
        features[:, 2] = features[:, 2] * 1.5
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
        labels = kmeans.fit_predict(features)
        
        c0 = np.mean(lip_pixels_lab[labels == 0], axis=0)
        c1 = np.mean(lip_pixels_lab[labels == 1], axis=0)
        
        if (c0[0] + c0[1]) > (c1[0] + c1[1]):
            main_lab, dark_lab = c0, c1
            main_lbl, dark_lbl = 0, 1
        else:
            main_lab, dark_lab = c1, c0
            main_lbl, dark_lbl = 1, 0
            
        main_mask = np.zeros((h, w), dtype=bool)
        dark_mask = np.zeros((h, w), dtype=bool)
        main_mask[lip_mask] = (labels == main_lbl)
        dark_mask[lip_mask] = (labels == dark_lbl)
        full_lip_mask = lip_mask
        is_two_tone = (main_lab[1] - dark_lab[1]) > 1.8
        
    else:
        # 얼굴을 찾지 못한 경우 (접사 사진 등) 기존 전체 이미지 기반 분석(Fallback) 실행
        features = img_lab.copy()
        features[:, :, 0] = features[:, :, 0] * 0.1  
        features[:, :, 1] = features[:, :, 1] * 2.0  
        features[:, :, 2] = features[:, :, 2] * 1.5
        
        pixels_features = features.reshape(-1, 3)
        pixels_original = img_lab.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels_features)
        
        actual_centers = []
        for i in range(4):
            cluster_pixels = pixels_original[labels == i]
            actual_centers.append(np.mean(cluster_pixels, axis=0) if len(cluster_pixels) > 0 else np.array([0, 0, 0]))
        actual_centers = np.array(actual_centers)
        
        sorted_by_a_idx = np.argsort(actual_centers[:, 1])[::-1]
        lip_idx_1, lip_idx_2 = sorted_by_a_idx[0], sorted_by_a_idx[1]
        
        if (actual_centers[lip_idx_1][0] + actual_centers[lip_idx_1][1]) > (actual_centers[lip_idx_2][0] + actual_centers[lip_idx_2][1]):
            main_idx, dark_idx = lip_idx_1, lip_idx_2
        else:
            main_idx, dark_idx = lip_idx_2, lip_idx_1
            
        main_lab, dark_lab = actual_centers[main_idx], actual_centers[dark_idx]
        is_two_tone = (main_lab[1] - dark_lab[1]) > 1.8
        
        labels_2d = labels.reshape(h, w)
        main_mask, dark_mask = (labels_2d == main_idx), (labels_2d == dark_idx)
        full_lip_mask = main_mask | dark_mask
    
    return img_array, main_mask, dark_mask, full_lip_mask, main_lab, dark_lab, is_two_tone, skin_tone_result

@st.cache_data
def get_single_dominant_lab(image_file):
    image = Image.open(image_file).convert('RGB')
    image.thumbnail((150, 150))
    img_array = np.array(image)
    img_lab = color.rgb2lab(img_array)
    
    features = img_lab.copy()
    features[:, :, 0] = features[:, :, 0] * 0.1
    features[:, :, 1] = features[:, :, 1] * 2.0
    features[:, :, 2] = features[:, :, 2] * 1.5
    
    pixels_features = features.reshape(-1, 3)
    pixels_original = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels_features)
    
    actual_centers = []
    for i in range(3):
        cluster_pixels = pixels_original[labels == i]
        actual_centers.append(np.mean(cluster_pixels, axis=0) if len(cluster_pixels) > 0 else np.array([0, 0, 0]))
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
    c_main, c_dark = np.array(mcolors.to_rgb('#ff9ff3')) * 255, np.array(mcolors.to_rgb('#54a0ff')) * 255
    for c in range(3):
        overlay[main_mask, c] = (base_img[main_mask, c] * 0.4 + c_main[c] * 0.6).astype(np.uint8)
        overlay[dark_mask, c] = (base_img[dark_mask, c] * 0.4 + c_dark[c] * 0.6).astype(np.uint8)
    return overlay

def analyze_lip_tone_detailed(main_lab, dark_lab, is_two_tone):
    l, a, b = main_lab
    if l < 43:
        lightness, l_reason = "어두운 톤", f"명도(L*)가 {l:.1f}로 낮아 전체적으로 어둡습니다."
    elif l > 60:
        lightness, l_reason = "밝은 톤", f"명도(L*)가 {l:.1f}로 높아 색소 발색이 유리합니다."
    else:
        lightness, l_reason = "중간 밝기 톤", f"명도(L*)가 {l:.1f}로 평균적인 밝기입니다."
        
    if b < 5:
        hue, h_reason = "쿨톤 (푸른기/보랏빛)", f"노란/푸른기(b*)가 {b:.1f}로 낮아 차가운 온도를 띱니다."
    elif b > 18:
        hue, h_reason = "웜톤 (오렌지/노란기)", f"노란/푸른기(b*)가 {b:.1f}로 높아 따뜻한 온도를 띱니다."
    else:
        hue, h_reason = "뉴트럴톤 (자연스러움)", f"노란/푸른기(b*)가 {b:.1f}로 중립적입니다."
        
    if is_two_tone:
        uni, u_reason = "투톤 (테두리 착색)", f"테두리의 붉은기가 메인보다 {main_lab[1]-dark_lab[1]:.1f} 소실되어 진짜 착색으로 판별됨."
    else:
        uni, u_reason = "균일한 톤", "부위별 붉은기 편차가 적어 그림자를 제외하면 균일한 상태임."
        
    return lightness, l_reason, hue, h_reason, uni, u_reason

def get_neutralizer_guide_detailed(main_lab, dark_lab, is_two_tone):
    target_lab = dark_lab if is_two_tone else main_lab
    l, a, b = target_lab
    
    if is_two_tone:
        if l < 45 or b < 8:
            return {
                "needed": True, "type": "투톤 / 짙은 테두리 착색", 
                "mfg": "Perma Blend(퍼마블렌드)", "name": "Orange Crush (또는 브라이트 오렌지)", "hex": "#F27930",
                "diag_reason": "테두리의 붉은기가 크게 소실되었고, 명도가 낮거나 푸른기가 강합니다.",
                "col_reason": "강한 푸른기와 어두움을 상쇄하려면 고채도의 오렌지가 필수입니다.",
                "guide": "파란색 표시 영역에 타겟팅하여 얇게 주입하세요."
            }
        else:
            return {
                "needed": True, "type": "투톤 / 옅은 테두리 착색", 
                "mfg": "Perma Blend(퍼마블렌드)", "name": "Sweet Melissa (또는 웜 코랄)", "hex": "#E08B9B",
                "diag_reason": "테두리 붉은기가 미세하게 소실되어 약간의 탁함이 존재합니다.",
                "col_reason": "과도한 중화보다는 부드러운 살몬/코랄로 자연스러운 혈색 보완이 적합합니다.",
                "guide": "파란색 표시 영역에 가볍게 터치하세요."
            }
    elif l < 48 and b < 10:
        return {
            "needed": True, "type": "전체 다크 / 보랏빛", 
            "mfg": "Perma Blend(퍼마블렌드)", "name": "Orange Crush (또는 브라이트 오렌지)", "hex": "#F27930",
            "diag_reason": "전체적으로 명도가 낮고 차가운 보랏빛을 띠고 있습니다.",
            "col_reason": "어두운 쿨톤 베이스를 웜톤으로 끌어올리기 위해 오렌지 코렉터가 필요합니다.",
            "guide": "전체 영역에 얇게 깔아주세요."
        }
    elif b < 8:
        return {
            "needed": True, "type": "창백 / 푸른빛", 
            "mfg": "Perma Blend(퍼마블렌드)", "name": "Sweet Melissa (또는 웜 코랄)", "hex": "#E08B9B",
            "diag_reason": "명도는 양호하나 전체적으로 혈색이 없는 차가운 톤입니다.",
            "col_reason": "시각적 온도를 자연스럽게 높여주기 위해 웜톤의 코랄 계열을 사용합니다.",
            "guide": "전체 영역에 가볍게 깔아주세요."
        }
    else:
        return {"needed": False, "type": "완벽한 균일 톤", "mfg": "-", "name": "-", "hex": None, "diag_reason": "붉은기 소실이나 푸른기 등 착색 징후가 전혀 발견되지 않았습니다.", "col_reason": "-", "guide": "사전 중화 전면 생략"}

def get_color_box_by_hex(hex_code, size=25):
    return f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; border-radius: 4px; border: 1px solid #999; display: inline-block; vertical-align: middle;"></div>'

def get_color_box(name, size=25):
    match = re.search(r'#([A-Fa-f0-9]{6})', name)
    return get_color_box_by_hex(f"#{match.group(1)}" if match else "#CCCCCC", size)

def lab_to_hex(lab_array):
    try:
        return mcolors.to_hex(color.lab2rgb(np.array([[lab_array]]))[0][0])
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
    .reason-text { font-size: 0.85em; color: #555555; margin-top: 2px; margin-bottom: 10px; padding-left: 10px; border-left: 2px solid #ddd; }
    .legend-box { display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; align-items: center; justify-content: center; background-color: #f1f2f6; color: #111111 !important; padding: 8px; border-radius: 5px; }
    .legend-color { width: 16px; height: 16px; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 5px; border: 1px solid #999; }
    </style>
""", unsafe_allow_html=True)

st.title("💋 PMU Lip Color Match Pro (AI 2.0)")
st.markdown("안면 인식 AI가 치아와 피부를 완벽히 제외하고, 주변 피부톤(퍼스널 컬러)까지 함께 분석합니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 고객 사진 (얼굴 포함 권장)", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()
    
    with st.spinner('구글 안면 인식 AI가 입술 점막과 퍼스널 컬러를 분석 중입니다...'):
        base_img, main_mask, dark_mask, full_lip_mask, curr_main_lab, curr_dark_lab, is_two_tone, skin_tone_result = analyze_and_mask_lip(current_file)
        targ_lab = get_single_dominant_lab(target_file)
        
        main_hex = lab_to_hex(curr_main_lab)
        dark_hex = lab_to_hex(curr_dark_lab)
        targ_hex = lab_to_hex(targ_lab)
        
        lightness, l_reason, hue, h_reason, uni, u_reason = analyze_lip_tone_detailed(curr_main_lab, curr_dark_lab, is_two_tone)
        neutralizer = get_neutralizer_guide_detailed(curr_main_lab, curr_dark_lab, is_two_tone)
        
        best_mix = find_best_mix(targ_lab)
        mix_hex = targ_hex if best_mix["p2"] is None else mcolors.to_hex(color.lab2rgb(np.array([[best_mix["mixed_lab"]]]))[0][0])

    st.markdown('<div class="step-header">🔍 분석 1. 시술 전 색상 분포도 & 퍼스널 컬러</div>', unsafe_allow_html=True)
    
    # 💡 [신규] 퍼스널 컬러 안내 배너
    st.info(f"👱‍♀️ **고객 주변 피부톤 분석:** {skin_tone_result} (색소 선택 시 참고하세요)")
    
    st.markdown("""
    <div class="legend-box">
        <div><span class="legend-color" style="background-color: #ff9ff3;"></span><strong>정상 발색 영역 (메인 톤)</strong></div>
        <div><span class="legend-color" style="background-color: #54a0ff;"></span><strong>붉은기 소실/착색 영역 (다크 톤)</strong></div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
    
    dist_img = generate_distribution_map(base_img, main_mask, dark_mask)
    c2.image(dist_img, caption="[분석] AI 맵핑 분포도", use_container_width=True)
    
    c3.markdown(f"**명도 상태:** {lightness}")
    c3.markdown(f"<div class='reason-text'>↳ {l_reason}</div>", unsafe_allow_html=True)
    
    c3.markdown(f"**온도/색상:** {hue}")
    c3.markdown(f"<div class='reason-text'>↳ {h_reason}</div>", unsafe_allow_html=True)
    
    c3.markdown(f"**균일도:** <span style='color:#e74c3c; font-weight:bold;'>{uni}</span>", unsafe_allow_html=True)
    c3.markdown(f"<div class='reason-text'>↳ {u_reason}</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 진단 및 타겟 시각화</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    
    if neutralizer["needed"]:
        c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
        target_mask = dark_mask if is_two_tone else full_lip_mask
        neu_img = apply_color_overlay(base_img, target_mask, neutralizer["hex"], alpha=0.7)
        c2.image(neu_img, caption="[타겟] 중화 컬러 주입 부위", use_container_width=True)
        
        c3.warning(f"**진단 타입:** {neutralizer['type']}")
        c3.markdown(f"<div class='reason-text'><strong>[진단 이유]</strong> {neutralizer['diag_reason']}</div>", unsafe_allow_html=True)
        
        c3.markdown(f"**추천 컬러:** {get_color_box_by_hex(neutralizer['hex'], 20)} <strong>{neutralizer['mfg']} - {neutralizer['name']}</strong>", unsafe_allow_html=True)
        c3.markdown(f"<div class='reason-text'><strong>[추천 근거]</strong> {neutralizer['col_reason']}</div>", unsafe_allow_html=True)
        c3.info(f"**시술 가이드:** {neutralizer['guide']}")
    else:
        c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
        c2.image(base_img, caption="중화 불필요", use_container_width=True)
        c3.success(f"✨ {neutralizer['type']}")
        c3.markdown(f"<div class='reason-text'><strong>[진단 이유]</strong> {neutralizer['diag_reason']}</div>", unsafe_allow_html=True)

    st.markdown('<div class="step-header">🎨 분석 3&4. 본 컬러 배합 가이드</div>', unsafe_allow_html=True)
    p1, p2 = best_mix["p1"], best_mix["p2"]
    
    if p2 is None:
        st.success(f"단일 색상: {p1['mfg']} - {p1['name']} (100%)")
    else:
        st.markdown(f"**목표 컬러:** {get_color_box_by_hex(targ_hex, 25)} &nbsp; | &nbsp; **오차율(Delta E):** {best_mix['delta_e']:.2f}", unsafe_allow_html=True)
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

    st.markdown('<div class="step-header">✨ 분석 5&6. 초기 상태 ➔ 주입 타겟 ➔ 예상 결과 시뮬레이션</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.image(base_img, caption="[원본] 처리 전 초기 입술", use_container_width=True)
    c2.image(apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.9), caption="[타겟] 메인 컬러 주입 부위", use_container_width=True)
    c3.image(apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.45), caption="[예상] 시술 후 예상 결과", use_container_width=True)
    st.caption("※ 시각화된 예상 이미지는 픽셀 블렌딩을 통한 시뮬레이션이며, 실제 피부 두께 및 탈각 과정에 따라 최종 발색은 다를 수 있습니다.")
