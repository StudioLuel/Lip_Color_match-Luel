# -*- coding: utf-8 -*-
"""
PMU Lip Color Match Pro (접사 정밀 AI) — 개선판
=================================================

[이번 개선의 핵심 방향]
1. "인간의 인지 반응"을 최종 판단 기준으로 통일
   - 색상 진단(투톤 여부, 중화 필요 여부)을 임의의 L*/a*/b* 축 숫자 임계값이 아니라,
     CIEDE2000(ΔE00) 지각 색차 지표를 기준으로 재설계했습니다.
   - ΔE00 ≈ 2.3 은 색채과학에서 통용되는 "평균 관찰자의 JND(Just Noticeable Difference)"이며,
     이 값을 넘으면 사람 눈에 "다른 색"으로 인지된다는 것이 일반적으로 받아들여지는 기준입니다.
   - 색소 배합 최적화(find_best_mix)는 기존에도 ΔE2000을 쓰고 있었으므로,
     이번 개선으로 "진단 → 중화 → 배합" 전 과정이 하나의 지각 기준으로 통일되었습니다.

2. 시술 전 입술 사진 분석의 정확도/견고성 강화
   - 조명 편차를 줄이기 위한 Gray-World 화이트밸런스 보정 (토글 가능)
   - 마스크 크기가 비정상적으로 작을 때 경고 (세그멘테이션 실패 감지)
   - 예외 처리 (이미지 로딩 실패, 클러스터링 실패, 빈 데이터베이스 등)
   - 중복 코드 제거 (K-means 클러스터링 공통 함수화)
   - 파일 바이트 기반 캐싱 (재업로드 시에도 정상 동작)
   - 색소 데이터베이스를 CSV로 교체 가능하게 하여, 실제 분광측색(spectrophotometer) 실측값을
     사용할 수 있도록 확장 (기존 내장값은 "추정치"임을 명시)

※ 주의: 색소 Lab 기본값은 여전히 추정치입니다. 실제 시술에 사용하시려면
   반드시 분광측색계로 측정한 실측 Lab 값으로 교체(CSV 업로드)하시길 권장합니다.
"""

import io
import itertools
import re

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from skimage import color
from sklearn.cluster import KMeans

# =========================================================
# 0. 상수 / 색채과학 기준값
# =========================================================

# CIEDE2000 기준 "평균 관찰자가 겨우 인지 가능한 차이(JND)".
# 색채과학에서 통용되는 근사치이며, 이 값을 넘으면 사람 눈에 서로 다른 색으로 인지됩니다.
# (참고: NBS/ISCC 색차 등급, CIE 관련 문헌에서 ΔE00 2~2.3 구간을 "인지 가능"의 기준으로 사용)
PERCEPTUAL_JND = 2.3

# 진단 신뢰도 판정을 위한 여유값(margin). 1, 2위 후보의 ΔE00 차이가 이보다 작으면
# "경계선 케이스"로 간주하고 사용자에게 낮은 신뢰도임을 알립니다.
CONFIDENCE_MARGIN = 1.5

# 마스크(입술 영역)가 전체 이미지 대비 이 비율보다 작으면 세그멘테이션 실패 가능성으로 경고합니다.
MIN_LIP_AREA_RATIO = 0.03

# ---------------------------------------------------------
# 색소 데이터베이스 (기본값 = 추정치)
# ⚠️ 실제 시술 가이드로 사용하려면 분광측색계로 측정한 실측 Lab 값으로 교체해야 합니다.
#    (아래 UI에서 CSV 업로드로 교체 가능: 컬럼 = brand,name,hex,L,a,b)
# ---------------------------------------------------------
DEFAULT_PIGMENT_DB = {
    "Perma Blend": {
        "Sweet Melissa (#E08B9B)": [65.0, 30.0, 5.0],
        "Bazooka (#E65C7B)": [55.0, 45.0, 10.0],
        "Date Night (#B03A48)": [42.0, 40.0, 15.0],
        "Orange Crush (#F27930)": [60.0, 40.0, 50.0],
        "Tres Pink (#E89EB3)": [70.0, 25.0, -2.0],
        "Pillow Talk (#D47883)": [62.0, 35.0, 8.0],
        "Passion Red (#BA2737)": [45.0, 50.0, 20.0],
        "French Fancy (#D66B85)": [58.0, 38.0, 2.0],
    }
}

# ---------------------------------------------------------
# 인지 기준 레퍼런스 톤
# "건강한 발색"과 "교정이 필요한 착색 유형"을 Lab 공간의 대표점으로 정의하고,
# 실제 측정된 입술 색상과의 ΔE00 거리로 가장 가까운 카테고리를 찾습니다.
# ⚠️ 이 레퍼런스 값 역시 실측 기반 캘리브레이션이 이상적이며, 현재는 근사치입니다.
#    다만 "진단 로직"이 단일 지각 지표(ΔE00)로 통일되어, 추후 레퍼런스 값만 교체하면
#    전체 진단 정확도를 개선할 수 있는 구조입니다.
# ---------------------------------------------------------
REFERENCE_TONES = {
    "healthy_warm": {
        "lab": [58.0, 35.0, 16.0],
        "healthy": True,
        "label_ko": "건강한 웜톤",
    },
    "healthy_cool": {
        "lab": [58.0, 32.0, 9.0],
        "healthy": True,
        "label_ko": "건강한 쿨톤 (선호 발색)",
    },
    "pale_cool": {
        "lab": [60.0, 14.0, 4.0],
        "healthy": False,
        "label_ko": "창백 / 푸른기",
        "pigment_hex": "#E08B9B",
        "pigment_name": "Sweet Melissa",
        "pigment_mfg": "Perma Blend",
        "reason": "혈색이 부족하고 차가운 톤입니다. 웜코랄 계열로 혈색을 보완하는 것이 적합합니다.",
    },
    "dark_purple": {
        "lab": [38.0, 22.0, 2.0],
        "healthy": False,
        "label_ko": "어두운 보랏빛 착색",
        "pigment_hex": "#F27930",
        "pigment_name": "Orange Crush",
        "pigment_mfg": "Perma Blend",
        "reason": "명도가 낮고 보랏빛이 강합니다. 고채도 오렌지 계열로 중화가 필요합니다.",
    },
    "dark_warm_uneven": {
        "lab": [40.0, 32.0, 14.0],
        "healthy": False,
        "label_ko": "어둡지만 붉은기 잔존",
        "pigment_hex": "#E08B9B",
        "pigment_name": "Sweet Melissa",
        "pigment_mfg": "Perma Blend",
        "reason": "붉은기는 남아있으나 명도가 낮습니다. 밝기 보정 위주의 접근이 적합합니다.",
    },
}


# =========================================================
# 1. 이미지 로딩 / 전처리 유틸
# =========================================================

def _load_rgb_array(file_bytes: bytes, max_side: int = 500) -> np.ndarray:
    """바이트에서 이미지를 로딩하고 RGB numpy 배열로 변환. 실패 시 예외를 던짐."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("이미지 파일을 읽을 수 없습니다. 손상되었거나 지원하지 않는 형식입니다.") from exc
    image.thumbnail((max_side, max_side))
    return np.array(image)


def gray_world_white_balance(img_array: np.ndarray) -> np.ndarray:
    """
    간이 Gray-World 화이트밸런스 보정.
    촬영 조명(색온도)에 따른 편차를 줄여, 서로 다른 사진 간 Lab 측정값의
    일관성을 높이기 위한 전처리입니다.

    한계: 완벽한 색보정이 아니며, 극단적인 단색 배경/조명에서는 오히려
    왜곡을 유발할 수 있습니다. UI에서 토글로 켜고 끌 수 있습니다.
    """
    img = img_array.astype(np.float32)
    channel_avg = img.reshape(-1, 3).mean(axis=0)
    gray_avg = channel_avg.mean()
    scale = gray_avg / np.clip(channel_avg, 1e-6, None)
    balanced = np.clip(img * scale, 0, 255).astype(np.uint8)
    return balanced


def _weighted_kmeans_lab(img_array: np.ndarray, n_clusters: int, seed: int = 42):
    """
    공통 K-means 클러스터링 헬퍼 (기존 코드의 중복 로직을 통합).
    그림자 영향을 줄이기 위해 L*는 축소, a*/b*는 증폭한 가중 특징으로 군집화하되,
    실제 대표값(클러스터 중심)은 가중치 적용 전 원본 Lab 평균으로 계산합니다.
    """
    img_lab = color.rgb2lab(img_array)
    features = img_lab.copy()
    features[:, :, 0] *= 0.1
    features[:, :, 1] *= 2.0
    features[:, :, 2] *= 1.5

    pixels_features = features.reshape(-1, 3)
    pixels_original = img_lab.reshape(-1, 3)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
        labels = kmeans.fit_predict(pixels_features)
    except Exception as exc:
        raise RuntimeError("색상 군집화(K-means)에 실패했습니다. 이미지 해상도나 형식을 확인해주세요.") from exc

    centers = []
    for i in range(n_clusters):
        cluster_pixels = pixels_original[labels == i]
        centers.append(np.mean(cluster_pixels, axis=0) if len(cluster_pixels) > 0 else np.array([0.0, 0.0, 0.0]))
    centers = np.array(centers)

    return img_lab, labels, centers


def _refine_mask(mask_uint8: np.ndarray) -> np.ndarray:
    """모폴로지 연산으로 노이즈 제거 후 가장 큰 연결 성분(입술 덩어리)만 추출."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels_img == largest_label).astype(bool)
    return opened.astype(bool)


# =========================================================
# 2. 핵심 분석 함수 (캐시는 바이트 기반으로 안정화)
# =========================================================

@st.cache_data(show_spinner=False)
def analyze_and_mask_lip(file_bytes: bytes, use_white_balance: bool = True):
    """시술 전 입술 접사 사진을 분석하여 메인/다크 톤 마스크와 Lab 값을 추출."""
    img_array = _load_rgb_array(file_bytes)
    if use_white_balance:
        img_array = gray_world_white_balance(img_array)

    h, w, _ = img_array.shape
    img_lab, labels, actual_centers = _weighted_kmeans_lab(img_array, n_clusters=4)

    sorted_by_a_idx = np.argsort(actual_centers[:, 1])[::-1]
    lip_idx_1, lip_idx_2 = sorted_by_a_idx[0], sorted_by_a_idx[1]

    if (actual_centers[lip_idx_1][0] + actual_centers[lip_idx_1][1]) > \
       (actual_centers[lip_idx_2][0] + actual_centers[lip_idx_2][1]):
        main_idx, dark_idx = lip_idx_1, lip_idx_2
    else:
        main_idx, dark_idx = lip_idx_2, lip_idx_1

    main_lab = actual_centers[main_idx]
    dark_lab = actual_centers[dark_idx]

    labels_2d = labels.reshape(h, w)
    raw_main_mask = (labels_2d == main_idx).astype(np.uint8)
    raw_dark_mask = (labels_2d == dark_idx).astype(np.uint8)

    main_mask = _refine_mask(raw_main_mask)
    dark_mask = _refine_mask(raw_dark_mask)
    full_lip_mask = main_mask | dark_mask

    # 세그멘테이션 품질 체크: 입술 영역이 비정상적으로 작으면 경고 플래그
    lip_area_ratio = float(np.sum(full_lip_mask)) / (h * w)
    quality_warning = lip_area_ratio < MIN_LIP_AREA_RATIO

    # 두 톤이 "사람 눈에 실제로 다르게 보이는지"를 ΔE00 기준으로 판정 (JND 사용)
    delta_e_main_dark = color.deltaE_ciede2000(main_lab, dark_lab)
    is_two_tone = bool(delta_e_main_dark > PERCEPTUAL_JND)

    # 주변 피부톤(퍼스널 컬러) 유추
    skin_mask = ~full_lip_mask
    skin_tone_result = "분석 불가"
    if np.any(skin_mask):
        skin_lab = np.mean(img_lab[skin_mask], axis=0)
        sl, _sa, sb = skin_lab
        st_type = "웜톤(Warm)" if sb > 13 else "쿨톤(Cool)" if sb < 10 else "뉴트럴(Neutral)"
        st_light = "밝은 피부" if sl > 65 else "어두운 피부" if sl < 45 else "중간 피부"
        skin_tone_result = f"{st_light} / {st_type}"

    return {
        "base_img": img_array,
        "main_mask": main_mask,
        "dark_mask": dark_mask,
        "full_lip_mask": full_lip_mask,
        "main_lab": main_lab,
        "dark_lab": dark_lab,
        "is_two_tone": is_two_tone,
        "delta_e_main_dark": float(delta_e_main_dark),
        "skin_tone_result": skin_tone_result,
        "lip_area_ratio": lip_area_ratio,
        "quality_warning": quality_warning,
    }


@st.cache_data(show_spinner=False)
def get_single_dominant_lab(file_bytes: bytes, use_white_balance: bool = True) -> np.ndarray:
    """목표 색상 사진에서 가장 지배적인(붉은기가 강한) 색을 추출."""
    img_array = _load_rgb_array(file_bytes, max_side=150)
    if use_white_balance:
        img_array = gray_world_white_balance(img_array)

    _img_lab, _labels, centers = _weighted_kmeans_lab(img_array, n_clusters=3)
    lip_cluster = max(centers, key=lambda c: c[1])
    return np.array(lip_cluster)


# =========================================================
# 3. 인지 기준(ΔE00) 진단 로직
# =========================================================

def classify_perceptual_tone(lab: np.ndarray) -> dict:
    """
    측정된 Lab 값을 정의된 레퍼런스 톤들과 ΔE00으로 비교하여 가장 가까운 카테고리를 찾음.
    "인간의 인지 반응"을 기준으로 삼기 위해, 임의의 L*/b* 축 임계값 대신
    지각적으로 균일한 CIEDE2000 거리를 유일한 판단 척도로 사용합니다.
    """
    distances = []
    for key, ref in REFERENCE_TONES.items():
        d = color.deltaE_ciede2000(lab, np.array(ref["lab"]))
        distances.append((key, float(d)))
    distances.sort(key=lambda x: x[1])

    best_key, best_dist = distances[0]
    second_dist = distances[1][1] if len(distances) > 1 else best_dist + CONFIDENCE_MARGIN + 1
    margin = second_dist - best_dist
    confidence = "높음" if margin >= CONFIDENCE_MARGIN else "낮음(경계선 케이스)"

    ref = REFERENCE_TONES[best_key]
    return {
        "key": best_key,
        "delta_e": best_dist,
        "confidence": confidence,
        "margin": margin,
        **ref,
    }


def get_neutralizer_guide_detailed(main_lab, dark_lab, is_two_tone, delta_e_main_dark):
    """
    ΔE00 기반 인지 진단 결과를 사전 중화 가이드 형태로 변환.
    - 투톤인 경우: 다크(테두리) 영역의 톤을 진단
    - 단일톤인 경우: 메인 영역의 톤을 진단
    """
    target_lab = dark_lab if is_two_tone else main_lab
    diagnosis = classify_perceptual_tone(target_lab)

    base_info = {
        "region": "테두리(다크 톤)" if is_two_tone else "입술 전체",
        "confidence": diagnosis["confidence"],
        "delta_e": diagnosis["delta_e"],
    }

    if diagnosis["healthy"]:
        return {
            "needed": False,
            "type": diagnosis["label_ko"],
            "mfg": "-",
            "name": "-",
            "hex": None,
            "diag_reason": (
                f"가장 가까운 기준 톤은 '{diagnosis['label_ko']}'이며 "
                f"ΔE00={diagnosis['delta_e']:.2f}로 건강한 발색 범주에 해당합니다."
            ),
            "col_reason": "-",
            "guide": "사전 중화 생략",
            **base_info,
        }

    return {
        "needed": True,
        "type": diagnosis["label_ko"],
        "mfg": diagnosis["pigment_mfg"],
        "name": diagnosis["pigment_name"],
        "hex": diagnosis["pigment_hex"],
        "diag_reason": (
            f"{diagnosis['reason']} (기준 톤과의 지각 색차 ΔE00={diagnosis['delta_e']:.2f}, "
            f"진단 신뢰도: {diagnosis['confidence']})"
        ),
        "col_reason": f"'{diagnosis['label_ko']}' 카테고리에 대한 표준 교정 색상입니다.",
        "guide": f"{base_info['region']} 영역에 타겟팅하여 얇게 주입하세요.",
        **base_info,
    }


def analyze_lip_tone_detailed(main_lab, dark_lab, is_two_tone, delta_e_main_dark):
    """
    ※ 이 함수는 참고용 서술(descriptive) 정보만 제공합니다.
    실제 중화/교정 여부에 대한 '판단'은 classify_perceptual_tone()의 ΔE00 기준을 따릅니다.
    """
    l, _a, b = main_lab
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
        uni = "투톤 (테두리 착색)"
        u_reason = (
            f"메인 톤과 테두리 톤의 지각 색차가 ΔE00={delta_e_main_dark:.2f}로, "
            f"사람 눈이 인지 가능한 기준(JND≈{PERCEPTUAL_JND})을 초과하여 실제 착색으로 판별됨."
        )
    else:
        uni = "균일한 톤"
        u_reason = (
            f"메인 톤과 테두리 톤의 지각 색차가 ΔE00={delta_e_main_dark:.2f}로, "
            f"인지 가능 기준(JND≈{PERCEPTUAL_JND}) 이하라 균일한 상태로 판별됨."
        )

    return lightness, l_reason, hue, h_reason, uni, u_reason


# =========================================================
# 4. 시각화 유틸
# =========================================================

def apply_color_overlay(base_img, mask, hex_color, alpha=0.6):
    overlay = base_img.copy()
    rgb_color = mcolors.to_rgb(hex_color)
    color_uint8 = (np.array(rgb_color) * 255).astype(np.uint8)
    for c in range(3):
        overlay[mask, c] = (base_img[mask, c] * (1 - alpha) + color_uint8[c] * alpha).astype(np.uint8)
    return overlay


def generate_distribution_map(base_img, main_mask, dark_mask):
    overlay = base_img.copy()
    c_main = np.array(mcolors.to_rgb("#ff9ff3")) * 255
    c_dark = np.array(mcolors.to_rgb("#54a0ff")) * 255
    for c in range(3):
        overlay[main_mask, c] = (base_img[main_mask, c] * 0.4 + c_main[c] * 0.6).astype(np.uint8)
        overlay[dark_mask, c] = (base_img[dark_mask, c] * 0.4 + c_dark[c] * 0.6).astype(np.uint8)
    return overlay


def get_color_box_by_hex(hex_code, size=25):
    return (
        f'<div style="background-color: {hex_code}; width: {size}px; height: {size}px; '
        f'border-radius: 4px; border: 1px solid #999; display: inline-block; '
        f'vertical-align: middle;"></div>'
    )


def get_color_box(name, size=25):
    match = re.search(r"#([A-Fa-f0-9]{6})", name)
    return get_color_box_by_hex(f"#{match.group(1)}" if match else "#CCCCCC", size)


def lab_to_hex(lab_array):
    try:
        return mcolors.to_hex(color.lab2rgb(np.array([[lab_array]]))[0][0])
    except Exception:
        return "#CCCCCC"


# =========================================================
# 5. 색소 배합 로직 (ΔE2000 기반 — 인지 기준 최적화)
# =========================================================

def build_pigment_list(pigment_db: dict) -> list:
    return [
        {"mfg": m, "name": n, "lab": np.array(l)}
        for m, pigs in pigment_db.items()
        for n, l in pigs.items()
    ]


def find_best_mix(target_lab, pigment_db: dict):
    """
    목표 색상(target_lab)에 가장 가까운 단일/2종 배합을 ΔE2000(인지 색차) 기준으로 탐색.
    ΔE2000을 손실함수로 사용하는 것 자체가 "인간이 실제로 느끼는 색차"를 최소화하는
    방향이므로, 이 부분은 기존 설계를 유지하되 예외 처리만 보강했습니다.
    """
    all_pigments = build_pigment_list(pigment_db)
    if not all_pigments:
        return None

    best_match = None
    min_delta_e = float("inf")

    for pig in all_pigments:
        delta_e = color.deltaE_ciede2000(target_lab, pig["lab"])
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = {"p1": pig, "p2": None, "r1": 100, "r2": 0, "delta_e": delta_e}

    # 2종 배합 탐색.
    # 주의: Lab 공간에서의 선형보간은 실제 안료의 감법혼색(Kubelka-Munk 등)과 정확히 일치하지
    # 않는 근사치입니다. 다만 탐색 자체는 여전히 ΔE2000(지각 색차)을 최소화하는 방향으로 이뤄지므로,
    # "최종적으로 눈에 가장 가깝게 보이는 배합"을 찾는다는 목적에는 부합합니다.
    for p1, p2 in itertools.combinations(all_pigments, 2):
        for ratio in range(10, 100, 10):
            mixed_lab = (p1["lab"] * (ratio / 100.0)) + (p2["lab"] * ((100 - ratio) / 100.0))
            delta_e = color.deltaE_ciede2000(target_lab, mixed_lab)
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match = {
                    "p1": p1, "p2": p2, "r1": ratio, "r2": 100 - ratio,
                    "delta_e": delta_e, "mixed_lab": mixed_lab,
                }

    return best_match


def load_pigment_db_from_csv(uploaded_file) -> dict:
    """
    사용자가 업로드한 CSV(brand,name,hex,L,a,b)로 색소 DB를 구성.
    실제 분광측색 데이터를 반영할 수 있도록 하기 위한 확장 포인트입니다.
    """
    df = pd.read_csv(uploaded_file)
    required_cols = {"brand", "name", "hex", "L", "a", "b"}
    missing = required_cols - set(df.columns.str.strip())
    if missing:
        raise ValueError(f"CSV에 다음 컬럼이 없습니다: {', '.join(missing)}")

    db: dict = {}
    for _, row in df.iterrows():
        brand = str(row["brand"]).strip()
        display_name = f"{str(row['name']).strip()} ({str(row['hex']).strip()})"
        db.setdefault(brand, {})[display_name] = [float(row["L"]), float(row["a"]), float(row["b"])]
    return db


# =========================================================
# 6. UI
# =========================================================

st.set_page_config(page_title="PMU 컬러 매치 프로", page_icon="💋", layout="centered")

st.markdown(
    """
    <style>
    .step-header { background-color: #e2e8f0; color: #111111 !important; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
    .extract-box { padding: 12px; background-color: #f8f9fa; color: #111111 !important; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px; font-size: 0.95em; }
    .reason-text { font-size: 0.85em; color: #555555; margin-top: 2px; margin-bottom: 10px; padding-left: 10px; border-left: 2px solid #ddd; }
    .legend-box { display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; align-items: center; justify-content: center; background-color: #f1f2f6; color: #111111 !important; padding: 8px; border-radius: 5px; }
    .legend-color { width: 16px; height: 16px; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 5px; border: 1px solid #999; }
    .warn-box { background-color: #fff3cd; color: #664d03 !important; padding: 10px; border-radius: 5px; border: 1px solid #ffe69c; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💋 PMU Lip Color Match Pro (접사 정밀 AI)")
st.caption(
    "이 도구는 참고용 보조 분석이며, 최종 색상 판단과 시술 결정은 반드시 숙련된 시술자가 육안 확인을 병행해야 합니다."
)
st.markdown("입술 접사(확대) 사진에 최적화된 AI 알고리즘이 착색과 톤을 지각(ΔE2000) 기준으로 정밀하게 진단합니다.")

with st.expander("⚙️ 고급 설정 (화이트밸런스 · 색소 데이터베이스)"):
    use_white_balance = st.checkbox(
        "촬영 조명 편차 보정 (Gray-World 화이트밸런스)", value=True,
        help="사진마다 다른 조명색을 완화하여 Lab 측정 일관성을 높입니다. 단색 배경 등 특수한 경우 오히려 왜곡될 수 있어 끌 수 있습니다.",
    )
    st.markdown(
        "**색소 데이터베이스**: 기본값은 추정치입니다. 분광측색계로 측정한 실제 Lab 값이 있다면 "
        "`brand,name,hex,L,a,b` 컬럼을 가진 CSV로 교체하세요."
    )
    pigment_csv = st.file_uploader("색소 데이터베이스 CSV (선택)", type=["csv"], key="pigment_csv")

    pigment_db = DEFAULT_PIGMENT_DB
    if pigment_csv is not None:
        try:
            pigment_db = load_pigment_db_from_csv(pigment_csv)
            st.success(f"실측 색소 데이터베이스 {sum(len(v) for v in pigment_db.values())}종을 불러왔습니다.")
        except Exception as exc:
            st.error(f"CSV를 불러오는 중 오류가 발생했습니다: {exc}\n기본(추정) 데이터베이스를 사용합니다.")
            pigment_db = DEFAULT_PIGMENT_DB
    else:
        st.info("현재 기본(추정) 색소 데이터베이스를 사용 중입니다.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("STEP 1. 현재 입술")
    current_file = st.file_uploader("시술 전 입술 확대 사진", type=["jpg", "jpeg", "png"], key="current")
with col2:
    st.subheader("STEP 2. 목표 색상")
    target_file = st.file_uploader("원하는 컬러 사진", type=["jpg", "jpeg", "png"], key="target")

if current_file and target_file:
    st.divider()

    current_bytes = current_file.getvalue()
    target_bytes = target_file.getvalue()

    try:
        with st.spinner("입술 접사 사진의 노이즈를 제거하고 지각(ΔE2000) 기준 톤 분석을 진행 중입니다..."):
            result = analyze_and_mask_lip(current_bytes, use_white_balance=use_white_balance)
            targ_lab = get_single_dominant_lab(target_bytes, use_white_balance=use_white_balance)
    except (ValueError, RuntimeError) as exc:
        st.error(f"이미지 분석 중 오류가 발생했습니다: {exc}")
        st.stop()

    base_img = result["base_img"]
    main_mask = result["main_mask"]
    dark_mask = result["dark_mask"]
    full_lip_mask = result["full_lip_mask"]
    curr_main_lab = result["main_lab"]
    curr_dark_lab = result["dark_lab"]
    is_two_tone = result["is_two_tone"]
    delta_e_main_dark = result["delta_e_main_dark"]
    skin_tone_result = result["skin_tone_result"]

    if result["quality_warning"]:
        st.markdown(
            f'<div class="warn-box">⚠️ <strong>세그멘테이션 품질 경고:</strong> '
            f"추출된 입술 영역이 전체 이미지의 {result['lip_area_ratio']*100:.1f}%로 매우 작습니다. "
            f"조명이 고르고 입술이 프레임을 채우는 접사 사진으로 다시 촬영하시면 정확도가 높아집니다.</div>",
            unsafe_allow_html=True,
        )

    main_hex = lab_to_hex(curr_main_lab)
    dark_hex = lab_to_hex(curr_dark_lab)
    targ_hex = lab_to_hex(targ_lab)

    lightness, l_reason, hue, h_reason, uni, u_reason = analyze_lip_tone_detailed(
        curr_main_lab, curr_dark_lab, is_two_tone, delta_e_main_dark
    )
    neutralizer = get_neutralizer_guide_detailed(curr_main_lab, curr_dark_lab, is_two_tone, delta_e_main_dark)

    with st.spinner("목표 색상과 가장 가까운(지각적으로) 배합을 탐색 중입니다..."):
        best_mix = find_best_mix(targ_lab, pigment_db)

    if best_mix is None:
        st.error("색소 데이터베이스가 비어 있습니다. CSV를 확인해주세요.")
        st.stop()

    mix_hex = targ_hex if best_mix["p2"] is None else mcolors.to_hex(
        color.lab2rgb(np.array([[best_mix["mixed_lab"]]]))[0][0]
    )

    # ---------------- 분석 1: 색상 분포도 & 퍼스널 컬러 ----------------
    st.markdown('<div class="step-header">🔍 분석 1. 시술 전 색상 분포도 & 퍼스널 컬러</div>', unsafe_allow_html=True)
    st.info(f"👱‍♀️ **입술 주변 피부톤 힌트:** {skin_tone_result} (색소 선택 시 참고하세요)")
    st.markdown(
        """
        <div class="legend-box">
            <div><span class="legend-color" style="background-color: #ff9ff3;"></span><strong>정상 발색 영역 (메인 톤)</strong></div>
            <div><span class="legend-color" style="background-color: #54a0ff;"></span><strong>붉은기 소실/착색 영역 (다크 톤)</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
    dist_img = generate_distribution_map(base_img, main_mask, dark_mask)
    c2.image(dist_img, caption="[분석] 정밀 맵핑 분포도", use_container_width=True)

    c3.markdown(f"**명도 상태:** {lightness}")
    c3.markdown(f"<div class='reason-text'>↳ {l_reason}</div>", unsafe_allow_html=True)
    c3.markdown(f"**온도/색상:** {hue}")
    c3.markdown(f"<div class='reason-text'>↳ {h_reason}</div>", unsafe_allow_html=True)
    c3.markdown(f"**균일도:** <span style='color:#e74c3c; font-weight:bold;'>{uni}</span>", unsafe_allow_html=True)
    c3.markdown(f"<div class='reason-text'>↳ {u_reason}</div>", unsafe_allow_html=True)

    # ---------------- 분석 2: 사전 중화 진단 (ΔE00 기준) ----------------
    st.markdown(
        '<div class="step-header">🛠️ 분석 2. 사전 중화(Neutralizer) 진단 및 타겟 시각화 (인지 기준 ΔE00)</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1.2])

    if neutralizer["needed"]:
        c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
        target_mask = dark_mask if is_two_tone else full_lip_mask
        neu_img = apply_color_overlay(base_img, target_mask, neutralizer["hex"], alpha=0.7)
        c2.image(neu_img, caption="[타겟] 중화 컬러 주입 부위", use_container_width=True)

        c3.warning(f"**진단 타입:** {neutralizer['type']} (신뢰도: {neutralizer['confidence']})")
        c3.markdown(f"<div class='reason-text'><strong>[진단 이유]</strong> {neutralizer['diag_reason']}</div>", unsafe_allow_html=True)
        c3.markdown(
            f"**추천 컬러:** {get_color_box_by_hex(neutralizer['hex'], 20)} <strong>{neutralizer['mfg']} - {neutralizer['name']}</strong>",
            unsafe_allow_html=True,
        )
        c3.markdown(f"<div class='reason-text'><strong>[추천 근거]</strong> {neutralizer['col_reason']}</div>", unsafe_allow_html=True)
        c3.info(f"**시술 가이드:** {neutralizer['guide']}")
        if neutralizer["confidence"] != "높음":
            c3.caption("⚠️ 경계선 케이스입니다. 육안으로 재확인 후 결정하세요.")
    else:
        c1.image(base_img, caption="[원본] 초기 입술", use_container_width=True)
        c2.image(base_img, caption="중화 불필요", use_container_width=True)
        c3.success(f"✨ {neutralizer['type']} (신뢰도: {neutralizer['confidence']})")
        c3.markdown(f"<div class='reason-text'><strong>[진단 이유]</strong> {neutralizer['diag_reason']}</div>", unsafe_allow_html=True)

    # ---------------- 분석 3&4: 배합 가이드 ----------------
    st.markdown('<div class="step-header">🎨 분석 3&4. 본 컬러 배합 가이드 (ΔE2000 최소화)</div>', unsafe_allow_html=True)
    p1, p2 = best_mix["p1"], best_mix["p2"]

    if p2 is None:
        st.success(f"단일 색상: {p1['mfg']} - {p1['name']} (100%) · ΔE00={best_mix['delta_e']:.2f}")
    else:
        st.markdown(
            f"**목표 컬러:** {get_color_box_by_hex(targ_hex, 25)} &nbsp; | &nbsp; "
            f"**지각 오차(ΔE00):** {best_mix['delta_e']:.2f} "
            f"{'(사람 눈으로 거의 구별 불가)' if best_mix['delta_e'] < PERCEPTUAL_JND else '(약간의 차이가 인지될 수 있음)'}",
            unsafe_allow_html=True,
        )
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
        st.caption(
            "※ 배합 비율은 Lab 공간 선형보간에 대한 ΔE2000 최소화 결과입니다. "
            "실제 안료의 감법혼색(피부 침투 후 발색)은 이론값과 다를 수 있으므로 패치 테스트를 권장합니다."
        )

    # ---------------- 분석 5&6: 시뮬레이션 ----------------
    st.markdown(
        '<div class="step-header">✨ 분석 5&6. 초기 상태 ➔ 주입 타겟 ➔ 예상 결과 시뮬레이션</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.image(base_img, caption="[원본] 처리 전 초기 입술", use_container_width=True)
    c2.image(apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.9), caption="[타겟] 메인 컬러 주입 부위", use_container_width=True)
    c3.image(apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.45), caption="[예상] 시술 후 예상 결과", use_container_width=True)
    st.caption(
        "※ 시각화된 예상 이미지는 픽셀 블렌딩을 통한 시뮬레이션이며, 실제 피부 두께 및 탈각 과정에 "
        "따라 최종 발색은 다를 수 있습니다. 모든 진단·추천은 참고용이며 최종 판단은 시술자의 육안 확인을 거쳐야 합니다."
    )
