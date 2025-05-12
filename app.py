# 모듈 불러오기
import pandas as pd
import numpy as np
import streamlit as st
import requests
import joblib
import ast

from PIL import Image
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import pickle
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity



# 넥슨 API_KEY
API_KEY = "live_894cbc7c1e6fc3db5b54614eb927bcd9ae15c0eeddf53f337288526957c78db5efe8d04e6d233bd35cf2fabdeb93fb0d"
headers = {"x-nxopen-api-key": API_KEY}

# Streamlit 인터페이스 구성
st.set_page_config(page_title="🍁 메이플스토리 분석기", layout="wide")
st.title("🍁 메이플스토리 분석기")

tab1, tab2, tab3 = st.tabs(["🥷 캐릭터 정보", "⚔️ 장비 추천", "📊 시뮬레이션"])

# -------------------- API 요청 --------------------

def character_data(ocid: str, headers: dict):
    # 캐릭터 기본, 장비, 어빌리티 정보 요청
    base_url = "https://open.api.nexon.com/maplestory/v1/character"
    info = requests.get(f"{base_url}/basic?ocid={ocid}", headers=headers).json()
    equip = requests.get(f"{base_url}/item-equipment?ocid={ocid}", headers=headers).json()
    ability = requests.get(f"{base_url}/ability?ocid={ocid}", headers=headers).json()
    stat = requests.get(f"{base_url}/stat?ocid={ocid}", headers=headers).json()

    presets = {
        0: equip.get("item_equipment", []),
        1: equip.get("item_equipment_preset_1", []),
        2: equip.get("item_equipment_preset_2", []),
        3: equip.get("item_equipment_preset_3", []),
    }
    return info, presets, ability, stat

# -------------------- 캐릭터 장비 정보 --------------------

def parse_equipment_to_df(equip_items):
    return pd.DataFrame([
        {
            "장비 부위": item.get("item_equipment_slot"),
            "장비 이름": item.get("item_name"),
            "장비 아이콘": item.get("item_icon"),
            "스타포스": item.get("starforce"),
            "장비 등급": item.get("potential_option_grade"),
            "장비 최종 옵션": item.get("item_total_option", {}),
            "장비 기본 옵션": item.get("item_base_option", {}),
            "장비 추가 옵션": item.get("item_add_option", {}),
            "장비 기타 옵션": item.get("item_etc_option", {}),
            "장비 스타포스 옵션" : item.get("item_starforce_option", {}),
            "잠재능력": ", ".join([opt for opt in [
                item.get("potential_option_1", ""),
                item.get("potential_option_2", ""),
                item.get("potential_option_3", "")
            ] if opt]),
            "에디셔널": ", ".join([opt for opt in [
                item.get("additional_potential_option_1", ""),
                item.get("additional_potential_option_2", ""),
                item.get("additional_potential_option_3", "")
            ] if opt])
        }
        for item in equip_items
    ])

# -------------------- 캐릭터 장비 시각화 --------------------

stat_name_map = {
    "str": "STR",
    "dex": "DEX",
    "int": "INT",
    "luk": "LUK",
    "max_hp": "최대 HP",
    "max_mp": "최대 MP",
    "attack_power": "공격력",
    "magic_power": "마력",
    "armor": "방어력",
    "speed": "이동속도",
    "jump": "점프력",
    "boss_damage": "보스 공격력",
    "ignore_monster_armor": "몬스터 방어율 무시",
    "all_stat": "올스탯",
    "damage": "데미지",
    "max_hp_rate": "최대 HP",
    "max_mp_rate": "최대 MP",
    "equipment_level_increase": "착용 레벨 증가",
    "equipment_level_decrease": "착용 레벨 감소"
}

def render_equipment_grid(equip_df, num_cols=6):
    rows = [equip_df[i:i+num_cols] for i in range(0, len(equip_df), num_cols)]
    for row_ in rows:
        cols = st.columns(num_cols)
        for i, (_, item) in enumerate(row_.iterrows()):
            with cols[i]:
                try:
                    response = requests.get(item['장비 아이콘'])
                    img = Image.open(BytesIO(response.content)).resize((50, 50))
                    st.image(img)
                except:
                    st.write("이미지 불러오기 실패")

                with st.expander(item['장비 이름']):
                    st.markdown(f"**장비 등급 :** {item['장비 등급']}")
                    st.markdown(f"**스타포스 :** {item['스타포스']}")
                    
                    # 제외 옵션
                    exclude_keys = {"base_equipment_level", "equipment_level_increase", "equipment_level_decrease"}

                    base = item["장비 기본 옵션"]
                    add = item["장비 추가 옵션"]
                    etc = item["장비 기타 옵션"]
                    star = item["장비 스타포스 옵션"]

                    st.markdown("**최종 옵션**")
                    for key in stat_name_map:
                        if key in exclude_keys:
                            continue

                        b = base.get(key, 0)
                        a = add.get(key, 0)
                        e = etc.get(key, 0)
                        s = star.get(key, 0)

                        try:
                            fb, fa, fe, fs = map(float, (b, a, e, s))
                            total = fb + fa + fe + fs
                        except:
                            continue

                        if total == 0:
                            continue

                        is_percent = "%" in str(b) or "%" in str(a) or "%" in str(e) or "%" in str(s)
                        unit = "%" if is_percent else ""
                        display_k = stat_name_map[key]

                        def fmt(v):
                            return f"{float(v):.0f}" if float(v).is_integer() else f"{float(v):.1f}"

                        colored_parts = (
                            f"<span style='color:#000000'>{fmt(b)}</span>"
                            f"<span style='color:#2ecc71'>+{fmt(a)}</span>"
                            f"<span style='color:#9b59b6'>+{fmt(e)}</span>"
                            f"<span style='color:#f1c40f'>+{fmt(s)}</span>"
                        )

                        st.markdown(
                            f"<div style='margin-bottom:3px; font-size: 12px;'> <b style='font-size: 14px;'>{display_k}</b> : +{fmt(total)}{unit} "
                            f"(<code>{colored_parts}</code>){unit}</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("---")
                    st.markdown(
                    f"**잠재 능력**<br><span style='font-size: 12px;'>{item['잠재능력'].replace(', ', '<br>')}</span>",
                    unsafe_allow_html=True
                    )
                    st.markdown("---")
                    st.markdown(
                    f"**에디셔널**<br><span style='font-size: 12px;'>{item['에디셔널'].replace(', ', '<br>')}</span>",
                    unsafe_allow_html=True
                    )

# -------------------- 캐릭터 어빌리티 시각화 --------------------

def render_ability_info(ability_json):
    st.markdown("**어빌리티**")
    grade_colors = {
        "레전드리": "#2ecc71",
        "유니크": "#FF8C00",
        "에픽": "#9400D3",
        "레어": "#1E90FF"
    }

    for ability in ability_json.get('ability_info', []):
        grade = ability.get('ability_grade', '알 수 없음')
        value = ability.get('ability_value', '')
        color = grade_colors.get(grade, '#DDDDDD')

        # 능력 등급과 값을 출력하는 부분
        st.markdown(f"""
        <div style='padding: 5px 10px; border-radius: 8px; border: 1px solid black; background-color: {color}; color: white; font-weight: bold; text-align: center; margin-bottom: 10px; height: 40px; font-size: 15px;'>
            {value}
        </div>
        """, unsafe_allow_html=True)

# -------------------- 캐릭터 전투력 시각화 --------------------

# 한국식 단위로 변환하는 함수
def format_korean_number(num):
    num = int(num)
    eok = num // 100_000_000
    man = (num % 100_000_000) // 10_000
    rest = num % 10_000

    parts = []
    if eok > 0:
        parts.append(f"{eok}억")
    if man > 0 or (eok > 0 and rest > 0):
        parts.append(f"{man}만")
    if rest > 0 or not parts:
        parts.append(str(rest).zfill(4) if man > 0 else str(rest))

    return " ".join(parts)

def render_combat_power(stat_json):
    combat_stat = next((item for item in stat_json.get("final_stat", []) if item["stat_name"] == "전투력"), None)
    
    if combat_stat:
        raw_value = combat_stat["stat_value"]
        formatted_value = format_korean_number(raw_value)
        
        st.markdown(f"""
        <div style='margin-top: 10px; padding: 10px; border-radius: 10px; border: 1px solid black;
                    background-color: #4444AA; color: white; font-weight: bold; text-align: center;
                    font-size: 18px;'>
            전투력 : {formatted_value}
        </div>
        """, unsafe_allow_html=True)

# -------------------- 장비 평균 옵션 계산 --------------------

def average_dicts(list_of_dicts):
    count_dict = {}
    sum_dict = {}
    for d in list_of_dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    val = float(v)
                    sum_dict[k] = sum_dict.get(k, 0) + val
                    count_dict[k] = count_dict.get(k, 0) + 1
                except:
                    continue
    return {k: sum_dict[k] / count_dict[k] for k in sum_dict}

def render_aggregated_equipment_grid(equip_df, num_cols=6):
    grouped = equip_df.groupby('장비 이름').agg({
        '장비 등급': lambda x: x.mode().iloc[0] if not x.mode().empty else '-',
        '스타포스': 'mean',
        '장비 아이콘': 'first',
        '잠재능력': lambda x: x.mode().iloc[0] if not x.mode().empty else '-',
        '에디셔널': lambda x: x.mode().iloc[0] if not x.mode().empty else '-',
        '장비 부위': 'first',
        '장비 기본 옵션': lambda x: average_dicts(x.dropna()),
        '장비 추가 옵션': lambda x: average_dicts(x.dropna()),
        '장비 기타 옵션': lambda x: average_dicts(x.dropna()),
        '장비 스타포스 옵션': lambda x: average_dicts(x.dropna())
    }).reset_index()

    rows = [grouped[i:i + num_cols] for i in range(0, len(grouped), num_cols)]

    item_counts = equip_df['장비 이름'].value_counts(normalize=True) * 100
    item_share = item_counts.to_dict()
    
    for row_ in rows:
        cols = st.columns(num_cols)
        for i, (_, item) in enumerate(row_.iterrows()):
            with cols[i]:
                try:
                    response = requests.get(item['장비 아이콘'])
                    img = Image.open(BytesIO(response.content)).resize((50, 50))
                    st.image(img)
                except:
                    st.write("이미지 불러오기 실패")

                item_name = item['장비 이름']
                item_pct = item_share.get(item_name, 0)

                with st.expander(f"{item_name} {item_pct:.1f}%"):
                    st.markdown(f"**장비 등급 (최빈값) :** {item['장비 등급']}")
                    st.markdown(f"**스타포스 (평균) :** {item['스타포스']:.1f}")

                    exclude_keys = {"base_equipment_level", "equipment_level_increase", "equipment_level_decrease"}

                    base = item["장비 기본 옵션"]
                    add = item["장비 추가 옵션"]
                    etc = item["장비 기타 옵션"]
                    star = item["장비 스타포스 옵션"]

                    st.markdown("**최종 옵션 (평균)**")
                    for key in stat_name_map:
                        if key in exclude_keys:
                            continue

                        b_value = base.get(key, 0)
                        a_value = add.get(key, 0)
                        e_value = etc.get(key, 0)
                        s_value = star.get(key, 0)

                        total = 0
                        try:
                            b_value = float(b_value)
                            a_value = float(a_value)
                            e_value = float(e_value)
                            s_value = float(s_value)
                            total = b_value + a_value + e_value + s_value
                        except (ValueError, TypeError):
                            continue

                        if total == 0:
                            continue

                        is_percent = "%" in key or key.endswith("_rate")
                        unit = "%" if is_percent else ""
                        display_k = stat_name_map.get(key, key)

                        def fmt(v):
                            return f"{float(v):.0f}" if float(v).is_integer() else f"{float(v):.1f}"

                        colored_parts = (
                            f"<span style='color:#000000'>{fmt(b_value)}</span>"
                            f"<span style='color:#2ecc71'>+{fmt(a_value)}</span>"
                            f"<span style='color:#9b59b6'>+{fmt(e_value)}</span>"
                            f"<span style='color:#f1c40f'>+{fmt(s_value)}</span>"
                        )

                        st.markdown(
                            f"<div style='margin-bottom:3px; font-size: 12px;'> <b style='font-size: 14px;'>{display_k}</b> : +{fmt(total)}{unit} "
                            f"(<code>{colored_parts}</code>){unit}</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("---")
                    st.markdown(
                        f"**잠재 능력 (최빈값)**<br><span style='font-size: 12px;'>{item['잠재능력'].replace(', ', '<br>')}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
                    st.markdown(
                        f"**에디셔널 (최빈값)**<br><span style='font-size: 12px;'>{item['에디셔널'].replace(', ', '<br>')}</span>",
                        unsafe_allow_html=True
                    )

# -------------------- TAB 1 --------------------

with tab1:
    st.header("🥷 캐릭터 정보")
    nickname = st.text_input("캐릭터 닉네임을 입력하세요")

    if nickname:
        st.write(f"**{nickname}** 님의 정보를 조회 중입니다...")
        url_id = f"https://open.api.nexon.com/maplestory/v1/id?character_name={nickname}"
        res = requests.get(url_id, headers=headers)

        if res.status_code == 200:
            ocid = res.json()["ocid"]
            info, equip_items, ability_json, stat = character_data(ocid, headers)
            
            # selected_preset이 session_state에 없으면 초기화
            if "selected_preset" not in st.session_state:
                st.session_state["selected_preset"] = 0

            selected_preset = st.session_state["selected_preset"]
            selected_preset_items = equip_items.get(selected_preset, [])
            equip_df = parse_equipment_to_df(selected_preset_items)

            st.session_state["equip_df"] = equip_df
            st.session_state["character_info"] = info

            left_col, right_col = st.columns([1, 3])
            with left_col:
                st.image(info.get("character_image"), width=120)
                st.markdown(f"""
                    **닉네임** : {info.get("character_name")}  
                    **직업** : {info.get("character_class")}  
                    **월드** : {info.get("world_name")}  
                    **레벨** : {info.get("character_level")}
                """)

                render_combat_power(stat) # 전투력 정보 표시
                st.markdown("---")
                render_ability_info(ability_json) # 어빌리티 정보 표시

            with right_col:
                st.subheader("장비 목록")

                # selectbox로 프리셋을 선택
                preset_names = {
                    0: "0",
                    1: "1",
                    2: "2",
                    3: "3"
                }
                
                selected_preset = st.selectbox(
                    "프리셋 선택", 
                    options=[0, 1, 2, 3], 
                    format_func=lambda x: preset_names[x], 
                    key="selected_preset_tab1"
                )

                # 장비 부위 순서 정렬용 우선순위 딕셔너리
                part_priority = {
                    "무기": 1, "보조무기": 2, "엠블렘": 3, "모자": 4, 
                    "상의": 5, "하의": 6, "신발": 7, "장갑": 8, 
                    "망토": 9, "어깨장식": 10, "얼굴장식": 11, "눈장식": 12, 
                    "귀고리": 13, "벨트": 14, "펜던트": 15, "펜던트2": 16, 
                    "반지1": 17, "반지2": 18, "반지3": 19, "반지4": 20,
                    "포켓 아이템": 21, "기계 심장": 22, "뱃지": 23, "훈장": 24
                }
                equip_df["sort_order"] = equip_df["장비 부위"].map(part_priority).fillna(999)
                equip_df = equip_df.sort_values("sort_order")

                render_equipment_grid(equip_df)

        else:
            st.error("캐릭터 조회 실패. 닉네임을 다시 확인해주세요.")

# -------------------- TAB 2 --------------------

def get_equipment_data_from_api(ocid: str, headers: dict):
    url = f"https://open.api.nexon.com/maplestory/v1/character/item-equipment?ocid={ocid}"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json().get("item_equipment", [])
    else:
        return None

with tab2:
    st.header("⚔️ 장비 추천")
    st.write("직업과 목표 전투력을 설정하고 비슷한 상위 유저들의 장비를 참고해 육성 방향을 설정하세요.")

    # 상위 유저 데이터 불러오기
    df = pd.read_csv("streamlit/power.csv")

    # 직업 선택
    subclasses = sorted(df["subclass"].dropna().unique())
    selected_subclass = st.selectbox("직업 선택", subclasses)

    df_filtered = df[df["subclass"] == selected_subclass]

    if df_filtered.empty:
        st.warning("선택한 직업의 데이터가 없습니다.")
        st.stop()

    # 전투력 슬라이더
    target_cp = st.slider(
        "목표 전투력을 설정하세요",
        min_value=int(df_filtered["전투력"].min()),
        max_value=int(df_filtered["전투력"].max()),
        value=int(df_filtered["전투력"].mean())
    )

    # 유사 유저 5명 추출 (전투력 차이 기준)
    df_filtered["전투력차이"] = abs(df_filtered["전투력"] - target_cp)
    top5_users = df_filtered.sort_values("전투력차이").head(5)
    st.subheader("전투력이 비슷한 유저")
    st.dataframe(top5_users[["nickname", "전투력"]])

    # 장비 조회
    all_equips = []

    for nickname in top5_users["nickname"]:
        # st.write(f"{nickname} 유저 장비 조회 중...")
        url_id = f"https://open.api.nexon.com/maplestory/v1/id?character_name={nickname}"
        res = requests.get(url_id, headers=headers)

        if res.status_code == 200:
            ocid = res.json().get("ocid")
            if ocid:
                # 유저의 장비 데이터를 가져옴
                equip_items = get_equipment_data_from_api(ocid, headers)
                if equip_items:
                    equip_df = parse_equipment_to_df(equip_items)
                    equip_df["nickname"] = nickname
                    all_equips.append(equip_df)
                else:
                    st.warning(f"{nickname}의 장비 정보를 불러오는 데 실패했습니다.")
            else:
                st.warning(f"{nickname}의 ocid를 찾을 수 없습니다.")
        else:
            st.warning(f"{nickname} 조회 실패")

    if all_equips:
        final_df = pd.concat(all_equips, ignore_index=True)

        # 스타포스는 수치형으로 변환
        final_df['스타포스'] = pd.to_numeric(final_df['스타포스'], errors='coerce')

        # 장비 등급은 최빈값으로 처리
        grade_mode = final_df['장비 등급'].mode().iloc[0] if not final_df['장비 등급'].mode().empty else '-'
        final_df['장비 등급'] = final_df['장비 등급'].fillna(grade_mode)

        # 장비 부위 정렬 우선순위 정의
        part_priority = {
            "무기": 1, "보조무기": 2, "엠블렘": 3, "모자": 4,
            "상의": 5, "하의": 6, "신발": 7, "장갑": 8,
            "망토": 9, "어깨장식": 10, "얼굴장식": 11, "눈장식": 12,
            "귀고리": 13, "벨트": 14, "펜던트": 15, "펜던트2": 16,
            "반지1": 17, "반지2": 18, "반지3": 19, "반지4": 20, 
            "포켓 아이템": 21, "기계 심장": 22, "뱃지": 23, "훈장": 24
        }
        part_order = list(part_priority.keys())

        # 카테고리형으로 정렬 순서 고정
        final_df["장비 부위"] = pd.Categorical(final_df["장비 부위"], categories=part_order, ordered=True)
        final_df = final_df.sort_values("장비 부위")

        # 시각화
        st.markdown("### 장비 점유율")

        for part, group in final_df.groupby("장비 부위"):
            if group.empty:
                continue

            st.markdown(f"{part}")
            render_aggregated_equipment_grid(group)

# -------------------- TAB 3 --------------------

with tab3:
    st.header("📊 시뮬레이션(Beta)")
    st.write("직업과 장비를 세부적으로 설정하면 딥러닝 모델을 통해 유사한 고레벨 유저를 추천합니다.")
    st.write("장비 세트는 파프니르, 앱솔랩스, 도전자를 선택할 수 있습니다.")
    
    # ------------------------------
    # 모델 정의 (Deepsets 구조)
    class DeepMaskedModel(nn.Module):
        def __init__(self, embedding_info, num_cont_features, emb_dim=8):
            super().__init__()
            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(cardinality, emb_dim)
                for name, cardinality in embedding_info.items()
            })
            total_emb_dim = emb_dim * len(embedding_info)
            self.input_dim = total_emb_dim + num_cont_features
            self.phi = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.rho = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x_cat, x_cont, mask):
            emb = [self.embeddings[name](x_cat[:, :, i]) for i, name in enumerate(self.embeddings)]
            emb = torch.cat(emb, dim=-1)
            x = torch.cat([emb, x_cont], dim=-1)
            encoded = self.phi(x)
            masked = encoded * mask.unsqueeze(-1)
            pooled = masked.sum(dim=1)
            return pooled

    # ------------------------------
    # 장비 인코딩 함수 (연속형 19차원 벡터만 추출)
    def encode_item_to_cont_vector(item):
        option_map = {'레전드리': 3, '유니크': 2, '에픽': 1, '레어': 0}
        grade_map = {'S': 3, 'A': 2, 'B': 1, '기타': 0}
        item_group_map = {'파프니르': 0, '앱솔랩스': 1, '도전자': 2}

        vec = [
            item['boss_dmg'], item['ignore_def'], item['all_stat_total'], item['damage'],
            0, 0, item['all_stat_total'], item['starforce'], 0,
            0, item['mainstat_total'], item['power_total'], 0, 0,
            0, 0, 0, 0, item['item_count']
        ]
        return vec

    # ------------------------------
    # 전역 embedding 정보 정의
    embedding_info = {
        'subclass': 46,
        'equipment_slot': 24,
        'main_stat_type': 6,
        'item_group': 15,
        'starforce_scroll_flag': 2,
        'potential_option_grade': 6,
        'additional_potential_option_grade': 6,
        'main_pot_grade_summary': 42,
        'add_pot_grade_summary': 55,
        'potential_status': 27
    }

    # ------------------------------
    @st.cache_resource
    def load_model():
        model = DeepMaskedModel(embedding_info, num_cont_features=19)
        model.load_state_dict(torch.load("streamlit/best_model_r2_0.7083_rmse_0.70.pt", map_location="cpu"))
        model.eval()
        return model

    @st.cache_resource
    def load_high_level_user_vectors():
        vecs = np.load("streamlit/high_level_user_vectors.npy")
        nicks = np.load("streamlit/high_level_user_nicks.npy", allow_pickle=True)
        return vecs, nicks

    @st.cache_resource
    def load_job_avg_vectors():
        with open("streamlit/subclass_profiles.pkl", "rb") as f:
            return pickle.load(f)

    model = load_model()
    hl_vecs, hl_nicks = load_high_level_user_vectors()
    job_avg_vectors = load_job_avg_vectors()

    # ------------------------------
    # 사용자 입력 받기
    job_list = sorted(job_avg_vectors.keys())
    user_job = st.selectbox("직업을 선택하세요", job_list)
    top_slots = st.multiselect("장비 부위를 선택하세요", ['무기', '모자', '장갑', '신발', '망토', '상의', '하의'], default=['무기', '모자', '장갑'])

    user_input = {}
    for part in top_slots:
        with st.expander(f"{part} 정보 입력"):
            item = {}
            item['item_group'] = st.selectbox(f"{part} - 장비 세트", ['파프니르', '앱솔랩스', '도전자'], key=part)
            item['starforce'] = st.number_input(f"{part} - 스타포스", 0, 25, 15, key=part+"sf")
            item['mainstat_total'] = st.number_input(f"{part} - 주스탯 합", 0, 9999, 100, key=part+"main")
            item['power_total'] = st.number_input(f"{part} - 공격력/마력 합", 0, 999, 80, key=part+"pow")
            item['all_stat_total'] = st.number_input(f"{part} - 올스탯 합", 0, 99, 0, key=part+"all")
            item['potential_option_grade'] = st.selectbox(f"{part} - 잠재옵션 등급", ['레전드리', '유니크', '에픽', '레어'], key=part+"p1")
            item['additional_potential_option_grade'] = st.selectbox(f"{part} - 에디셔널 등급", ['레전드리', '유니크', '에픽', '레어'], key=part+"p2")
            for i in range(1, 4):
                item[f'potential_option_{i}_grade'] = st.selectbox(f"{part} - 잠재옵션 {i}", ['S', 'A', 'B', '기타'], key=f"{part}po{i}")
            for i in range(1, 4):
                item[f'additional_potential_option_{i}_grade'] = st.selectbox(f"{part} - 에디셔널 옵션 {i}", ['S', 'A', 'B', '기타'], key=f"{part}apo{i}")
            item['boss_dmg'] = st.slider(f"{part} - 보스 데미지 총합", 0.0, 100.0, 30.0, key=part+"bd")
            item['ignore_def'] = st.slider(f"{part} - 방무 총합", 0.0, 100.0, 20.0, key=part+"id")
            item['damage'] = st.slider(f"{part} - 데미지 총합", 0.0, 100.0, 25.0, key=part+"dg")
            item['item_count'] = 1  # 각 아이템 1개로 가정
            user_input[part] = item

    # ------------------------------
    # --- 유사 유저 추천 처리 ---
    if st.button("🔍 유사 유저 추천"):
        cont_vecs = [encode_item_to_cont_vector(user_input[part]) for part in top_slots]
        avg_vec = np.mean(cont_vecs, axis=0)

        x_cont = torch.tensor(avg_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 19]
        x_cat = torch.zeros((1, 1, len(embedding_info)), dtype=torch.long)
        mask = torch.tensor([[1.0]])

        with torch.no_grad():
            user_vec = model(x_cat, x_cont, mask).cpu().numpy().reshape(1, -1)
            similarities = cosine_similarity(user_vec, hl_vecs)[0]
            top5_idx = np.argsort(similarities)[::-1][:5]

        st.subheader("추천 유사 유저 Top5")
        for idx in top5_idx:
            st.write(f"{hl_nicks[idx]} (유사도: {similarities[idx]:.3f})")

# -------------------- 푸터 --------------------

st.markdown("""
    <style>
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 16px;
        margin-top: 20px;
        color: rgba(169, 169, 169, 0.7);
    }
    hr {
        width: 100%;
        border-top: 2px solid rgba(169, 169, 169, 0.7);
        margin: 10px 0;
    }
    </style>
    <div class="centered">
        <hr>
        <p>by 챌린저스@늡딴, 챌린저스@생갈치122호, 챌린저스@비딜러</p>
        <p>Data base on NEXON OPEN API</p>
        <p>This site is not an official site of NEXON and does not provide any warranty.</p>
    </div>
""", unsafe_allow_html=True)
