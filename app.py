import io
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="태린이아빠 Market Dashboard",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DEFAULT_FILE = BASE_DIR / "sample_dashboard_data.xlsx"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {
        background: #121a2d;
        border: 1px solid #27324c;
        padding: 16px;
        border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] {color:#aeb9cc;}
      div[data-testid="stMetricValue"] {font-weight:800;}
      .hero {
        border:1px solid #27324c;
        background: linear-gradient(90deg,#101a30,#141e33);
        border-radius:16px;
        padding:18px 20px;
        margin-bottom:16px;
      }
      .hero h1 {margin:0 0 5px 0;font-size:28px;}
      .muted {color:#97a4b8;font-size:13px;}
      .comment {
        border-left:4px solid #6ea8fe;
        background:#121a2d;
        padding:16px 18px;
        border-radius:10px;
        line-height:1.7;
      }
      .section-note {color:#94a3b8;font-size:12px;margin-top:-6px;margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_excel(source):
    """Read the workbook into a dict of dataframes."""
    return pd.read_excel(source, sheet_name=None)


def first_row(df, defaults=None):
    defaults = defaults or {}
    if df is None or df.empty:
        return defaults
    row = df.iloc[0].to_dict()
    return {**defaults, **row}


def get_sheet(book, names):
    lower = {str(k).lower(): v for k, v in book.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()].copy()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_default_book():
    return read_excel(DEFAULT_FILE)


with st.sidebar:
    st.title("태린이아빠")
    st.caption("Market Dashboard · Sample")
    st.divider()

    uploaded = st.file_uploader(
        "관리자 데이터 업로드",
        type=["xlsx"],
        help="샘플 엑셀과 같은 시트 구조의 파일을 올리면 대시보드가 즉시 변경됩니다.",
    )

    if uploaded is not None:
        try:
            book = read_excel(io.BytesIO(uploaded.getvalue()))
            st.success(f"업로드 완료: {uploaded.name}")
        except Exception as e:
            st.error(f"엑셀을 읽지 못했습니다: {e}")
            book = load_default_book()
    else:
        book = load_default_book()
        st.info("현재 샘플 데이터를 표시 중입니다.")

    st.divider()
    st.caption("실제 서비스에서는 이 업로드 영역을 관리자에게만 보이도록 숨길 수 있습니다.")

# Sheets
Dashboard = get_sheet(book, ["Dashboard", "대시보드"])
Macro = get_sheet(book, ["Macro", "매크로"])
Sector = get_sheet(book, ["Sector", "업종"])
ETF = get_sheet(book, ["ETF", "액티브ETF"])
Semi = get_sheet(book, ["Semi", "반도체"])
Comment = get_sheet(book, ["Comment", "코멘트"])

metrics = first_row(
    Dashboard,
    {"breadth": "-", "equity": "-", "realrate": "-", "concentration": "-"},
)
comments = first_row(Comment, {"daily": "", "weekly": ""})

st.markdown(
    """
    <div class="hero">
      <h1>📈 태린이아빠 Market Dashboard</h1>
      <div class="muted">매크로 · 시장 위험 · 주도 업종 · 액티브 ETF · 반도체 포지션을 한 화면에서 확인하는 샘플</div>
    </div>
    """,
    unsafe_allow_html=True,
)

home, macro_tab, risk_tab, sector_tab, etf_tab, semi_tab, comment_tab = st.tabs(
    ["오늘", "매크로", "시장 위험", "주도 업종", "액티브 ETF", "반도체", "코멘트"]
)

with home:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("120MA Breadth", str(metrics.get("breadth", "-")))
    c2.metric("주식 비중 신호", str(metrics.get("equity", "-")))
    c3.metric("정책금리 - Core PCE", str(metrics.get("realrate", "-")))
    c4.metric("업종 쏠림 지수", str(metrics.get("concentration", "-")))

    st.subheader("오늘의 시장 코멘트")
    st.markdown(f'<div class="comment">{comments.get("daily", "")}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("주도 업종 Top 5")
        if not Sector.empty:
            cols = [c for c in ["rank", "sector", "rs", "flow", "status"] if c in Sector.columns]
            st.dataframe(Sector[cols].head(5), use_container_width=True, hide_index=True)
    with right:
        st.subheader("업종 상대강도")
        if not Sector.empty and {"sector", "rs"}.issubset(Sector.columns):
            chart_df = Sector[["sector", "rs"]].dropna().set_index("sector")
            st.bar_chart(chart_df, horizontal=True)

with macro_tab:
    st.subheader("매크로 환경")
    st.markdown('<div class="section-note">실제 서비스에서는 Colab 결과를 이 시트에 자동 저장하도록 연결할 수 있습니다.</div>', unsafe_allow_html=True)
    if not Macro.empty:
        st.dataframe(Macro, use_container_width=True, hide_index=True)

        if {"indicator", "value"}.issubset(Macro.columns):
            numeric = Macro.copy()
            numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
            numeric = numeric.dropna(subset=["value"])
            if not numeric.empty:
                st.bar_chart(numeric.set_index("indicator")[["value"]])
    else:
        st.warning("Macro 시트가 없습니다.")

with risk_tab:
    st.subheader("시장 위험관리")
    c1, c2, c3 = st.columns(3)
    c1.metric("Breadth", str(metrics.get("breadth", "-")))
    c2.metric("위험자산 비중", str(metrics.get("equity", "-")))
    c3.metric("쏠림 지수", str(metrics.get("concentration", "-")))

    st.markdown("#### 샘플 운용 규칙")
    risk_rules = pd.DataFrame(
        [
            ["Breadth ≥ 60%", "주식 100%"],
            ["Breadth 40~60%", "주식 50%"],
            ["Breadth < 40%", "현금 100%"],
            ["월말 Breadth 70↓ + KOSPI 20일선 이탈", "조기 회피"],
        ],
        columns=["조건", "대응"],
    )
    st.dataframe(risk_rules, use_container_width=True, hide_index=True)

with sector_tab:
    st.subheader("주도 업종")
    if not Sector.empty:
        st.dataframe(Sector, use_container_width=True, hide_index=True)
        if {"sector", "rs"}.issubset(Sector.columns):
            st.bar_chart(Sector.set_index("sector")[["rs"]], horizontal=True)
    else:
        st.warning("Sector 시트가 없습니다.")

with etf_tab:
    st.subheader("액티브 ETF 최근 변화")
    st.caption("신규 편입 / 비중 확대 / 비중 축소를 한 번에 보는 영역")
    if not ETF.empty:
        direction = st.multiselect(
            "변화 유형 필터",
            options=sorted([x for x in ETF.get("direction", pd.Series(dtype=str)).dropna().astype(str).unique()]),
            default=[],
        )
        view = ETF.copy()
        if direction and "direction" in view.columns:
            view = view[view["direction"].astype(str).isin(direction)]
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.warning("ETF 시트가 없습니다.")

with semi_tab:
    st.subheader("반도체 포지션 맵")
    if not Semi.empty:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.dataframe(Semi, use_container_width=True, hide_index=True)
        with c2:
            if {"area", "score"}.issubset(Semi.columns):
                chart = Semi[["area", "score"]].copy()
                chart["score"] = pd.to_numeric(chart["score"], errors="coerce")
                chart = chart.dropna().set_index("area")
                st.bar_chart(chart, horizontal=True)
    else:
        st.warning("Semi 시트가 없습니다.")

with comment_tab:
    st.subheader("오늘의 코멘트")
    st.markdown(f'<div class="comment">{comments.get("daily", "")}</div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("주간 코멘트")
    st.markdown(f'<div class="comment">{comments.get("weekly", "")}</div>', unsafe_allow_html=True)

st.divider()
st.caption("샘플 버전 · 개별 종목 매수/매도 지시가 아닌 시장 데이터 시각화 예시")
