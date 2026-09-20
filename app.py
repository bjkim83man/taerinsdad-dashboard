import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, contextlib, traceback, time
import base64, pickle, requests, gzip, re

st.set_page_config(page_title="태린이아빠 Market Dashboard", page_icon="📊", layout="wide")


# ============================================================
# v10: GitHub 영구 저장
# - 앱 접속은 저장된 결과만 읽음
# - 관리자 업데이트 때만 계산 후 GitHub data repo에 저장
# - 별도 data repo 사용 권장: 저장할 때 Streamlit 앱 자체가 재배포되지 않음
# ============================================================
GH_OWNER = st.secrets.get("GITHUB_OWNER", "bjkim83man")
GH_DATA_REPO = st.secrets.get("GITHUB_DATA_REPO", "taerinsdad-dashboard-data")
GH_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GH_TOKEN = st.secrets.get("GITHUB_TOKEN", "")

def _gh_headers():
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    return h

def _figure_marker(fig):
    """Matplotlib figure를 이미지 대신 숫자 좌표 중심으로 저장한다."""
    try:
        axes = []
        positions = []
        figure_has_bars = any(len(ax.patches) > 0 for ax in fig.axes)
        for idx, ax in enumerate(fig.axes):
            pos = tuple(round(float(v), 5) for v in ax.get_position().bounds)
            sharex_with = None
            for j, p in enumerate(positions):
                if all(abs(a-b) < 1e-4 for a,b in zip(pos,p)):
                    sharex_with = j
                    break
            positions.append(pos)
            axd = {
                "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                "position": pos, "sharex_with": sharex_with,
                "ylim": tuple(float(v) for v in ax.get_ylim()),
                "lines": [], "bars": [],
            }
            for line in ax.get_lines():
                try:
                    x = np.asarray(line.get_xdata())
                    y = np.asarray(line.get_ydata())
                    # Gap30 figures contain hundreds/thousands of daily bars.
                    # The actual heat check uses 247 trading days, so keeping the most recent 400
                    # preserves the useful notebook view while greatly reducing persistent storage.
                    if figure_has_bars and len(x) > 400 and len(y) > 400:
                        x = x[-400:]
                        y = y[-400:]
                    is_hline = False
                    try:
                        xx = np.asarray(x, dtype=float)
                        is_hline = (len(xx) == 2 and np.allclose(xx, [0.0, 1.0]) and len(y) == 2 and float(y[0]) == float(y[1]) and line.get_transform() != ax.transData)
                    except Exception:
                        pass
                    axd["lines"].append({
                        "x": x, "y": y, "label": line.get_label(),
                        "linewidth": float(line.get_linewidth()),
                        "linestyle": line.get_linestyle(), "alpha": line.get_alpha(),
                        "color": line.get_color(), "is_hline": is_hline,
                    })
                except Exception:
                    pass
            _patches = list(ax.patches)
            if figure_has_bars and len(_patches) > 400:
                _patches = _patches[-400:]
            for patch in _patches:
                try:
                    if patch.__class__.__name__ != 'Rectangle':
                        continue
                    axd["bars"].append({
                        "x": float(patch.get_x()), "width": float(patch.get_width()),
                        "height": float(patch.get_height()), "bottom": float(patch.get_y()),
                        "alpha": patch.get_alpha(), "facecolor": tuple(float(v) for v in patch.get_facecolor()),
                    })
                except Exception:
                    pass
            axes.append(axd)
        if axes and any(a["lines"] or a["bars"] for a in axes):
            return {"__dashboard_chartdata__": True, "axes": axes}
    except Exception:
        pass
    # Very unusual figure fallback only.
    try:
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=72, bbox_inches="tight")
        return {"__dashboard_png__": True, "data": b.getvalue()}
    except Exception:
        return None

def _storage_safe(obj, depth=0):
    """계산 namespace에서 화면 재현에 필요한 직렬화 가능한 값만 남긴다."""
    if depth > 16:
        return None
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index, pd.Timestamp)):
        return obj
    if isinstance(obj, datetime):
        return obj
    # matplotlib figure -> PNG bytes (가볍고 재시작 후에도 안전)
    if hasattr(obj, "savefig") and obj.__class__.__name__ == "Figure":
        return _figure_marker(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if callable(v):
                continue
            sv = _storage_safe(v, depth + 1)
            if sv is not None:
                out[k] = sv
        return out
    if isinstance(obj, (list, tuple)):
        # tuple/list는 위치 자체가 결과 포맷일 수 있으므로 None도 보존한다.
        # (한국 Fear & Greed의 마지막 err=None이 사라지면 저장 후 tuple 길이가 달라지는 문제 방지)
        vals = [_storage_safe(v, depth + 1) for v in obj]
        return tuple(vals) if isinstance(obj, tuple) else vals
    # numpy arrays are useful in some results
    if isinstance(obj, np.ndarray):
        return obj
    return None

def _render_saved_fig(fig):
    if isinstance(fig, dict) and fig.get("__dashboard_chartdata__"):
        axes_specs = fig.get("axes", [])
        if not axes_specs:
            return
        rf = plt.figure(figsize=(12, 4.8))
        made = []
        for i, axd in enumerate(axes_specs):
            share = axd.get("sharex_with")
            if i == 0 or share is None or share >= len(made):
                rax = rf.add_subplot(111) if i == 0 else made[0].twinx()
            else:
                rax = made[share].twinx()
            made.append(rax)
            for line in axd.get("lines", []):
                try:
                    label = line.get("label", "")
                    if isinstance(label, str) and label.startswith("_"):
                        label = None
                    kw = {"label": label, "linewidth": line.get("linewidth", 1.8)}
                    if line.get("linestyle") is not None: kw["linestyle"] = line.get("linestyle")
                    if line.get("alpha") is not None: kw["alpha"] = line.get("alpha")
                    if line.get("color") is not None: kw["color"] = line.get("color")
                    if line.get("is_hline"):
                        yy = np.asarray(line.get("y"))
                        if len(yy): rax.axhline(float(yy[0]), **kw)
                    else:
                        rax.plot(line.get("x"), line.get("y"), **kw)
                except Exception:
                    pass
            bars = axd.get("bars", [])
            if bars:
                try:
                    xs=[b["x"] for b in bars]; hs=[b["height"] for b in bars]
                    ws=[b["width"] for b in bars]; bs=[b["bottom"] for b in bars]
                    fc=bars[0].get("facecolor"); alpha=bars[0].get("alpha")
                    rax.bar(xs, hs, width=ws, bottom=bs, color=fc, alpha=alpha)
                except Exception:
                    pass
            if axd.get("title"): rax.set_title(axd["title"])
            if axd.get("xlabel"): rax.set_xlabel(axd["xlabel"])
            if axd.get("ylabel"): rax.set_ylabel(axd["ylabel"])
            try:
                yl=axd.get("ylim")
                if yl and np.isfinite(yl).all(): rax.set_ylim(*yl)
            except Exception:
                pass
            rax.grid(alpha=0.2)
        try:
            handles=[]; labels=[]
            for a in made:
                h,l=a.get_legend_handles_labels(); handles += h; labels += l
            pairs=[(h,l) for h,l in zip(handles,labels) if l and not str(l).startswith('_')]
            if pairs:
                h,l=zip(*pairs); made[0].legend(h,l,loc="best")
        except Exception:
            pass
        rf.tight_layout()
        st.pyplot(rf, use_container_width=True)
        plt.close(rf)
    elif isinstance(fig, dict) and fig.get("__dashboard_png__"):
        st.image(fig["data"], use_container_width=True)
    else:
        try:
            st.pyplot(fig, use_container_width=True)
        except Exception:
            pass

def github_save_result(key, result, updated):
    if not GH_TOKEN:
        return False, "GITHUB_TOKEN이 Secrets에 없습니다."
    payload_obj = {
        "key": key,
        "updated": updated,
        "result": _storage_safe(result),
        "format_version": 1,
    }
    # 장기 운영용: pickle 자체를 gzip 압축해서 GitHub 저장 용량과 전송량을 줄임
    packed = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    raw = gzip.compress(packed, compresslevel=6)
    path = f"dashboard_data/{key}.pkl"
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_DATA_REPO}/contents/{path}"

    # 기존 파일이면 sha가 필요함
    g = requests.get(url, headers=_gh_headers(), params={"ref": GH_BRANCH}, timeout=30)
    sha = g.json().get("sha") if g.status_code == 200 else None

    body = {
        "message": f"Update dashboard result: {key} ({updated})",
        "content": base64.b64encode(raw).decode("ascii"),
        "branch": GH_BRANCH,
    }
    if sha:
        body["sha"] = sha

    r = requests.put(url, headers=_gh_headers(), json=body, timeout=90)
    if r.status_code in (200, 201):
        return True, f"GitHub 저장 완료 · 경량 저장 {len(raw)/1024/1024:.2f} MB"
    try:
        detail = r.json().get("message", r.text)
    except Exception:
        detail = r.text
    return False, f"GitHub 저장 실패 HTTP {r.status_code}: {detail}"

def github_load_result(key):
    """GitHub의 마지막 결과를 읽는다.
    1MB를 넘는 파일도 읽을 수 있도록 raw media type을 사용한다.
    """
    path = f"dashboard_data/{key}.pkl"
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_DATA_REPO}/contents/{path}"
    try:
        headers = _gh_headers().copy()
        headers["Accept"] = "application/vnd.github.raw+json"
        r = requests.get(url, headers=headers, params={"ref": GH_BRANCH}, timeout=60)
        if r.status_code != 200:
            return None, None
        raw = r.content
        # v11은 gzip 압축. 기존 v10/v10.2의 비압축 pickle도 그대로 호환
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        obj = pickle.loads(raw)
        return obj.get("result"), obj.get("updated")
    except Exception:
        return None, None

def persist_current_result(key):
    updated = st.session_state.get(key + "_updated")
    result = st.session_state.get(key + "_result")
    if result is None or not updated:
        return False, "저장할 결과 또는 업데이트 시간이 없습니다."
    ok, msg = github_save_result(key, result, updated)
    if ok:
        st.toast("마지막 결과를 GitHub에 저장했습니다.", icon="💾")
    else:
        st.error(msg)
    return ok, msg

def load_persisted_once():
    # 각 브라우저 세션 시작 시 GitHub의 마지막 저장 결과를 1회 읽음.
    # 방문자는 계산하지 않으며 저장 결과만 조회.
    if st.session_state.get("_persistent_loaded"):
        return
    for key in ["liq","fg","canary","trend","rotation","ai","us_sector","kr_sector","kr_fg","tw_revenue"]:
        result, updated = github_load_result(key)
        if result is not None:
            st.session_state[key + "_result"] = result
            st.session_state[key + "_updated"] = updated
    st.session_state["_persistent_loaded"] = True


st.markdown("""
<style>
:root{
  --td-bg:#07111f;
  --td-panel:#101a31;
  --td-panel-2:#0f1728;
  --td-card:#121c32;
  --td-border:#2a3958;
  --td-text:#f5f7fb;
  --td-muted:#b9c5d8;
  --td-sub:#8e9db4;
  --td-link:#7db8ff;
}
html, body, [class*="css"]{
  color:var(--td-text) !important;
}
body{
  background:var(--td-bg) !important;
}
.stApp, [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"]{
  background:var(--td-bg) !important;
  color:var(--td-text) !important;
}
[data-testid="stSidebar"]{
  background:var(--td-panel-2) !important;
  color:var(--td-text) !important;
}
[data-testid="stSidebar"] *{
  color:var(--td-text) !important;
}
.block-container{padding-top:1.15rem;max-width:1500px}
.hero{background:var(--td-panel);border-radius:18px;padding:22px;margin-bottom:14px}
.hero h1{color:var(--td-text);margin:0}
.hero p{color:var(--td-muted);margin:6px 0 0}
.card{background:var(--td-card);border:1px solid var(--td-border);border-radius:14px;padding:15px;min-height:108px}
.label{color:#aab8cd;font-size:.85rem}
.value{color:var(--td-text);font-size:1.55rem;font-weight:800;margin-top:5px}
.sub{color:var(--td-sub);font-size:.76rem;margin-top:5px}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;text-align:left;border-radius:9px}

h1,h2,h3,h4,h5,h6,p,span,label,div,li,small,strong{
  color:var(--td-text);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label{
  color:var(--td-text) !important;
}

a{color:var(--td-link) !important;}
hr{border-color:var(--td-border) !important;}

.stButton>button,
button[kind="secondary"],
button[kind="primary"]{
  background:#182645 !important;
  color:var(--td-text) !important;
  border:1px solid var(--td-border) !important;
}
.stButton>button:hover{
  background:#21325b !important;
  color:var(--td-text) !important;
  border:1px solid #355184 !important;
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div{
  background:#0c1527 !important;
  color:var(--td-text) !important;
  border-color:var(--td-border) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder{color:var(--td-sub) !important;}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"]{
  color:var(--td-text) !important;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span{
  color:var(--td-text);
}

div.stAlert{
  background:#13233e !important;
  color:var(--td-text) !important;
  border:1px solid var(--td-border) !important;
}

div.stAlert *{color:var(--td-text) !important;}

pre, code{
  color:#e8edf7 !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"]{
  color:var(--td-text) !important;
}

[data-testid="stExpander"] details{
  background:#0d1628 !important;
  border:1px solid var(--td-border) !important;
  border-radius:10px !important;
}


.fixed-result-output{
  background:#091322 !important;
  color:#e9f0fb !important;
  border:1px solid #263a5e !important;
  border-radius:10px !important;
  padding:14px 16px !important;
  margin:4px 0 !important;
  white-space:pre-wrap !important;
  overflow-x:auto !important;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace !important;
  font-size:.90rem !important;
  line-height:1.48 !important;
}
div[data-testid="stCode"],
div[data-testid="stCode"] pre,
div[data-testid="stCode"] code{
  background:#091322 !important;
  color:#e9f0fb !important;
}

[data-testid="stFileUploader"] section{
  background:#0d1628 !important;
  border:1px solid var(--td-border) !important;
  color:var(--td-text) !important;
}
</style>
""", unsafe_allow_html=True)

def card(a,b,c=""):
    st.markdown(f'<div class="card"><div class="label">{a}</div><div class="value">{b}</div><div class="sub">{c}</div></div>',unsafe_allow_html=True)

def run_source(source):
    buf=io.StringIO(); shown=[]; figs=[]
    def _display(obj):
        try:
            shown.append(obj.data if hasattr(obj,"data") else obj)
        except Exception:
            shown.append(str(obj))
    old_show=plt.show
    def _show(*args, **kwargs):
        try:
            figs.append(plt.gcf())
        except Exception:
            pass
    plt.show=_show
    ns={"display":_display}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(source,ns,ns)
        ok=True
    except Exception:
        ok=False
        buf.write("\n"+traceback.format_exc())
    finally:
        plt.show=old_show
    return ok,ns,buf.getvalue(),shown,figs

def render_run(result, max_chars=22000):
    ok,ns,txt,shown,figs=result
    for fig in figs:
        _render_saved_fig(fig)
    for obj in shown[-8:]:
        if isinstance(obj,pd.DataFrame):
            st.dataframe(obj.tail(80),use_container_width=True)
        elif isinstance(obj,pd.Series):
            st.dataframe(obj.tail(80).to_frame(),use_container_width=True)

    if txt.strip():
        import html as _html
        with st.expander("계산 상세 결과", expanded=(len(figs)==0)):
            _safe_txt = _html.escape(txt[-max_chars:])
            st.markdown(
                f"""
                <pre class="fixed-result-output">{_safe_txt}</pre>
                """,
                unsafe_allow_html=True,
            )
    return ok,ns,txt,shown,figs

LIQ_SRC='# ============================================================\n# 미국 매크로 · 유동성 현재상황 판정\n# Google Colab용\n#\n# [기존 유지]\n# Liquidity Score : 4점\n#   1) 10Y 실질금리\n#   2) 2Y 금리\n#   3) 10Y-2Y Curve\n#   4) Fed/TGA/RRP Liquidity Proxy\n#\n# [신규 추가]\n# Private Credit Score : 2점\n#   1) 미국 상업은행 대출\n#   2) M2 광의통화\n#\n# [별도 경고등]\n#   - 은행 지급준비금\n#   - RRP 고갈 여부\n#\n# [별도 확인]\n#   - 30Y\n#   - 30Y-10Y\n#\n# [신규 별도 경고등]\n# Yen Carry Trade Unwind Risk\n#   - USD/JPY (엔화 강세)\n#   - 미국 2Y 급락\n#   - 미·일 단기금리차 축소(보조)\n#   - VIX 급등\n#   - Nikkei 급락\n#   - Sahm Rule 상승\n#   - 2024-08-05 엔캐리 청산 국면과 유사도 비교\n# ============================================================\n\n\nimport pandas as pd\nimport numpy as np\nfrom pandas_datareader import data as web\n\nSTART_DATE = "2023-01-01"\n\n# ============================================================\n# 1. FRED 데이터\n# ============================================================\n\nseries = {\n    # 금리\n    "Real10Y": "DFII10",\n    "Treasury2Y": "DGS2",\n    "Treasury10Y": "DGS10",\n    "Treasury30Y": "DGS30",\n\n    # Fed / Treasury Liquidity\n    "FedAssets": "WALCL",\n    "TGA": "WTREGEN",\n    "RRP": "RRPONTSYD",\n\n    # 신규 : 민간 신용 / 광의통화\n    "BankLoans": "TOTLL",      # Loans & Leases, All Commercial Banks\n    "Reserves": "WRESBAL",     # Reserve Balances with Federal Reserve Banks\n    "M2": "M2SL",              # M2 Money Stock\n\n    # 엔캐리 트레이드 모니터\n    # DEXJPUS = 1달러당 엔화. 하락하면 엔화 강세\n    "USDJPY": "DEXJPUS",\n    "VIX": "VIXCLS",\n    "Nikkei225": "NIKKEI225",\n    "Sahm": "SAHMREALTIME",\n\n    # 일본 단기금리. 월간 데이터라 실시간 판정보다는 보조지표로 사용\n    "JapanCall": "IRSTCI01JPM156N",\n}\n\ndownloaded = {}\n\nfor name, code in series.items():\n    try:\n        s = web.DataReader(code, "fred", START_DATE)[code]\n        downloaded[name] = s\n        print(f"{name:15s} 다운로드 완료")\n    except Exception as e:\n        print(f"{name} 다운로드 실패:", e)\n\n# 중요:\n# 각 시리즈의 날짜 빈도가 다르므로 concat으로 union index 생성\ndf = pd.concat(downloaded, axis=1).sort_index()\n\n# 핵심 시리즈가 다운로드되지 않았으면 뒤에서 애매한 KeyError가 나지 않도록\n# 여기서 명확하게 중단합니다.\nrequired_base = [\n    "Real10Y", "Treasury2Y", "Treasury10Y", "Treasury30Y",\n    "FedAssets", "TGA", "RRP", "BankLoans", "Reserves", "M2"\n]\n\nrequired_carry = [\n    "USDJPY", "VIX", "Nikkei225", "Sahm"\n]\n\nmissing_base = [x for x in required_base if x not in df.columns]\nmissing_carry = [x for x in required_carry if x not in df.columns]\n\nif missing_base:\n    raise ValueError(\n        f"필수 매크로 시리즈 다운로드 실패: {missing_base}"\n    )\n\nif missing_carry:\n    raise ValueError(\n        f"엔캐리 모니터 필수 시리즈 다운로드 실패: {missing_carry}"\n    )\n\n# ============================================================\n# 2. 단위 변환\n# ============================================================\n\n# WALCL / TGA / WRESBAL = Million USD\ndf["FedAssets_T"] = df["FedAssets"] / 1_000_000\ndf["TGA_T"] = df["TGA"] / 1_000_000\ndf["Reserves_T"] = df["Reserves"] / 1_000_000\n\n# RRP / TOTLL / M2 = Billion USD\ndf["RRP_T"] = df["RRP"] / 1_000\ndf["BankLoans_T"] = df["BankLoans"] / 1_000\ndf["M2_T"] = df["M2"] / 1_000\n\n# ============================================================\n# 3. Yield Curve\n# ============================================================\n\ndf["Spread10_2"] = df["Treasury10Y"] - df["Treasury2Y"]\ndf["Spread30_10"] = df["Treasury30Y"] - df["Treasury10Y"]\n\n# ============================================================\n# 4. 기존 Fed/Treasury Liquidity Proxy\n#\n# 기존 Net Liquidity 공식 유지\n# 다만 RRP가 거의 0이 된 환경에서는\n# "전체 유동성"이 아니라 Fed/Treasury Liquidity Proxy로 해석\n# ============================================================\n\ndf["NetLiquidity"] = (\n    df["FedAssets_T"]\n    - df["TGA_T"]\n    - df["RRP_T"]\n)\n\n# ============================================================\n# 5. 주간 데이터\n# ============================================================\n\nweekly_cols = [\n    "Real10Y",\n    "Treasury2Y",\n    "Treasury10Y",\n    "Treasury30Y",\n    "Spread10_2",\n    "Spread30_10",\n    "FedAssets_T",\n    "TGA_T",\n    "RRP_T",\n    "NetLiquidity",\n    "BankLoans_T",\n    "Reserves_T"\n]\n\nweekly = (\n    df[weekly_cols]\n    .resample("W-FRI")\n    .last()\n    .ffill()\n)\n\nif len(weekly) < 53:\n    raise ValueError("YoY 및 최근 4주 변화를 계산하기 위한 데이터가 부족합니다.")\n\nnow = weekly.iloc[-1]\nprev4 = weekly.iloc[-5]\nprev52 = weekly.iloc[-53]\n\n# ============================================================\n# 6. 최근 4주 금리 변화\n# ============================================================\n\nreal_change = now["Real10Y"] - prev4["Real10Y"]\ny2_change = now["Treasury2Y"] - prev4["Treasury2Y"]\ny10_change = now["Treasury10Y"] - prev4["Treasury10Y"]\ny30_change = now["Treasury30Y"] - prev4["Treasury30Y"]\n\nspread10_2_change = (\n    now["Spread10_2"] - prev4["Spread10_2"]\n)\n\nspread30_10_change = (\n    now["Spread30_10"] - prev4["Spread30_10"]\n)\n\nliq_change = (\n    now["NetLiquidity"] - prev4["NetLiquidity"]\n)\n\n# ============================================================\n# 7. 은행 대출 / 지급준비금 변화\n# ============================================================\n\nbankloan_4w_pct = (\n    now["BankLoans_T"] / prev4["BankLoans_T"] - 1\n) * 100\n\nbankloan_yoy = (\n    now["BankLoans_T"] / prev52["BankLoans_T"] - 1\n) * 100\n\nreserve_4w_change = (\n    now["Reserves_T"] - prev4["Reserves_T"]\n)\n\nreserve_4w_pct = (\n    now["Reserves_T"] / prev4["Reserves_T"] - 1\n) * 100\n\n# ============================================================\n# 8. M2\n#\n# M2는 월간 데이터이므로\n# 최근 4주 변화가 아니라\n#\n# 1) 3개월 연율화\n# 2) YoY\n#\n# 로 판단\n# ============================================================\n\nm2_monthly = (\n    df["M2_T"]\n    .dropna()\n    .resample("ME")\n    .last()\n    .dropna()\n)\n\nif len(m2_monthly) < 13:\n    raise ValueError("M2 YoY 계산을 위한 데이터가 부족합니다.")\n\nm2_now = m2_monthly.iloc[-1]\nm2_prev3 = m2_monthly.iloc[-4]\nm2_prev12 = m2_monthly.iloc[-13]\n\nm2_3m_ann = (\n    (m2_now / m2_prev3) ** 4 - 1\n) * 100\n\nm2_yoy = (\n    m2_now / m2_prev12 - 1\n) * 100\n\nm2_date = m2_monthly.index[-1]\n\n# ============================================================\n# 9. 기존 Liquidity Score (4점)\n# ============================================================\n\nscore = 0\nsignals = []\n\n# ① 10Y 실질금리\nif real_change < 0:\n    score += 1\n    signals.append(\n        "10Y 실질금리 하락 → 위험자산 할인율 부담 완화 (+)"\n    )\nelse:\n    signals.append(\n        "10Y 실질금리 상승 → 위험자산 할인율 부담 (-)"\n    )\n\n# ② 2Y\nif y2_change < 0:\n    score += 1\n    signals.append(\n        "2Y 금리 하락 → Fed 완화 기대 강화 (+)"\n    )\nelse:\n    signals.append(\n        "2Y 금리 상승 → Fed 긴축/고금리 기대 (-)"\n    )\n\n# ③ 좋은 Steepening\nif spread10_2_change > 0 and y2_change < 0:\n    score += 1\n    signals.append(\n        "2Y 하락형 10Y-2Y Steepening → 금융여건 완화 (+)"\n    )\n\nelif spread10_2_change > 0 and y10_change > 0:\n    signals.append(\n        "10Y 상승형 10Y-2Y Steepening "\n        "→ 재정·Term Premium 부담 (-)"\n    )\n\nelse:\n    signals.append(\n        "10Y-2Y → 뚜렷한 완화형 Steepening 없음"\n    )\n\n# ④ 기존 Net Liquidity\nif liq_change > 0:\n    score += 1\n    signals.append(\n        "Fed/Treasury Liquidity Proxy 증가 (+)"\n    )\nelse:\n    signals.append(\n        "Fed/Treasury Liquidity Proxy 감소 (-)"\n    )\n\n# ============================================================\n# 10. 기존 4점 판정\n# ============================================================\n\nif score == 4:\n    regime = "★★★★★ 매우 우호적"\nelif score == 3:\n    regime = "★★★★☆ 우호적"\nelif score == 2:\n    regime = "★★★☆☆ 혼조"\nelif score == 1:\n    regime = "★★☆☆☆ 비우호적"\nelse:\n    regime = "★☆☆☆☆ 긴축적"\n\n# ============================================================\n# 11. 신규 Private Credit Score (2점)\n#\n# 기존 4점 점수에는 섞지 않음.\n# "Fed 유동성"과 "민간 신용"을 분리해서 보기 위함.\n# ============================================================\n\ncredit_score = 0\ncredit_signals = []\n\n# 은행 대출\nif bankloan_4w_pct > 0 and bankloan_yoy > 0:\n    credit_score += 1\n    credit_signals.append(\n        "은행대출 증가 → 민간 신용창출 확대 (+)"\n    )\n\nelif bankloan_4w_pct <= 0 and bankloan_yoy > 0:\n    credit_signals.append(\n        "은행대출 YoY는 증가하지만 최근 4주는 둔화 → 신용확장 모멘텀 점검"\n    )\n\nelse:\n    credit_signals.append(\n        "은행대출 감소/부진 → 민간 신용창출 약화 (-)"\n    )\n\n# M2\nif m2_3m_ann > 0 and m2_yoy > 0:\n    credit_score += 1\n    credit_signals.append(\n        "M2 증가 → 광의통화 확장 (+)"\n    )\n\nelif m2_yoy > 0:\n    credit_signals.append(\n        "M2 YoY는 증가하지만 최근 3개월 모멘텀 둔화"\n    )\n\nelse:\n    credit_signals.append(\n        "M2 YoY 감소 → 광의통화 위축 (-)"\n    )\n\nif credit_score == 2:\n    credit_regime = "★★★★★ 민간 신용 강한 확장"\nelif credit_score == 1:\n    credit_regime = "★★★☆☆ 민간 신용 혼조"\nelse:\n    credit_regime = "★☆☆☆☆ 민간 신용 위축"\n\n# ============================================================\n# 12. 지급준비금 경고등\n#\n# 준비금은 단순히 \'증가=무조건 호재\'가 아니므로\n# 점수에는 넣지 않고 Stress Indicator로 사용\n# ============================================================\n\nif reserve_4w_pct <= -5:\n    reserve_signal = (\n        "⚠️ 경고 — 은행 지급준비금이 최근 4주 급감 "\n        "→ 단기 자금시장 스트레스 점검 필요"\n    )\n\nelif reserve_4w_pct < 0:\n    reserve_signal = (\n        "주의 — 지급준비금 감소 중 "\n        "→ 아직 즉각적 위험 신호는 아니지만 추세 점검"\n    )\n\nelse:\n    reserve_signal = (\n        "양호 — 지급준비금 안정/증가 "\n        "→ 은행시스템 유동성 부담 제한적"\n    )\n\n# ============================================================\n# 13. RRP 상태\n#\n# RRP가 거의 고갈되면 과거처럼\n# RRP 감소가 QT를 흡수하는 효과가 사라짐\n# ============================================================\n\nif now["RRP_T"] < 0.05:\n    rrp_signal = (\n        "⚠️ RRP Buffer 거의 소진 "\n        "→ 향후 유동성 판단에서 은행대출·M2·준비금 중요도 상승"\n    )\nelif now["RRP_T"] < 0.20:\n    rrp_signal = (\n        "RRP Buffer 낮음 "\n        "→ 과거 대비 QT 완충 능력 크게 감소"\n    )\nelse:\n    rrp_signal = (\n        "RRP Buffer 존재 "\n        "→ 일부 유동성 완충 여력 존재"\n    )\n\n# ============================================================\n# 14. Fed + Private Credit 종합판정\n# ============================================================\n\nif score >= 3 and credit_score == 2:\n    total_liquidity_view = (\n        "★★★★★ 매우 우호적 — "\n        "금융여건과 민간 신용이 동시에 확장"\n    )\n\nelif score <= 1 and credit_score == 0:\n    total_liquidity_view = (\n        "★☆☆☆☆ 매우 비우호적 — "\n        "Fed/Treasury 환경과 민간 신용이 동시에 약화"\n    )\n\nelif score <= 1 and credit_score == 2:\n    total_liquidity_view = (\n        "★★★☆☆ 체제 전환형 혼조 — "\n        "Fed/Treasury 지표는 약하지만 은행 신용이 이를 보완"\n    )\n\nelif score >= 3 and credit_score == 0:\n    total_liquidity_view = (\n        "★★★☆☆ 정책완화 선행 — "\n        "시장금리/Fed 환경은 좋아지지만 민간 신용 확산은 아직 미확인"\n    )\n\nelif score == 2 and credit_score == 2:\n    total_liquidity_view = (\n        "★★★★☆ 우호적 — "\n        "기존 유동성은 혼조지만 민간 신용 확장이 강함"\n    )\n\nelse:\n    total_liquidity_view = (\n        "★★★☆☆ 혼조 — "\n        "Fed/Treasury와 민간 신용 신호가 완전히 일치하지 않음"\n    )\n\n# ============================================================\n# 15. 기존 종합 매크로 해석\n# ============================================================\n\nif real_change < 0 and y2_change < 0 and liq_change > 0:\n    macro_comment = (\n        "금리환경과 Fed/Treasury 유동성 Proxy가 동시에 개선되고 있습니다. "\n        "기존 유동성 기준으로 Risk-On 환경에 가깝습니다."\n    )\n\nelif real_change < 0 and y2_change < 0 and liq_change <= 0:\n    macro_comment = (\n        "Fed 완화 기대와 실질금리 하락은 나타나지만 "\n        "Fed/Treasury Liquidity Proxy는 아직 증가하지 않습니다."\n    )\n\nelif real_change > 0 and liq_change < 0:\n    macro_comment = (\n        "실질금리가 오르고 Fed/Treasury Liquidity Proxy도 감소해 "\n        "기존 프레임상 위험자산에 비우호적입니다."\n    )\n\nelif liq_change > 0 and real_change > 0:\n    macro_comment = (\n        "유동성 Proxy는 증가하지만 실질금리 상승이 이를 일부 상쇄합니다."\n    )\n\nelse:\n    macro_comment = (\n        "금리와 Fed/Treasury 유동성 신호가 엇갈립니다."\n    )\n\n# ============================================================\n# 16. 30Y / 30Y-10Y 판정\n# ============================================================\n\nif y30_change < 0 and spread30_10_change < 0:\n    long_rate_signal = (\n        "★★★★★ 매우 긍정적 — 30Y 하락 + 30Y-10Y 축소 "\n        "→ 초장기 인플레·재정·Term Premium 부담 완화"\n    )\n\nelif y30_change < 0 and y10_change < 0:\n    long_rate_signal = (\n        "★★★★☆ 긍정적 — 10Y·30Y 동반 하락 "\n        "→ 장기 금융여건 완화"\n    )\n\nelif y10_change < 0 and y30_change >= 0:\n    long_rate_signal = (\n        "★★☆☆☆ 주의 — 10Y는 하락하지만 30Y는 버팀/상승 "\n        "→ Fed 완화 기대와 달리 초장기 재정 부담 지속"\n    )\n\nelif y30_change > 0 and spread30_10_change > 0:\n    long_rate_signal = (\n        "★☆☆☆☆ 부정적 — 30Y 상승 + 30Y-10Y 확대 "\n        "→ 장기 인플레·재정·Term Premium 부담 증가"\n    )\n\nelse:\n    long_rate_signal = (\n        "★★★☆☆ 중립 — 초장기금리 방향 추가 확인 필요"\n    )\n\n# ============================================================\n# 17. 금리 구조 종합\n# ============================================================\n\nif (\n    real_change < 0\n    and y2_change < 0\n    and y30_change < 0\n    and spread30_10_change < 0\n):\n    rate_structure_comment = (\n        "단기금리·실질금리·초장기금리가 함께 완화됩니다. "\n        "Fed 완화 기대와 장기 위험 프리미엄 완화가 동시에 나타납니다."\n    )\n\nelif y2_change < 0 and y10_change < 0 and y30_change >= 0:\n    rate_structure_comment = (\n        "2Y와 10Y는 하락하지만 30Y가 따라오지 않습니다. "\n        "통화정책 완화 기대는 있으나 초장기 재정/국채공급 부담이 남아 있습니다."\n    )\n\nelif y2_change < 0 and y30_change > 0 and spread30_10_change > 0:\n    rate_structure_comment = (\n        "Fed 완화 기대와 초장기 재정 부담이 동시에 나타나는 "\n        "분화된 금리시장입니다."\n    )\n\nelif y2_change > 0 and real_change > 0 and y30_change > 0:\n    rate_structure_comment = (\n        "단기·실질·초장기금리가 모두 상승합니다. "\n        "금융여건의 긴축 압력이 강합니다."\n    )\n\nelse:\n    rate_structure_comment = (\n        "금리곡선 내부 신호가 혼재되어 있습니다."\n    )\n\n# ============================================================\n# 18. 자산별 해석\n# ============================================================\n\n# Gold\nif real_change < 0 and y30_change < 0:\n    gold_view = (\n        "매우 우호적 — 실질금리 하락 + 초장기금리 하락"\n    )\n\nelif real_change < 0:\n    gold_view = (\n        "우호적 — 실질금리 하락"\n    )\n\nelse:\n    gold_view = (\n        "부담 — 실질금리 상승"\n    )\n\n# Bitcoin\nif (\n    (liq_change > 0 or credit_score == 2)\n    and real_change < 0\n    and y2_change < 0\n):\n    btc_view = (\n        "매우 우호적 — 금리환경 개선 + 유동성/민간신용 확장"\n    )\n\nelif real_change < 0 and y2_change < 0:\n    btc_view = (\n        "우호적 — 금리환경 개선, 유동성 확산 추가 확인"\n    )\n\nelif liq_change < 0 and credit_score == 0:\n    btc_view = (\n        "주의 — Fed/Treasury 유동성과 민간신용 모두 약화"\n    )\n\nelse:\n    btc_view = (\n        "중립 — 뚜렷한 방향 확인 필요"\n    )\n\n# 중소형주\nif (\n    y2_change < 0\n    and credit_score == 2\n    and y30_change < 0\n):\n    smallcap_view = (\n        "매우 우호적 — 단기금리 하락 + 민간신용 확장 + 장기금리 안정"\n    )\n\nelif y2_change < 0 and credit_score == 2:\n    smallcap_view = (\n        "우호적 — 단기금리 하락 + 은행신용/M2 확장"\n    )\n\nelif y2_change < 0 and real_change < 0:\n    smallcap_view = (\n        "개선 가능 — 금융여건 완화 기대가 먼저 나타남"\n    )\n\nelse:\n    smallcap_view = (\n        "아직 확인 필요"\n    )\n\n# ============================================================\n# 19. 엔캐리 트레이드 청산 위험\n#\n# 기준 사건 : 2024-08-05 전후 글로벌 엔캐리 청산\n#\n# 핵심 논리\n# 1) 엔화 급등          : USDJPY 급락\n# 2) 미국 단기금리 하락 : 미국 2Y 하락 → 미일 금리차 축소 압력\n# 3) VIX 급등          : 위험회피 / 레버리지 청산 확인\n# 4) Nikkei 급락       : 일본 위험자산 스트레스 확인\n# 5) Sahm Rule 상승    : 미국 경기둔화 우려 확인\n# 6) 일본 단기금리     : 월간 보조지표\n#\n# 주의:\n# VIX 급등이나 Nikkei 급락만으로는 엔캐리 청산이라고 판정하지 않습니다.\n# 반드시 \'엔화 강세\'가 먼저 확인되어야 합니다.\n# ============================================================\n\n\ndef _clean_series(name):\n    return df[name].dropna().sort_index()\n\n\ndef _latest_before(name, date=None):\n    s = _clean_series(name)\n\n    if date is not None:\n        s = s[s.index <= pd.Timestamp(date)]\n\n    if len(s) == 0:\n        return np.nan\n\n    return float(s.iloc[-1])\n\n\ndef _value_days_before(name, date=None, days=28):\n    s = _clean_series(name)\n\n    if len(s) == 0:\n        return np.nan\n\n    if date is None:\n        end_date = s.index[-1]\n    else:\n        end_date = pd.Timestamp(date)\n\n    target_date = end_date - pd.Timedelta(days=days)\n    old = s[s.index <= target_date]\n\n    if len(old) == 0:\n        return np.nan\n\n    return float(old.iloc[-1])\n\n\ndef _safe_pct_change(now_value, old_value):\n    if pd.isna(now_value) or pd.isna(old_value) or old_value == 0:\n        return np.nan\n    return (now_value / old_value - 1) * 100\n\n\ndef _safe_diff(now_value, old_value):\n    if pd.isna(now_value) or pd.isna(old_value):\n        return np.nan\n    return now_value - old_value\n\n\ndef carry_snapshot(date=None):\n    # USD/JPY\n    usd_now = _latest_before("USDJPY", date)\n    usd_5d = _value_days_before("USDJPY", date, 7)\n    usd_4w = _value_days_before("USDJPY", date, 28)\n\n    # 미국 2Y\n    us2_now = _latest_before("Treasury2Y", date)\n    us2_4w = _value_days_before("Treasury2Y", date, 28)\n\n    # VIX\n    vix_now = _latest_before("VIX", date)\n    vix_5d = _value_days_before("VIX", date, 7)\n\n    # Nikkei\n    nikkei_now = _latest_before("Nikkei225", date)\n    nikkei_5d = _value_days_before("Nikkei225", date, 7)\n\n    # Sahm Rule\n    sahm_now = _latest_before("Sahm", date)\n\n    # 일본 단기금리: 월간 데이터라 있으면 보조적으로만 사용\n    if "JapanCall" in df.columns:\n        japan_now = _latest_before("JapanCall", date)\n        japan_4w = _value_days_before("JapanCall", date, 28)\n    else:\n        japan_now = np.nan\n        japan_4w = np.nan\n\n    jpy_5d_pct = _safe_pct_change(usd_now, usd_5d)\n    jpy_4w_pct = _safe_pct_change(usd_now, usd_4w)\n    us2_4w_change = _safe_diff(us2_now, us2_4w)\n    vix_5d_pct = _safe_pct_change(vix_now, vix_5d)\n    nikkei_5d_pct = _safe_pct_change(nikkei_now, nikkei_5d)\n\n    # 단순 미일 단기금리 차이(미국 2Y - 일본 단기금리)\n    # 일본 단기금리는 월간이라 보조지표입니다.\n    if (\n        not pd.isna(us2_now)\n        and not pd.isna(japan_now)\n    ):\n        carry_gap_now = us2_now - japan_now\n    else:\n        carry_gap_now = np.nan\n\n    if (\n        not pd.isna(us2_4w)\n        and not pd.isna(japan_4w)\n    ):\n        carry_gap_4w = us2_4w - japan_4w\n    else:\n        carry_gap_4w = np.nan\n\n    carry_gap_change = _safe_diff(\n        carry_gap_now,\n        carry_gap_4w\n    )\n\n    return {\n        "USDJPY": usd_now,\n        "JPY_5D": jpy_5d_pct,\n        "JPY_4W": jpy_4w_pct,\n        "US2Y": us2_now,\n        "US2Y_4W": us2_4w_change,\n        "VIX": vix_now,\n        "VIX_5D": vix_5d_pct,\n        "Nikkei": nikkei_now,\n        "Nikkei_5D": nikkei_5d_pct,\n        "Sahm": sahm_now,\n        "JapanRate": japan_now,\n        "CarryGap": carry_gap_now,\n        "CarryGapChange": carry_gap_change,\n    }\n\n\ncarry_now = carry_snapshot()\ncarry_2024 = carry_snapshot("2024-08-05")\n\n\n# ------------------------------------------------------------\n# 19-1. 현재 엔캐리 청산 Trigger\n# ------------------------------------------------------------\n\n# ① 엔화 급등\n# USDJPY가 내려가면 엔화 강세입니다.\nyen_trigger = (\n    (\n        not pd.isna(carry_now["JPY_5D"])\n        and carry_now["JPY_5D"] <= -2.5\n    )\n    or\n    (\n        not pd.isna(carry_now["JPY_4W"])\n        and carry_now["JPY_4W"] <= -5.0\n    )\n)\n\n# ② 미국 2Y 급락\n# Fed 완화 기대 / 미국 성장 우려로 미일 금리차가 빠르게 좁아지는 경우\nrate_trigger = (\n    not pd.isna(carry_now["US2Y_4W"])\n    and carry_now["US2Y_4W"] <= -0.30\n)\n\n# ③ 미일 단기금리차 축소\n# 일본 데이터가 월간이므로 반드시 보조 신호로만 사용\ngap_trigger = (\n    not pd.isna(carry_now["CarryGapChange"])\n    and carry_now["CarryGapChange"] <= -0.25\n)\n\n# ④ VIX 급등\nvol_trigger = (\n    (\n        not pd.isna(carry_now["VIX"])\n        and carry_now["VIX"] >= 25\n    )\n    or\n    (\n        not pd.isna(carry_now["VIX_5D"])\n        and carry_now["VIX_5D"] >= 50\n    )\n)\n\n# ⑤ Nikkei 급락\nnikkei_trigger = (\n    not pd.isna(carry_now["Nikkei_5D"])\n    and carry_now["Nikkei_5D"] <= -7\n)\n\n# ⑥ 미국 경기둔화 경고\ngrowth_trigger = (\n    not pd.isna(carry_now["Sahm"])\n    and carry_now["Sahm"] >= 0.40\n)\n\ncarry_score = int(sum([\n    yen_trigger,\n    rate_trigger,\n    gap_trigger,\n    vol_trigger,\n    nikkei_trigger,\n    growth_trigger,\n]))\n\n\n# ------------------------------------------------------------\n# 19-2. 엔캐리 위험 판정\n# ------------------------------------------------------------\n\n# 핵심 원칙:\n# 엔화가 강해지지 않으면 VIX/Nikkei가 흔들려도\n# \'엔캐리 청산\'이라고 부르지 않습니다.\nif not yen_trigger:\n    carry_regime = (\n        "🟢 낮음 — 엔화 급등 신호 없음"\n    )\n\nelif (\n    yen_trigger\n    and (rate_trigger or gap_trigger)\n    and vol_trigger\n    and nikkei_trigger\n):\n    carry_regime = (\n        "🔴 매우 높음 — 엔화 급등 + 금리차 축소 + "\n        "VIX/일본증시 스트레스 동시 발생"\n    )\n\nelif (\n    yen_trigger\n    and (rate_trigger or gap_trigger)\n    and (vol_trigger or nikkei_trigger)\n):\n    carry_regime = (\n        "🔴 높음 — 엔화 강세 + 금리차 축소 + "\n        "위험자산 스트레스 확인"\n    )\n\nelif (\n    yen_trigger\n    and (rate_trigger or gap_trigger)\n):\n    carry_regime = (\n        "🟠 높아지는 중 — 엔화 강세와 "\n        "Carry 금리차 축소가 동시 발생"\n    )\n\nelse:\n    carry_regime = (\n        "🟡 관찰 — 엔화 강세는 있으나 "\n        "금리차 축소/시장 스트레스 확인 필요"\n    )\n\n\n# ------------------------------------------------------------\n# 19-3. 2024-08-05 엔캐리 청산 국면과 현재 유사도\n# ------------------------------------------------------------\n\n# 2024-08-05 당시와 현재를 절대 레벨 그대로 비교하면\n# 시대별 금리/환율 레벨 차이 때문에 왜곡될 수 있으므로,\n# \'스트레스 방향의 변화폭\'을 비교합니다.\n\n\ndef _positive_stress(value, reverse=False, baseline=0.0):\n    if pd.isna(value):\n        return 0.0\n\n    if reverse:\n        return max(0.0, -(value - baseline))\n\n    return max(0.0, value - baseline)\n\n\ndef _stress_ratio(current, reference):\n    # 기준 사건에서 해당 신호가 거의 없었다면 유사도 계산에서 0 처리\n    if pd.isna(reference) or reference <= 0:\n        return 0.0\n\n    if pd.isna(current):\n        return 0.0\n\n    # 2024 사건보다 더 심해도 유사도는 최대 100%로 제한\n    return min(max(current / reference, 0.0), 1.0)\n\n\ncurrent_stress = {\n    # USDJPY 하락폭: 마이너스가 위험 방향이므로 부호 반전\n    "JPY": _positive_stress(\n        carry_now["JPY_4W"],\n        reverse=True\n    ),\n\n    # 미국 2Y 하락폭\n    "US2Y": _positive_stress(\n        carry_now["US2Y_4W"],\n        reverse=True\n    ),\n\n    # VIX는 평시 15를 기준으로 초과분 사용\n    "VIX": _positive_stress(\n        carry_now["VIX"],\n        baseline=15.0\n    ),\n\n    # Nikkei 5일 하락폭\n    "Nikkei": _positive_stress(\n        carry_now["Nikkei_5D"],\n        reverse=True\n    ),\n\n    # Sahm Rule\n    "Sahm": _positive_stress(\n        carry_now["Sahm"]\n    ),\n}\n\n\nevent_stress = {\n    "JPY": _positive_stress(\n        carry_2024["JPY_4W"],\n        reverse=True\n    ),\n    "US2Y": _positive_stress(\n        carry_2024["US2Y_4W"],\n        reverse=True\n    ),\n    "VIX": _positive_stress(\n        carry_2024["VIX"],\n        baseline=15.0\n    ),\n    "Nikkei": _positive_stress(\n        carry_2024["Nikkei_5D"],\n        reverse=True\n    ),\n    "Sahm": _positive_stress(\n        carry_2024["Sahm"]\n    ),\n}\n\n\n# 엔화 + 미국 단기금리를 가장 중요하게 둠\ncarry_weights = {\n    "JPY": 0.30,\n    "US2Y": 0.25,\n    "VIX": 0.20,\n    "Nikkei": 0.15,\n    "Sahm": 0.10,\n}\n\ncarry_similarity = 0.0\n\nfor key, weight in carry_weights.items():\n    carry_similarity += (\n        _stress_ratio(\n            current_stress[key],\n            event_stress[key]\n        )\n        * weight\n    )\n\ncarry_similarity *= 100\n\n\n# ------------------------------------------------------------\n# 19-4. 유사도 레이블\n# ------------------------------------------------------------\n\nif carry_similarity >= 75:\n    carry_similarity_view = (\n        "🔴 2024년 8월 청산국면과 매우 유사"\n    )\nelif carry_similarity >= 50:\n    carry_similarity_view = (\n        "🟠 2024년 8월 청산국면과 유사성 상승"\n    )\nelif carry_similarity >= 25:\n    carry_similarity_view = (\n        "🟡 일부 유사 신호 존재"\n    )\nelse:\n    carry_similarity_view = (\n        "🟢 2024년 8월과 유사성 낮음"\n    )\n\n\n# ------------------------------------------------------------\n# 19-5. 현재 상황 설명\n# ------------------------------------------------------------\n\ncarry_signals = []\n\ncarry_signals.append(\n    f"엔화 5일 {carry_now[\'JPY_5D\']:+.2f}%, "\n    f"4주 {carry_now[\'JPY_4W\']:+.2f}% "\n    "(USDJPY 기준, 마이너스=엔화 강세)"\n)\n\nif yen_trigger:\n    carry_signals.append(\n        "엔화가 빠르게 강해져 Carry 포지션 청산 압력이 커지는 방향"\n    )\nelse:\n    carry_signals.append(\n        "엔화 급등 조건은 아직 충족하지 않음"\n    )\n\nif rate_trigger:\n    carry_signals.append(\n        "미국 2Y가 빠르게 하락 → 미일 금리차 축소 압력 확대"\n    )\nelse:\n    carry_signals.append(\n        "미국 2Y 급락 조건 없음 → 2024년 8월과 핵심 차이"\n    )\n\nif vol_trigger:\n    carry_signals.append(\n        "VIX 스트레스 확인 → 레버리지 축소/위험회피 동반"\n    )\nelse:\n    carry_signals.append(\n        "VIX 급등 신호 없음"\n    )\n\nif nikkei_trigger:\n    carry_signals.append(\n        "Nikkei 급락 확인 → 일본 위험자산 청산 압력 동반"\n    )\nelse:\n    carry_signals.append(\n        "Nikkei 급락 조건 없음"\n    )\n\nif growth_trigger:\n    carry_signals.append(\n        "Sahm Rule 상승 → 미국 경기둔화 우려가 Carry 청산을 자극할 수 있음"\n    )\nelse:\n    carry_signals.append(\n        "Sahm Rule 기준 경기침체 스트레스 낮음"\n    )\n\n\n# ============================================================\n# 20. 출력\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("현재 미국 매크로 / 유동성 상황")\nprint("=" * 90)\n\nprint(f"기준일                    : {weekly.index[-1].date()}")\n\nprint()\nprint("[금리]")\nprint(f"10Y 실질금리              : {now[\'Real10Y\']:.2f}%")\nprint(f"2Y 국채금리               : {now[\'Treasury2Y\']:.2f}%")\nprint(f"10Y 국채금리              : {now[\'Treasury10Y\']:.2f}%")\nprint(f"30Y 국채금리              : {now[\'Treasury30Y\']:.2f}%")\nprint(f"10Y-2Y                    : {now[\'Spread10_2\']:.2f}%p")\nprint(f"30Y-10Y                   : {now[\'Spread30_10\']:.2f}%p")\n\nprint()\nprint("[Fed / Treasury]")\nprint(f"Fed Assets                : ${now[\'FedAssets_T\']:.2f}T")\nprint(f"TGA                       : ${now[\'TGA_T\']:.2f}T")\nprint(f"RRP                       : ${now[\'RRP_T\']:.3f}T")\nprint(f"Liquidity Proxy           : ${now[\'NetLiquidity\']:.2f}T")\n\nprint()\nprint("[은행 / 민간 신용]")\nprint(f"은행대출                  : ${now[\'BankLoans_T\']:.2f}T")\nprint(f"은행 지급준비금           : ${now[\'Reserves_T\']:.2f}T")\nprint(f"M2 ({m2_date.date()})     : ${m2_now:.2f}T")\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("최근 4주 변화")\nprint("=" * 90)\n\nprint(f"10Y 실질금리              : {real_change:+.2f}%p")\nprint(f"2Y                        : {y2_change:+.2f}%p")\nprint(f"10Y                       : {y10_change:+.2f}%p")\nprint(f"30Y                       : {y30_change:+.2f}%p")\nprint(f"10Y-2Y                    : {spread10_2_change:+.2f}%p")\nprint(f"30Y-10Y                   : {spread30_10_change:+.2f}%p")\nprint(f"Liquidity Proxy           : {liq_change:+.3f}T")\n\nprint()\nprint(f"은행대출 4주              : {bankloan_4w_pct:+.2f}%")\nprint(f"은행대출 YoY              : {bankloan_yoy:+.2f}%")\nprint(f"지급준비금 4주            : {reserve_4w_change:+.3f}T")\nprint(f"지급준비금 4주 %          : {reserve_4w_pct:+.2f}%")\n\nprint()\nprint(f"M2 3개월 연율화           : {m2_3m_ann:+.2f}%")\nprint(f"M2 YoY                    : {m2_yoy:+.2f}%")\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("기존 Liquidity Score")\nprint("=" * 90)\n\nfor signal in signals:\n    print("•", signal)\n\nprint()\nprint(f"Liquidity Score           : {score} / 4")\nprint(f"판정                      : {regime}")\nprint()\nprint(macro_comment)\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("신규 Private Credit / 광의통화")\nprint("=" * 90)\n\nfor signal in credit_signals:\n    print("•", signal)\n\nprint()\nprint(f"Private Credit Score      : {credit_score} / 2")\nprint(f"판정                      : {credit_regime}")\n\nprint()\nprint("•", reserve_signal)\nprint("•", rrp_signal)\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("Fed + 은행신용 종합 유동성 판정")\nprint("=" * 90)\n\nprint(total_liquidity_view)\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("초장기금리 / 재정·Term Premium")\nprint("=" * 90)\n\nprint(long_rate_signal)\nprint()\nprint(rate_structure_comment)\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("자산별 현재 해석")\nprint("=" * 90)\n\nprint(f"Gold                      : {gold_view}")\nprint(f"Bitcoin                   : {btc_view}")\nprint(f"중소형주                  : {smallcap_view}")\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("엔캐리 트레이드 청산 위험")\nprint("=" * 90)\n\nprint(f"USDJPY                    : {carry_now[\'USDJPY\']:.2f}")\nprint(f"엔화 5일 변화             : {carry_now[\'JPY_5D\']:+.2f}%")\nprint(f"엔화 4주 변화             : {carry_now[\'JPY_4W\']:+.2f}%")\n\nprint()\nprint(f"미국 2Y                   : {carry_now[\'US2Y\']:.2f}%")\nprint(f"미국 2Y 4주 변화          : {carry_now[\'US2Y_4W\']:+.2f}%p")\n\nif not pd.isna(carry_now["JapanRate"]):\n    print(f"일본 단기금리(보조)       : {carry_now[\'JapanRate\']:.2f}%")\n\nif not pd.isna(carry_now["CarryGap"]):\n    print(f"미·일 단기금리차(보조)    : {carry_now[\'CarryGap\']:+.2f}%p")\n\nif not pd.isna(carry_now["CarryGapChange"]):\n    print(f"금리차 4주 변화(보조)     : {carry_now[\'CarryGapChange\']:+.2f}%p")\n\nprint()\nprint(f"VIX                       : {carry_now[\'VIX\']:.2f}")\nprint(f"VIX 5일 변화              : {carry_now[\'VIX_5D\']:+.1f}%")\nprint(f"Nikkei 5일 변화           : {carry_now[\'Nikkei_5D\']:+.2f}%")\nprint(f"Sahm Rule                 : {carry_now[\'Sahm\']:.2f}")\n\nprint()\nfor signal in carry_signals:\n    print("•", signal)\n\nprint()\nprint(f"엔캐리 위험점수           : {carry_score} / 6")\nprint(f"현재 판정                 : {carry_regime}")\nprint(\n    f"2024-08-05 유사도         : "\n    f"{carry_similarity:.0f}%"\n)\nprint(f"유사도 해석               : {carry_similarity_view}")\n\nprint()\nprint("[2024-08-05 기준 사건]")\nprint(f"엔화 4주 변화             : {carry_2024[\'JPY_4W\']:+.2f}%")\nprint(f"미국 2Y 4주 변화          : {carry_2024[\'US2Y_4W\']:+.2f}%p")\nprint(f"VIX                       : {carry_2024[\'VIX\']:.2f}")\nprint(f"Nikkei 5일 변화           : {carry_2024[\'Nikkei_5D\']:+.2f}%")\nprint(f"Sahm Rule                 : {carry_2024[\'Sahm\']:.2f}")\n\n# ============================================================\n\nprint()\nprint("=" * 90)\nprint("최종 한눈에 보기")\nprint("=" * 90)\n\nprint(f"기존 유동성               : {regime}")\nprint(f"민간 신용                 : {credit_regime}")\nprint(f"종합 유동성               : {total_liquidity_view}")\nprint(f"지급준비금                : {reserve_signal}")\nprint(f"RRP                       : {rrp_signal}")\nprint(f"초장기금리                : {long_rate_signal}")\nprint(f"엔캐리 청산               : {carry_regime}")\nprint(f"2024 엔캐리 유사도        : {carry_similarity:.0f}% — {carry_similarity_view}")\n\nprint()\nprint(f"Gold                      : {gold_view}")\nprint(f"Bitcoin                   : {btc_view}")\nprint(f"중소형주                  : {smallcap_view}")'
FG_SRC='# ============================================================\n# Full Code\n# US Market Fear & Greed Index + EMA20 + Oscillator + Index\n# + QQQ/SPY SuperMA & Gap\n# + Elder Impulse\n# + DeMark TD Setup\n#\n# Fear & Greed 구성:\n# 1. 125일 모멘텀\n# 2. RSI 10일\n# 3. 10년-5년 금리 스프레드\n# 4. VIX\n# 5. Risk Appetite (HYG / IEF)\n#\n# 그래프 표시 기간:\n# DISPLAY_MONTHS 값을 변경\n# 6 = 최근 6개월\n# 12 = 최근 1년\n# 18 = 최근 1년 6개월\n# ============================================================\n\nimport yfinance as yf\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.preprocessing import MinMaxScaler\nimport numpy as np\n\n\n# ============================================================\n# 0) 기본 설정\n# ============================================================\n\nplt.rcParams[\'axes.unicode_minus\'] = False\n\nDISPLAY_MONTHS = 6\n\n# 색상\nCOL_FG_INDEX = \'#1565C0\'\nCOL_FG_EMA20 = \'#F57C00\'\nCOL_OSC = \'#6A5ACD\'\nCOL_PRICE = \'#555555\'\nGRID_ALPHA = 0.30\n\n\n# ============================================================\n# 1) 데이터 다운로드\n# ============================================================\n\ntickers = [\n    \'^GSPC\',\n    \'^IXIC\',\n    \'^VIX\',\n    \'^TNX\',\n    \'^FVX\',\n    \'HYG\',\n    \'IEF\',\n    \'QQQ\',\n    \'SPY\'\n]\n\nraw = yf.download(\n    tickers,\n    start=\'2024-01-01\',\n    auto_adjust=False,\n    progress=False\n)\n\n# yfinance 결과가 MultiIndex일 때 Close만 추출\nif isinstance(raw.columns, pd.MultiIndex):\n    data = raw[\'Close\'].copy()\nelse:\n    data = raw.copy()\n\ndata = data.rename(\n    columns={\n        \'^GSPC\': \'S&P500\',\n        \'^IXIC\': \'NASDAQ\',\n        \'^VIX\': \'VIX\',\n        \'^TNX\': \'10Y\',\n        \'^FVX\': \'5Y\'\n    }\n)\n\nrequired_cols = [\n    \'S&P500\',\n    \'NASDAQ\',\n    \'VIX\',\n    \'10Y\',\n    \'5Y\',\n    \'HYG\',\n    \'IEF\',\n    \'QQQ\',\n    \'SPY\'\n]\n\nmissing_cols = [\n    col for col in required_cols\n    if col not in data.columns\n]\n\nif missing_cols:\n    raise KeyError(\n        f\'다운로드되지 않은 컬럼이 있습니다: {missing_cols}\'\n    )\n\ndata = data[required_cols].copy()\n\nfor col in required_cols:\n    data[col] = pd.to_numeric(\n        data[col],\n        errors=\'coerce\'\n    )\n\ndata = data.replace(\n    [np.inf, -np.inf],\n    np.nan\n)\n\ndata = data.dropna().copy()\n\n# Risk Appetite\ndata[\'Risk_Appetite\'] = (\n    data[\'HYG\']\n    / data[\'IEF\'].replace(0, np.nan)\n)\n\n\n# ============================================================\n# 2) RSI 함수\n# ============================================================\n\ndef calculate_rsi(\n    series,\n    window=10\n):\n    delta = series.diff()\n\n    gain = (\n        delta.where(delta > 0, 0)\n        .rolling(window=window)\n        .mean()\n    )\n\n    loss = (\n        -delta.where(delta < 0, 0)\n        .rolling(window=window)\n        .mean()\n    )\n\n    rs = (\n        gain\n        / loss.replace(0, np.nan)\n    )\n\n    rsi = (\n        100\n        - (100 / (1 + rs))\n    )\n\n    rsi.loc[\n        (loss == 0)\n        & (gain > 0)\n    ] = 100\n\n    return rsi\n\n\n# ============================================================\n# 3) MACD 함수\n# ============================================================\n\ndef calculate_macd_components(\n    series\n):\n    ema12 = (\n        series\n        .ewm(\n            span=12,\n            adjust=False\n        )\n        .mean()\n    )\n\n    ema26 = (\n        series\n        .ewm(\n            span=26,\n            adjust=False\n        )\n        .mean()\n    )\n\n    macd = ema12 - ema26\n\n    signal = (\n        macd\n        .ewm(\n            span=9,\n            adjust=False\n        )\n        .mean()\n    )\n\n    oscillator = macd - signal\n\n    return macd, signal, oscillator\n\n\ndef calculate_macd(\n    df,\n    column\n):\n    _, _, oscillator = (\n        calculate_macd_components(\n            df[column]\n        )\n    )\n\n    return oscillator\n\n\n# ============================================================\n# 4) Fear & Greed 계산\n# ============================================================\n\ndef calculate_fear_greed(\n    df,\n    index_col,\n    label\n):\n    df = df.copy()\n\n    # 125일 모멘텀\n    df[f\'{label}_125MA\'] = (\n        df[index_col]\n        .rolling(window=125)\n        .mean()\n    )\n\n    df[f\'{label}_Momentum\'] = (\n        (\n            df[index_col]\n            - df[f\'{label}_125MA\']\n        )\n        / df[f\'{label}_125MA\']\n        * 100\n    )\n\n    # RSI\n    df[f\'{label}_RSI\'] = (\n        calculate_rsi(\n            df[index_col],\n            window=10\n        )\n    )\n\n    # 금리 스프레드\n    df[f\'{label}_BondSpread\'] = (\n        df[\'10Y\']\n        - df[\'5Y\']\n    )\n\n    # VIX\n    df[f\'{label}_VIX\'] = (\n        df[\'VIX\']\n    )\n\n    # Risk Appetite\n    df[f\'{label}_RiskAppetite\'] = (\n        df[\'Risk_Appetite\']\n    )\n\n    feature_cols = [\n        f\'{label}_Momentum\',\n        f\'{label}_RSI\',\n        f\'{label}_BondSpread\',\n        f\'{label}_VIX\',\n        f\'{label}_RiskAppetite\'\n    ]\n\n    valid_index = (\n        df\n        .dropna(\n            subset=feature_cols\n        )\n        .index\n    )\n\n    # 결과 컬럼 초기화\n    df[f\'{label}_FGI\'] = np.nan\n    df[f\'{label}_FG_Score\'] = np.nan\n    df[f\'{label}_FG_EMA20\'] = np.nan\n    df[f\'{label}_FG_EMA20_Score\'] = np.nan\n    df[f\'{label}_MACD\'] = np.nan\n    df[f\'{label}_Signal\'] = np.nan\n    df[f\'{label}_Oscillator\'] = np.nan\n\n    if len(valid_index) == 0:\n        return df\n\n    # 유효 구간만 Min-Max 정규화\n    scaler = MinMaxScaler()\n\n    scaled = scaler.fit_transform(\n        df.loc[\n            valid_index,\n            feature_cols\n        ]\n    )\n\n    scaled_cols = [\n        f\'{label}_Momentum_Scaled\',\n        f\'{label}_RSI_Scaled\',\n        f\'{label}_BondSpread_Scaled\',\n        f\'{label}_VIX_Scaled\',\n        f\'{label}_RiskAppetite_Scaled\'\n    ]\n\n    scaled_df = pd.DataFrame(\n        scaled,\n        index=valid_index,\n        columns=scaled_cols\n    )\n\n    for col in scaled_cols:\n        df.loc[\n            valid_index,\n            col\n        ] = scaled_df[col]\n\n    # Fear & Greed Index: 0~1\n    df.loc[\n        valid_index,\n        f\'{label}_FGI\'\n    ] = (\n        df.loc[\n            valid_index,\n            f\'{label}_Momentum_Scaled\'\n        ] * 0.20\n        +\n        df.loc[\n            valid_index,\n            f\'{label}_RiskAppetite_Scaled\'\n        ] * 0.20\n        +\n        (\n            1\n            - df.loc[\n                valid_index,\n                f\'{label}_VIX_Scaled\'\n            ]\n        ) * 0.20\n        +\n        df.loc[\n            valid_index,\n            f\'{label}_BondSpread_Scaled\'\n        ] * 0.20\n        +\n        df.loc[\n            valid_index,\n            f\'{label}_RSI_Scaled\'\n        ] * 0.20\n    )\n\n    # 화면 표시용 0~100\n    df[f\'{label}_FG_Score\'] = (\n        df[f\'{label}_FGI\']\n        * 100\n    )\n\n    # Fear & Greed EMA20\n    df[f\'{label}_FG_EMA20\'] = (\n        df[f\'{label}_FGI\']\n        .ewm(\n            span=20,\n            adjust=False\n        )\n        .mean()\n    )\n\n    df[f\'{label}_FG_EMA20_Score\'] = (\n        df[f\'{label}_FG_EMA20\']\n        * 100\n    )\n\n    # Fear & Greed MACD\n    macd, signal, oscillator = (\n        calculate_macd_components(\n            df[f\'{label}_FGI\']\n        )\n    )\n\n    df[f\'{label}_MACD\'] = macd\n    df[f\'{label}_Signal\'] = signal\n    df[f\'{label}_Oscillator\'] = oscillator\n\n    return df\n\n\n# S&P500 및 NASDAQ Fear & Greed\ndata = calculate_fear_greed(\n    data,\n    index_col=\'S&P500\',\n    label=\'SPX\'\n)\n\ndata = calculate_fear_greed(\n    data,\n    index_col=\'NASDAQ\',\n    label=\'NDX\'\n)\n\n\n# ============================================================\n# 5) QQQ / SPY SuperMA + 괴리율\n# ============================================================\n\ndef add_super_ma_gap(\n    df,\n    price_col,\n    label,\n    windows=(20, 60, 120, 200)\n):\n    df = df.copy()\n\n    for window in windows:\n        df[f\'{label}_MA{window}\'] = (\n            df[price_col]\n            .rolling(window)\n            .mean()\n        )\n\n    ma_cols = [\n        f\'{label}_MA{window}\'\n        for window in windows\n    ]\n\n    df[f\'{label}_SuperMA\'] = (\n        df[ma_cols]\n        .mean(axis=1)\n    )\n\n    df[f\'{label}_GapPct\'] = (\n        (\n            df[price_col]\n            - df[f\'{label}_SuperMA\']\n        )\n        / df[f\'{label}_SuperMA\']\n        * 100\n    )\n\n    return df\n\n\ndata = add_super_ma_gap(\n    data,\n    price_col=\'QQQ\',\n    label=\'QQQ\'\n)\n\ndata = add_super_ma_gap(\n    data,\n    price_col=\'SPY\',\n    label=\'SPY\'\n)\n\n\n# ============================================================\n# 6) Elder Impulse\n# ============================================================\n\ndef add_impulse_components(\n    df,\n    price_col,\n    label\n):\n    df = df.copy()\n\n    df[f\'{label}_EMA13\'] = (\n        df[price_col]\n        .ewm(\n            span=13,\n            adjust=False\n        )\n        .mean()\n    )\n\n    df[f\'{label}_MACD_Hist\'] = (\n        calculate_macd(\n            df,\n            price_col\n        )\n    )\n\n    return df\n\n\ndef get_impulse_colors(\n    df,\n    ema_col,\n    macd_col\n):\n    colors = []\n\n    ema = df[ema_col]\n    macd = df[macd_col]\n\n    for i in range(len(df)):\n        if i == 0:\n            colors.append(\'gray\')\n            continue\n\n        ema_up = (\n            ema.iloc[i]\n            > ema.iloc[i - 1]\n        )\n\n        ema_down = (\n            ema.iloc[i]\n            < ema.iloc[i - 1]\n        )\n\n        macd_up = (\n            macd.iloc[i]\n            > macd.iloc[i - 1]\n        )\n\n        macd_down = (\n            macd.iloc[i]\n            < macd.iloc[i - 1]\n        )\n\n        if ema_up and macd_up:\n            colors.append(\'green\')\n\n        elif ema_down and macd_down:\n            colors.append(\'red\')\n\n        else:\n            colors.append(\'blue\')\n\n    return colors\n\n\ndata = add_impulse_components(\n    data,\n    price_col=\'QQQ\',\n    label=\'QQQ\'\n)\n\ndata = add_impulse_components(\n    data,\n    price_col=\'SPY\',\n    label=\'SPY\'\n)\n\n\n# ============================================================\n# 7) DeMark TD Setup\n# ============================================================\n\ndef add_td_setup_counts(\n    df,\n    price_col,\n    label\n):\n    df = df.copy()\n\n    n = len(df)\n\n    sell = np.zeros(n)\n    buy = np.zeros(n)\n\n    prices = df[price_col].values\n\n    for i in range(n):\n\n        # Sell Setup\n        if (\n            i >= 4\n            and np.isfinite(prices[i])\n            and np.isfinite(prices[i - 4])\n            and prices[i] > prices[i - 4]\n        ):\n            sell[i] = sell[i - 1] + 1\n        else:\n            sell[i] = 0\n\n        # Buy Setup\n        if (\n            i >= 2\n            and np.isfinite(prices[i])\n            and np.isfinite(prices[i - 2])\n            and prices[i] < prices[i - 2]\n        ):\n            buy[i] = buy[i - 1] + 1\n        else:\n            buy[i] = 0\n\n    df[f\'{label}_TD_SellSetup\'] = sell\n    df[f\'{label}_TD_BuySetup\'] = buy\n\n    return df\n\n\ndata = add_td_setup_counts(\n    data,\n    price_col=\'SPY\',\n    label=\'SPY\'\n)\n\ndata = add_td_setup_counts(\n    data,\n    price_col=\'QQQ\',\n    label=\'QQQ\'\n)\n\n\n# ============================================================\n# 8) 표시 기간 필터\n# ============================================================\n\ncutoff_date = (\n    data.index.max()\n    - pd.DateOffset(\n        months=DISPLAY_MONTHS\n    )\n)\n\nrecent = data[\n    data.index >= cutoff_date\n].copy()\n\n\n# ============================================================\n# 9) Fear & Greed 통합 그래프 함수\n# ============================================================\n\ndef plot_fear_greed_combined(\n    df,\n    label,\n    price_col,\n    market_name\n):\n    required = [\n        f\'{label}_FG_Score\',\n        f\'{label}_FG_EMA20_Score\',\n        f\'{label}_Oscillator\',\n        price_col\n    ]\n\n    chart_df = (\n        df\n        .dropna(\n            subset=required\n        )\n        .copy()\n    )\n\n    if chart_df.empty:\n        print(\n            f\'{market_name}: \'\n            f\'그래프로 표시할 데이터가 없습니다.\'\n        )\n        return\n\n    fig, ax_fg = plt.subplots(\n        figsize=(16, 8)\n    )\n\n    fig.subplots_adjust(\n        right=0.82\n    )\n\n    # ----------------------------------\n    # 왼쪽 축: F&G Index + EMA20\n    # ----------------------------------\n\n    line_fg, = ax_fg.plot(\n        chart_df.index,\n        chart_df[\n            f\'{label}_FG_Score\'\n        ],\n        color=COL_FG_INDEX,\n        linewidth=2.5,\n        label=\'Fear & Greed Index\'\n    )\n\n    line_ema, = ax_fg.plot(\n        chart_df.index,\n        chart_df[\n            f\'{label}_FG_EMA20_Score\'\n        ],\n        color=COL_FG_EMA20,\n        linewidth=2.1,\n        label=\'F&G EMA20\'\n    )\n\n    ax_fg.set_ylim(\n        0,\n        100\n    )\n\n    ax_fg.set_ylabel(\n        \'Fear & Greed Index\',\n        color=COL_FG_INDEX\n    )\n\n    ax_fg.tick_params(\n        axis=\'y\',\n        labelcolor=COL_FG_INDEX\n    )\n\n    for level in [20, 50, 80]:\n        ax_fg.axhline(\n            level,\n            color=\'gray\',\n            linestyle=\'--\',\n            linewidth=0.8,\n            alpha=0.5,\n            label=\'_nolegend_\'\n        )\n\n    ax_fg.grid(\n        True,\n        alpha=GRID_ALPHA\n    )\n\n    ax_fg.set_xlabel(\'Date\')\n\n\n    # ----------------------------------\n    # 오른쪽 축 1: 시장 지수\n    # ----------------------------------\n\n    ax_price = ax_fg.twinx()\n\n    line_price, = ax_price.plot(\n        chart_df.index,\n        chart_df[price_col],\n        color=COL_PRICE,\n        linewidth=1.5,\n        alpha=0.50,\n        label=market_name\n    )\n\n    ax_price.set_ylabel(\n        market_name,\n        color=COL_PRICE\n    )\n\n    ax_price.tick_params(\n        axis=\'y\',\n        labelcolor=COL_PRICE\n    )\n\n\n    # ----------------------------------\n    # 오른쪽 축 2: Oscillator\n    # ----------------------------------\n\n    ax_osc = ax_fg.twinx()\n\n    ax_osc.spines[\n        \'right\'\n    ].set_position(\n        (\'axes\', 1.11)\n    )\n\n    line_osc, = ax_osc.plot(\n        chart_df.index,\n        chart_df[\n            f\'{label}_Oscillator\'\n        ],\n        color=COL_OSC,\n        linewidth=2.0,\n        label=\'Fear & Greed Oscillator\'\n    )\n\n    ax_osc.axhline(\n        0,\n        color=COL_OSC,\n        linewidth=0.9,\n        alpha=0.7,\n        label=\'_nolegend_\'\n    )\n\n    ax_osc.set_ylabel(\n        \'Oscillator\',\n        color=COL_OSC\n    )\n\n    ax_osc.tick_params(\n        axis=\'y\',\n        labelcolor=COL_OSC\n    )\n\n\n    # ----------------------------------\n    # 최신값\n    # ----------------------------------\n\n    latest = chart_df.iloc[-1]\n\n    latest_fg = (\n        latest[\n            f\'{label}_FG_Score\'\n        ]\n    )\n\n    latest_ema = (\n        latest[\n            f\'{label}_FG_EMA20_Score\'\n        ]\n    )\n\n    latest_osc = (\n        latest[\n            f\'{label}_Oscillator\'\n        ]\n    )\n\n\n    # ----------------------------------\n    # 범례\n    # ----------------------------------\n\n    lines = [\n        line_fg,\n        line_ema,\n        line_osc,\n        line_price\n    ]\n\n    labels = [\n        line.get_label()\n        for line in lines\n    ]\n\n    ax_fg.legend(\n        lines,\n        labels,\n        loc=\'upper left\',\n        ncol=2,\n        frameon=True\n    )\n\n\n    # ----------------------------------\n    # 제목\n    # ----------------------------------\n\n    ax_fg.set_title(\n        f\'{market_name} – \'\n        f\'Fear & Greed Index + EMA20 + Oscillator\\n\'\n        f\'Fear & Greed {latest_fg:.1f} | \'\n        f\'EMA20 {latest_ema:.1f} | \'\n        f\'Oscillator {latest_osc:.4f}\',\n        fontsize=14\n    )\n\n    plt.show()\n\n\n# S&P500 통합 그래프\nplot_fear_greed_combined(\n    recent,\n    label=\'SPX\',\n    price_col=\'S&P500\',\n    market_name=\'S&P500\'\n)\n\n# NASDAQ 통합 그래프\nplot_fear_greed_combined(\n    recent,\n    label=\'NDX\',\n    price_col=\'NASDAQ\',\n    market_name=\'NASDAQ\'\n)\n\n\n# ============================================================\n# 10) QQQ SuperMA + Gap%\n# ============================================================\n\ndef plot_super_ma_gap(\n    df,\n    price_col,\n    label,\n    title\n):\n    chart_df = df.dropna(\n        subset=[\n            price_col,\n            f\'{label}_SuperMA\',\n            f\'{label}_GapPct\'\n        ]\n    ).copy()\n\n    if chart_df.empty:\n        print(\n            f\'{title}: 표시할 데이터가 없습니다.\'\n        )\n        return\n\n    fig, ax1 = plt.subplots(\n        figsize=(14, 7)\n    )\n\n    ax1.plot(\n        chart_df.index,\n        chart_df[price_col],\n        label=f\'{label} Price\',\n        color=\'black\'\n    )\n\n    ax1.plot(\n        chart_df.index,\n        chart_df[\n            f\'{label}_SuperMA\'\n        ],\n        label=(\n            f\'{label} SuperMA \'\n            f\'(20/60/120/200 avg)\'\n        ),\n        linestyle=\'--\',\n        color=\'orange\'\n    )\n\n    ax1.set_ylabel(\n        f\'{label} Price\'\n    )\n\n    ax1.grid(True)\n\n    ax2 = ax1.twinx()\n\n    ax2.plot(\n        chart_df.index,\n        chart_df[\n            f\'{label}_GapPct\'\n        ],\n        label=(\n            f\'{label} Gap% \'\n            f\'vs SuperMA\'\n        ),\n        color=\'purple\',\n        alpha=0.7\n    )\n\n    ax2.set_ylabel(\n        \'Gap %\',\n        color=\'purple\'\n    )\n\n    lines1, labels1 = (\n        ax1.get_legend_handles_labels()\n    )\n\n    lines2, labels2 = (\n        ax2.get_legend_handles_labels()\n    )\n\n    ax1.legend(\n        lines1 + lines2,\n        labels1 + labels2,\n        loc=\'upper left\'\n    )\n\n    plt.title(title)\n\n    plt.show()\n\n\nplot_super_ma_gap(\n    recent,\n    price_col=\'QQQ\',\n    label=\'QQQ\',\n    title=(\n        f\'QQQ vs SuperMA and Gap% \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\nplot_super_ma_gap(\n    recent,\n    price_col=\'SPY\',\n    label=\'SPY\',\n    title=(\n        f\'SPY vs SuperMA and Gap% \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\n\n# ============================================================\n# 11) Elder Impulse 그래프\n# ============================================================\n\ndef plot_impulse(\n    df,\n    price_col,\n    ema_col,\n    macd_col,\n    title\n):\n    chart_df = df.dropna(\n        subset=[\n            price_col,\n            ema_col,\n            macd_col\n        ]\n    ).copy()\n\n    if chart_df.empty:\n        print(\n            f\'{title}: 표시할 데이터가 없습니다.\'\n        )\n        return\n\n    impulse_colors = (\n        get_impulse_colors(\n            chart_df,\n            ema_col,\n            macd_col\n        )\n    )\n\n    fig, ax = plt.subplots(\n        figsize=(14, 7)\n    )\n\n    ax.plot(\n        chart_df.index,\n        chart_df[price_col],\n        color=\'black\',\n        label=f\'{price_col} Price\'\n    )\n\n    ax.scatter(\n        chart_df.index,\n        chart_df[price_col],\n        c=impulse_colors,\n        s=20,\n        label=\'Impulse\'\n    )\n\n    ax.set_ylabel(\n        f\'{price_col} Price\'\n    )\n\n    ax.grid(True)\n\n    ax.legend(\n        loc=\'upper left\'\n    )\n\n    plt.title(title)\n\n    plt.show()\n\n\nplot_impulse(\n    recent,\n    price_col=\'QQQ\',\n    ema_col=\'QQQ_EMA13\',\n    macd_col=\'QQQ_MACD_Hist\',\n    title=(\n        f\'QQQ Price with Elder Impulse System \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\nplot_impulse(\n    recent,\n    price_col=\'SPY\',\n    ema_col=\'SPY_EMA13\',\n    macd_col=\'SPY_MACD_Hist\',\n    title=(\n        f\'SPY Price with Elder Impulse System \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\n\n# ============================================================\n# 12) DeMark TD Setup 그래프\n# ============================================================\n\ndef plot_td_setup(\n    df,\n    price_col,\n    label,\n    title\n):\n    sell_col = (\n        f\'{label}_TD_SellSetup\'\n    )\n\n    buy_col = (\n        f\'{label}_TD_BuySetup\'\n    )\n\n    chart_df = df.dropna(\n        subset=[\n            price_col,\n            sell_col,\n            buy_col\n        ]\n    ).copy()\n\n    if chart_df.empty:\n        print(\n            f\'{title}: 표시할 데이터가 없습니다.\'\n        )\n        return\n\n    fig, ax1 = plt.subplots(\n        figsize=(14, 7)\n    )\n\n    line_price, = ax1.plot(\n        chart_df.index,\n        chart_df[price_col],\n        color=\'black\',\n        label=f\'{label} Price\'\n    )\n\n    ax1.set_ylabel(\n        f\'{label} Price\',\n        color=\'black\'\n    )\n\n    ax1.tick_params(\n        axis=\'y\',\n        labelcolor=\'black\'\n    )\n\n    ax1.grid(True)\n\n\n    ax2 = ax1.twinx()\n\n    line_sell, = ax2.plot(\n        chart_df.index,\n        chart_df[sell_col],\n        color=\'red\',\n        label=\'TD Sell Setup\'\n    )\n\n    line_buy, = ax2.plot(\n        chart_df.index,\n        chart_df[buy_col],\n        color=\'blue\',\n        label=\'TD Buy Setup\'\n    )\n\n    ax2.set_ylabel(\n        \'TD Setup Count\',\n        color=\'gray\'\n    )\n\n    ax2.tick_params(\n        axis=\'y\',\n        labelcolor=\'gray\'\n    )\n\n    max_td = (\n        chart_df[\n            [\n                sell_col,\n                buy_col\n            ]\n        ]\n        .max()\n        .max()\n    )\n\n    ax2.set_ylim(\n        0,\n        max_td + 2\n    )\n\n    lines = [\n        line_price,\n        line_sell,\n        line_buy\n    ]\n\n    ax1.legend(\n        lines,\n        [\n            line.get_label()\n            for line in lines\n        ],\n        loc=\'upper left\'\n    )\n\n    plt.title(title)\n\n    plt.show()\n\n\nplot_td_setup(\n    recent,\n    price_col=\'SPY\',\n    label=\'SPY\',\n    title=(\n        f\'SPY Price with DeMark TD Setup Counts \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\nplot_td_setup(\n    recent,\n    price_col=\'QQQ\',\n    label=\'QQQ\',\n    title=(\n        f\'QQQ Price with DeMark TD Setup Counts \'\n        f\'(Last {DISPLAY_MONTHS} Months)\'\n    )\n)\n\n\n# ============================================================\n# 13) 최신 Fear & Greed 수치 출력\n# ============================================================\n\ndef print_latest_fear_greed(\n    df,\n    label,\n    market_name,\n    price_col\n):\n    required = [\n        f\'{label}_FG_Score\',\n        f\'{label}_FG_EMA20_Score\',\n        f\'{label}_MACD\',\n        f\'{label}_Signal\',\n        f\'{label}_Oscillator\',\n        price_col\n    ]\n\n    valid = (\n        df\n        .dropna(\n            subset=required\n        )\n        .copy()\n    )\n\n    if valid.empty:\n        print(\n            f\'{market_name}: \'\n            f\'계산 가능한 데이터가 없습니다.\'\n        )\n        return\n\n    latest = valid.iloc[-1]\n\n    print(\'=\' * 60)\n    print(f\'{market_name} 최신 Fear & Greed\')\n    print(\'=\' * 60)\n\n    print(\n        f\'기준일: \'\n        f\'{latest.name.strftime("%Y-%m-%d")}\'\n    )\n\n    print(\n        f\'{market_name}: \'\n        f\'{latest[price_col]:,.2f}\'\n    )\n\n    print(\n        f\'Fear & Greed Index: \'\n        f\'{latest[f"{label}_FG_Score"]:.2f}\'\n    )\n\n    print(\n        f\'F&G EMA20: \'\n        f\'{latest[f"{label}_FG_EMA20_Score"]:.2f}\'\n    )\n\n    print(\n        f\'MACD: \'\n        f\'{latest[f"{label}_MACD"]:.6f}\'\n    )\n\n    print(\n        f\'Signal: \'\n        f\'{latest[f"{label}_Signal"]:.6f}\'\n    )\n\n    print(\n        f\'Oscillator: \'\n        f\'{latest[f"{label}_Oscillator"]:.6f}\'\n    )\n\n\nprint_latest_fear_greed(\n    data,\n    label=\'SPX\',\n    market_name=\'S&P500\',\n    price_col=\'S&P500\'\n)\n\nprint_latest_fear_greed(\n    data,\n    label=\'NDX\',\n    market_name=\'NASDAQ\',\n    price_col=\'NASDAQ\'\n)'
AI_SRC='# ============================================================\n# AI Tech Breadth & Peak-out Dashboard\n# ============================================================\n\n\nimport yfinance as yf\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# ============================================================\n# 1. Universe\n# ============================================================\n\nstocks = {\n    # US\n    "NVDA": "NVDA",\n    "AVGO": "AVGO",\n    "AMD": "AMD",\n    "TSM": "TSM",\n    "ASML": "ASML",\n    "MU": "MU",\n    "ARM": "ARM",\n    "QCOM": "QCOM",\n    "MRVL": "MRVL",\n    "LRCX": "LRCX",\n    "AMAT": "AMAT",\n    "KLAC": "KLAC",\n    "CDNS": "CDNS",\n    "SNPS": "SNPS",\n    "ANET": "ANET",\n    "TXN": "TXN",\n    "ON": "ON",\n    "DELL": "DELL",\n\n    # Added US\n    "Sandisk": "SNDK",\n    "Intel": "INTC",\n    "Amkor": "AMKR",\n    "WesternDigital": "WDC",\n    "Seagate": "STX",\n    "Lumentum": "LITE",\n    "Corning": "GLW",\n    "AsteraLabs": "ALAB",\n\n    # Korea\n    "Samsung": "005930.KS",\n    "SKHynix": "000660.KS",\n    "SamsungElectroMechanics": "009150.KS",\n\n    # Japan\n    "TokyoElectron": "8035.T",\n    "Advantest": "6857.T",\n    "Disco": "6146.T",\n    "Lasertec": "6920.T",\n    "SCREEN": "7735.T",\n    "Socionext": "6526.T",\n    "Murata": "6981.T",\n    "Kioxia": "285A.T",\n\n    # Taiwan / ADR\n    "UMC": "UMC",\n    "ASE": "ASX",\n    "Himax": "HIMX",\n    "SiliconMotion": "SIMO"\n}\n\n# ============================================================\n# 2. Download\n# ============================================================\n\nraw = yf.download(\n    list(stocks.values()),\n    period="3y",\n    auto_adjust=True,\n    progress=False,\n    group_by="column"\n)\n\nprices = raw["Close"].copy()\n\nreverse = {\n    ticker: name\n    for name, ticker in stocks.items()\n}\n\nprices = prices.rename(columns=reverse)\nprices = prices.sort_index()\n\n# 국가별 휴장일 차이를 고려\nprices = prices.ffill(limit=5)\n\n# 200일선 계산이 가능한 종목만 사용\nminimum_data = 220\n\nvalid_columns = prices.columns[\n    prices.notna().sum() >= minimum_data\n]\n\nexcluded_columns = [\n    col for col in prices.columns\n    if col not in valid_columns\n]\n\nprices = prices[valid_columns]\n\nprint(f"분석 종목 수: {len(prices.columns)}개")\n\nif excluded_columns:\n    print("데이터 부족으로 제외:", excluded_columns)\n\nprint(\n    f"분석 기간: "\n    f"{prices.index.min().date()} ~ "\n    f"{prices.index.max().date()}"\n)\n\n# ============================================================\n# 3. Positive / Negative Counts\n# ============================================================\n\nperiods = {\n    "1M": 21,\n    "2M": 42,\n    "3M": 63,\n    "6M": 126\n}\n\ncount_rows = []\n\nfor period_name, days in periods.items():\n\n    period_return = (\n        prices\n        .pct_change(days, fill_method=None)\n        .iloc[-1]\n        .dropna()\n    )\n\n    positive_count = int((period_return > 0).sum())\n    negative_count = int((period_return < 0).sum())\n    zero_count = int((period_return == 0).sum())\n    total_count = len(period_return)\n\n    count_rows.append({\n        "Period": period_name,\n        "Positive": positive_count,\n        "Negative": negative_count,\n        "Zero": zero_count,\n        "Total": total_count,\n        "Positive_Ratio": (\n            positive_count / total_count * 100\n            if total_count > 0 else np.nan\n        ),\n        "Negative_Ratio": (\n            negative_count / total_count * 100\n            if total_count > 0 else np.nan\n        )\n    })\n\ncount_table = pd.DataFrame(\n    count_rows\n).set_index("Period")\n\nprint("\\n기간별 상승·하락 종목 수")\n\ndisplay(\n    count_table.style.format({\n        "Positive": "{:.0f}",\n        "Negative": "{:.0f}",\n        "Zero": "{:.0f}",\n        "Total": "{:.0f}",\n        "Positive_Ratio": "{:.1f}%",\n        "Negative_Ratio": "{:.1f}%"\n    })\n)\n\n# ============================================================\n# 4. Breadth Indicators\n# ============================================================\n\nma20 = prices.rolling(\n    20,\n    min_periods=15\n).mean()\n\nma60 = prices.rolling(\n    60,\n    min_periods=45\n).mean()\n\nma200 = prices.rolling(\n    200,\n    min_periods=160\n).mean()\n\nret60 = prices.pct_change(\n    60,\n    fill_method=None\n)\n\n# 각 날짜마다 실제 데이터가 있는 종목만 계산\nvalid_count = prices.notna().sum(axis=1)\n\nabove_ma20_count = (\n    (prices > ma20) &\n    prices.notna() &\n    ma20.notna()\n).sum(axis=1)\n\nabove_ma60_count = (\n    (prices > ma60) &\n    prices.notna() &\n    ma60.notna()\n).sum(axis=1)\n\nabove_ma200_count = (\n    (prices > ma200) &\n    prices.notna() &\n    ma200.notna()\n).sum(axis=1)\n\npositive_60d_count = (\n    (ret60 > 0) &\n    ret60.notna()\n).sum(axis=1)\n\nma20_available = (\n    prices.notna() &\n    ma20.notna()\n).sum(axis=1)\n\nma60_available = (\n    prices.notna() &\n    ma60.notna()\n).sum(axis=1)\n\nma200_available = (\n    prices.notna() &\n    ma200.notna()\n).sum(axis=1)\n\nret60_available = ret60.notna().sum(axis=1)\n\nabove_ma20 = (\n    above_ma20_count /\n    ma20_available.replace(0, np.nan) *\n    100\n)\n\nabove_ma60 = (\n    above_ma60_count /\n    ma60_available.replace(0, np.nan) *\n    100\n)\n\nabove_ma200 = (\n    above_ma200_count /\n    ma200_available.replace(0, np.nan) *\n    100\n)\n\npositive_60d = (\n    positive_60d_count /\n    ret60_available.replace(0, np.nan) *\n    100\n)\n\nbreadth = pd.DataFrame({\n    "Above_MA20": above_ma20,\n    "Above_MA60": above_ma60,\n    "Above_MA200": above_ma200,\n    "Positive_60D": positive_60d\n})\n\n# ============================================================\n# 5. Breadth Score\n# ============================================================\n\n# 단기 20%, 중기 35%, 장기 25%, 수익률 확산 20%\nbreadth["Breadth_Score"] = (\n    breadth["Above_MA20"] * 0.20 +\n    breadth["Above_MA60"] * 0.35 +\n    breadth["Above_MA200"] * 0.25 +\n    breadth["Positive_60D"] * 0.20\n)\n\n# 노이즈 축소\nbreadth["Breadth_Score_MA5"] = (\n    breadth["Breadth_Score"]\n    .rolling(5, min_periods=3)\n    .mean()\n)\n\nbreadth["Breadth_Score_MA10"] = (\n    breadth["Breadth_Score"]\n    .rolling(10, min_periods=5)\n    .mean()\n)\n\n# 당일 점수와 5일 평균의 차이\nbreadth["Breadth_Diff"] = (\n    breadth["Breadth_Score"] -\n    breadth["Breadth_Score_MA5"]\n)\n\n# 기울기\nbreadth["Slope_3D"] = (\n    breadth["Breadth_Score_MA5"] -\n    breadth["Breadth_Score_MA5"].shift(3)\n)\n\nbreadth["Slope_5D"] = (\n    breadth["Breadth_Score_MA5"] -\n    breadth["Breadth_Score_MA5"].shift(5)\n)\n\nbreadth["Slope_10D"] = (\n    breadth["Breadth_Score_MA5"] -\n    breadth["Breadth_Score_MA5"].shift(10)\n)\n\n# 최근 10일 중 점수가 하락한 날짜 수\nbreadth["Falling_Days_10D"] = (\n    breadth["Breadth_Score_MA5"]\n    .diff()\n    .lt(0)\n    .rolling(10)\n    .sum()\n)\n\n# ============================================================\n# 6. Equal-weight AI Tech Index\n# ============================================================\n\n# 개별 종목을 시작점 100으로 정규화\nnormalized_prices = (\n    prices /\n    prices.apply(\n        lambda col: col.dropna().iloc[0]\n        if not col.dropna().empty\n        else np.nan\n    )\n) * 100\n\n# 동일가중 지수\nbreadth["AI_Tech_Index"] = (\n    normalized_prices.mean(axis=1)\n)\n\nbreadth["AI_Index_MA20"] = (\n    breadth["AI_Tech_Index"]\n    .rolling(20)\n    .mean()\n)\n\nbreadth["AI_Index_20D_Return"] = (\n    breadth["AI_Tech_Index"]\n    .pct_change(20) *\n    100\n)\n\nbreadth["AI_Index_60D_Return"] = (\n    breadth["AI_Tech_Index"]\n    .pct_change(60) *\n    100\n)\n\n# ============================================================\n# 7. Divergence Detection\n# ============================================================\n\n# 지수는 20일 신고가 부근인데\n# Breadth Score는 최근 20일 고점 대비 약해지는 경우\nindex_high_20 = (\n    breadth["AI_Tech_Index"]\n    .rolling(20)\n    .max()\n)\n\nbreadth_high_20 = (\n    breadth["Breadth_Score_MA5"]\n    .rolling(20)\n    .max()\n)\n\nbreadth["Index_Near_20D_High"] = (\n    breadth["AI_Tech_Index"] >=\n    index_high_20 * 0.99\n)\n\nbreadth["Breadth_From_20D_High"] = (\n    breadth["Breadth_Score_MA5"] -\n    breadth_high_20\n)\n\nbreadth["Bearish_Divergence"] = (\n    breadth["Index_Near_20D_High"] &\n    (breadth["Breadth_From_20D_High"] <= -10) &\n    (breadth["Slope_5D"] < 0)\n)\n\n# 강한 다이버전스:\n# 지수는 고점권인데 Breadth가 고점보다 20점 이상 낮음\nbreadth["Strong_Bearish_Divergence"] = (\n    breadth["Index_Near_20D_High"] &\n    (breadth["Breadth_From_20D_High"] <= -20) &\n    (breadth["Slope_10D"] < 0)\n)\n\n# ============================================================\n# 8. Risk Level\n# ============================================================\n\ndef classify_risk(row):\n\n    score = row["Breadth_Score_MA5"]\n    slope_5d = row["Slope_5D"]\n    slope_10d = row["Slope_10D"]\n    divergence = row["Bearish_Divergence"]\n    strong_divergence = row["Strong_Bearish_Divergence"]\n\n    if pd.isna(score):\n        return "데이터 부족"\n\n    if strong_divergence:\n        return "피크아웃 강경고"\n\n    if divergence:\n        return "피크아웃 경고"\n\n    if score >= 70 and slope_5d >= 0:\n        return "강세"\n\n    if score >= 70 and slope_5d < 0:\n        return "고점 경계"\n\n    if score >= 60:\n        return "정상"\n\n    if score >= 50:\n        if slope_5d < 0:\n            return "주의"\n        return "중립"\n\n    if score >= 35:\n        if slope_10d < 0:\n            return "위험"\n        return "약세 반등"\n\n    if score < 35 and slope_5d < 0:\n        return "투매 진행"\n\n    if score < 35 and slope_5d >= 0:\n        return "투매 후 반등"\n\n    return "중립"\n\nbreadth["Risk_Level"] = breadth.apply(\n    classify_risk,\n    axis=1\n)\n\n# ============================================================\n# 9. Current Summary\n# ============================================================\n\nlatest = breadth.dropna(\n    subset=["Breadth_Score_MA5"]\n).iloc[-1]\n\nlatest_date = breadth.dropna(\n    subset=["Breadth_Score_MA5"]\n).index[-1]\n\nlatest_summary = pd.DataFrame({\n    "현재값": {\n        "20일선 위 종목 비율": latest["Above_MA20"],\n        "60일선 위 종목 비율": latest["Above_MA60"],\n        "200일선 위 종목 비율": latest["Above_MA200"],\n        "60일 수익률 플러스 비율": latest["Positive_60D"],\n        "Breadth Score": latest["Breadth_Score"],\n        "Breadth Score MA5": latest["Breadth_Score_MA5"],\n        "Breadth Diff": latest["Breadth_Diff"],\n        "3일 기울기": latest["Slope_3D"],\n        "5일 기울기": latest["Slope_5D"],\n        "10일 기울기": latest["Slope_10D"],\n        "최근 10일 하락 일수": latest["Falling_Days_10D"],\n        "20일 고점 대비 Breadth": latest["Breadth_From_20D_High"]\n    }\n})\n\nprint(f"\\n기준일: {latest_date.date()}")\n\ndisplay(\n    latest_summary.style.format("{:.1f}")\n)\n\nprint(f"현재 위험 단계: {latest[\'Risk_Level\']}")\nprint(\n    "Bearish Divergence:",\n    bool(latest["Bearish_Divergence"])\n)\nprint(\n    "Strong Bearish Divergence:",\n    bool(latest["Strong_Bearish_Divergence"])\n)\n\n# ============================================================\n# 10. Positive / Negative Count Chart\n# ============================================================\n\ncount_plot = count_table[\n    ["Positive", "Negative"]\n]\n\nax = count_plot.plot(\n    kind="bar",\n    figsize=(10, 5)\n)\n\nax.set_title(\n    "AI Tech Positive / Negative Count"\n)\nax.set_xlabel("")\nax.set_ylabel("Number of Stocks")\nax.tick_params(\n    axis="x",\n    rotation=0\n)\nax.grid(\n    axis="y",\n    alpha=0.3\n)\n\nplt.tight_layout()\nplt.show()\n\n# ============================================================\n# 11. Moving Average Breadth\n# ============================================================\n\nplt.figure(figsize=(15, 6))\n\nplt.plot(\n    breadth.index,\n    breadth["Above_MA20"],\n    label="Above MA20"\n)\n\nplt.plot(\n    breadth.index,\n    breadth["Above_MA60"],\n    label="Above MA60"\n)\n\nplt.plot(\n    breadth.index,\n    breadth["Above_MA200"],\n    label="Above MA200"\n)\n\nplt.axhline(\n    70,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    50,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    30,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.title(\n    "AI Tech Moving Average Breadth"\n)\nplt.ylabel("Stocks (%)")\nplt.ylim(0, 100)\nplt.legend()\nplt.grid(alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\n# ============================================================\n# 12. Breadth Score\n# ============================================================\n\nplt.figure(figsize=(15, 6))\n\nplt.plot(\n    breadth.index,\n    breadth["Breadth_Score"],\n    alpha=0.30,\n    label="Daily Breadth Score"\n)\n\nplt.plot(\n    breadth.index,\n    breadth["Breadth_Score_MA5"],\n    linewidth=2,\n    label="Breadth Score MA5"\n)\n\nplt.axhline(\n    70,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    60,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    50,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    35,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.title("AI Tech Breadth Score")\nplt.ylabel("Score")\nplt.ylim(0, 100)\nplt.legend()\nplt.grid(alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\n# ============================================================\n# 13. Breadth Slope\n# ============================================================\n\nplt.figure(figsize=(15, 5))\n\nplt.plot(\n    breadth.index,\n    breadth["Slope_5D"],\n    label="5D Breadth Slope"\n)\n\nplt.plot(\n    breadth.index,\n    breadth["Slope_10D"],\n    label="10D Breadth Slope"\n)\n\nplt.axhline(\n    0,\n    linewidth=1\n)\n\nplt.axhline(\n    -10,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.axhline(\n    10,\n    linestyle="--",\n    alpha=0.5\n)\n\nplt.title("AI Tech Breadth Slope")\nplt.ylabel("Score Change")\nplt.legend()\nplt.grid(alpha=0.3)\n\nplt.tight_layout()\nplt.show()\n\n# ============================================================\n# 14. AI Tech Index vs Breadth Score\n# ============================================================\n\nfig, ax1 = plt.subplots(figsize=(15, 6))\n\n# 왼쪽 축: 동일가중 지수\nline1 = ax1.plot(\n    breadth.index,\n    breadth["AI_Tech_Index"],\n    color="black",\n    linewidth=1.8,\n    label="AI Tech Equal-weight Index"\n)\n\nax1.set_ylabel(\n    "AI Tech Index",\n    color="black"\n)\n\nax1.tick_params(\n    axis="y",\n    labelcolor="black"\n)\n\nax1.grid(alpha=0.3)\n\n# 오른쪽 축: Breadth Score\nax2 = ax1.twinx()\n\nline2 = ax2.plot(\n    breadth.index,\n    breadth["Breadth_Score_MA5"],\n    color="orange",\n    linewidth=2.0,\n    label="Breadth Score MA5"\n)\n\nax2.set_ylabel(\n    "Breadth Score",\n    color="orange"\n)\n\nax2.tick_params(\n    axis="y",\n    labelcolor="orange"\n)\n\nax2.set_ylim(0, 100)\n\n# 강한 다이버전스 구간 표시\nstrong_divergence_dates = breadth.index[\n    breadth["Strong_Bearish_Divergence"].fillna(False)\n]\n\nfor date in strong_divergence_dates:\n    ax1.axvline(\n        date,\n        color="red",\n        alpha=0.10\n    )\n\n# 범례 합치기\nlines = line1 + line2\nlabels = [\n    line.get_label()\n    for line in lines\n]\n\nax1.legend(\n    lines,\n    labels,\n    loc="upper left"\n)\n\nplt.title(\n    "AI Tech Equal-weight Index vs Breadth"\n)\n\nplt.tight_layout()\nplt.show()\n\n# ============================================================\n# 15. Recent 30 Trading Days\n# ============================================================\n\nrecent_columns = [\n    "Above_MA20",\n    "Above_MA60",\n    "Above_MA200",\n    "Positive_60D",\n    "Breadth_Score_MA5",\n    "Breadth_Diff",\n    "Slope_5D",\n    "Slope_10D",\n    "Breadth_From_20D_High",\n    "Bearish_Divergence",\n    "Strong_Bearish_Divergence",\n    "Risk_Level"\n]\n\nrecent_table = (\n    breadth[recent_columns]\n    .tail(30)\n)\n\ndisplay(\n    recent_table.style.format({\n        "Above_MA20": "{:.1f}",\n        "Above_MA60": "{:.1f}",\n        "Above_MA200": "{:.1f}",\n        "Positive_60D": "{:.1f}",\n        "Breadth_Score_MA5": "{:.1f}",\n        "Breadth_Diff": "{:+.1f}",\n        "Slope_5D": "{:+.1f}",\n        "Slope_10D": "{:+.1f}",\n        "Breadth_From_20D_High": "{:+.1f}"\n    })\n)'
ROT_SRC='# ============================================================\n# 미국 업종 ETF MASTER\n# 52W FIXED + Rotation_B FIXED\n#\n# 목적\n# ① 순수 52주 신고가 전략 (BUG FIX)\n# ② 52주 신고가 + Weak Regime Rotation_B (BUG FIX)\n# ③ SPY Buy & Hold\n#\n# ------------------------------------------------------------\n# 공통 52W 전략\n# - 6개월 SPY 대비 RS Top20\n# - 종가 > 직전 252거래일 종가 최고치\n# - 최대 4종목\n# - 종목당 최대 25%\n# - Close < 설정된 매도 이동평균(SELL_MA_DAYS) 이면 매도\n# - 그룹 제한 유지\n#\n# ------------------------------------------------------------\n# Rotation_B\n#\n# Weak Regime 판단:\n# - "순수 52W FIXED 전략"의 최근 20거래일 수익률\n#   minus SPY 최근 20거래일 수익률\n# - 초과수익 <= -1% 이면 WEAK_ROTATION\n# - 그 외 NORMAL_52W\n#\n# WEAK_ROTATION에서 신규매수 후보:\n# - 5D 수익률이 SPY 5D보다 강함\n# - 20D RS 순위가 10거래일 전보다 10등 이상 개선\n# - 당일 종가가 직전 20거래일 종가 최고치 돌파\n#\n# 후보 우선순위:\n# 1) 10일간 RS순위 개선폭 큰 순\n# 2) 5D SPY 대비 RS 강한 순\n# 3) 현재 20D RS 순위 좋은 순\n#\n# 중요:\n# - Regime이 바뀌어도 기존 보유종목은 강제매도하지 않음.\n# - 기존 보유종목은 설정된 매도 이동평균(SELL_MA_DAYS) 이탈 때만 매도.\n# - Regime은 "빈 슬롯에 무엇을 새로 살지"만 결정.\n#\n# ------------------------------------------------------------\n# ★ BUG FIX\n#\n# 과거 문제:\n# 신규종목을 무조건 25% 추가하면서\n# 기존 보유 비중 drift 때문에 총 투자비중이 100%를 넘을 수 있었음.\n#\n# 이번 버전:\n# available_cash = 1 - 현재 투자비중\n# buy_weight = min(25%, available_cash)\n#\n# → 52W와 Rotation_B 모두 동일하게 수정\n# → 총 투자비중은 절대 100% 초과 불가\n#\n# ------------------------------------------------------------\n# 체결 가정\n# - signal_date 종가에서 신호 확인\n# - signal_date 종가 체결 가정\n# - 다음 거래일부터 수익률 반영\n#\n# 최신일 신호는 성과에 소급하지 않고\n# Final_Weights에만 즉시 반영\n# ============================================================\n\n\n# ============================================================\n# 0. 설치 / Import\n# ============================================================\n\n\nimport time\nimport numpy as np\nimport pandas as pd\nimport yfinance as yf\nimport matplotlib.pyplot as plt\n\n\n\n# ============================================================\n# 1. 기본 설정\n# ============================================================\n\nDOWNLOAD_PERIOD = "3y"\nDOWNLOAD_INTERVAL = "1d"\n\nBATCH_SIZE = 10\nMAX_RETRIES = 3\nRETRY_WAIT_SECONDS = 4\nBATCH_WAIT_SECONDS = 2\n\nBENCHMARK = "SPY"\n\n# 52W 전략\nRS_LOOKBACK = 126\nHIGH_LOOKBACK = 252\nSELL_MA_DAYS = 20\nRS_TOP_N = 20\n\nMAX_HOLDINGS = 4\nMAX_WEIGHT_PER_ETF = 0.25\n\nTRANSACTION_COST = 0.001\n\nUSE_GROUP_LIMIT = True\n\n# Rotation_B / Regime\nWEAK_LOOKBACK = 20\nWEAK_THRESHOLD = -0.01\n\nROT_RS_SHORT = 5\nROT_RS_MEDIUM = 20\n\nROT_RANK_LOOKBACK = 10\nROT_RANK_IMPROVE = 10\n\nROT_BREAKOUT = 20\n\nEPS = 1e-10\n\n\n# ============================================================\n# 2. 미국 업종 ETF 유니버스\n# ============================================================\n\nindustry_etfs = {\n\n    "SPY": "S&P 500",\n\n    "XLK": "Technology",\n    "XLC": "Communication Services",\n    "XLY": "Consumer Discretionary",\n    "XLP": "Consumer Staples",\n    "XLI": "Industrials",\n    "XLB": "Materials",\n    "XLE": "Energy",\n    "XLF": "Financials",\n    "XLV": "Health Care",\n    "XLU": "Utilities",\n    "XLRE": "Real Estate",\n\n    "SOXX": "Semiconductors",\n    "IGV": "Software",\n    "SKYY": "Cloud Computing",\n    "FDN": "Internet",\n\n    "XOP": "Oil & Gas Exploration",\n    "OIH": "Oil Services",\n    "CRAK": "Oil Refining",\n    "AMLP": "Energy Infrastructure",\n    "URA": "Uranium",\n\n    "KRE": "Regional Banks",\n    "IAI": "Broker Dealers & Exchanges",\n    "KIE": "Insurance",\n    "BIZD": "Business Development Companies",\n\n    "IBB": "Biotechnology",\n    "IHI": "Medical Devices",\n    "IHF": "Health Care Providers",\n    "IHE": "Pharmaceuticals",\n\n    "XRT": "Retail",\n    "XHB": "Homebuilders",\n    "PEJ": "Leisure & Entertainment",\n    "PBJ": "Food & Beverage",\n\n    "ITA": "Aerospace & Defense",\n    "IYT": "Transportation",\n    "TOLZ": "Global Infrastructure",\n    "CGW": "Global Water",\n    "AIRR": "US Industrial Renaissance",\n\n    "GDX": "Gold Miners",\n    "SIL": "Silver Miners",\n    "SLX": "Steel",\n    "COPX": "Copper Miners",\n    "REMX": "Rare Earth & Strategic Metals",\n    "MOO": "Agribusiness",\n    "WOOD": "Timber & Forestry",\n\n    "ICLN": "Global Clean Energy",\n    "TAN": "Solar Energy",\n    "FAN": "Wind Energy",\n    "GRID": "Smart Grid",\n\n    "IYZ": "Telecom",\n    "LIT": "Lithium & Battery Tech",\n    "SEA": "Shipping",\n    "CARZ": "Global Auto",\n    "UFO": "Space & Satellite",\n    "ROBO": "Robotics & Automation",\n}\n\n\n# ============================================================\n# 3. 그룹 제한\n# ============================================================\n\nGROUP_MAP = {\n\n    "GDX": "Mining",\n    "SIL": "Mining",\n    "COPX": "Mining",\n    "REMX": "Mining",\n\n    "ICLN": "Clean_Energy",\n    "TAN": "Clean_Energy",\n    "FAN": "Clean_Energy",\n    "GRID": "Clean_Energy",\n}\n\n\nfor ticker in industry_etfs:\n\n    GROUP_MAP.setdefault(\n        ticker,\n        ticker\n    )\n\n\nall_tickers = list(\n    industry_etfs.keys()\n)\n\n\n# ============================================================\n# 4. Yahoo Finance 다운로드\n# ============================================================\n\ndef split_batches(items, size):\n\n    return [\n\n        items[\n            i:i + size\n        ]\n\n        for i in range(\n            0,\n            len(items),\n            size\n        )\n    ]\n\n\ndef extract_close(downloaded, requested):\n\n    if (\n        downloaded is None\n        or\n        downloaded.empty\n    ):\n\n        return pd.DataFrame()\n\n\n    if isinstance(\n        downloaded.columns,\n        pd.MultiIndex\n    ):\n\n        if (\n            "Close"\n            in downloaded.columns.get_level_values(0)\n        ):\n\n            close = (\n                downloaded[\n                    "Close"\n                ]\n                .copy()\n            )\n\n        elif (\n            "Close"\n            in downloaded.columns.get_level_values(1)\n        ):\n\n            close = (\n                downloaded.xs(\n                    "Close",\n                    axis=1,\n                    level=1\n                )\n                .copy()\n            )\n\n        else:\n\n            return pd.DataFrame()\n\n    else:\n\n        if (\n            "Close"\n            not in downloaded.columns\n        ):\n\n            return pd.DataFrame()\n\n\n        close = (\n            downloaded[\n                ["Close"]\n            ]\n            .copy()\n        )\n\n\n        close.columns = [\n            requested[0]\n        ]\n\n\n    if isinstance(\n        close,\n        pd.Series\n    ):\n\n        close = (\n            close.to_frame()\n        )\n\n\n    return close\n\n\ndef download_prices(tickers):\n\n    parts = []\n    failed = []\n\n\n    batches = split_batches(\n        tickers,\n        BATCH_SIZE\n    )\n\n\n    for batch_no, batch in enumerate(\n        batches,\n        start=1\n    ):\n\n\n        print(\n\n            f"다운로드 "\n            f"{batch_no}/{len(batches)}: "\n\n            + ", ".join(\n                batch\n            )\n        )\n\n\n        close = pd.DataFrame()\n\n\n        for attempt in range(\n            1,\n            MAX_RETRIES + 1\n        ):\n\n\n            try:\n\n                raw = yf.download(\n\n                    tickers=batch,\n\n                    period=DOWNLOAD_PERIOD,\n\n                    interval=DOWNLOAD_INTERVAL,\n\n                    auto_adjust=True,\n\n                    progress=False,\n\n                    threads=False,\n\n                    group_by="column",\n                )\n\n\n                close = extract_close(\n                    raw,\n                    batch\n                )\n\n\n                if (\n                    not close.empty\n                ):\n\n                    break\n\n\n            except Exception as error:\n\n                print(\n\n                    f"시도 {attempt} 실패: "\n                    f"{error}"\n                )\n\n\n            if (\n                attempt\n                < MAX_RETRIES\n            ):\n\n                time.sleep(\n                    RETRY_WAIT_SECONDS\n                )\n\n\n        usable = [\n\n            ticker\n\n            for ticker\n            in batch\n\n            if (\n\n                ticker\n                in close.columns\n\n                and\n\n                not close[\n                    ticker\n                ].dropna().empty\n            )\n        ]\n\n\n        failed.extend([\n\n            ticker\n\n            for ticker\n            in batch\n\n            if (\n                ticker\n                not in usable\n            )\n        ])\n\n\n        if usable:\n\n            parts.append(\n                close[\n                    usable\n                ]\n            )\n\n\n        time.sleep(\n            BATCH_WAIT_SECONDS\n        )\n\n\n    if not parts:\n\n        raise RuntimeError(\n            "Yahoo Finance 가격 다운로드 실패"\n        )\n\n\n    prices = pd.concat(\n        parts,\n        axis=1\n    )\n\n\n    prices = prices.loc[\n        :,\n        ~prices.columns.duplicated()\n    ]\n\n\n    prices = (\n        prices\n        .sort_index()\n        .dropna(\n            how="all"\n        )\n    )\n\n\n    return (\n        prices,\n        sorted(\n            set(\n                failed\n            )\n        )\n    )\n\n\n# ============================================================\n# 5. 다운로드 실행\n# ============================================================\n\nprices, failed_tickers = (\n    download_prices(\n        all_tickers\n    )\n)\n\n\nif (\n    BENCHMARK\n    not in prices.columns\n):\n\n    raise ValueError(\n        "SPY 다운로드 실패"\n    )\n\n\ndates = (\n\n    prices[\n        BENCHMARK\n    ]\n\n    .dropna()\n\n    .index\n)\n\n\nprices = (\n\n    prices\n\n    .reindex(\n        dates\n    )\n\n    .ffill(\n        limit=3\n    )\n)\n\n\nsector_cols = [\n\n    ticker\n\n    for ticker\n    in industry_etfs\n\n    if (\n\n        ticker\n\n        and\n\n        ticker\n        in prices.columns\n\n        and\n\n        prices[\n            ticker\n        ].notna().sum()\n        >= HIGH_LOOKBACK + 5\n    )\n]\n\n\nprint(\n    "\\n"\n    + "=" * 90\n)\n\nprint(\n    "다운로드 결과"\n)\n\nprint(\n    "=" * 90\n)\n\n\nprint(\n\n    f"데이터 기간: "\n    f"{dates.min().date()} "\n    f"~ "\n    f"{dates.max().date()}"\n)\n\n\nprint(\n\n    f"사용 업종 ETF 수: "\n    f"{len(sector_cols)}"\n)\n\n\nif failed_tickers:\n\n    print(\n        "다운로드 실패:",\n        failed_tickers\n    )\n\n\n# ============================================================\n# 6. 공통 지표\n# ============================================================\n\nsector_prices = prices[\n    sector_cols\n]\n\n\nspy = prices[\n    BENCHMARK\n]\n\n\nsector_returns = (\n\n    sector_prices\n\n    .pct_change(\n        fill_method=None\n    )\n)\n\n\nspy_returns = (\n\n    spy\n\n    .pct_change(\n        fill_method=None\n    )\n)\n\n\n# ------------------------------------------------------------\n# 6-1. 6개월 SPY 대비 RS\n# ------------------------------------------------------------\n\nrelative_ratio = (\n\n    sector_prices\n\n    .div(\n        spy,\n        axis=0\n    )\n)\n\n\nrelative_momentum = (\n\n    relative_ratio\n\n    / relative_ratio.shift(\n        RS_LOOKBACK\n    )\n\n    - 1\n)\n\n\n# ------------------------------------------------------------\n# 6-2. 52주 신고가\n# 오늘 제외, 직전 252거래일 종가 최고치\n# ------------------------------------------------------------\n\nprevious_52w_high = (\n\n    sector_prices\n\n    .shift(1)\n\n    .rolling(\n        HIGH_LOOKBACK,\n        min_periods=HIGH_LOOKBACK\n    )\n\n    .max()\n)\n\n\nbreakout_signal = (\n\n    sector_prices\n\n    > previous_52w_high\n)\n\n\n# ------------------------------------------------------------\n# 6-3. 매도 이동평균 (SELL_MA_DAYS)\n# ------------------------------------------------------------\n\nsell_ma = (\n\n    sector_prices\n\n    .rolling(\n        SELL_MA_DAYS,\n        min_periods=SELL_MA_DAYS\n    )\n\n    .mean()\n)\n\n\nbelow_sell_ma = (\n\n    sector_prices\n\n    < sell_ma\n)\n\n\n# ============================================================\n# 7. Rotation_B 지표\n# ============================================================\n\n# ------------------------------------------------------------\n# 7-1. 5D SPY 대비 RS\n# ------------------------------------------------------------\n\nrot_ret_5d = (\n\n    sector_prices\n\n    / sector_prices.shift(\n        ROT_RS_SHORT\n    )\n\n    - 1\n)\n\n\nrot_spy_ret_5d = (\n\n    spy\n\n    / spy.shift(\n        ROT_RS_SHORT\n    )\n\n    - 1\n)\n\n\nrot_rs_5d = (\n\n    rot_ret_5d\n\n    .sub(\n        rot_spy_ret_5d,\n        axis=0\n    )\n)\n\n\n# ------------------------------------------------------------\n# 7-2. 20D SPY 대비 RS\n# ------------------------------------------------------------\n\nrot_ret_20d = (\n\n    sector_prices\n\n    / sector_prices.shift(\n        ROT_RS_MEDIUM\n    )\n\n    - 1\n)\n\n\nrot_spy_ret_20d = (\n\n    spy\n\n    / spy.shift(\n        ROT_RS_MEDIUM\n    )\n\n    - 1\n)\n\n\nrot_rs_20d = (\n\n    rot_ret_20d\n\n    .sub(\n        rot_spy_ret_20d,\n        axis=0\n    )\n)\n\n\n# 현재 20D RS 순위\nrot_rs20_rank = (\n\n    rot_rs_20d\n\n    .rank(\n        axis=1,\n        ascending=False,\n        method="min"\n    )\n)\n\n\n# 10거래일 전 순위 - 현재 순위\n# 양수일수록 순위가 개선됨\nrot_rank_improve_10d = (\n\n    rot_rs20_rank.shift(\n        ROT_RANK_LOOKBACK\n    )\n\n    - rot_rs20_rank\n)\n\n\n# ------------------------------------------------------------\n# 7-3. 직전 20거래일 종가 최고치\n# ------------------------------------------------------------\n\nrot_prev_high_20 = (\n\n    sector_prices\n\n    .shift(1)\n\n    .rolling(\n        ROT_BREAKOUT,\n        min_periods=ROT_BREAKOUT\n    )\n\n    .max()\n)\n\n\nrot_breakout_20 = (\n\n    sector_prices\n\n    > rot_prev_high_20\n)\n\n\n# ------------------------------------------------------------\n# 7-4. Rotation_B 최종 신호\n# ------------------------------------------------------------\n\nrotation_b_signal = (\n\n    (\n        rot_rs_5d\n        > 0\n    )\n\n    &\n\n    (\n        rot_rank_improve_10d\n        >= ROT_RANK_IMPROVE\n    )\n\n    &\n\n    rot_breakout_20\n)\n\n\n# ============================================================\n# 8. 공통 안전 함수\n# ============================================================\n\ndef check_exposure(\n    weights,\n    date,\n    label\n):\n\n\n    exposure = float(\n        weights.sum()\n    )\n\n\n    if (\n        exposure\n        > 1.0 + 1e-8\n    ):\n\n        raise RuntimeError(\n\n            f"[{label}] "\n            f"투자비중 100% 초과 | "\n            f"{pd.Timestamp(date).date()} | "\n            f"{exposure:.4%}"\n        )\n\n\n    return exposure\n\n\ndef get_available_cash(\n    weights\n):\n\n\n    return max(\n\n        0.0,\n\n        1.0\n\n        - float(\n            weights.sum()\n        )\n    )\n\n\n# ============================================================\n# 9. 순수 52W FIXED 백테스트\n#\n# 이 Equity가 Rotation_B의 Weak Regime 기준이 됨.\n# ============================================================\n\ndef run_52w_fixed():\n\n\n    equity = pd.Series(\n        1.0,\n        index=dates,\n        dtype=float\n    )\n\n\n    weights = pd.Series(\n        0.0,\n        index=sector_cols,\n        dtype=float\n    )\n\n\n    turnover_series = pd.Series(\n        0.0,\n        index=dates,\n        dtype=float\n    )\n\n\n    trade_rows = []\n\n    daily_rows = []\n\n\n    first_signal_position = max(\n\n        RS_LOOKBACK,\n\n        HIGH_LOOKBACK,\n\n        SELL_MA_DAYS\n    )\n\n\n    for i in range(\n        1,\n        len(dates)\n    ):\n\n\n        date = dates[i]\n\n        signal_date = dates[\n            i - 1\n        ]\n\n\n        previous_equity = (\n            equity.iloc[\n                i - 1\n            ]\n        )\n\n\n        if (\n            i - 1\n            >= first_signal_position\n        ):\n\n\n            old_weights = (\n                weights.copy()\n            )\n\n\n            # ========================================================\n            # SELL\n            # ========================================================\n\n            holdings = (\n\n                weights[\n                    weights > EPS\n                ]\n\n                .index\n\n                .tolist()\n            )\n\n\n            sell_tickers = [\n\n                ticker\n\n                for ticker\n                in holdings\n\n                if bool(\n\n                    below_sell_ma.loc[\n                        signal_date,\n                        ticker\n                    ]\n                )\n            ]\n\n\n            for ticker in sell_tickers:\n\n\n                old_weight = float(\n                    weights.loc[\n                        ticker\n                    ]\n                )\n\n\n                weights.loc[\n                    ticker\n                ] = 0.0\n\n\n                trade_rows.append({\n\n                    "Signal_Date":\n                        signal_date,\n\n                    "Execution_Date":\n                        date,\n\n                    "Ticker":\n                        ticker,\n\n                    "Industry":\n                        industry_etfs.get(\n                            ticker,\n                            ""\n                        ),\n\n                    "Strategy":\n                        "52W",\n\n                    "Action":\n                        "SELL",\n\n                    "Reason":\n                        f"Close below MA{SELL_MA_DAYS}",\n\n                    "Trade_Weight":\n                        -old_weight,\n\n                    "RS_Rank":\n                        np.nan,\n                })\n\n\n            remaining = (\n\n                weights[\n                    weights > EPS\n                ]\n\n                .index\n\n                .tolist()\n            )\n\n\n            # ========================================================\n            # RS Top20\n            # ========================================================\n\n            rs_today = (\n\n                relative_momentum\n\n                .loc[\n                    signal_date\n                ]\n\n                .dropna()\n\n                .sort_values(\n                    ascending=False\n                )\n            )\n\n\n            rs_top = (\n\n                rs_today\n\n                .head(\n                    RS_TOP_N\n                )\n            )\n\n\n            rank_map = {\n\n                ticker:\n                    rank\n\n                for rank, ticker\n\n                in enumerate(\n                    rs_today.index,\n                    start=1\n                )\n            }\n\n\n            candidates = [\n\n                ticker\n\n                for ticker\n                in rs_top.index\n\n                if (\n\n                    ticker\n                    not in remaining\n\n                    and\n\n                    ticker\n                    not in sell_tickers\n\n                    and\n\n                    bool(\n\n                        breakout_signal.loc[\n                            signal_date,\n                            ticker\n                        ]\n                    )\n                )\n            ]\n\n\n            candidates = sorted(\n\n                candidates,\n\n                key=lambda ticker:\n\n                    rank_map.get(\n                        ticker,\n                        999\n                    )\n            )\n\n\n            used_groups = {\n\n                GROUP_MAP[\n                    ticker\n                ]\n\n                for ticker\n                in remaining\n            }\n\n\n            available_slots = max(\n\n                0,\n\n                MAX_HOLDINGS\n\n                - len(\n                    remaining\n                )\n            )\n\n\n            buys = []\n\n\n            for ticker in candidates:\n\n\n                if (\n                    len(\n                        buys\n                    )\n                    >= available_slots\n                ):\n\n                    break\n\n\n                group = (\n                    GROUP_MAP[\n                        ticker\n                    ]\n                )\n\n\n                if (\n\n                    USE_GROUP_LIMIT\n\n                    and\n\n                    group\n                    in used_groups\n                ):\n\n                    continue\n\n\n                # ====================================================\n                # ★ BUG FIX\n                # 남은 현금까지만 매수\n                # ====================================================\n\n                available_cash = (\n                    get_available_cash(\n                        weights\n                    )\n                )\n\n\n                buy_weight = min(\n\n                    MAX_WEIGHT_PER_ETF,\n\n                    available_cash\n                )\n\n\n                if (\n                    buy_weight\n                    <= EPS\n                ):\n\n                    break\n\n\n                weights.loc[\n                    ticker\n                ] = buy_weight\n\n\n                buys.append(\n                    ticker\n                )\n\n\n                used_groups.add(\n                    group\n                )\n\n\n                trade_rows.append({\n\n                    "Signal_Date":\n                        signal_date,\n\n                    "Execution_Date":\n                        date,\n\n                    "Ticker":\n                        ticker,\n\n                    "Industry":\n                        industry_etfs.get(\n                            ticker,\n                            ""\n                        ),\n\n                    "Strategy":\n                        "52W",\n\n                    "Action":\n                        "BUY",\n\n                    "Reason":\n                        "RS Top20 + 52W breakout",\n\n                    "Trade_Weight":\n                        buy_weight,\n\n                    "RS_Rank":\n                        rank_map.get(\n                            ticker,\n                            np.nan\n                        ),\n                })\n\n\n            check_exposure(\n                weights,\n                signal_date,\n                "52W FIXED after signal"\n            )\n\n\n            # ========================================================\n            # 거래비용\n            # ========================================================\n\n            turnover = (\n\n                weights\n\n                - old_weights\n\n            ).abs().sum()\n\n\n            if (\n                turnover\n                > 0\n            ):\n\n\n                previous_equity *= max(\n\n                    0.0,\n\n                    1.0\n\n                    - TRANSACTION_COST\n                    * turnover\n                )\n\n\n                turnover_series.loc[\n                    date\n                ] = turnover\n\n\n        # ============================================================\n        # 오늘 수익률\n        # ============================================================\n\n        day_returns = (\n\n            sector_returns\n\n            .loc[\n                date\n            ]\n\n            .reindex(\n                sector_cols\n            )\n\n            .fillna(\n                0.0\n            )\n        )\n\n\n        exposure = (\n            check_exposure(\n                weights,\n                date,\n                "52W FIXED before return"\n            )\n        )\n\n\n        cash_weight = max(\n\n            0.0,\n\n            1.0\n\n            - exposure\n        )\n\n\n        portfolio_multiple = (\n\n            cash_weight\n\n            +\n\n            (\n                weights\n\n                * (\n                    1\n                    + day_returns\n                )\n\n            ).sum()\n        )\n\n\n        equity.loc[\n            date\n        ] = (\n\n            previous_equity\n\n            * portfolio_multiple\n        )\n\n\n        # ============================================================\n        # 비중 Drift\n        # ============================================================\n\n        if (\n            portfolio_multiple\n            > 0\n        ):\n\n\n            weights = (\n\n                weights\n\n                * (\n                    1\n                    + day_returns\n                )\n\n                / portfolio_multiple\n            )\n\n\n        check_exposure(\n            weights,\n            date,\n            "52W FIXED after drift"\n        )\n\n\n        held = (\n\n            weights[\n                weights > EPS\n            ]\n\n            .sort_values(\n                ascending=False\n            )\n        )\n\n\n        daily_rows.append({\n\n            "Date":\n                date,\n\n            "Regime":\n                "52W",\n\n            "Holding_Count":\n                len(held),\n\n            "Invested_Exposure":\n                float(\n                    held.sum()\n                ),\n\n            "Cash_Exposure":\n                max(\n                    0.0,\n                    1.0\n                    - float(\n                        held.sum()\n                    )\n                ),\n\n            "Holdings":\n                " | ".join(\n\n                    f"{t}:{w:.1%}"\n\n                    for t, w\n                    in held.items()\n                ),\n        })\n\n\n    # ------------------------------------------------------------\n    # 최신일 신호를 Final_Weights에만 즉시 반영\n    # 과거 Equity에는 소급하지 않음\n    # ------------------------------------------------------------\n\n    live_date = dates[-1]\n\n    live_weights = (\n        weights.copy()\n    )\n\n\n    live_rows = []\n\n\n    # 최신 SELL\n    live_holdings = (\n\n        live_weights[\n            live_weights > EPS\n        ]\n\n        .index\n\n        .tolist()\n    )\n\n\n    live_sells = [\n\n        ticker\n\n        for ticker\n        in live_holdings\n\n        if bool(\n\n            below_sell_ma.loc[\n                live_date,\n                ticker\n            ]\n        )\n    ]\n\n\n    for ticker in live_sells:\n\n\n        old_weight = float(\n            live_weights.loc[\n                ticker\n            ]\n        )\n\n\n        live_weights.loc[\n            ticker\n        ] = 0.0\n\n\n        live_rows.append({\n\n            "Signal_Date":\n                live_date,\n\n            "Execution_Date":\n                live_date,\n\n            "Ticker":\n                ticker,\n\n            "Industry":\n                industry_etfs.get(\n                    ticker,\n                    ""\n                ),\n\n            "Strategy":\n                "52W",\n\n            "Action":\n                "SELL",\n\n            "Reason":\n                f"Close below MA{SELL_MA_DAYS} (latest/live)",\n\n            "Trade_Weight":\n                -old_weight,\n\n            "RS_Rank":\n                np.nan,\n        })\n\n\n    remaining = (\n\n        live_weights[\n            live_weights > EPS\n        ]\n\n        .index\n\n        .tolist()\n    )\n\n\n    rs_today = (\n\n        relative_momentum\n\n        .loc[\n            live_date\n        ]\n\n        .dropna()\n\n        .sort_values(\n            ascending=False\n        )\n    )\n\n\n    rs_top = (\n\n        rs_today\n\n        .head(\n            RS_TOP_N\n        )\n    )\n\n\n    rank_map = {\n\n        ticker:\n            rank\n\n        for rank, ticker\n\n        in enumerate(\n            rs_today.index,\n            start=1\n        )\n    }\n\n\n    candidates = [\n\n        ticker\n\n        for ticker\n        in rs_top.index\n\n        if (\n\n            ticker\n            not in remaining\n\n            and\n\n            ticker\n            not in live_sells\n\n            and\n\n            bool(\n\n                breakout_signal.loc[\n                    live_date,\n                    ticker\n                ]\n            )\n        )\n    ]\n\n\n    candidates = sorted(\n\n        candidates,\n\n        key=lambda ticker:\n\n            rank_map.get(\n                ticker,\n                999\n            )\n    )\n\n\n    used_groups = {\n\n        GROUP_MAP[\n            ticker\n        ]\n\n        for ticker\n        in remaining\n    }\n\n\n    slots = max(\n\n        0,\n\n        MAX_HOLDINGS\n\n        - len(\n            remaining\n        )\n    )\n\n\n    live_buys = []\n\n\n    for ticker in candidates:\n\n\n        if (\n            len(\n                live_buys\n            )\n            >= slots\n        ):\n\n            break\n\n\n        group = (\n            GROUP_MAP[\n                ticker\n            ]\n        )\n\n\n        if (\n\n            USE_GROUP_LIMIT\n\n            and\n\n            group\n            in used_groups\n        ):\n\n            continue\n\n\n        available_cash = (\n            get_available_cash(\n                live_weights\n            )\n        )\n\n\n        buy_weight = min(\n\n            MAX_WEIGHT_PER_ETF,\n\n            available_cash\n        )\n\n\n        if (\n            buy_weight\n            <= EPS\n        ):\n\n            break\n\n\n        live_weights.loc[\n            ticker\n        ] = buy_weight\n\n\n        live_buys.append(\n            ticker\n        )\n\n\n        used_groups.add(\n            group\n        )\n\n\n        live_rows.append({\n\n            "Signal_Date":\n                live_date,\n\n            "Execution_Date":\n                live_date,\n\n            "Ticker":\n                ticker,\n\n            "Industry":\n                industry_etfs.get(\n                    ticker,\n                    ""\n                ),\n\n            "Strategy":\n                "52W",\n\n            "Action":\n                "BUY",\n\n            "Reason":\n                "RS Top20 + 52W breakout (latest/live)",\n\n            "Trade_Weight":\n                buy_weight,\n\n            "RS_Rank":\n                rank_map.get(\n                    ticker,\n                    np.nan\n                ),\n        })\n\n\n    check_exposure(\n        live_weights,\n        live_date,\n        "52W FIXED live"\n    )\n\n\n    all_trades = pd.DataFrame(\n        trade_rows\n        + live_rows\n    )\n\n\n    return {\n\n        "Equity":\n            equity,\n\n        "Trades":\n            all_trades,\n\n        "Turnover":\n            turnover_series,\n\n        "Daily_Holdings":\n            pd.DataFrame(\n                daily_rows\n            ).set_index(\n                "Date"\n            ),\n\n        "Final_Weights":\n            live_weights.copy(),\n\n        "Live_Date":\n            live_date,\n\n        "Live_SELL":\n            live_sells,\n\n        "Live_BUY":\n            live_buys,\n    }\n\n\n# ============================================================\n# 10. 순수 52W FIXED 실행\n# ============================================================\n\nbaseline_result = (\n    run_52w_fixed()\n)\n\n\nbaseline_equity = (\n    baseline_result[\n        "Equity"\n    ]\n)\n\n\n# ============================================================\n# 11. Weak Regime 계산\n#\n# 기준은 반드시 "버그 수정된 52W FIXED" Equity\n# ============================================================\n\nbaseline_20d = (\n\n    baseline_equity\n\n    / baseline_equity.shift(\n        WEAK_LOOKBACK\n    )\n\n    - 1\n)\n\n\nspy_20d = (\n\n    spy\n\n    / spy.shift(\n        WEAK_LOOKBACK\n    )\n\n    - 1\n)\n\n\nbaseline_excess_20d = (\n\n    baseline_20d\n\n    - spy_20d\n)\n\n\nweak_regime = (\n\n    baseline_excess_20d\n\n    <= WEAK_THRESHOLD\n)\n\n\n# ============================================================\n# 12. Rotation_B 후보 정렬\n# ============================================================\n\ndef sort_rotation_candidates(\n    signal_date,\n    candidates\n):\n\n\n    def key(ticker):\n\n\n        improve = (\n            rot_rank_improve_10d.loc[\n                signal_date,\n                ticker\n            ]\n        )\n\n\n        rs5 = (\n            rot_rs_5d.loc[\n                signal_date,\n                ticker\n            ]\n        )\n\n\n        rank_now = (\n            rot_rs20_rank.loc[\n                signal_date,\n                ticker\n            ]\n        )\n\n\n        improve = (\n\n            float(\n                improve\n            )\n\n            if pd.notna(\n                improve\n            )\n\n            else -999.0\n        )\n\n\n        rs5 = (\n\n            float(\n                rs5\n            )\n\n            if pd.notna(\n                rs5\n            )\n\n            else -999.0\n        )\n\n\n        rank_now = (\n\n            float(\n                rank_now\n            )\n\n            if pd.notna(\n                rank_now\n            )\n\n            else 999.0\n        )\n\n\n        return (\n\n            -improve,\n\n            -rs5,\n\n            rank_now\n        )\n\n\n    return sorted(\n        candidates,\n        key=key\n    )\n\n\n# ============================================================\n# 13. Regime / 신규매수 후보 함수\n# ============================================================\n\ndef get_regime(\n    signal_date\n):\n\n\n    excess = (\n\n        baseline_excess_20d.loc[\n            signal_date\n        ]\n    )\n\n\n    is_weak = bool(\n\n        pd.notna(\n            excess\n        )\n\n        and\n\n        excess\n        <= WEAK_THRESHOLD\n    )\n\n\n    regime_name = (\n\n        "WEAK_ROTATION"\n\n        if is_weak\n\n        else "NORMAL_52W"\n    )\n\n\n    return (\n        is_weak,\n        regime_name,\n        excess\n    )\n\n\ndef get_final_candidates(\n    signal_date,\n    remaining,\n    sold_today=None\n):\n\n\n    sold_today = set(\n        sold_today\n        or []\n    )\n\n\n    is_weak, regime_name, excess = (\n        get_regime(\n            signal_date\n        )\n    )\n\n\n    # ========================================================\n    # NORMAL = 52W\n    # ========================================================\n\n    if (\n        not is_weak\n    ):\n\n\n        rs_today = (\n\n            relative_momentum\n\n            .loc[\n                signal_date\n            ]\n\n            .dropna()\n\n            .sort_values(\n                ascending=False\n            )\n        )\n\n\n        rs_top = (\n\n            rs_today\n\n            .head(\n                RS_TOP_N\n            )\n        )\n\n\n        rank_map = {\n\n            ticker:\n                rank\n\n            for rank, ticker\n\n            in enumerate(\n                rs_today.index,\n                start=1\n            )\n        }\n\n\n        candidates = [\n\n            ticker\n\n            for ticker\n            in rs_top.index\n\n            if (\n\n                ticker\n                not in remaining\n\n                and\n\n                ticker\n                not in sold_today\n\n                and\n\n                bool(\n\n                    breakout_signal.loc[\n                        signal_date,\n                        ticker\n                    ]\n                )\n            )\n        ]\n\n\n        candidates = sorted(\n\n            candidates,\n\n            key=lambda ticker:\n\n                rank_map.get(\n                    ticker,\n                    999\n                )\n        )\n\n\n        origin = (\n            "52W"\n        )\n\n\n    # ========================================================\n    # WEAK = Rotation_B\n    # ========================================================\n\n    else:\n\n\n        candidates = [\n\n            ticker\n\n            for ticker\n            in sector_cols\n\n            if (\n\n                ticker\n                not in remaining\n\n                and\n\n                ticker\n                not in sold_today\n\n                and\n\n                bool(\n\n                    rotation_b_signal.loc[\n                        signal_date,\n                        ticker\n                    ]\n                )\n            )\n        ]\n\n\n        candidates = (\n            sort_rotation_candidates(\n                signal_date,\n                candidates\n            )\n        )\n\n\n        origin = (\n            "ROTATION_B"\n        )\n\n\n    return (\n\n        candidates,\n\n        origin,\n\n        regime_name,\n\n        excess\n    )\n\n\n# ============================================================\n# 14. 한 날짜 신호 적용\n#\n# 52W / Rotation_B 동일하게 BUG FIX 적용\n# ============================================================\n\ndef apply_final_signal(\n    signal_date,\n    weights,\n    origins,\n    execution_date,\n    record_trades=True\n):\n\n\n    target = (\n        weights.copy()\n    )\n\n\n    origins = (\n        origins.copy()\n    )\n\n\n    rows = []\n\n\n    # ========================================================\n    # 14-1. SELL\n    # Regime과 무관하게 설정된 매도 이동평균 이탈\n    # ========================================================\n\n    holdings = (\n\n        target[\n            target > EPS\n        ]\n\n        .index\n\n        .tolist()\n    )\n\n\n    sell_tickers = [\n\n        ticker\n\n        for ticker\n        in holdings\n\n        if (\n\n            pd.notna(\n\n                sell_ma.loc[\n                    signal_date,\n                    ticker\n                ]\n            )\n\n            and\n\n            sector_prices.loc[\n                signal_date,\n                ticker\n            ]\n\n            <\n\n            sell_ma.loc[\n                signal_date,\n                ticker\n            ]\n        )\n    ]\n\n\n    for ticker in sell_tickers:\n\n\n        old_origin = (\n            origins.pop(\n                ticker,\n                ""\n            )\n        )\n\n\n        old_weight = float(\n            target.loc[\n                ticker\n            ]\n        )\n\n\n        target.loc[\n            ticker\n        ] = 0.0\n\n\n        if record_trades:\n\n\n            rows.append({\n\n                "Signal_Date":\n                    signal_date,\n\n                "Execution_Date":\n                    execution_date,\n\n                "Ticker":\n                    ticker,\n\n                "Industry":\n                    industry_etfs.get(\n                        ticker,\n                        ""\n                    ),\n\n                "Strategy":\n                    old_origin,\n\n                "Action":\n                    "SELL",\n\n                "Reason":\n                    f"Close below MA{SELL_MA_DAYS}",\n\n                "Trade_Weight":\n                    -old_weight,\n\n                "Regime":\n                    None,\n            })\n\n\n    remaining = (\n\n        target[\n            target > EPS\n        ]\n\n        .index\n\n        .tolist()\n    )\n\n\n    slots = max(\n\n        0,\n\n        MAX_HOLDINGS\n\n        - len(\n            remaining\n        )\n    )\n\n\n    # ========================================================\n    # 14-2. Regime에 따른 신규매수 후보\n    # ========================================================\n\n    (\n        candidates,\n        origin,\n        regime_name,\n        excess\n    ) = get_final_candidates(\n\n        signal_date,\n        remaining,\n        sell_tickers\n    )\n\n\n    used_groups = {\n\n        GROUP_MAP[\n            ticker\n        ]\n\n        for ticker\n        in remaining\n    }\n\n\n    buys = []\n\n\n    for ticker in candidates:\n\n\n        if (\n            len(\n                buys\n            )\n            >= slots\n        ):\n\n            break\n\n\n        group = (\n            GROUP_MAP[\n                ticker\n            ]\n        )\n\n\n        if (\n\n            USE_GROUP_LIMIT\n\n            and\n\n            group\n            in used_groups\n        ):\n\n            continue\n\n\n        # ====================================================\n        # ★ BUG FIX\n        # 52W든 Rotation_B든 무조건 남은 현금까지만 매수\n        # ====================================================\n\n        available_cash = (\n            get_available_cash(\n                target\n            )\n        )\n\n\n        buy_weight = min(\n\n            MAX_WEIGHT_PER_ETF,\n\n            available_cash\n        )\n\n\n        if (\n            buy_weight\n            <= EPS\n        ):\n\n            break\n\n\n        target.loc[\n            ticker\n        ] = buy_weight\n\n\n        origins[\n            ticker\n        ] = origin\n\n\n        buys.append(\n            ticker\n        )\n\n\n        used_groups.add(\n            group\n        )\n\n\n        if record_trades:\n\n\n            rows.append({\n\n                "Signal_Date":\n                    signal_date,\n\n                "Execution_Date":\n                    execution_date,\n\n                "Ticker":\n                    ticker,\n\n                "Industry":\n                    industry_etfs.get(\n                        ticker,\n                        ""\n                    ),\n\n                "Strategy":\n                    origin,\n\n                "Action":\n                    "BUY",\n\n                "Reason":\n\n                    (\n                        "RS Top20 + 52W breakout"\n\n                        if origin\n                        == "52W"\n\n                        else\n                        (\n                            "20D breakout + "\n                            "RS5>SPY + "\n                            "RS rank improve 10D >=10"\n                        )\n                    ),\n\n                "Trade_Weight":\n                    buy_weight,\n\n                "Regime":\n                    regime_name,\n            })\n\n\n    check_exposure(\n        target,\n        signal_date,\n        "FINAL after signal"\n    )\n\n\n    return (\n\n        target,\n\n        origins,\n\n        rows,\n\n        sell_tickers,\n\n        buys,\n\n        regime_name,\n\n        excess,\n\n        origin\n    )\n\n\n# ============================================================\n# 15. FINAL 52주 신고가 + Rotation_B FIXED 백테스트\n# ============================================================\n\ndef run_final_rotation_fixed():\n\n\n    equity = pd.Series(\n        1.0,\n        index=dates,\n        dtype=float\n    )\n\n\n    weights = pd.Series(\n        0.0,\n        index=sector_cols,\n        dtype=float\n    )\n\n\n    turnover_series = pd.Series(\n        0.0,\n        index=dates,\n        dtype=float\n    )\n\n\n    origins = {}\n\n\n    trade_rows = []\n\n    daily_rows = []\n\n\n    first_signal_position = max(\n\n        HIGH_LOOKBACK,\n\n        RS_LOOKBACK,\n\n        SELL_MA_DAYS,\n\n        WEAK_LOOKBACK,\n\n        ROT_RS_MEDIUM\n        + ROT_RANK_LOOKBACK,\n\n        ROT_BREAKOUT\n    )\n\n\n    for i in range(\n        1,\n        len(dates)\n    ):\n\n\n        date = dates[i]\n\n        signal_date = dates[\n            i - 1\n        ]\n\n\n        previous_equity = (\n            equity.iloc[\n                i - 1\n            ]\n        )\n\n\n        regime_name = (\n            "WARMUP"\n        )\n\n\n        excess = (\n            np.nan\n        )\n\n\n        # ========================================================\n        # 신호 적용\n        # ========================================================\n\n        if (\n            i - 1\n            >= first_signal_position\n        ):\n\n\n            old_weights = (\n                weights.copy()\n            )\n\n\n            (\n                weights,\n                origins,\n                rows,\n                _,\n                _,\n                regime_name,\n                excess,\n                _\n            ) = apply_final_signal(\n\n                signal_date,\n                weights,\n                origins,\n                execution_date=date,\n                record_trades=True\n            )\n\n\n            trade_rows.extend(\n                rows\n            )\n\n\n            turnover = (\n\n                weights\n\n                - old_weights\n\n            ).abs().sum()\n\n\n            if (\n                turnover\n                > 0\n            ):\n\n\n                previous_equity *= max(\n\n                    0.0,\n\n                    1.0\n\n                    - TRANSACTION_COST\n                    * turnover\n                )\n\n\n                turnover_series.loc[\n                    date\n                ] = turnover\n\n\n        # ========================================================\n        # 오늘 수익률\n        # ========================================================\n\n        day_returns = (\n\n            sector_returns\n\n            .loc[\n                date\n            ]\n\n            .reindex(\n                sector_cols\n            )\n\n            .fillna(\n                0.0\n            )\n        )\n\n\n        exposure = (\n            check_exposure(\n                weights,\n                date,\n                "FINAL before return"\n            )\n        )\n\n\n        cash_weight = max(\n\n            0.0,\n\n            1.0\n\n            - exposure\n        )\n\n\n        portfolio_multiple = (\n\n            cash_weight\n\n            +\n\n            (\n                weights\n\n                * (\n                    1\n                    + day_returns\n                )\n\n            ).sum()\n        )\n\n\n        equity.loc[\n            date\n        ] = (\n\n            previous_equity\n\n            * portfolio_multiple\n        )\n\n\n        # ========================================================\n        # 비중 drift\n        # ========================================================\n\n        if (\n            portfolio_multiple\n            > 0\n        ):\n\n\n            weights = (\n\n                weights\n\n                * (\n                    1\n                    + day_returns\n                )\n\n                / portfolio_multiple\n            )\n\n\n        check_exposure(\n            weights,\n            date,\n            "FINAL after drift"\n        )\n\n\n        held = (\n\n            weights[\n                weights > EPS\n            ]\n\n            .sort_values(\n                ascending=False\n            )\n        )\n\n\n        daily_rows.append({\n\n            "Date":\n                date,\n\n            "Regime":\n                regime_name,\n\n            "52W_20D_Excess_vs_SPY":\n                baseline_excess_20d.loc[\n                    date\n                ],\n\n            "Holding_Count":\n                len(\n                    held\n                ),\n\n            "Invested_Exposure":\n                float(\n                    held.sum()\n                ),\n\n            "Cash_Exposure":\n                max(\n                    0.0,\n                    1.0\n                    - float(\n                        held.sum()\n                    )\n                ),\n\n            "Holdings":\n                " | ".join(\n\n                    f"{ticker}:{weight:.1%}"\n                    f"[{origins.get(ticker, \'\')}]"\n\n                    for ticker, weight\n\n                    in held.items()\n                ),\n        })\n\n\n    # ========================================================\n    # 최신일 신호 즉시 반영\n    # Equity에는 소급하지 않고 Final_Weights만 갱신\n    # ========================================================\n\n    live_date = dates[-1]\n\n\n    (\n        live_weights,\n        live_origins,\n        live_rows,\n        live_sells,\n        live_buys,\n        live_regime,\n        live_excess,\n        live_origin\n    ) = apply_final_signal(\n\n        live_date,\n        weights,\n        origins,\n        execution_date=live_date,\n        record_trades=True\n    )\n\n\n    # 중복 최신일 거래 방지\n    existing_keys = {\n\n        (\n            row.get(\n                "Signal_Date"\n            ),\n\n            row.get(\n                "Execution_Date"\n            ),\n\n            row.get(\n                "Ticker"\n            ),\n\n            row.get(\n                "Action"\n            )\n        )\n\n        for row\n        in trade_rows\n    }\n\n\n    for row in live_rows:\n\n\n        key = (\n\n            row.get(\n                "Signal_Date"\n            ),\n\n            row.get(\n                "Execution_Date"\n            ),\n\n            row.get(\n                "Ticker"\n            ),\n\n            row.get(\n                "Action"\n            )\n        )\n\n\n        if (\n            key\n            not in existing_keys\n        ):\n\n\n            trade_rows.append(\n                row\n            )\n\n\n            existing_keys.add(\n                key\n            )\n\n\n    check_exposure(\n        live_weights,\n        live_date,\n        "FINAL live"\n    )\n\n\n    return {\n\n        "Equity":\n            equity,\n\n        "Trades":\n            pd.DataFrame(\n                trade_rows\n            ),\n\n        "Turnover":\n            turnover_series,\n\n        "Daily_Holdings":\n            pd.DataFrame(\n                daily_rows\n            ).set_index(\n                "Date"\n            ),\n\n        "Final_Weights":\n            live_weights.copy(),\n\n        "Origins":\n            live_origins.copy(),\n\n        "Live_Date":\n            live_date,\n\n        "Live_Regime":\n            live_regime,\n\n        "Live_Excess_20D":\n            live_excess,\n\n        "Live_SELL":\n            live_sells,\n\n        "Live_BUY":\n            live_buys,\n\n        "Live_Origin":\n            live_origin,\n    }\n\n\n# ============================================================\n# 16. FINAL 실행\n# ============================================================\n\nfinal_result = (\n    run_final_rotation_fixed()\n)\n\n\nfinal_equity = (\n    final_result[\n        "Equity"\n    ]\n)\n\n\n# ============================================================\n# 17. 동일 시작일 비교\n# ============================================================\n\nbaseline_trades = (\n    baseline_result[\n        "Trades"\n    ]\n)\n\n\nfinal_trades = (\n    final_result[\n        "Trades"\n    ]\n)\n\n\nhistorical_baseline_trades = (\n\n    baseline_trades[\n        pd.to_datetime(\n            baseline_trades[\n                "Execution_Date"\n            ]\n        )\n\n        < pd.Timestamp(\n            dates[-1]\n        )\n    ]\n)\n\n\nhistorical_final_trades = (\n\n    final_trades[\n        pd.to_datetime(\n            final_trades[\n                "Execution_Date"\n            ]\n        )\n\n        < pd.Timestamp(\n            dates[-1]\n        )\n    ]\n)\n\n\ncandidate_starts = []\n\n\nif (\n    not historical_baseline_trades.empty\n):\n\n    candidate_starts.append(\n\n        pd.Timestamp(\n\n            historical_baseline_trades[\n                "Execution_Date"\n            ].min()\n        )\n    )\n\n\nif (\n    not historical_final_trades.empty\n):\n\n    candidate_starts.append(\n\n        pd.Timestamp(\n\n            historical_final_trades[\n                "Execution_Date"\n            ].min()\n        )\n    )\n\n\nif not candidate_starts:\n\n    raise RuntimeError(\n        "비교 가능한 거래가 없습니다."\n    )\n\n\ncomparison_start = max(\n    candidate_starts\n)\n\n\ndef normalize_from(\n    curve,\n    start_date\n):\n\n\n    pos = (\n        curve.index.get_loc(\n            start_date\n        )\n    )\n\n\n    base = (\n\n        curve.iloc[\n            pos - 1\n        ]\n\n        if (\n            pos > 0\n        )\n\n        else curve.iloc[\n            pos\n        ]\n    )\n\n\n    return (\n\n        curve.loc[\n            start_date:\n        ]\n\n        / base\n    )\n\n\nbaseline_live = (\n    normalize_from(\n        baseline_equity,\n        comparison_start\n    )\n)\n\n\nfinal_live = (\n    normalize_from(\n        final_equity,\n        comparison_start\n    )\n)\n\n\nspy_full = (\n\n    1\n\n    + spy_returns.fillna(\n        0.0\n    )\n\n).cumprod()\n\n\nspy_live = (\n    normalize_from(\n        spy_full,\n        comparison_start\n    )\n)\n\n\n# ============================================================\n# 18. 성과 계산\n# ============================================================\n\ndef calc_stats(\n    curve\n):\n\n\n    curve = (\n        curve\n        .dropna()\n    )\n\n\n    returns = (\n\n        curve\n\n        .pct_change()\n\n        .dropna()\n    )\n\n\n    years = (\n\n        (\n            curve.index[-1]\n\n            - curve.index[0]\n        ).days\n\n        / 365.25\n    )\n\n\n    cagr = (\n\n        curve.iloc[-1]\n\n        / curve.iloc[0]\n\n    ) ** (\n\n        1 / years\n\n    ) - 1\n\n\n    drawdown = (\n\n        curve\n\n        / curve.cummax()\n\n        - 1\n    )\n\n\n    mdd = (\n        drawdown.min()\n    )\n\n\n    sharpe = (\n\n        np.sqrt(\n            252\n        )\n\n        * returns.mean()\n\n        / returns.std()\n\n        if (\n            returns.std()\n            > 0\n        )\n\n        else np.nan\n    )\n\n\n    downside = (\n\n        returns[\n            returns < 0\n        ].std()\n    )\n\n\n    sortino = (\n\n        np.sqrt(\n            252\n        )\n\n        * returns.mean()\n\n        / downside\n\n        if (\n\n            pd.notna(\n                downside\n            )\n\n            and\n\n            downside > 0\n        )\n\n        else np.nan\n    )\n\n\n    calmar = (\n\n        cagr\n\n        / abs(\n            mdd\n        )\n\n        if (\n            mdd < 0\n        )\n\n        else np.nan\n    )\n\n\n    return {\n\n        "CAGR":\n            cagr,\n\n        "MDD":\n            mdd,\n\n        "Sharpe":\n            sharpe,\n\n        "Sortino":\n            sortino,\n\n        "Calmar":\n            calmar,\n\n        "Ending_Multiple":\n            curve.iloc[-1]\n\n            / curve.iloc[0],\n    }\n\n\nsummary = pd.DataFrame([\n\n    {\n        "Strategy":\n            "52W FIXED",\n\n        **calc_stats(\n            baseline_live\n        )\n    },\n\n    {\n        "Strategy":\n            "52주 신고가 + Rotation_B FIXED",\n\n        **calc_stats(\n            final_live\n        )\n    },\n\n    {\n        "Strategy":\n            "SPY Buy & Hold",\n\n        **calc_stats(\n            spy_live\n        )\n    },\n])\n\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "★ BUG FIX 후 전략 성과 비교"\n)\n\nprint(\n    f"공통 비교 시작일: "\n    f"{comparison_start.date()}"\n)\n\nprint(\n    "=" * 120\n)\n\n\ndisplay(\n\n    summary.style.format({\n\n        "CAGR":\n            "{:.2%}",\n\n        "MDD":\n            "{:.2%}",\n\n        "Sharpe":\n            "{:.2f}",\n\n        "Sortino":\n            "{:.2f}",\n\n        "Calmar":\n            "{:.2f}",\n\n        "Ending_Multiple":\n            "{:.2f}x",\n    })\n)\n\n\n# ============================================================\n# 19. 연도별 수익률\n# ============================================================\n\ndef yearly_return(\n    curve\n):\n\n\n    returns = (\n\n        curve\n\n        .pct_change()\n\n        .fillna(\n            0.0\n        )\n    )\n\n\n    return (\n\n        (\n            1\n            + returns\n        )\n\n        .groupby(\n            returns.index.year\n        )\n\n        .prod()\n\n        - 1\n    )\n\n\nyearly = pd.concat(\n\n    [\n\n        yearly_return(\n            baseline_live\n        ),\n\n        yearly_return(\n            final_live\n        ),\n\n        yearly_return(\n            spy_live\n        ),\n    ],\n\n    axis=1\n)\n\n\nyearly.columns = [\n\n    "52W_FIXED",\n\n    "52W_ROTATION_B_FIXED",\n\n    "SPY"\n]\n\n\nprint(\n    "\\n연도별 수익률"\n)\n\n\ndisplay(\n\n    yearly.style.format(\n        "{:.2%}"\n    )\n)\n\n\n# ============================================================\n# 20. Exposure Audit\n# ============================================================\n\ndef exposure_audit(\n    name,\n    daily_df\n):\n\n\n    if (\n        daily_df.empty\n    ):\n\n        return\n\n\n    max_exposure = float(\n\n        daily_df[\n            "Invested_Exposure"\n        ].max()\n    )\n\n\n    min_cash = float(\n\n        daily_df[\n            "Cash_Exposure"\n        ].min()\n    )\n\n\n    over_count = int(\n\n        (\n            daily_df[\n                "Invested_Exposure"\n            ]\n\n            > 1.0 + 1e-8\n        ).sum()\n    )\n\n\n    print(\n        f"\\n[{name}]"\n    )\n\n\n    print(\n        f"최대 투자비중 : "\n        f"{max_exposure:.4%}"\n    )\n\n\n    print(\n        f"최소 현금비중 : "\n        f"{min_cash:.4%}"\n    )\n\n\n    print(\n        f"100% 초과 일수 : "\n        f"{over_count}일"\n    )\n\n\n    if (\n        over_count\n        == 0\n    ):\n\n        print(\n            "✅ 정상"\n        )\n\n    else:\n\n        print(\n            "❌ 오류"\n        )\n\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "★ EXPOSURE BUG AUDIT"\n)\n\nprint(\n    "=" * 120\n)\n\n\nexposure_audit(\n\n    "52W FIXED",\n\n    baseline_result[\n        "Daily_Holdings"\n    ].loc[\n        comparison_start:\n    ]\n)\n\n\nexposure_audit(\n\n    "52주 신고가 + Rotation_B FIXED",\n\n    final_result[\n        "Daily_Holdings"\n    ].loc[\n        comparison_start:\n    ]\n)\n\n\n# ============================================================\n# 21. Rotation_B Regime 비중\n# ============================================================\n\nregime_df = (\n\n    final_result[\n        "Daily_Holdings"\n    ]\n\n    .loc[\n        comparison_start:\n    ]\n)\n\n\nregime_counts = (\n\n    regime_df[\n        "Regime"\n    ]\n\n    .value_counts()\n\n    .rename(\n        "Days"\n    )\n\n    .to_frame()\n)\n\n\nregime_counts[\n    "Pct"\n] = (\n\n    regime_counts[\n        "Days"\n    ]\n\n    / regime_counts[\n        "Days"\n    ].sum()\n)\n\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "Rotation_B Regime 비중"\n)\n\nprint(\n    "=" * 120\n)\n\n\ndisplay(\n\n    regime_counts.style.format({\n\n        "Pct":\n            "{:.2%}"\n    })\n)\n\n\n# ============================================================\n# 22. 현재 FINAL Regime\n# ============================================================\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "★ 현재 FINAL Regime"\n)\n\nprint(\n    "=" * 120\n)\n\n\nprint(\n    "Yahoo 최신 기준일 :",\n    final_result[\n        "Live_Date"\n    ].date()\n)\n\n\nprint(\n\n    "52W 최근 20일 SPY 대비 초과수익 :",\n\n    (\n        f"{final_result[\'Live_Excess_20D\']:+.2%}"\n\n        if pd.notna(\n            final_result[\n                "Live_Excess_20D"\n            ]\n        )\n\n        else "NaN"\n    )\n)\n\n\nprint(\n\n    "현재 Regime :",\n\n    final_result[\n        "Live_Regime"\n    ]\n)\n\n\nprint(\n\n    "현재 신규매수 방식 :",\n\n    final_result[\n        "Live_Origin"\n    ]\n)\n\n\nprint(\n\n    "SELL TODAY :",\n\n    (\n        ", ".join(\n            final_result[\n                "Live_SELL"\n            ]\n        )\n\n        if final_result[\n            "Live_SELL"\n        ]\n\n        else "없음"\n    )\n)\n\n\nprint(\n\n    "BUY TODAY :",\n\n    (\n        ", ".join(\n            final_result[\n                "Live_BUY"\n            ]\n        )\n\n        if final_result[\n            "Live_BUY"\n        ]\n\n        else "없음"\n    )\n)\n\n\n# ============================================================\n# 23. 현재 FINAL 포트\n# ============================================================\n\nfinal_weights = (\n\n    final_result[\n        "Final_Weights"\n    ]\n\n    .loc[\n        lambda x:\n            x > EPS\n    ]\n\n    .sort_values(\n        ascending=False\n    )\n)\n\n\norigins = (\n    final_result[\n        "Origins"\n    ]\n)\n\n\ncurrent_rows = []\n\n\nfor ticker, weight in final_weights.items():\n\n\n    current_rows.append({\n\n        "Ticker":\n            ticker,\n\n        "Industry":\n            industry_etfs.get(\n                ticker,\n                ""\n            ),\n\n        "Origin":\n            origins.get(\n                ticker,\n                ""\n            ),\n\n        "Weight":\n            weight,\n    })\n\n\ncurrent_df = pd.DataFrame(\n    current_rows\n)\n\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "★ 현재 FINAL 포트폴리오"\n)\n\nprint(\n    "=" * 120\n)\n\n\nif (\n    current_df.empty\n):\n\n    print(\n        "보유종목 없음"\n    )\n\nelse:\n\n    display(\n\n        current_df.style.format({\n\n            "Weight":\n                "{:.2%}"\n        })\n    )\n\n\nprint(\n\n    "투자비중 :",\n\n    f"{final_weights.sum():.2%}"\n)\n\n\nprint(\n\n    "현금비중 :",\n\n    f"{max(0.0, 1.0-final_weights.sum()):.2%}"\n)\n\n\n# ============================================================\n# 24. 최근 거래 비교\n# ============================================================\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "최근 거래 - 52W FIXED"\n)\n\nprint(\n    "=" * 120\n)\n\n\ndisplay(\n\n    baseline_result[\n        "Trades"\n    ]\n\n    .tail(\n        30\n    )\n)\n\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "최근 거래 - 52주 신고가 + Rotation_B FIXED"\n)\n\nprint(\n    "=" * 120\n)\n\n\ndisplay(\n\n    final_result[\n        "Trades"\n    ]\n\n    .tail(\n        40\n    )\n)\n\n\n# ============================================================\n# 25. 최근 FINAL 보유현황\n# ============================================================\n\nprint(\n    "\\n"\n    + "=" * 120\n)\n\nprint(\n    "최근 30거래일 FINAL 포지션"\n)\n\nprint(\n    "=" * 120\n)\n\n\ndisplay(\n\n    final_result[\n        "Daily_Holdings"\n    ]\n\n    .loc[\n        comparison_start:\n    ]\n\n    .tail(\n        30\n    )\n\n    .style.format({\n\n        "52W_20D_Excess_vs_SPY":\n            "{:+.2%}",\n\n        "Invested_Exposure":\n            "{:.2%}",\n\n        "Cash_Exposure":\n            "{:.2%}",\n    })\n)\n\n\n# ============================================================\n# 26. Equity Curve\n# ============================================================\n\nplt.figure(\n    figsize=(\n        15,\n        7\n    )\n)\n\n\nplt.plot(\n\n    baseline_live.index,\n\n    baseline_live,\n\n    label="52W FIXED"\n)\n\n\nplt.plot(\n\n    final_live.index,\n\n    final_live,\n\n    label="52주 신고가 + Rotation_B FIXED"\n)\n\n\nplt.plot(\n\n    spy_live.index,\n\n    spy_live,\n\n    label="SPY"\n)\n\n\nplt.title(\n    "52W Fixed vs Rotation_B Fixed"\n)\n\n\nplt.ylabel(\n    "Growth of $1"\n)\n\n\nplt.grid(\n    alpha=0.3\n)\n\n\nplt.legend()\n\n\nplt.show()\n\n\n# ============================================================\n# 27. Drawdown 비교\n# ============================================================\n\nbaseline_dd = (\n\n    baseline_live\n\n    / baseline_live.cummax()\n\n    - 1\n)\n\n\nfinal_dd = (\n\n    final_live\n\n    / final_live.cummax()\n\n    - 1\n)\n\n\nspy_dd = (\n\n    spy_live\n\n    / spy_live.cummax()\n\n    - 1\n)\n\n\nplt.figure(\n    figsize=(\n        15,\n        6\n    )\n)\n\n\nplt.plot(\n\n    baseline_dd.index,\n\n    baseline_dd,\n\n    label="52W FIXED"\n)\n\n\nplt.plot(\n\n    final_dd.index,\n\n    final_dd,\n\n    label="52주 신고가 + Rotation_B FIXED"\n)\n\n\nplt.plot(\n\n    spy_dd.index,\n\n    spy_dd,\n\n    label="SPY"\n)\n\n\nplt.title(\n    "Drawdown Comparison"\n)\n\n\nplt.ylabel(\n    "Drawdown"\n)\n\n\nplt.grid(\n    alpha=0.3\n)\n\n\nplt.legend()\n\n\nplt.show()\n\n\nprint(\n    "\\n=== FINISHED ==="\n)'

def run_liquidity(): return run_source(LIQ_SRC)
def run_fg(): return run_source(FG_SRC)
def run_ai(): return run_source(AI_SRC)
def run_rotation(): return run_source(ROT_SRC)

@st.cache_data(ttl=3600,show_spinner=False)
def canary():
    raw=yf.download(["QQQ","TIP"],period="3y",auto_adjust=False,progress=False,threads=False)
    p=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
    vals={}
    for t in ["QQQ","TIP"]:
        s=p[t].dropna()
        vals[t]=float(np.mean([s.iloc[-1]/s.shift(n).iloc[-1]-1 for n in [21,63,126,252]]))
    return p.index[-1],vals,("공격 모드" if vals["QQQ"]>0 and vals["TIP"]>0 else "방어 모드")

@st.cache_data(ttl=3600,show_spinner=False)
def trend_backtest():
    tickers=["QQQ","SOXX","QLD","PSQ","GLD","SPMO","TIP"]
    today=datetime.today()
    first=today.replace(day=1)
    last_complete=first-timedelta(days=1)
    raw=yf.download(tickers,start="2010-01-01",end=last_complete.strftime("%Y-%m-%d"),
                    interval="1mo",auto_adjust=False,progress=False)
    p=raw["Close"].dropna(how="all")
    if isinstance(p.columns,pd.MultiIndex): p.columns=p.columns.droplevel(0)
    q=p["QQQ"].dropna(); tip=p["TIP"].dropna()
    qma=q.rolling(6).mean(); tma=tip.rolling(6).mean()
    up=(q>qma)&(tip>tma)
    d=up.dropna().index[-1]; is_up=bool(up.loc[d])
    alloc="QQQ 60% · QLD 15% · SOXX 25%" if is_up else "SPMO 50% · PSQ 50%"
    mr=p.pct_change().dropna()
    sig=up.shift(1).dropna()
    w=pd.DataFrame(0.0,index=mr.index,columns=mr.columns)
    for date in w.index:
        if date not in sig.index: continue
        if bool(sig.loc[date]):
            for t,x in [("QQQ",.60),("QLD",.15),("SOXX",.25)]:
                if t in w.columns:w.loc[date,t]=x
        else:
            for t,x in [("SPMO",.50),("PSQ",.50)]:
                if t in w.columns:w.loc[date,t]=x
    shifted=w.shift(1).dropna()
    pr=(shifted*mr).sum(axis=1)
    cum=(1+pr).cumprod()
    years=len(cum)/12
    cagr=float(cum.iloc[-1]**(1/years)-1)
    dd=cum/cum.cummax()-1; mdd=float(dd.min())
    qret=mr["QQQ"].reindex(cum.index).fillna(0)
    qcum=(1+qret).cumprod()
    return d,is_up,alloc,float(q.loc[d]),float(qma.loc[d]),float(tip.loc[d]),float(tma.loc[d]),cum,qcum,cagr,mdd


KR_SECTOR_SRC='# -*- coding: utf-8 -*-\n# ============================================================\n# ETF/섹터 분석 코드\n# - 6개월 Mansfield RS\n# - 6개월 변동성 조정 모멘텀\n# - 6개월 Sortino / ETR Comfort\n# - 업종 쏠림지수\n# - 최종 그래프: Sortino 10주선(50일선) 이상 섹터만\n#   30일 이격도 + 과거 247거래일 97% 초과 이격도 비교\n# ============================================================\n\n# ----------------------------\n# 1. 라이브러리 임포트\n# ----------------------------\nimport pandas as pd\nimport numpy as np\n\nimport io\n\nimport matplotlib.pyplot as plt\nimport matplotlib.ticker as mticker\nimport logging\n\n# Matplotlib 한글 폰트 경고 숨김: 그래프는 영문/숫자 중심으로 표기\nlogging.getLogger(\'matplotlib.font_manager\').setLevel(logging.ERROR)\nplt.rcParams[\'axes.unicode_minus\'] = False\n\n# ------------------------------------------------------------\n# 참고\n# 그래프에서 한글 폰트가 깨질 수 있으므로,\n# 최종 그래프의 제목/범례는 영어로 표시합니다.\n# 한글 설명은 DataFrame 표로 표시하므로 깨지지 않습니다.\n# ------------------------------------------------------------\n\n# ----------------------------\n# 2. 엑셀 업로드 및 불러오기\n# ----------------------------\ndf = pd.read_excel(io.BytesIO(EXCEL_BYTES), sheet_name="데이터")\n\n# ----------------------------\n# 3. 데이터 전처리\n# ----------------------------\ndf = df.dropna(subset=[\'DATE\'])\ndf[\'DATE\'] = pd.to_datetime(df[\'DATE\'])\ndf.set_index(\'DATE\', inplace=True)\n\nkospi = df[\'코스피\'].dropna()\netf_data = df.drop(columns=[\'코스피\'])\n\n# ----------------------------\n# 4. Mansfield RS 계산 함수\n# ----------------------------\ndef compute_mansfield_rs(etf, benchmark, ma_period=52):\n    """\n    etf와 benchmark(코스피)의 비율(relative)을 ma_period 이동평균으로 나눈 편차(%)를 Mansfield RS로 계산\n    Note: 일간 데이터 기준 ma_period=52면 52일 이동평균\n    """\n    relative = etf / benchmark\n    ma = relative.rolling(window=ma_period, min_periods=ma_period).mean()\n    rs = ((relative / ma) - 1) * 100\n    return rs\n\n\ndef normalize_to_100_scale(x, scale=12):\n    """\n    Mansfield RS(%)를 0~100 스케일로 변환 (시그모이드)\n    scale이 작을수록 민감도 증가\n    """\n    return 100 * (1 / (1 + np.exp(-x / scale)))\n\n# ----------------------------\n# 5. RS 계산 실행 (6개월 = 126거래일)\n# ----------------------------\nRS_WINDOWS = [126]\n\nmansfield_rows = []\n\nfor column in etf_data.columns:\n    series = etf_data[column].dropna()\n    common_idx = series.index.intersection(kospi.index)\n    etf_series = series.loc[common_idx]\n    kospi_series = kospi.loc[common_idx]\n\n    if len(etf_series) < min(RS_WINDOWS):\n        continue\n\n    latest = {}\n\n    for win in RS_WINDOWS:\n        if len(etf_series) >= win:\n            rs_series = compute_mansfield_rs(etf_series, kospi_series, ma_period=win).dropna()\n            if not rs_series.empty:\n                raw = float(rs_series.iloc[-1])\n                latest[f"RS_{win}d"] = round(raw, 2)\n                latest[f"Norm_RS_{win}d"] = round(normalize_to_100_scale(raw), 2)\n\n    raw_keys = [k for k in latest.keys() if k.startswith("RS_") and k.endswith("d")]\n    if len(raw_keys) == 0:\n        continue\n\n    raw_values = [latest[k] for k in raw_keys]\n    rs_avg = round(sum(raw_values) / len(raw_values), 2)\n    norm_rs_avg = round(normalize_to_100_scale(rs_avg), 2)\n\n    row = {\n        "ETF": column,\n        "RS_126d": np.nan,\n        "Norm_RS_126d": np.nan,\n        "RS_avg": rs_avg,\n        "Norm_RS_avg": norm_rs_avg,\n    }\n\n    for win in RS_WINDOWS:\n        if f"RS_{win}d" in latest:\n            row[f"RS_{win}d"] = latest[f"RS_{win}d"]\n            row[f"Norm_RS_{win}d"] = latest[f"Norm_RS_{win}d"]\n\n    mansfield_rows.append(row)\n\nmansfield_df = pd.DataFrame(mansfield_rows).sort_values(by="Norm_RS_avg", ascending=False)\n\n# ----------------------------\n# 6. 변동성 조정 모멘텀 계산 함수\n# ----------------------------\ndef calculate_risk_adjusted_momentum(data, last_n_days=10, ma_filter_days=None, top_n=20):\n    """\n    최근 6개월(126거래일) 누적수익률을\n    같은 기간의 일간수익률 표준편차로 나눈 6개월 위험조정 모멘텀 점수.\n\n    score = 6개월 누적수익률 / 6개월 일간수익률 표준편차\n\n    ma_filter_days가 주어지면 해당 이동평균 하회 종목 제외.\n    """\n    daily_top = {}\n    trading_days = data.dropna(how=\'all\').index[-last_n_days:]\n\n    for day in trading_days:\n        if day not in data.index:\n            continue\n\n        end_idx = data.index.get_indexer_for([day])[0]\n        start_6m = max(0, end_idx - 125)\n        window_6m = data.iloc[start_6m:end_idx + 1]\n\n        # 6개월 누적수익률\n        returns_6m = (\n            window_6m.iloc[-1] / window_6m.iloc[0] - 1\n        )\n\n        # 6개월 일간수익률 변동성\n        daily_returns_6m = window_6m.pct_change(fill_method=None)\n        vol_6m = daily_returns_6m.std(skipna=True)\n\n        valid = (\n            returns_6m.notna()\n            & vol_6m.notna()\n            & (vol_6m > 0)\n        )\n\n        if ma_filter_days is not None:\n            ma_ok = []\n            for col in data.columns:\n                series = data[col].iloc[:end_idx + 1].dropna()\n                if len(series) < ma_filter_days:\n                    ma_ok.append(False)\n                else:\n                    ma_value = series.rolling(\n                        window=ma_filter_days\n                    ).mean().iloc[-1]\n                    price_value = series.iloc[-1]\n                    ma_ok.append(price_value >= ma_value)\n\n            ma_ok = pd.Series(ma_ok, index=data.columns)\n            valid &= ma_ok\n\n        risk_adj = (\n            returns_6m[valid] / vol_6m[valid]\n        )\n\n        top_list = (\n            risk_adj\n            .sort_values(ascending=False)\n            .head(top_n)\n            .index\n            .tolist()\n        )\n\n        while len(top_list) < top_n:\n            top_list.append(None)\n\n        daily_top[day] = top_list\n\n    df_top = pd.DataFrame.from_dict(\n        daily_top,\n        orient="index",\n        columns=[f"Top {i+1}" for i in range(top_n)]\n    )\n    df_top.index.name = "Date"\n    return df_top\n\n# ============================================================\n# 7. Sortino / ETR Comfort 계산 함수\n# ============================================================\ndef _mdd_from_returns(ret: pd.Series) -> float:\n    cum = (1 + ret.fillna(0)).cumprod()\n    peak = cum.cummax()\n    dd = (cum - peak) / peak\n    return float(dd.min())\n\n\ndef _sortino_from_returns(ret: pd.Series, mar: float = 0.0, periods_per_year: int = 252) -> float:\n    ret = ret.dropna()\n    if len(ret) < 5:\n        return np.nan\n\n    mar_daily = (1 + mar) ** (1 / periods_per_year) - 1\n    excess = ret - mar_daily\n    downside = excess[excess < 0]\n\n    if len(downside) < 5:\n        return np.nan\n\n    downside_dev = downside.std(ddof=1) * np.sqrt(periods_per_year)\n    if downside_dev == 0 or np.isnan(downside_dev):\n        return np.nan\n\n    annual_excess = excess.mean() * periods_per_year\n    return float(annual_excess / downside_dev)\n\n\ndef _cagr_from_prices_window(prices: pd.Series, periods_per_year: int = 252) -> float:\n    prices = prices.dropna()\n    if len(prices) < 2:\n        return np.nan\n\n    n = len(prices)\n    return float((prices.iloc[-1] / prices.iloc[0]) ** (periods_per_year / (n - 1)) - 1)\n\n\ndef _etr_comfort_from_prices_window(prices: pd.Series, periods_per_year: int = 252) -> float:\n    prices = prices.dropna()\n    if len(prices) < 60:\n        return np.nan\n\n    ret = prices.pct_change(fill_method=None).dropna()\n    if len(ret) < 5:\n        return np.nan\n\n    cagr = _cagr_from_prices_window(prices, periods_per_year=periods_per_year)\n    vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year))\n    mdd = _mdd_from_returns(ret)\n    denom = abs(mdd) * vol\n\n    if denom == 0 or np.isnan(denom) or np.isnan(cagr):\n        return np.nan\n\n    return float(cagr / denom)\n\n\ndef _ma_ok_mask(close_df: pd.DataFrame, end_idx: int, ma_days: int) -> pd.Series:\n    ok = {}\n    for col in close_df.columns:\n        s = close_df[col].iloc[:end_idx + 1].dropna()\n        if len(s) < ma_days:\n            ok[col] = False\n        else:\n            ma = s.rolling(ma_days).mean().iloc[-1]\n            ok[col] = bool(s.iloc[-1] >= ma)\n    return pd.Series(ok)\n\n\ndef rank_sortino_or_etr_like_momentum(\n    data: pd.DataFrame,\n    metric: str = "sortino",\n    last_n_days: int = 10,\n    ma_filter_days: int = 10,\n    top_n: int = 20,\n    mar: float = 0.0,\n    periods_per_year: int = 252,\n    period_6m: int = 126,\n) -> pd.DataFrame:\n    """\n    최근 6개월(126거래일)만 사용해 Sortino 또는 ETR Comfort 순위를 계산합니다.\n    """\n\n    daily_top = {}\n    trading_days = data.dropna(how="all").index[-last_n_days:]\n\n    for day in trading_days:\n        end_idx = data.index.get_indexer_for([day])[0]\n        ok = _ma_ok_mask(\n            data,\n            end_idx=end_idx,\n            ma_days=ma_filter_days\n        )\n\n        scores = {}\n\n        for col in data.columns:\n            if not ok.get(col, False):\n                continue\n\n            start_idx = max(\n                0,\n                end_idx - (period_6m - 1)\n            )\n\n            w_prices = (\n                data[col]\n                .iloc[start_idx:end_idx + 1]\n                .dropna()\n            )\n\n            if len(w_prices) < max(\n                20,\n                int(period_6m * 0.8)\n            ):\n                continue\n\n            if metric == "sortino":\n                r = (\n                    w_prices\n                    .pct_change(fill_method=None)\n                    .dropna()\n                )\n                score = _sortino_from_returns(\n                    r,\n                    mar=mar,\n                    periods_per_year=periods_per_year\n                )\n\n            elif metric == "etr":\n                score = _etr_comfort_from_prices_window(\n                    w_prices,\n                    periods_per_year=periods_per_year\n                )\n\n            else:\n                raise ValueError(\n                    "metric은 \'sortino\' 또는 \'etr\'만 지원합니다."\n                )\n\n            if pd.notna(score):\n                scores[col] = float(score)\n\n        if len(scores) == 0:\n            daily_top[day] = [None] * top_n\n            continue\n\n        ranked = (\n            pd.Series(scores)\n            .sort_values(ascending=False)\n            .head(top_n)\n            .index\n            .tolist()\n        )\n\n        while len(ranked) < top_n:\n            ranked.append(None)\n\n        daily_top[day] = ranked\n\n    out = pd.DataFrame.from_dict(\n        daily_top,\n        orient="index",\n        columns=[f"Top {i+1}" for i in range(top_n)]\n    )\n    out.index.name = "Date"\n    return out\n\n# ----------------------------\n# 8. 기본 출력 테이블\n# ----------------------------\nmomentum_df = calculate_risk_adjusted_momentum(etf_data, last_n_days=10, ma_filter_days=None, top_n=20)\nprint("\\n📌 최근 10일간 변동성 조정 모멘텀 Top 20 ETF (전체):")\ndisplay(momentum_df)\n\nsortino_ma10 = rank_sortino_or_etr_like_momentum(etf_data, metric="sortino", last_n_days=10, ma_filter_days=10, top_n=20)\nprint("\\n📌 최근 10일간 Sortino Top 20 ETF (10일선 이상 / 6M):")\ndisplay(sortino_ma10)\n\nsortino_ma20 = rank_sortino_or_etr_like_momentum(etf_data, metric="sortino", last_n_days=10, ma_filter_days=20, top_n=20)\nprint("\\n📌 최근 10일간 Sortino Top 20 ETF (20일선 이상 / 6M):")\ndisplay(sortino_ma20)\n\nsortino_ma10w = rank_sortino_or_etr_like_momentum(etf_data, metric="sortino", last_n_days=10, ma_filter_days=50, top_n=20)\nprint("\\n📌 최근 10일간 Sortino Top 20 ETF (10주선(50일) 이상 / 6M):")\ndisplay(sortino_ma10w)\n\netr_ma10 = rank_sortino_or_etr_like_momentum(etf_data, metric="etr", last_n_days=10, ma_filter_days=10, top_n=20)\nprint("\\n📌 최근 10일간 ETR Comfort Top 20 ETF (10일선 이상 / 6M):")\ndisplay(etr_ma10)\n\netr_ma20 = rank_sortino_or_etr_like_momentum(etf_data, metric="etr", last_n_days=10, ma_filter_days=20, top_n=20)\nprint("\\n📌 최근 10일간 ETR Comfort Top 20 ETF (20일선 이상 / 6M):")\ndisplay(etr_ma20)\n\netr_ma10w = rank_sortino_or_etr_like_momentum(etf_data, metric="etr", last_n_days=10, ma_filter_days=50, top_n=20)\nprint("\\n📌 최근 10일간 ETR Comfort Top 20 ETF (10주선(50일) 이상 / 6M):")\ndisplay(etr_ma10w)\n\nprint("\\n📌 Mansfield RS (126일(6개월)) 결과 (Norm_RS_avg 내림차순):")\ndisplay(mansfield_df)\n\n\n# ------------------------------------------------------------\n# 6개월 절대수익률 + 6개월 KOSPI 대비 상대수익률 표\n# ------------------------------------------------------------\nLOOKBACK_6M = 126\n\nreturn_6m = (\n    etf_data / etf_data.shift(LOOKBACK_6M) - 1\n)\n\nkospi_return_6m = (\n    kospi / kospi.shift(LOOKBACK_6M) - 1\n)\n\nrelative_return_6m = return_6m.sub(\n    kospi_return_6m,\n    axis=0\n)\n\nlatest_6m_date = return_6m.dropna(how="all").index[-1]\n\nsix_month_summary = pd.DataFrame({\n    "6M_Return": return_6m.loc[latest_6m_date],\n    "6M_RS_vs_KOSPI": relative_return_6m.loc[latest_6m_date],\n}).dropna()\n\nsix_month_summary = six_month_summary.sort_values(\n    "6M_RS_vs_KOSPI",\n    ascending=False\n)\n\nprint(\n    f"\\n📌 기준일 {latest_6m_date.date()} "\n    f"6개월 수익률 및 KOSPI 대비 RS"\n)\n\ndisplay(\n    six_month_summary.style.format({\n        "6M_Return": "{:.2%}",\n        "6M_RS_vs_KOSPI": "{:+.2%}",\n    })\n)\n\nmansfield_df_70 = mansfield_df[mansfield_df["Norm_RS_avg"] >= 70].copy()\nprint("\\n📌 Normalized RS 평균 ≥ 70 (필터링):")\ndisplay(mansfield_df_70)\n\n# ============================================================\n# 9. 업종(테마) 쏠림지수\n# ============================================================\ndef compute_crowding_index(\n    price_df: pd.DataFrame,\n    lookback_days: int = 126,\n    min_universe: int = 10,\n    score_mode: str = "linear",\n    scale: float = 100.0\n):\n    ret6m = price_df.pct_change(lookback_days, fill_method=None)\n\n    crowding_vals = []\n    valid_dates = ret6m.index\n\n    for dt in valid_dates:\n        r = ret6m.loc[dt].dropna()\n        n = len(r)\n        if n < min_universe:\n            crowding_vals.append(np.nan)\n            continue\n\n        rank = r.rank(ascending=False, method="average")\n\n        if score_mode == "linear":\n            score = (n - rank) / (n - 1)\n        elif score_mode == "centered":\n            score = ((n + 1) / 2 - rank) / ((n - 1) / 2)\n        else:\n            raise ValueError("score_mode는 \'linear\' 또는 \'centered\'만 지원합니다.")\n\n        weighted = score * r\n        crowd = float(weighted.std(ddof=1) * scale)\n        crowding_vals.append(crowd)\n\n    crowding_series = pd.Series(crowding_vals, index=valid_dates, name="CrowdingIndex").dropna()\n    return crowding_series, ret6m\n\n\ncrowding_series, ret6m = compute_crowding_index(\n    etf_data,\n    lookback_days=126,\n    min_universe=10,\n    score_mode="centered",\n    scale=100.0\n)\n\nprint("\\n📌 업종(테마) 쏠림지수(6개월 수익률 기반) 최신값:")\nlatest_dt = crowding_series.index[-1]\nlatest_crowd = crowding_series.iloc[-1]\nprint(f"- Date: {latest_dt.date()}  |  CrowdingIndex: {latest_crowd:.2f}")\nprint("  (참고: 35 이상을 과열/쏠림 신호로 참고)")\ndisplay(crowding_series.tail(20))\n\n\ndef show_leaders_laggards(ret6m: pd.DataFrame, date: pd.Timestamp, top_k: int = 10):\n    r = ret6m.loc[date].dropna().sort_values(ascending=False)\n\n    top = r.head(top_k)\n    bottom = r.tail(top_k).sort_values(ascending=True)\n\n    print(f"\\n📈 주도 업종 Top {top_k} (6개월 수익률)")\n    display(top.to_frame("6M Return"))\n\n    print(f"\\n📉 소외 업종 Bottom {top_k} (6개월 수익률)")\n    display(bottom.to_frame("6M Return"))\n\n\nprint("\\n📌 최신일 기준 6개월 수익률 주도/소외 업종:")\nshow_leaders_laggards(ret6m, latest_dt, top_k=10)\n\nthreshold = 35.0\nsignal = "🔥 과열/쏠림" if latest_crowd >= threshold else "✅ 비과열/확산"\nprint(f"\\n📌 쏠림 판정: {signal} (threshold={threshold}, value={latest_crowd:.2f})")\n# ============================================================\n# 10. ETF/섹터 Breadth\n#     전체 ETF 중 이동평균선 위에 있는 종목 비율\n#\n#     Above MA10  : 10일선 위 종목 비율\n#     Above MA20  : 20일선 위 종목 비율\n#     Above MA50  : 10주선(50일선) 위 종목 비율\n#     Above MA120 : 120일선 위 종목 비율  ★ 역추세 장중 체크용\n# ============================================================\n\ndef compute_ma_breadth(\n    price_df: pd.DataFrame,\n    ma_windows=(10, 20, 50, 120),\n    min_valid_assets=10,\n    smooth_days=3\n):\n    """\n    날짜별로 전체 ETF 중 각 이동평균선 위에 있는 종목 비율을 계산합니다.\n    """\n\n    price_df = price_df.sort_index().copy()\n    breadth_df = pd.DataFrame(index=price_df.index)\n\n    for window in ma_windows:\n\n        moving_average = price_df.rolling(\n            window=window,\n            min_periods=window\n        ).mean()\n\n        valid_mask = (\n            price_df.notna() &\n            moving_average.notna()\n        )\n\n        valid_count = valid_mask.sum(axis=1)\n\n        above_count = (\n            (price_df >= moving_average) &\n            valid_mask\n        ).sum(axis=1)\n\n        breadth_ratio = (\n            above_count /\n            valid_count.replace(0, np.nan) *\n            100\n        )\n\n        breadth_ratio = breadth_ratio.where(\n            valid_count >= min_valid_assets\n        )\n\n        breadth_df[f"Above_MA{window}"] = breadth_ratio\n        breadth_df[f"Count_Above_MA{window}"] = above_count\n        breadth_df[f"Valid_Count_MA{window}"] = valid_count\n\n        # 하루 단위 휩쏘 완화\n        breadth_df[f"Above_MA{window}_Smooth"] = (\n            breadth_ratio\n            .rolling(\n                smooth_days,\n                min_periods=1\n            )\n            .mean()\n        )\n\n    return breadth_df\n\n\n# ------------------------------------------------------------\n# Breadth 계산\n# ------------------------------------------------------------\nsector_breadth = compute_ma_breadth(\n    price_df=etf_data,\n    ma_windows=(10, 20, 50, 120),\n    min_valid_assets=10,\n    smooth_days=3\n)\n\n\n# ------------------------------------------------------------\n# 종합 Breadth Score\n#\n# 10일선 35%\n# 20일선 35%\n# 10주선 30%\n# ------------------------------------------------------------\nsector_breadth["Breadth_Score"] = (\n    sector_breadth["Above_MA10_Smooth"] * 0.35 +\n    sector_breadth["Above_MA20_Smooth"] * 0.35 +\n    sector_breadth["Above_MA50_Smooth"] * 0.30\n)\n\nsector_breadth["Breadth_Score_MA5"] = (\n    sector_breadth["Breadth_Score"]\n    .rolling(5, min_periods=1)\n    .mean()\n)\n\nsector_breadth["Breadth_Slope_5D"] = (\n    sector_breadth["Breadth_Score_MA5"] -\n    sector_breadth["Breadth_Score_MA5"].shift(5)\n)\n\nsector_breadth["Breadth_Slope_10D"] = (\n    sector_breadth["Breadth_Score_MA5"] -\n    sector_breadth["Breadth_Score_MA5"].shift(10)\n)\n\n\n# ------------------------------------------------------------\n# 최신 Breadth 요약표\n# ------------------------------------------------------------\nvalid_breadth = sector_breadth.dropna(\n    subset=["Breadth_Score_MA5"]\n)\n\nlatest_breadth = valid_breadth.iloc[-1]\nlatest_breadth_date = valid_breadth.index[-1]\n\nbreadth_summary = pd.DataFrame({\n    "현재값": {\n        "10일선 위 종목 비율": latest_breadth["Above_MA10"],\n        "20일선 위 종목 비율": latest_breadth["Above_MA20"],\n        "10주선(50일선) 위 종목 비율": latest_breadth["Above_MA50"],\n        "120일선 위 종목 비율": latest_breadth["Above_MA120"],\n        "종합 Breadth Score": latest_breadth["Breadth_Score_MA5"],\n        "Breadth 5일 변화": latest_breadth["Breadth_Slope_5D"],\n        "Breadth 10일 변화": latest_breadth["Breadth_Slope_10D"],\n    }\n})\n\nprint(f"\\n📌 ETF/섹터 Breadth 기준일: {latest_breadth_date.date()}")\n\ndisplay(\n    breadth_summary.style.format("{:.1f}")\n)\n\n\n# ------------------------------------------------------------\n# 최신 이동평균선 위 종목 수\n# ------------------------------------------------------------\nlatest_count_table = pd.DataFrame({\n    "이동평균": [\n        "MA10 (10일선)",\n        "MA20 (20일선)",\n        "MA50 (10주선)",\n        "MA120 (120일선) ★ 역추세용"\n    ],\n    "이평선 위 종목 수": [\n        int(latest_breadth["Count_Above_MA10"]),\n        int(latest_breadth["Count_Above_MA20"]),\n        int(latest_breadth["Count_Above_MA50"]),\n        int(latest_breadth["Count_Above_MA120"]),\n    ],\n    "분석 가능 종목 수": [\n        int(latest_breadth["Valid_Count_MA10"]),\n        int(latest_breadth["Valid_Count_MA20"]),\n        int(latest_breadth["Valid_Count_MA50"]),\n        int(latest_breadth["Valid_Count_MA120"]),\n    ],\n    "Breadth 비율": [\n        latest_breadth["Above_MA10"],\n        latest_breadth["Above_MA20"],\n        latest_breadth["Above_MA50"],\n        latest_breadth["Above_MA120"],\n    ]\n})\n\nprint("\\n📌 현재 이동평균선 위 ETF/섹터 수")\n\ndisplay(\n    latest_count_table.style.format({\n        "Breadth 비율": "{:.1f}%"\n    })\n)\n\n\n# ------------------------------------------------------------\n# 그래프 1: 10일선·20일선·10주선 Breadth\n# ------------------------------------------------------------\nplt.figure(figsize=(16, 6))\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Above_MA10_Smooth"],\n    linewidth=1.8,\n    label="Above MA10"\n)\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Above_MA20_Smooth"],\n    linewidth=2.0,\n    label="Above MA20"\n)\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Above_MA50_Smooth"],\n    linewidth=2.3,\n    label="Above MA50 (10 Weeks)"\n)\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Above_MA120_Smooth"],\n    linewidth=2.5,\n    label="Above MA120 (Reverse)"\n)\n\nplt.axhline(70, linestyle="--", alpha=0.5)\nplt.axhline(50, linestyle="--", alpha=0.5)\nplt.axhline(30, linestyle="--", alpha=0.5)\n\nplt.title("ETF Sector Moving Average Breadth")\nplt.ylabel("Sectors Above Moving Average (%)")\nplt.ylim(0, 100)\nplt.legend(loc="best")\nplt.grid(alpha=0.25)\n\nplt.tight_layout()\nplt.show()\n\n\n# ------------------------------------------------------------\n# 그래프 2: 종합 Breadth Score\n# ------------------------------------------------------------\nplt.figure(figsize=(16, 6))\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Breadth_Score"],\n    alpha=0.35,\n    linewidth=1.2,\n    label="Daily Breadth Score"\n)\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Breadth_Score_MA5"],\n    linewidth=2.4,\n    label="Breadth Score MA5"\n)\n\nplt.axhline(70, linestyle="--", alpha=0.5)\nplt.axhline(60, linestyle="--", alpha=0.5)\nplt.axhline(50, linestyle="--", alpha=0.5)\nplt.axhline(35, linestyle="--", alpha=0.5)\n\nplt.title("ETF Sector Breadth Score")\nplt.ylabel("Breadth Score")\nplt.ylim(0, 100)\nplt.legend(loc="best")\nplt.grid(alpha=0.25)\n\nplt.tight_layout()\nplt.show()\n\n\n# ------------------------------------------------------------\n# 그래프 3: Breadth 기울기\n# ------------------------------------------------------------\nplt.figure(figsize=(16, 5))\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Breadth_Slope_5D"],\n    linewidth=1.8,\n    label="5D Breadth Slope"\n)\n\nplt.plot(\n    sector_breadth.index,\n    sector_breadth["Breadth_Slope_10D"],\n    linewidth=1.8,\n    label="10D Breadth Slope"\n)\n\nplt.axhline(0, linewidth=1)\nplt.axhline(10, linestyle="--", alpha=0.5)\nplt.axhline(-10, linestyle="--", alpha=0.5)\n\nplt.title("ETF Sector Breadth Slope")\nplt.ylabel("Score Change")\nplt.legend(loc="best")\nplt.grid(alpha=0.25)\n\nplt.tight_layout()\nplt.show()\n\n\n# ------------------------------------------------------------\n# 최근 20거래일 Breadth 변화표\n# ------------------------------------------------------------\nrecent_breadth_cols = [\n    "Above_MA10",\n    "Above_MA20",\n    "Above_MA50",\n    "Above_MA120",\n    "Breadth_Score_MA5",\n    "Breadth_Slope_5D",\n    "Breadth_Slope_10D"\n]\n\nprint("\\n📌 최근 20거래일 ETF/섹터 Breadth")\n\ndisplay(\n    sector_breadth[recent_breadth_cols]\n    .tail(20)\n    .style\n    .format({\n        "Above_MA10": "{:.1f}",\n        "Above_MA20": "{:.1f}",\n        "Above_MA50": "{:.1f}",\n        "Above_MA120": "{:.1f}",\n        "Breadth_Score_MA5": "{:.1f}",\n        "Breadth_Slope_5D": "{:+.1f}",\n        "Breadth_Slope_10D": "{:+.1f}"\n    })\n)\n\n# ============================================================\n# 10-1. 역추세용 120일선 ETF Breadth 장중 모니터\n# ============================================================\n#\n# 최신 행이 장중 가격을 포함하고 있다면,\n# 아래 값도 장중 현재가격을 포함한 MA120 Breadth입니다.\n#\n# 역추세 기준 참고:\n# - Above_MA120 < 30% : 시장 내부가 매우 약한 구간\n# - 단독 매수 신호가 아니라 KOSPI 252일 DD 조건과 함께 확인\n# ============================================================\n\nlatest_ma120_breadth = float(latest_breadth["Above_MA120"])\nlatest_ma120_count = int(latest_breadth["Count_Above_MA120"])\nlatest_ma120_valid = int(latest_breadth["Valid_Count_MA120"])\n\nif latest_ma120_breadth < 30:\n    reverse_breadth_signal = "🔥 역추세 조건 충족 (MA120 Breadth < 30%)"\nelif latest_ma120_breadth < 40:\n    reverse_breadth_signal = "⚠️ 역추세 접근 구간 (30~40%)"\nelse:\n    reverse_breadth_signal = "대기"\n\nprint("\\n" + "=" * 90)\nprint("📌 역추세용 120일선 ETF Breadth 장중 모니터")\nprint("=" * 90)\nprint(f"기준일: {latest_breadth_date.date()}")\nprint(f"MA120 위 ETF: {latest_ma120_count}/{latest_ma120_valid}개")\nprint(f"MA120 Breadth: {latest_ma120_breadth:.1f}%")\nprint(f"판정: {reverse_breadth_signal}")\nprint("※ 실제 역추세 매수는 KOSPI 252일 DD 조건과 함께 확인")\n\n# ============================================================\n# 11. 최종 그래프\n#     Sortino 10주선(50일선) 이상 섹터만 출력\n#     30일 이격도 + 과거 247거래일 97% 초과 이격도 비교\n#\n#     구성:\n#     1) 먼저 전체 요약표 1개 출력\n#     2) 그 다음 섹터별 가로형 그래프 출력\n#     3) 그래프에는 한글을 최소화하여 폰트 깨짐 방지\n# ============================================================\n\nLOOKBACK_DAYS = 247    # 최근 247거래일 기준\nGAP_MA_DAYS = 30      # 30일 이격도\nGRAPH_YLIM = (90, 160)\nBAR_ALPHA = 0.15\n\n\ndef make_sortino_10w_gap30_summary(price_df: pd.DataFrame, sortino_top_df: pd.DataFrame) -> pd.DataFrame:\n    """\n    Sortino 10주선(50일선) 이상 섹터만 대상으로\n    30일 이격도와 과거 247거래일 97% 초과 이격도 수준을 계산합니다.\n\n    표시용 컬럼은 % 기호를 붙여 눈으로 보기 쉽게 만들고,\n    그래프 계산용 숫자 컬럼은 앞에 \'_\'를 붙여 내부용으로 보관합니다.\n    """\n    latest_sortino_list = sortino_top_df.iloc[-1].dropna().tolist()\n\n    summary_rows = []\n\n    for rank_no, sector in enumerate(latest_sortino_list, start=1):\n        if sector not in price_df.columns:\n            continue\n\n        price = price_df[sector].dropna()\n        if len(price) < GAP_MA_DAYS + 5:\n            continue\n\n        ma30 = price.rolling(GAP_MA_DAYS).mean()\n        gap30 = price / ma30 * 100\n        gap30 = gap30.dropna()\n\n        if gap30.empty:\n            continue\n\n        recent_gap = gap30.tail(LOOKBACK_DAYS)\n\n        current_price = float(price.loc[gap30.index[-1]])\n        current_ma30 = float(ma30.loc[gap30.index[-1]])\n        current_gap = float(gap30.iloc[-1])\n        current_gap_excess = current_gap - 100\n\n        gap97 = float(recent_gap.quantile(0.97))\n        gap97_excess = gap97 - 100\n\n        # 현재 위치: 30일선 초과분 기준으로 비교\n        # 예: 현재 초과분 12, 247거래일 97% 초과분 10이면 120%\n        excess_heat_ratio = (\n            current_gap_excess / gap97_excess * 100\n            if gap97_excess != 0\n            else np.nan\n        )\n\n        기준일 = gap30.index[-1].strftime("%Y-%m-%d")\n\n        summary_rows.append({\n            "Sortino순위": rank_no,\n            "섹터": sector,\n            "종가": f"{current_price:,.0f}",\n            "MA30": f"{current_ma30:,.0f}",\n            "현재30일선초과이격도": round(current_gap_excess, 2),\n            "247거래일97%초과이격도": round(gap97_excess, 2),\n            "97%구간대비현재위치": f"{excess_heat_ratio:.1f}%",\n            "기준일": 기준일,\n            # 그래프 제목용 내부 숫자 컬럼\n            "_현재30일선초과이격도": round(current_gap_excess, 2),\n            "_247거래일97%초과이격도": round(gap97_excess, 2),\n            "_97%구간대비현재위치": round(excess_heat_ratio, 1),\n        })\n\n    summary_df = pd.DataFrame(summary_rows)\n    return summary_df\n\n\ndef plot_gap30_charts_after_summary(price_df: pd.DataFrame, summary_df: pd.DataFrame):\n    """\n    표는 한글로 출력하고, 그래프 내부 텍스트는 영문/숫자 중심으로 표시합니다.\n    그래서 NanumGothic 폰트가 없어도 경고가 나오지 않습니다.\n    """\n    for _, row in summary_df.iterrows():\n        sector = row["섹터"]\n        rank_no = int(row["Sortino순위"])\n\n        if sector not in price_df.columns:\n            continue\n\n        price = price_df[sector].dropna()\n        ma30 = price.rolling(GAP_MA_DAYS).mean()\n        gap30 = price / ma30 * 100\n\n        plot_df = pd.DataFrame({\n            "Price": price,\n            "MA30": ma30,\n            "Gap30": gap30,\n        }).dropna()\n\n        if plot_df.empty:\n            continue\n\n        fig, ax1 = plt.subplots(figsize=(16, 4))\n        ax2 = ax1.twinx()\n\n        # 이격도 막대: 뒤쪽에 연하게 표시\n        ax2.bar(\n            plot_df.index,\n            plot_df["Gap30"],\n            color="#FFC000",\n            alpha=BAR_ALPHA,\n            width=1.5,\n            label="Gap30",\n            zorder=1,\n        )\n        ax2.set_ylim(GRAPH_YLIM)\n\n        # 왼쪽 축을 앞으로 가져오기\n        ax1.set_zorder(ax2.get_zorder() + 1)\n        ax1.patch.set_visible(False)\n\n        # 가격과 MA30\n        ax1.plot(\n            plot_df.index,\n            plot_df["Price"],\n            color="black",\n            linewidth=2.8,\n            label="Price",\n            zorder=10,\n        )\n\n        ax1.plot(\n            plot_df.index,\n            plot_df["MA30"],\n            color="#8FA3BD",\n            linewidth=3.0,\n            label="MA30",\n            zorder=9,\n        )\n\n        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))\n        ax1.grid(True, axis="both", alpha=0.25, zorder=0)\n\n        # 그래프 제목은 영문/숫자 중심으로 표시해서 한글 폰트 깨짐 및 font warning 방지\n        ax1.set_title(\n            f"Sortino Rank #{rank_no} | Gap30 Excess={row[\'_현재30일선초과이격도\']:.1f} | P97 Excess={row[\'_247거래일97%초과이격도\']:.1f} | Heat={row[\'_97%구간대비현재위치\']:.1f}%",\n            fontsize=13,\n            fontweight="bold",\n        )\n\n        lines1, labels1 = ax1.get_legend_handles_labels()\n        lines2, labels2 = ax2.get_legend_handles_labels()\n\n        ax1.legend(\n            lines2 + lines1,\n            labels2 + labels1,\n            loc="upper center",\n            bbox_to_anchor=(0.5, 1.18),\n            ncol=3,\n            frameon=False,\n            fontsize=11,\n        )\n\n        plt.tight_layout()\n        plt.show()\n\n\n# ------------------------------------------------------------\n# 최종 실행\n# ------------------------------------------------------------\nsummary_df = make_sortino_10w_gap30_summary(\n    price_df=etf_data,\n    sortino_top_df=sortino_ma10w,\n)\n\nprint("\\n📌 Sortino 10주선(50일선) 이상 섹터: 30일선 초과 이격도 / 247거래일 97% 구간 비교")\ndisplay_cols = [c for c in summary_df.columns if not c.startswith("_")]\ndisplay(summary_df[display_cols])\n\nplot_gap30_charts_after_summary(\n    price_df=etf_data,\n    summary_df=summary_df,\n)\n\n# ============================================================\n# 12. 4주 / 13주 / 26주 / 52주 이동평균 정배열 종목군\n#\n# 일봉 데이터 기준:\n# 4주  = 20거래일\n# 13주 = 65거래일\n# 26주 = 130거래일\n# 52주 = 260거래일\n#\n# 완전 정배열:\n# 현재가 > MA20 > MA65 > MA130 > MA260\n#\n# 이평 정배열:\n# MA20 > MA65 > MA130 > MA260\n#\n# 추가:\n# - 정배열 시작일\n# - 현재까지 연속 유지 거래일\n# - 달력일 기준 유지기간\n# ============================================================\n\nMA_4W = 20\nMA_13W = 65\nMA_26W = 130\nMA_52W = 260\n\n\ndef _current_true_streak(mask: pd.Series):\n    """\n    최신일 기준 True가 연속으로 이어지는 거래일 수와 시작일 계산\n    """\n    mask = mask.dropna().astype(bool)\n\n    if mask.empty or not bool(mask.iloc[-1]):\n        return 0, pd.NaT\n\n    arr = mask.to_numpy()\n    false_positions = np.flatnonzero(~arr)\n\n    if len(false_positions) == 0:\n        start_pos = 0\n    else:\n        start_pos = false_positions[-1] + 1\n\n    streak_days = len(mask) - start_pos\n    start_date = mask.index[start_pos]\n\n    return int(streak_days), start_date\n\n\ndef get_weekly_ma_alignment(price_df: pd.DataFrame) -> pd.DataFrame:\n    """\n    각 ETF/섹터별 최신 유효 시점에서\n    4주(20일), 13주(65일), 26주(130일), 52주(260일)\n    이동평균 정배열 여부와 현재 연속 유지기간을 계산합니다.\n    """\n\n    price_df = price_df.sort_index().copy()\n\n    ma4 = price_df.rolling(\n        window=MA_4W,\n        min_periods=MA_4W\n    ).mean()\n\n    ma13 = price_df.rolling(\n        window=MA_13W,\n        min_periods=MA_13W\n    ).mean()\n\n    ma26 = price_df.rolling(\n        window=MA_26W,\n        min_periods=MA_26W\n    ).mean()\n\n    ma52 = price_df.rolling(\n        window=MA_52W,\n        min_periods=MA_52W\n    ).mean()\n\n    result_rows = []\n\n    for col in price_df.columns:\n\n        temp = pd.DataFrame({\n            "Price": price_df[col],\n            "MA4W": ma4[col],\n            "MA13W": ma13[col],\n            "MA26W": ma26[col],\n            "MA52W": ma52[col],\n        }).dropna()\n\n        if temp.empty:\n            continue\n\n        # 날짜별 이평 정배열\n        ma_mask = (\n            (temp["MA4W"] > temp["MA13W"])\n            & (temp["MA13W"] > temp["MA26W"])\n            & (temp["MA26W"] > temp["MA52W"])\n        )\n\n        # 날짜별 완전 정배열\n        full_mask = (\n            (temp["Price"] > temp["MA4W"])\n            & ma_mask\n        )\n\n        latest_date = temp.index[-1]\n        row = temp.iloc[-1]\n\n        ma_alignment = bool(ma_mask.iloc[-1])\n        full_alignment = bool(full_mask.iloc[-1])\n\n        ma_streak_days, ma_start_date = _current_true_streak(ma_mask)\n        full_streak_days, full_start_date = _current_true_streak(full_mask)\n\n        ma_calendar_days = (\n            (latest_date - ma_start_date).days + 1\n            if ma_alignment and pd.notna(ma_start_date)\n            else 0\n        )\n\n        full_calendar_days = (\n            (latest_date - full_start_date).days + 1\n            if full_alignment and pd.notna(full_start_date)\n            else 0\n        )\n\n        # 정배열 간격\n        gap_price_4w = (row["Price"] / row["MA4W"] - 1) * 100\n        gap_4w_13w = (row["MA4W"] / row["MA13W"] - 1) * 100\n        gap_13w_26w = (row["MA13W"] / row["MA26W"] - 1) * 100\n        gap_26w_52w = (row["MA26W"] / row["MA52W"] - 1) * 100\n\n        alignment_strength = (\n            gap_price_4w\n            + gap_4w_13w\n            + gap_13w_26w\n            + gap_26w_52w\n        )\n\n        result_rows.append({\n            "ETF": col,\n            "기준일": latest_date.date(),\n\n            "현재가": float(row["Price"]),\n            "MA4주": float(row["MA4W"]),\n            "MA13주": float(row["MA13W"]),\n            "MA26주": float(row["MA26W"]),\n            "MA52주": float(row["MA52W"]),\n\n            "현재가>4주": bool(row["Price"] > row["MA4W"]),\n            "4주>13주": bool(row["MA4W"] > row["MA13W"]),\n            "13주>26주": bool(row["MA13W"] > row["MA26W"]),\n            "26주>52주": bool(row["MA26W"] > row["MA52W"]),\n\n            "이평정배열": ma_alignment,\n            "완전정배열": full_alignment,\n\n            "이평정배열_시작일": (\n                ma_start_date.date()\n                if ma_alignment and pd.notna(ma_start_date)\n                else None\n            ),\n            "이평정배열_유지거래일": ma_streak_days,\n            "이평정배열_달력일": ma_calendar_days,\n\n            "완전정배열_시작일": (\n                full_start_date.date()\n                if full_alignment and pd.notna(full_start_date)\n                else None\n            ),\n            "완전정배열_유지거래일": full_streak_days,\n            "완전정배열_달력일": full_calendar_days,\n\n            "현재가_4주이격(%)": float(gap_price_4w),\n            "4주_13주이격(%)": float(gap_4w_13w),\n            "13주_26주이격(%)": float(gap_13w_26w),\n            "26주_52주이격(%)": float(gap_26w_52w),\n            "정배열강도": float(alignment_strength),\n        })\n\n    result_df = pd.DataFrame(result_rows)\n\n    if result_df.empty:\n        return result_df\n\n    return (\n        result_df\n        .sort_values(\n            by=[\n                "완전정배열",\n                "완전정배열_유지거래일",\n                "정배열강도"\n            ],\n            ascending=[False, False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n\n# ------------------------------------------------------------\n# 정배열 계산\n# ------------------------------------------------------------\n\nweekly_alignment_df = get_weekly_ma_alignment(etf_data)\n\n\nif weekly_alignment_df.empty:\n\n    print(\n        "\\n⚠️ 4주·13주·26주·52주 정배열을 계산할 수 있는 "\n        "충분한 데이터가 없습니다."\n    )\n\nelse:\n\n    # --------------------------------------------------------\n    # ① 완전 정배열\n    # --------------------------------------------------------\n\n    full_alignment_df = (\n        weekly_alignment_df[\n            weekly_alignment_df["완전정배열"]\n        ]\n        .copy()\n        .sort_values(\n            by=[\n                "완전정배열_유지거래일",\n                "정배열강도"\n            ],\n            ascending=[False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n    full_alignment_df.index += 1\n    full_alignment_df.index.name = "순위"\n\n    print(\n        "\\n📌 4주·13주·26주·52주 완전 정배열"\n        "\\n   현재가 > 4주선 > 13주선 > 26주선 > 52주선"\n        "\\n   ※ 정배열 시작일과 연속 유지기간 포함"\n    )\n\n    if full_alignment_df.empty:\n        print("해당 종목이 없습니다.")\n    else:\n        display(\n            full_alignment_df[\n                [\n                    "ETF",\n                    "기준일",\n                    "완전정배열_시작일",\n                    "완전정배열_유지거래일",\n                    "완전정배열_달력일",\n                    "현재가",\n                    "MA4주",\n                    "MA13주",\n                    "MA26주",\n                    "MA52주",\n                    "현재가_4주이격(%)",\n                    "4주_13주이격(%)",\n                    "13주_26주이격(%)",\n                    "26주_52주이격(%)",\n                    "정배열강도",\n                ]\n            ].style.format({\n                "현재가": "{:,.2f}",\n                "MA4주": "{:,.2f}",\n                "MA13주": "{:,.2f}",\n                "MA26주": "{:,.2f}",\n                "MA52주": "{:,.2f}",\n                "현재가_4주이격(%)": "{:+.2f}%",\n                "4주_13주이격(%)": "{:+.2f}%",\n                "13주_26주이격(%)": "{:+.2f}%",\n                "26주_52주이격(%)": "{:+.2f}%",\n                "정배열강도": "{:.2f}",\n            })\n        )\n\n\n    # --------------------------------------------------------\n    # ② 이동평균 정배열\n    # --------------------------------------------------------\n\n    ma_alignment_df = (\n        weekly_alignment_df[\n            weekly_alignment_df["이평정배열"]\n        ]\n        .copy()\n        .sort_values(\n            by=[\n                "이평정배열_유지거래일",\n                "정배열강도"\n            ],\n            ascending=[False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n    ma_alignment_df.index += 1\n    ma_alignment_df.index.name = "순위"\n\n    print(\n        "\\n📌 이동평균 정배열 종목군"\n        "\\n   4주선 > 13주선 > 26주선 > 52주선"\n        "\\n   ※ 정배열 시작일과 연속 유지기간 포함"\n    )\n\n    if ma_alignment_df.empty:\n        print("해당 종목이 없습니다.")\n    else:\n        display(\n            ma_alignment_df[\n                [\n                    "ETF",\n                    "기준일",\n                    "이평정배열_시작일",\n                    "이평정배열_유지거래일",\n                    "이평정배열_달력일",\n                    "현재가",\n                    "MA4주",\n                    "MA13주",\n                    "MA26주",\n                    "MA52주",\n                    "현재가>4주",\n                    "정배열강도",\n                ]\n            ].style.format({\n                "현재가": "{:,.2f}",\n                "MA4주": "{:,.2f}",\n                "MA13주": "{:,.2f}",\n                "MA26주": "{:,.2f}",\n                "MA52주": "{:,.2f}",\n                "정배열강도": "{:.2f}",\n            })\n        )\n\n\n    # --------------------------------------------------------\n    # ③ 전체 ETF/섹터 정배열 상태\n    # --------------------------------------------------------\n\n    print("\\n📌 전체 ETF/섹터 정배열 상태")\n\n    display(\n        weekly_alignment_df[\n            [\n                "ETF",\n                "기준일",\n                "이평정배열",\n                "이평정배열_시작일",\n                "이평정배열_유지거래일",\n                "완전정배열",\n                "완전정배열_시작일",\n                "완전정배열_유지거래일",\n                "정배열강도",\n            ]\n        ].style.format({\n            "정배열강도": "{:.2f}",\n        })\n    )\n\n\n    # --------------------------------------------------------\n    # ④ 간단 출력\n    # --------------------------------------------------------\n\n    print(\n        "\\n🔥 완전 정배열 종목 "\n        "(현재가 > 4주 > 13주 > 26주 > 52주)"\n    )\n\n    if len(full_alignment_df) > 0:\n        for i, (_, row) in enumerate(\n            full_alignment_df.iterrows(),\n            start=1\n        ):\n            print(\n                f"{i}. {row[\'ETF\']} | "\n                f"{row[\'완전정배열_시작일\']}부터 | "\n                f"{int(row[\'완전정배열_유지거래일\'])}거래일 유지"\n            )\n    else:\n        print("없음")\n\n\n    print(\n        "\\n📈 이평 정배열 종목 "\n        "(4주 > 13주 > 26주 > 52주)"\n    )\n\n    if len(ma_alignment_df) > 0:\n        for i, (_, row) in enumerate(\n            ma_alignment_df.iterrows(),\n            start=1\n        ):\n            print(\n                f"{i}. {row[\'ETF\']} | "\n                f"{row[\'이평정배열_시작일\']}부터 | "\n                f"{int(row[\'이평정배열_유지거래일\'])}거래일 유지"\n            )\n    else:\n        print("없음")\n\n\n    print(\n        f"\\n📊 정배열 현황"\n        f"\\n- 완전 정배열: {len(full_alignment_df)}개"\n        f"\\n- 이평 정배열: {len(ma_alignment_df)}개"\n        f"\\n- 분석 가능 전체: {len(weekly_alignment_df)}개"\n    )\n\n# ============================================================\n# 13. 최종 52주 신고가 전략 백테스트\n# ============================================================\n#\n# 최종 전략\n# ------------------------------------------------------------\n# 1) 6개월(126거래일) KOSPI 대비 상대수익률 RS\n# 2) RS Top12만 후보\n# 3) 당일 종가가 직전 252거래일 최고가 돌파\n# 4) 신호 발생일 종가 매수\n#    -> 다음 거래일 수익률부터 반영\n# 5) 보유 종목 종가가 MA10 아래면 당일 종가 매도\n#    -> 다음 거래일부터 제외\n# 6) 최대 4종목\n# 7) 신규 종목 목표비중 25%\n# 8) 기존 보유 종목은 비중 drift 허용\n# 9) 빈 자리는 현금\n# 10) 편도 거래비용 0.10%\n#\n# 최종:\n# 6M RS Top12\n# ∩ 52주 신고가\n# → 최대 4종목 × 25%\n# → MA10 이탈 매도\n# ============================================================\n\n\n# ------------------------------------------------------------\n# 설정\n# ------------------------------------------------------------\n\nBT_RS_LOOKBACK = 126\nBT_HIGH_LOOKBACK = 252\nBT_MA_EXIT = 10\n\nBT_RS_TOP_N = 12\n\nBT_MAX_HOLDINGS = 4\nBT_WEIGHT_PER_ASSET = 0.25\n\nBT_TRANSACTION_COST = 0.001\n\n\n# ------------------------------------------------------------\n# 백테스트 입력 데이터\n# ------------------------------------------------------------\n\nbt_prices = etf_data.copy().sort_index()\nbt_kospi = kospi.copy().sort_index()\n\nbt_common_index = (\n    bt_prices.index\n    .intersection(bt_kospi.index)\n)\n\nbt_prices = bt_prices.loc[bt_common_index]\nbt_kospi = bt_kospi.loc[bt_common_index]\n\n# 완전히 비어 있는 열 제거\nbt_prices = bt_prices.dropna(\n    axis=1,\n    how="all"\n)\n\nbt_returns = bt_prices.pct_change(\n    fill_method=None\n)\n\nbt_kospi_returns = bt_kospi.pct_change(\n    fill_method=None\n)\n\n\n# ============================================================\n# 1. 6개월 KOSPI 대비 RS\n# ============================================================\n\nbt_return_6m = (\n    bt_prices\n    / bt_prices.shift(BT_RS_LOOKBACK)\n    - 1\n)\n\nbt_kospi_return_6m = (\n    bt_kospi\n    / bt_kospi.shift(BT_RS_LOOKBACK)\n    - 1\n)\n\nbt_rs_6m = bt_return_6m.sub(\n    bt_kospi_return_6m,\n    axis=0\n)\n\n\n# ============================================================\n# 2. 52주 신고가\n#\n# 중요:\n# 당일을 제외한 직전 252거래일 최고가와 비교\n# ============================================================\n\nbt_prev_52w_high = (\n    bt_prices\n    .shift(1)\n    .rolling(\n        BT_HIGH_LOOKBACK,\n        min_periods=BT_HIGH_LOOKBACK\n    )\n    .max()\n)\n\nbt_breakout = (\n    bt_prices > bt_prev_52w_high\n)\n\n\n# ============================================================\n# 3. MA10\n# ============================================================\n\nbt_ma10 = (\n    bt_prices\n    .rolling(\n        BT_MA_EXIT,\n        min_periods=BT_MA_EXIT\n    )\n    .mean()\n)\n\nbt_below_ma10 = (\n    bt_prices < bt_ma10\n)\n\n\n# ============================================================\n# 4. 백테스트 함수\n# ============================================================\n\ndef run_final_52w_backtest():\n\n    dates = bt_prices.index\n\n    equity = pd.Series(\n        1.0,\n        index=dates,\n        dtype=float\n    )\n\n    current_weights = pd.Series(\n        0.0,\n        index=bt_prices.columns,\n        dtype=float\n    )\n\n    turnover = pd.Series(\n        0.0,\n        index=dates,\n        dtype=float\n    )\n\n    holding_count = pd.Series(\n        0,\n        index=dates,\n        dtype=int\n    )\n\n    trade_rows = []\n    holding_rows = []\n\n    first_signal_pos = max(\n        BT_RS_LOOKBACK,\n        BT_HIGH_LOOKBACK,\n        BT_MA_EXIT\n    )\n\n\n    # --------------------------------------------------------\n    # 날짜별 진행\n    # --------------------------------------------------------\n\n    for i in range(1, len(dates)):\n\n        date = dates[i]\n\n        # 전 거래일 종가에서 신호 발생\n        signal_date = dates[i - 1]\n\n        previous_equity = equity.iloc[i - 1]\n\n\n        # ====================================================\n        # 신호일 종가 기준 포지션 결정\n        # ====================================================\n\n        if i - 1 >= first_signal_pos:\n\n            current_holdings = (\n                current_weights[\n                    current_weights > 0\n                ]\n                .index\n                .tolist()\n            )\n\n            target_weights = (\n                current_weights.copy()\n            )\n\n\n            # =================================================\n            # SELL\n            #\n            # 신호일 종가 < MA10\n            # → 신호일 종가 매도 효과\n            # → 다음 거래일부터 포지션 없음\n            # =================================================\n\n            sell_tickers = []\n\n            for ticker in current_holdings:\n\n                if ticker not in bt_below_ma10.columns:\n                    continue\n\n                signal = bt_below_ma10.loc[\n                    signal_date,\n                    ticker\n                ]\n\n                if pd.notna(signal) and bool(signal):\n\n                    sell_tickers.append(ticker)\n\n\n            for ticker in sell_tickers:\n\n                trade_rows.append({\n\n                    "Signal_Date":\n                        signal_date,\n\n                    "Execution_Date":\n                        signal_date,\n\n                    "Ticker":\n                        ticker,\n\n                    "Action":\n                        "SELL",\n\n                    "Reason":\n                        "Close below MA10",\n\n                    "Signal_Close":\n                        bt_prices.loc[\n                            signal_date,\n                            ticker\n                        ],\n\n                    "Signal_MA10":\n                        bt_ma10.loc[\n                            signal_date,\n                            ticker\n                        ],\n\n                    "Prev_52W_High":\n                        np.nan,\n\n                    "RS_Rank":\n                        np.nan,\n                })\n\n                target_weights.loc[\n                    ticker\n                ] = 0.0\n\n\n            remaining_holdings = (\n                target_weights[\n                    target_weights > 0\n                ]\n                .index\n                .tolist()\n            )\n\n\n            # =================================================\n            # RS 순위\n            # =================================================\n\n            rs_today = (\n                bt_rs_6m\n                .loc[signal_date]\n                .dropna()\n                .sort_values(\n                    ascending=False\n                )\n            )\n\n            rs_top12 = (\n                rs_today\n                .head(BT_RS_TOP_N)\n            )\n\n            rank_map = {\n\n                ticker: rank\n\n                for rank, ticker\n                in enumerate(\n                    rs_today.index,\n                    start=1\n                )\n            }\n\n\n            # =================================================\n            # BUY 후보\n            #\n            # RS Top12\n            # ∩\n            # 52주 신고가\n            # =================================================\n\n            candidates = []\n\n            for ticker in rs_top12.index:\n\n                # 이미 보유 중이면 신규매수 안 함\n                if ticker in remaining_holdings:\n                    continue\n\n                if ticker not in bt_breakout.columns:\n                    continue\n\n                breakout_signal = (\n                    bt_breakout.loc[\n                        signal_date,\n                        ticker\n                    ]\n                )\n\n                if (\n                    pd.notna(breakout_signal)\n                    and bool(breakout_signal)\n                ):\n\n                    candidates.append(\n                        ticker\n                    )\n\n\n            # RS 순위가 높은 종목부터\n            candidates = sorted(\n\n                candidates,\n\n                key=lambda x:\n                    rank_map.get(\n                        x,\n                        999\n                    )\n            )\n\n\n            # =================================================\n            # 최대 4종목\n            # =================================================\n\n            available_slots = max(\n\n                0,\n\n                BT_MAX_HOLDINGS\n                - len(remaining_holdings)\n            )\n\n\n            buy_tickers = candidates[\n                :available_slots\n            ]\n\n\n            # =================================================\n            # 신규 편입\n            #\n            # 신규 종목 목표비중 25%\n            # 기존 종목은 drift 유지\n            # =================================================\n\n            for ticker in buy_tickers:\n\n                target_weights.loc[\n                    ticker\n                ] = BT_WEIGHT_PER_ASSET\n\n\n                trade_rows.append({\n\n                    "Signal_Date":\n                        signal_date,\n\n                    "Execution_Date":\n                        signal_date,\n\n                    "Ticker":\n                        ticker,\n\n                    "Action":\n                        "BUY",\n\n                    "Reason":\n                        "6M RS Top12 + 52W breakout",\n\n                    "Signal_Close":\n                        bt_prices.loc[\n                            signal_date,\n                            ticker\n                        ],\n\n                    "Signal_MA10":\n                        bt_ma10.loc[\n                            signal_date,\n                            ticker\n                        ],\n\n                    "Prev_52W_High":\n                        bt_prev_52w_high.loc[\n                            signal_date,\n                            ticker\n                        ],\n\n                    "RS_Rank":\n                        rank_map.get(\n                            ticker,\n                            np.nan\n                        ),\n                })\n\n\n            # =================================================\n            # 거래비용\n            # =================================================\n\n            trade_turnover = (\n                target_weights\n                - current_weights\n            ).abs().sum()\n\n\n            if trade_turnover > 0:\n\n                previous_equity *= max(\n\n                    0.0,\n\n                    1.0\n                    - BT_TRANSACTION_COST\n                    * trade_turnover\n                )\n\n                turnover.loc[\n                    date\n                ] = trade_turnover\n\n                current_weights = (\n                    target_weights.copy()\n                )\n\n\n        # ====================================================\n        # 수익률 계산\n        #\n        # signal_date 종가\n        #       ↓\n        # date 종가\n        #\n        # 신호 발생일 종가에서 매수했다고 간주하기 때문에\n        # 다음 거래일 수익률부터 반영\n        # ====================================================\n\n        day_returns = (\n            bt_returns\n            .loc[date]\n            .reindex(\n                bt_prices.columns\n            )\n            .fillna(0.0)\n        )\n\n\n        cash_weight = max(\n\n            0.0,\n\n            1.0\n            - current_weights.sum()\n        )\n\n\n        portfolio_multiple = (\n\n            cash_weight\n\n            + (\n\n                current_weights\n                * (1 + day_returns)\n\n            ).sum()\n        )\n\n\n        equity.loc[date] = (\n\n            previous_equity\n            * portfolio_multiple\n        )\n\n\n        # ====================================================\n        # 종가 기준 비중 drift\n        # ====================================================\n\n        if portfolio_multiple > 0:\n\n            current_weights = (\n\n                current_weights\n                * (1 + day_returns)\n                / portfolio_multiple\n            )\n\n\n        held = (\n\n            current_weights[\n                current_weights > 0\n            ]\n\n            .sort_values(\n                ascending=False\n            )\n        )\n\n\n        holding_count.loc[\n            date\n        ] = len(held)\n\n\n        holding_rows.append({\n\n            "Date":\n                date,\n\n            "Holding_Count":\n                len(held),\n\n            "Invested_Exposure":\n                held.sum(),\n\n            "Cash_Exposure":\n                max(\n                    0.0,\n                    1.0 - held.sum()\n                ),\n\n            "Holdings":\n                " | ".join(\n\n                    f"{ticker}:{weight:.1%}"\n\n                    for ticker, weight\n                    in held.items()\n                ),\n        })\n\n\n    trades = pd.DataFrame(\n        trade_rows\n    )\n\n\n    holdings_df = (\n\n        pd.DataFrame(\n            holding_rows\n        )\n        .set_index("Date")\n\n        if len(holding_rows) > 0\n\n        else pd.DataFrame()\n    )\n\n\n    return {\n\n        "Equity":\n            equity,\n\n        "Turnover":\n            turnover,\n\n        "Holding_Count":\n            holding_count,\n\n        "Trades":\n            trades,\n\n        "Daily_Holdings":\n            holdings_df,\n    }\n\n\n# ============================================================\n# 5. 성과 계산\n# ============================================================\n\ndef bt_performance_stats(\n    name,\n    equity_curve,\n    turnover=None\n):\n\n    equity_curve = (\n        equity_curve\n        .dropna()\n        .copy()\n    )\n\n    if len(equity_curve) < 2:\n\n        return {\n\n            "Strategy": name,\n            "CAGR": np.nan,\n            "MDD": np.nan,\n            "Sharpe": np.nan,\n            "Sortino": np.nan,\n            "Annual_Turnover": np.nan,\n            "Ending_Multiple": np.nan,\n        }\n\n\n    years = (\n\n        equity_curve.index[-1]\n        - equity_curve.index[0]\n\n    ).days / 365.25\n\n\n    if years <= 0:\n\n        years = (\n            len(equity_curve)\n            / 252\n        )\n\n\n    cagr = (\n\n        equity_curve.iloc[-1]\n        / equity_curve.iloc[0]\n\n    ) ** (1 / years) - 1\n\n\n    drawdown = (\n\n        equity_curve\n        / equity_curve.cummax()\n        - 1\n    )\n\n\n    daily_ret = (\n\n        equity_curve\n        .pct_change()\n        .dropna()\n    )\n\n\n    ret_std = daily_ret.std(\n        ddof=1\n    )\n\n\n    sharpe = (\n\n        daily_ret.mean()\n        / ret_std\n        * np.sqrt(252)\n\n        if pd.notna(ret_std)\n        and ret_std > 0\n\n        else np.nan\n    )\n\n\n    downside = daily_ret[\n        daily_ret < 0\n    ]\n\n\n    downside_std = downside.std(\n        ddof=1\n    )\n\n\n    sortino = (\n\n        daily_ret.mean()\n        / downside_std\n        * np.sqrt(252)\n\n        if pd.notna(downside_std)\n        and downside_std > 0\n\n        else np.nan\n    )\n\n\n    annual_turnover = np.nan\n\n\n    if (\n        turnover is not None\n        and years > 0\n    ):\n\n        annual_turnover = (\n\n            turnover\n            .reindex(\n                equity_curve.index\n            )\n            .fillna(0)\n            .sum()\n\n            / years\n        )\n\n\n    return {\n\n        "Strategy":\n            name,\n\n        "CAGR":\n            cagr,\n\n        "MDD":\n            drawdown.min(),\n\n        "Sharpe":\n            sharpe,\n\n        "Sortino":\n            sortino,\n\n        "Annual_Turnover":\n            annual_turnover,\n\n        "Ending_Multiple":\n            (\n                equity_curve.iloc[-1]\n                / equity_curve.iloc[0]\n            ),\n    }\n\n\n# ============================================================\n# 6. 최종 전략 실행\n# ============================================================\n\nprint(\n    "\\n"\n    + "=" * 110\n)\n\nprint(\n    "🚀 최종 전략 백테스트"\n)\n\nprint(\n    "6M RS Top12 ∩ 52주 신고가 "\n    "→ 최대 4종목 × 25% "\n    "→ MA10 이탈 매도"\n)\n\nprint(\n    "=" * 110\n)\n\n\nbt_result = (\n    run_final_52w_backtest()\n)\n\n\nbt_trades = (\n    bt_result["Trades"]\n)\n\nbt_holdings = (\n    bt_result["Daily_Holdings"]\n)\n\nbt_equity = (\n    bt_result["Equity"]\n)\n\nbt_turnover = (\n    bt_result["Turnover"]\n)\n\n\n# ============================================================\n# 7. 실제 전략 시작일\n# ============================================================\n\nif (\n    bt_trades is None\n    or bt_trades.empty\n):\n\n    print(\n        "\\n⚠️ 매매 신호가 없습니다."\n    )\n\nelse:\n\n    bt_start = pd.Timestamp(\n        bt_trades[\n            "Signal_Date"\n        ].min()\n    )\n\n\n    print(\n        f"\\n실제 백테스트 시작일: "\n        f"{bt_start.date()}"\n    )\n\n\n    # --------------------------------------------------------\n    # 전략 Normalize\n    # --------------------------------------------------------\n\n    start_pos = (\n        bt_equity.index\n        .get_loc(bt_start)\n    )\n\n\n    strategy_base = (\n\n        bt_equity.iloc[\n            start_pos - 1\n        ]\n\n        if start_pos > 0\n\n        else bt_equity.iloc[\n            start_pos\n        ]\n    )\n\n\n    strategy_live = (\n\n        bt_equity.loc[\n            bt_start:\n        ]\n\n        / strategy_base\n    )\n\n\n    # --------------------------------------------------------\n    # KOSPI Buy & Hold\n    # --------------------------------------------------------\n\n    kospi_full = (\n\n        1\n        + bt_kospi_returns.fillna(0)\n\n    ).cumprod()\n\n\n    kospi_pos = (\n        kospi_full.index\n        .get_loc(bt_start)\n    )\n\n\n    kospi_base = (\n\n        kospi_full.iloc[\n            kospi_pos - 1\n        ]\n\n        if kospi_pos > 0\n\n        else kospi_full.iloc[\n            kospi_pos\n        ]\n    )\n\n\n    kospi_live = (\n\n        kospi_full.loc[\n            bt_start:\n        ]\n\n        / kospi_base\n    )\n\n\n    # ========================================================\n    # 8. 성과 비교\n    # ========================================================\n\n    strategy_stats = (\n        bt_performance_stats(\n\n            name=(\n                "6M RS Top12 + "\n                "52W High + "\n                "MA10 + "\n                "25% x max4"\n            ),\n\n            equity_curve=\n                strategy_live,\n\n            turnover=\n                bt_turnover\n        )\n    )\n\n\n    kospi_stats = (\n        bt_performance_stats(\n\n            name=\n                "KOSPI Buy & Hold",\n\n            equity_curve=\n                kospi_live\n        )\n    )\n\n\n    summary = pd.DataFrame(\n        [\n            strategy_stats,\n            kospi_stats\n        ]\n    )\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "📊 최종 전략 성과"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n\n        summary.style.format({\n\n            "CAGR":\n                "{:.2%}",\n\n            "MDD":\n                "{:.2%}",\n\n            "Sharpe":\n                "{:.2f}",\n\n            "Sortino":\n                "{:.2f}",\n\n            "Annual_Turnover":\n                "{:.2f}x",\n\n            "Ending_Multiple":\n                "{:.2f}x",\n        })\n    )\n\n\n    # ========================================================\n    # 9. 연도별 수익률\n    # ========================================================\n\n    strategy_ret = (\n\n        strategy_live\n        .pct_change()\n        .fillna(0)\n    )\n\n\n    kospi_ret = (\n\n        kospi_live\n        .pct_change()\n        .fillna(0)\n    )\n\n\n    strategy_yearly = (\n\n        (1 + strategy_ret)\n\n        .groupby(\n            strategy_ret.index.year\n        )\n\n        .prod()\n\n        - 1\n    )\n\n\n    kospi_yearly = (\n\n        (1 + kospi_ret)\n\n        .groupby(\n            kospi_ret.index.year\n        )\n\n        .prod()\n\n        - 1\n    )\n\n\n    yearly = pd.concat(\n\n        [\n            strategy_yearly,\n            kospi_yearly\n        ],\n\n        axis=1\n    )\n\n\n    yearly.columns = [\n\n        "Strategy",\n        "KOSPI"\n    ]\n\n\n    yearly[\n        "Excess_Return"\n    ] = (\n\n        yearly["Strategy"]\n        - yearly["KOSPI"]\n    )\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "📅 연도별 수익률"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n\n        yearly.style.format(\n            "{:.2%}"\n        )\n    )\n\n\n    # ========================================================\n    # 10. 거래 통계\n    # ========================================================\n\n    buy_count = int(\n\n        (\n            bt_trades["Action"]\n            == "BUY"\n        ).sum()\n    )\n\n\n    sell_count = int(\n\n        (\n            bt_trades["Action"]\n            == "SELL"\n        ).sum()\n    )\n\n\n    live_holdings = (\n\n        bt_holdings\n        .loc[\n            bt_start:\n        ]\n    )\n\n\n    stats_table = pd.DataFrame({\n\n        "Metric": [\n\n            "Total Buys",\n            "Total Sells",\n            "Average Holdings",\n            "Maximum Holdings",\n            "Average Invested Exposure",\n            "Latest Holdings",\n        ],\n\n        "Value": [\n\n            buy_count,\n\n            sell_count,\n\n            live_holdings[\n                "Holding_Count"\n            ].mean(),\n\n            live_holdings[\n                "Holding_Count"\n            ].max(),\n\n            live_holdings[\n                "Invested_Exposure"\n            ].mean(),\n\n            live_holdings[\n                "Holdings"\n            ].iloc[-1],\n        ]\n    })\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "📊 거래 및 보유 통계"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n        stats_table\n    )\n\n\n    # ========================================================\n    # 11. 최근 거래 30건\n    # ========================================================\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "📌 최근 거래 30건"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n        bt_trades.tail(30)\n    )\n\n\n    # ========================================================\n    # 12. 최근 보유 현황\n    # ========================================================\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "📌 최근 보유 현황"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n\n        live_holdings\n        .tail(30)\n        .style\n        .format({\n\n            "Invested_Exposure":\n                "{:.2%}",\n\n            "Cash_Exposure":\n                "{:.2%}",\n        })\n    )\n\n\n    # ========================================================\n    # 13. 최신 RS Top12\n    # ========================================================\n\n    latest_date = (\n\n        bt_prices\n        .dropna(\n            how="all"\n        )\n        .index[-1]\n    )\n\n\n    latest_rs = (\n\n        bt_rs_6m\n        .loc[\n            latest_date\n        ]\n\n        .dropna()\n\n        .sort_values(\n            ascending=False\n        )\n    )\n\n\n    latest_top12 = (\n        latest_rs\n        .head(BT_RS_TOP_N)\n    )\n\n\n    top12_rows = []\n\n\n    for rank_no, ticker in enumerate(\n        latest_top12.index,\n        start=1\n    ):\n\n        close = (\n            bt_prices.loc[\n                latest_date,\n                ticker\n            ]\n        )\n\n        high52 = (\n            bt_prev_52w_high.loc[\n                latest_date,\n                ticker\n            ]\n        )\n\n        breakout = False\n\n        if pd.notna(\n            bt_breakout.loc[\n                latest_date,\n                ticker\n            ]\n        ):\n\n            breakout = bool(\n                bt_breakout.loc[\n                    latest_date,\n                    ticker\n                ]\n            )\n\n\n        top12_rows.append({\n\n            "RS_Rank":\n                rank_no,\n\n            "Ticker":\n                ticker,\n\n            "Close":\n                close,\n\n            "Prev_52W_High":\n                high52,\n\n            "52W_Breakout":\n                breakout,\n\n            "RS_6M_vs_KOSPI":\n                latest_rs.loc[\n                    ticker\n                ],\n        })\n\n\n    top12_df = pd.DataFrame(\n        top12_rows\n    )\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        f"📈 최신 6M RS Top12 "\n        f"| {latest_date.date()}"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    display(\n\n        top12_df.style.format({\n\n            "Close":\n                "{:,.2f}",\n\n            "Prev_52W_High":\n                "{:,.2f}",\n\n            "RS_6M_vs_KOSPI":\n                "{:+.2%}",\n        })\n    )\n\n\n    # ========================================================\n    # 14. 오늘 신규 52주 신고가 후보\n    # ========================================================\n\n    latest_candidates = (\n\n        top12_df[\n            top12_df[\n                "52W_Breakout"\n            ]\n        ]\n\n        .copy()\n    )\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        f"🚀 오늘 신규 매수 신호 "\n        f"| {latest_date.date()}"\n    )\n\n    print(\n        "조건: "\n        "6M RS Top12 ∩ 52주 신고가"\n    )\n\n    print(\n        "=" * 110\n    )\n\n\n    if latest_candidates.empty:\n\n        print(\n            "오늘 신규 52주 신고가 "\n            "후보가 없습니다."\n        )\n\n    else:\n\n        display(\n\n            latest_candidates\n            .style\n            .format({\n\n                "Close":\n                    "{:,.2f}",\n\n                "Prev_52W_High":\n                    "{:,.2f}",\n\n                "RS_6M_vs_KOSPI":\n                    "{:+.2%}",\n            })\n        )\n\n\n    # ========================================================\n    # 15. Equity Curve\n    # ========================================================\n\n    plt.figure(\n        figsize=(15, 6)\n    )\n\n\n    plt.plot(\n\n        strategy_live.index,\n        strategy_live,\n\n        linewidth=2.4,\n\n        label=(\n            "RS Top12 + "\n            "52W Breakout"\n        )\n    )\n\n\n    plt.plot(\n\n        kospi_live.index,\n        kospi_live,\n\n        linewidth=1.8,\n\n        label="KOSPI"\n    )\n\n\n    plt.title(\n        "52W Breakout Strategy vs KOSPI"\n    )\n\n    plt.ylabel(\n        "Growth of 1"\n    )\n\n    plt.grid(\n        alpha=0.25\n    )\n\n    plt.legend()\n\n    plt.tight_layout()\n\n    plt.show()\n\n\n    # ========================================================\n    # 16. Drawdown\n    # ========================================================\n\n    strategy_dd = (\n\n        strategy_live\n        / strategy_live.cummax()\n        - 1\n    )\n\n\n    kospi_dd = (\n\n        kospi_live\n        / kospi_live.cummax()\n        - 1\n    )\n\n\n    plt.figure(\n        figsize=(15, 5)\n    )\n\n\n    plt.plot(\n\n        strategy_dd.index,\n        strategy_dd,\n\n        linewidth=2.2,\n\n        label="RS Top12 + 52W"\n    )\n\n\n    plt.plot(\n\n        kospi_dd.index,\n        kospi_dd,\n\n        linewidth=1.6,\n\n        label="KOSPI"\n    )\n\n\n    plt.title(\n        "Drawdown Comparison"\n    )\n\n    plt.ylabel(\n        "Drawdown"\n    )\n\n    plt.grid(\n        alpha=0.25\n    )\n\n    plt.legend()\n\n    plt.tight_layout()\n\n    plt.show()\n\n\n    print(\n        "\\n"\n        + "=" * 110\n    )\n\n    print(\n        "✅ 최종 52주 신고가 전략 완료"\n    )\n\n    print(\n        "=" * 110\n    )\n\n    print(\n        "6M KOSPI 대비 RS Top12"\n    )\n\n    print(\n        "∩ 52주 신고가"\n    )\n\n    print(\n        "→ 최대 4종목 × 25%"\n    )\n\n    print(\n        "→ MA10 종가 이탈 매도"\n    )\n\n    print(\n        "→ 신호일 종가 체결"\n    )\n\n    print(\n        "→ 편도 거래비용 0.10%"\n    )\n\n'
US_SECTOR_SRC='# ----------------------------\n# 1. 라이브러리 설치 (Google Colab 전용)\n# ----------------------------\n\n# ----------------------------\n# 2. 라이브러리 임포트\n# ----------------------------\nimport yfinance as yf\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# ----------------------------\n# 3. 산업별 ETF 정의\n# ----------------------------\nindustry_etfs = {\n\n    # ------------------------------------------------\n    # 1. 벤치마크\n    # ------------------------------------------------\n    "SPY": "S&P 500",\n\n    # ------------------------------------------------\n    # 2. 미국 기본 11개 섹터 (대표 ETF)\n    # ------------------------------------------------\n    "XLK": "Technology",\n    "VOX": "Communication Services",\n    "XLY": "Consumer Discretionary",\n    "XLP": "Consumer Staples",\n    "XLI": "Industrials",\n    "XLB": "Materials",\n    "XLE": "Energy",\n    "XLF": "Financials",\n    "XLV": "Health Care",\n    "XLU": "Utilities",\n\n    # ------------------------------------------------\n    # 3. 테크 / 반도체 / 인터넷 / 클라우드\n    # ------------------------------------------------\n    "IGV": "Software",\n    "SOXX": "Semiconductors",\n    "XSD": "Semiconductors - Equal Weight",\n    "PSI": "Semiconductors - Dynamic",\n    "SKYY": "Cloud Computing",\n    "FDN": "Internet",\n    "PNQI": "Internet NASDAQ",\n    "SOCL": "Social Media",\n    "QTEC": "Tech - NASDAQ Equal Weight",\n    "TDIV": "Tech Dividend",\n\n    # ------------------------------------------------\n    # 4. 에너지 / 오일·가스 / MLP / 우라늄\n    # ------------------------------------------------\n    "XOP": "Oil & Gas Exploration",\n    "OIH": "Oil Services",\n    "IEZ": "Oil Equipment & Services",\n    "XES": "Oil & Gas Services",\n    "FCG": "Natural Gas",\n    "AMLP": "MLP Energy Infrastructure",\n    "MLPX": "MLP & Energy Infra",\n    "EMLP": "North American Energy Infra",\n    "ENFR": "Energy Infrastructure",\n    "URA": "Uranium",\n    "NLR": "Nuclear Energy",\n\n    # ------------------------------------------------\n    # 5. 금융 / 은행 / 보험 / 브로커 / BDC / PE\n    # ------------------------------------------------\n    "KBE": "Banks",\n    "KRE": "Regional Banks",\n    "IAI": "Broker-Dealers & Exchanges",\n    "KIE": "Insurance",\n    "IXG": "Global Financials",\n    "BIZD": "Business Development Companies",\n    "PSP": "Listed Private Equity",\n\n    # ------------------------------------------------\n    # 6. 헬스케어 / 의료기기 / 바이오\n    # ------------------------------------------------\n    "XBI": "Biotech Equal Weight",\n    "IBB": "Biotechnology",\n    "IHI": "Medical Devices",\n    "XHE": "Health Care Equipment",\n    "IHF": "Health Care Providers",\n    "XHS": "Health Care Services",\n    "IHE": "Pharmaceuticals",\n    "PJP": "Pharma Dynamic",\n\n    # ------------------------------------------------\n    # 7. 소비 / 리테일 / 주택 / 레저\n    # ------------------------------------------------\n    "XRT": "Retail",\n    "XHB": "Homebuilders",\n    "ITB": "Home Construction",\n    "PEJ": "Leisure & Entertainment",\n    "PBJ": "Food & Beverage",\n    "RXI": "Global Consumer Discretionary",\n    "CHIQ": "China Consumer",\n    "ECON": "Emerging Markets Consumer",\n    "INCO": "India Consumer",\n\n    # ------------------------------------------------\n    # 8. 산업 / 방산 / 운송 / 인프라 / 워터\n    # ------------------------------------------------\n    "ITA": "Aerospace & Defense",\n    "PPA": "Defense & Aerospace",\n    "XAR": "Aerospace & Defense Equal Weight",\n    "IYT": "Transportation",\n    "XTN": "Transportation Equal Weight",\n    "TOLZ": "Global Infrastructure",\n    "GII": "Global Infrastructure SPDR",\n    "EMIF": "Emerging Markets Infrastructure",\n    "CGW": "Global Water",\n    "FIW": "Water Resources",\n    "AIRR": "American Industrial Renaissance",\n\n    # ------------------------------------------------\n    # 9. 소재 / 자원 / 광물 / 금·은 / 농업\n    # ------------------------------------------------\n    "GNR": "Natural Resources",\n    "GUNR": "Global Upstream Natural Resources",\n    "HAP": "Natural Resources Producers",\n    "GDX": "Gold Miners",\n    "GDXJ": "Junior Gold Miners",\n    "SGDM": "Gold Miners - Sprott",\n    "SIL": "Silver Miners",\n    "SILJ": "Junior Silver Miners",\n    "SLVP": "Global Silver Miners",\n    "SLX": "Steel",\n    "COPX": "Copper Miners",\n    "REMX": "Rare Earth & Strategic Metals",\n    "PICK": "Global Metal & Mining",\n    "MOO": "Agribusiness",\n    "VEGI": "Agriculture Producers",\n    "WOOD": "Global Timber Forestry",\n    "CUT": "Timber & Forestry",\n\n    # ------------------------------------------------\n    # 10. 클린에너지 / 재생 / 그리드\n    # ------------------------------------------------\n    "ICLN": "Global Clean Energy",\n    "PBW": "Clean Energy - WilderHill",\n    "PBD": "Global Clean Energy",\n    "TAN": "Solar Energy",\n    "FAN": "Wind Energy",\n    "QCLN": "Clean Energy NASDAQ",\n    "GRID": "Smart Grid",\n\n    # ------------------------------------------------\n    # 11. 커뮤니케이션 / 글로벌 통신\n    # ------------------------------------------------\n    "IYZ": "Telecom",\n    "XTL": "Telecom SPDR",\n    "IXP": "Global Communication Services",\n\n    # ------------------------------------------------\n    # 12. 기타 테마\n    # ------------------------------------------------\n    "LIT": "Lithium & Battery Tech",\n    "SEA": "Shipping & Cargo",\n    "CARZ": "Global Auto",\n    "UFO": "Space & Satellite",\n    "ROBO" : "Robotics&Automation ETF"\n}\n\netf_list = list(industry_etfs.keys())\n\n# ----------------------------\n# 4. 데이터 다운로드 (250일 계산 안정화 위해 3년 권장)\n# ----------------------------\ndata = yf.download(etf_list, period="3y")[\'Close\'].sort_index()\n\nif \'SPY\' not in data.columns:\n    raise ValueError("SPY 데이터가 없습니다. 네트워크/티커 문제를 확인하세요.")\nspy = data[\'SPY\']\n\n# ----------------------------\n# 5. Mansfield RS 계산 함수 (6개월 = 126거래일)\n# ----------------------------\ndef compute_mansfield_rs(price_series, benchmark_series, ma_period):\n    relative = price_series / benchmark_series\n    ma = relative.rolling(window=ma_period, min_periods=ma_period).mean()\n    return ((relative / ma) - 1) * 100\n\ndef normalize_to_100_scale(x, scale=12):\n    return 100 * (1 / (1 + np.exp(-x / scale)))\n\n# ----------------------------\n# 6. RS 계산 실행 (6개월 = 126거래일)\n# ----------------------------\nRS_WINDOWS = [126]\nmansfield_rows = []\n\nfor ticker in etf_list:\n    if ticker == "SPY":\n        continue\n    if ticker not in data.columns:\n        continue\n\n    try:\n        price = data[ticker].dropna()\n        aligned = pd.concat([price, spy], axis=1, join=\'inner\')\n        aligned.columns = [\'etf\', \'spy\']\n\n        latest = {}\n        for win in RS_WINDOWS:\n            if len(aligned) >= win:\n                rs_series = compute_mansfield_rs(aligned[\'etf\'], aligned[\'spy\'], ma_period=win).dropna()\n                if not rs_series.empty:\n                    raw = float(rs_series.iloc[-1])\n                    latest[f"RS_{win}d"] = round(raw, 2)\n                    latest[f"Norm_RS_{win}d"] = round(normalize_to_100_scale(raw), 2)\n\n        raw_keys = [k for k in latest if k.startswith("RS_") and k.endswith("d")]\n        if not raw_keys:\n            continue\n\n        raw_values = [latest[k] for k in raw_keys]\n        rs_avg = round(sum(raw_values) / len(raw_values), 2)\n        norm_rs_avg = round(normalize_to_100_scale(rs_avg), 2)\n\n        row = {\n            "Ticker": ticker,\n            "Industry": industry_etfs.get(ticker, ""),\n            "RS_126d": np.nan,\n            "Norm_RS_126d": np.nan,\n            "RS_avg": rs_avg,\n            "Norm_RS_avg": norm_rs_avg,\n        }\n        for win in RS_WINDOWS:\n            if f"RS_{win}d" in latest:\n                row[f"RS_{win}d"] = latest[f"RS_{win}d"]\n                row[f"Norm_RS_{win}d"] = latest[f"Norm_RS_{win}d"]\n\n        mansfield_rows.append(row)\n\n    except Exception:\n        continue\n\nmansfield_df = pd.DataFrame(mansfield_rows).sort_values(by="Norm_RS_avg", ascending=False)\n\nprint("\\n📌 Mansfield RS 6개월(126거래일) 결과 (Norm_RS_avg 내림차순):")\ndisplay(mansfield_df)\n\nmansfield_df_70 = mansfield_df[mansfield_df["Norm_RS_avg"] >= 70].copy()\nif mansfield_df_70.empty:\n    print("\\n🚫 Normalized 6개월 RS ≥ 70 조건을 만족하는 ETF가 없습니다. 상위 10개를 보여드립니다:")\n    display(mansfield_df.head(10))\nelse:\n    print("\\n📌 Normalized 6개월 RS ≥ 70 (필터링):")\ndisplay(mansfield_df_70)\n\n# ------------------------------------------------------------\n# 6개월 절대수익률 + SPY 대비 6개월 상대수익률\n# ------------------------------------------------------------\nLOOKBACK_6M = 126\n\nreturn_6m = (\n    data / data.shift(LOOKBACK_6M) - 1\n)\n\nspy_return_6m = return_6m["SPY"]\n\nrelative_return_6m = return_6m.sub(\n    spy_return_6m,\n    axis=0\n)\n\nlatest_6m_date = return_6m.dropna(how="all").index[-1]\n\nsix_month_summary = pd.DataFrame({\n    "Ticker": [t for t in data.columns if t != "SPY"],\n})\n\nsix_month_summary["Industry"] = six_month_summary["Ticker"].map(\n    industry_etfs\n)\n\nsix_month_summary["6M_Return"] = six_month_summary["Ticker"].map(\n    return_6m.loc[latest_6m_date]\n)\n\nsix_month_summary["6M_RS_vs_SPY"] = six_month_summary["Ticker"].map(\n    relative_return_6m.loc[latest_6m_date]\n)\n\nsix_month_summary = (\n    six_month_summary\n    .dropna(subset=["6M_Return", "6M_RS_vs_SPY"])\n    .sort_values("6M_RS_vs_SPY", ascending=False)\n    .reset_index(drop=True)\n)\n\nsix_month_summary.index = six_month_summary.index + 1\nsix_month_summary.index.name = "Rank"\n\nprint(\n    f"\\n📌 기준일 {latest_6m_date.date()} "\n    f"미국 ETF 6개월 수익률 및 SPY 대비 RS"\n)\n\ndisplay(\n    six_month_summary.style.format({\n        "6M_Return": "{:.2%}",\n        "6M_RS_vs_SPY": "{:+.2%}",\n    })\n)\n\n# ----------------------------\n# 7. 변동성 조정 모멘텀 계산 함수 (+ 이동평균선 이탈 필터)\n#    ✅ FutureWarning 제거: pct_change(fill_method=None)\n#    ✅ vol=0 안전장치\n# ----------------------------\ndef calculate_risk_adjusted_momentum(\n    data,\n    period_6m=126,\n    last_n_days=10,\n    ma_filter_days=None,\n    top_n=20\n):\n    """\n    최근 6개월(126거래일) 누적수익률을\n    같은 기간 일간수익률 표준편차로 나눈 위험조정 모멘텀 점수.\n\n    score = 6개월 누적수익률 / 6개월 일간수익률 표준편차\n\n    ma_filter_days가 주어지면 해당 이동평균 하회 종목 제외.\n    """\n    daily_top_industries = {}\n    trading_days = data.index[-last_n_days:]\n\n    for day in trading_days:\n        end_idx = data.index.get_loc(day)\n        start_6m = max(0, end_idx - period_6m + 1)\n\n        window_6m = data.iloc[\n            start_6m:end_idx + 1\n        ]\n\n        if len(window_6m) < int(period_6m * 0.8):\n            daily_top_industries[day] = [None] * top_n\n            continue\n\n        # 6개월 누적수익률\n        returns_6m = (\n            window_6m.iloc[-1]\n            / window_6m.iloc[0]\n            - 1\n        )\n\n        # 6개월 일간수익률 변동성\n        daily_returns_6m = (\n            window_6m\n            .pct_change(fill_method=None)\n        )\n\n        vol_6m = (\n            daily_returns_6m\n            .std()\n            .replace(0, np.nan)\n        )\n\n        risk_adjusted_6m = (\n            returns_6m / vol_6m\n        )\n\n        # SPY는 벤치마크이므로 랭킹에서 제외\n        risk_adjusted_6m = risk_adjusted_6m.drop(\n            labels=["SPY"],\n            errors="ignore"\n        )\n\n        # 이동평균선 필터\n        if ma_filter_days is not None:\n            ma_ok = {}\n\n            for ticker in data.columns:\n                if ticker == "SPY":\n                    continue\n\n                series = (\n                    data[ticker]\n                    .iloc[:end_idx + 1]\n                    .dropna()\n                )\n\n                if len(series) < ma_filter_days:\n                    ma_ok[ticker] = False\n                else:\n                    ma_value = (\n                        series\n                        .rolling(ma_filter_days)\n                        .mean()\n                        .iloc[-1]\n                    )\n\n                    ma_ok[ticker] = bool(\n                        series.iloc[-1] >= ma_value\n                    )\n\n            ma_ok = pd.Series(ma_ok)\n\n            risk_adjusted_6m = risk_adjusted_6m[\n                ma_ok.reindex(\n                    risk_adjusted_6m.index\n                ).fillna(False)\n            ]\n\n        risk_adjusted_6m = (\n            risk_adjusted_6m\n            .dropna()\n            .sort_values(ascending=False)\n        )\n\n        top_list = (\n            risk_adjusted_6m\n            .head(top_n)\n            .index\n            .tolist()\n        )\n\n        top_names = [\n            f"{ticker} ({industry_etfs.get(ticker, \'\')})"\n            for ticker in top_list\n        ]\n\n        top_names += [None] * max(\n            0,\n            top_n - len(top_names)\n        )\n\n        daily_top_industries[day] = top_names\n\n    df = pd.DataFrame.from_dict(\n        daily_top_industries,\n        orient="index",\n        columns=[\n            f"Top {i+1}"\n            for i in range(top_n)\n        ]\n    )\n\n    df.index.name = "Date"\n    return df\n\n# ----------------------------\n# 8. 모멘텀 결과 출력 (Top 20 기준)\n# ----------------------------\nprint("\\n📌 6개월 변동성 조정 모멘텀 기준 상위 20 ETF (최근 10일 기준, 전체):")\nmomentum_df = calculate_risk_adjusted_momentum(data, last_n_days=10, top_n=20)\ndisplay(momentum_df)\n\nprint("\\n📌 6개월 변동성 조정 모멘텀 상위 20 ETF (10일선 이상):")\nmomentum_df_ma10 = calculate_risk_adjusted_momentum(data, last_n_days=10, ma_filter_days=10, top_n=20)\ndisplay(momentum_df_ma10)\n\nprint("\\n📌 6개월 변동성 조정 모멘텀 상위 20 ETF (20일선 이상):")\nmomentum_df_ma20 = calculate_risk_adjusted_momentum(data, last_n_days=10, ma_filter_days=20, top_n=20)\ndisplay(momentum_df_ma20)\n\nprint("\\n📌 6개월 변동성 조정 모멘텀 상위 20 ETF (60일선 이상):")\nmomentum_df_ma60 = calculate_risk_adjusted_momentum(data, last_n_days=10, ma_filter_days=60, top_n=20)\ndisplay(momentum_df_ma60)\n\n# ============================================================\n# 9. Sortino: 10일선 이상 기준만 계산\n#    - ETR Comfort 출력/기존 11~12번 추가 분석은 삭제\n#    - 최종 30일 초과 이격도 분석은 6개월 Sortino 10일선 이상 리스트 기준\n# ============================================================\n\nimport logging\nlogging.getLogger(\'matplotlib.font_manager\').setLevel(logging.ERROR)\nplt.rcParams[\'axes.unicode_minus\'] = False\n\n\ndef sortino_from_returns(ret: pd.Series, mar: float = 0.0, periods_per_year: int = 252) -> float:\n    """Sortino Ratio, annualized."""\n    ret = ret.dropna()\n    if len(ret) < 5:\n        return np.nan\n\n    mar_daily = (1 + mar) ** (1 / periods_per_year) - 1\n    excess = ret - mar_daily\n    downside = excess[excess < 0]\n\n    if len(downside) < 5:\n        return np.nan\n\n    downside_dev = downside.std(ddof=1) * np.sqrt(periods_per_year)\n    if downside_dev == 0 or np.isnan(downside_dev):\n        return np.nan\n\n    annual_excess = excess.mean() * periods_per_year\n    return float(annual_excess / downside_dev)\n\n\ndef ma_ok_mask(close_df: pd.DataFrame, end_idx: int, ma_days: int) -> pd.Series:\n    """end_idx 시점에서 종가 >= MA(ma_days)이면 True."""\n    ok = {}\n    for t in close_df.columns:\n        s = close_df[t].iloc[:end_idx + 1].dropna()\n        if len(s) < ma_days:\n            ok[t] = False\n        else:\n            ma = s.rolling(ma_days).mean().iloc[-1]\n            ok[t] = bool(s.iloc[-1] >= ma)\n    return pd.Series(ok)\n\n\ndef rank_sortino_ma_filter(\n    close_df: pd.DataFrame,\n    period_6m: int = 126,\n    last_n_days: int = 10,\n    ma_filter_days: int = 10,\n    top_n: int = 20,\n    periods_per_year: int = 252,\n    mar: float = 0.0,\n):\n    """\n    최근 6개월(126거래일) Sortino를 계산해 순위를 매깁니다.\n    ma_filter_days가 주어지면 해당 날짜 기준\n    종가 >= 이동평균선인 ETF만 통과합니다.\n    """\n    daily_top = {}\n    trading_days = (\n        close_df\n        .dropna(how="all")\n        .index[-last_n_days:]\n    )\n\n    for day in trading_days:\n        end_idx = close_df.index.get_loc(day)\n\n        ok = ma_ok_mask(\n            close_df,\n            end_idx=end_idx,\n            ma_days=ma_filter_days\n        )\n\n        scores = {}\n\n        for ticker in close_df.columns:\n            if not ok.get(ticker, False):\n                continue\n\n            start_idx = max(\n                0,\n                end_idx - period_6m + 1\n            )\n\n            prices_6m = (\n                close_df[ticker]\n                .iloc[start_idx:end_idx + 1]\n                .dropna()\n            )\n\n            if len(prices_6m) < max(\n                20,\n                int(period_6m * 0.8)\n            ):\n                continue\n\n            returns_6m = (\n                prices_6m\n                .pct_change(fill_method=None)\n                .dropna()\n            )\n\n            score = sortino_from_returns(\n                returns_6m,\n                mar=mar,\n                periods_per_year=periods_per_year\n            )\n\n            if pd.notna(score):\n                scores[ticker] = float(score)\n\n        if len(scores) == 0:\n            daily_top[day] = [None] * top_n\n            continue\n\n        ranked = (\n            pd.Series(scores)\n            .sort_values(ascending=False)\n            .head(top_n)\n            .index\n            .tolist()\n        )\n\n        row = [\n            f"{ticker} ({industry_etfs.get(ticker, \'\')})"\n            for ticker in ranked\n        ]\n\n        row += [None] * max(\n            0,\n            top_n - len(row)\n        )\n\n        daily_top[day] = row\n\n    out = pd.DataFrame.from_dict(\n        daily_top,\n        orient="index",\n        columns=[\n            f"Top {i+1}"\n            for i in range(top_n)\n        ]\n    )\n\n    out.index.name = "Date"\n    return out\n\n\n# SPY는 벤치마크이므로 업종 랭킹에서는 제외\nsector_data = data.drop(columns=["SPY"], errors="ignore").copy()\n\nprint("\\n📌 6개월 Sortino 상위 20 ETF (최근 10일, 10일선 이상):")\nsortino_ma10 = rank_sortino_ma_filter(\n    sector_data,\n    last_n_days=10,\n    ma_filter_days=10,\n    top_n=20\n)\ndisplay(sortino_ma10)\n\nprint("\\n📌 6개월 Sortino 상위 20 ETF (최근 10일, 20일선 이상):")\nsortino_ma20 = rank_sortino_ma_filter(\n    sector_data,\n    last_n_days=10,\n    ma_filter_days=20,\n    top_n=20\n)\ndisplay(sortino_ma20)\n\nprint("\\n📌 6개월 Sortino 상위 20 ETF (최근 10일, 10주(50일선) 이상):")\nsortino_ma50 = rank_sortino_ma_filter(\n    sector_data,\n    last_n_days=10,\n    ma_filter_days=50,\n    top_n=20\n)\ndisplay(sortino_ma50)\n\n\n# ============================================================\n# 10. 최종 분석\n#     6개월 Sortino 10일선 이상 ETF 기준\n#     30일 초과 이격도 + 최근 247거래일 97% 초과 이격도\n#\n# 계산 기준:\n# - 30일 초과 이격도 = (현재가 / MA30 - 1) * 100\n# - 최근247거래일97구간초과이격도 = 최근 247거래일 30일 초과 이격도의 97% 분위수\n# - 97구간대비현재위치 = 현재30일초과이격도 / 최근247거래일97구간초과이격도 * 100\n#\n# 표는 한글, 그래프는 영문 제목만 사용해서 Matplotlib 한글 폰트 경고를 방지\n# ============================================================\n\n\ndef _extract_ticker_from_label(x):\n    """\'SOXX (Semiconductors)\' 형태에서 SOXX만 추출."""\n    if pd.isna(x):\n        return None\n    return str(x).split(" ")[0].strip()\n\n\ndef build_sortino_ma10_gap247_p97_summary(\n    close_df: pd.DataFrame,\n    sortino_table: pd.DataFrame,\n    industry_map: dict,\n    ma_days: int = 30,\n    lookback_days: int = 247,\n    percentile: float = 0.97,\n    top_n: int = 20,\n):\n    latest_row = sortino_table.iloc[-1].dropna().tolist()\n    tickers = [_extract_ticker_from_label(x) for x in latest_row]\n    tickers = [x for x in tickers if x in close_df.columns]\n\n    if top_n is not None:\n        tickers = tickers[:top_n]\n\n    rows = []\n\n    for rank_no, ticker in enumerate(tickers, start=1):\n        price = close_df[ticker].dropna()\n        if len(price) < ma_days + 20:\n            continue\n\n        ma30 = price.rolling(ma_days).mean()\n        gap_excess = (price / ma30 - 1) * 100\n        gap_excess = gap_excess.dropna()\n\n        if len(gap_excess) < 20:\n            continue\n\n        hist = gap_excess.tail(lookback_days)\n        current_excess = float(gap_excess.iloc[-1])\n        p97_excess = float(hist.quantile(percentile))\n\n        if pd.notna(p97_excess) and p97_excess != 0:\n            heat_ratio = current_excess / p97_excess * 100\n        else:\n            heat_ratio = np.nan\n\n        rows.append({\n            "Sortino순위": rank_no,\n            "Ticker": ticker,\n            "Industry": industry_map.get(ticker, ""),\n            "Date": gap_excess.index[-1].date(),\n            "현재30일초과이격도": round(current_excess, 2),\n            "최근247거래일97구간초과이격도": round(p97_excess, 2),\n            "97구간대비현재위치": round(heat_ratio, 1) if pd.notna(heat_ratio) else np.nan,\n            "사용거래일수": len(hist),\n        })\n\n    out = pd.DataFrame(rows)\n\n    if not out.empty:\n        out = out[[\n            "Sortino순위",\n            "Ticker",\n            "Industry",\n            "Date",\n            "현재30일초과이격도",\n            "최근247거래일97구간초과이격도",\n            "97구간대비현재위치",\n            "사용거래일수",\n        ]]\n\n    return out\n\n\ndef plot_sortino_ma10_gap247_p97_graphs(\n    close_df: pd.DataFrame,\n    summary_df: pd.DataFrame,\n    ma_days: int = 30,\n    lookback_days: int = 247,\n    figsize=(16, 4),\n    bar_alpha: float = 0.15,\n):\n    if summary_df.empty:\n        print("그래프로 출력할 ETF가 없습니다.")\n        return\n\n    for _, row in summary_df.iterrows():\n        ticker = row["Ticker"]\n        industry = row["Industry"]\n\n        price = close_df[ticker].dropna()\n        ma30 = price.rolling(ma_days).mean()\n        gap_excess = (price / ma30 - 1) * 100\n\n        plot_df = pd.DataFrame({\n            "Price": price,\n            "MA30": ma30,\n            "GapExcess30": gap_excess,\n        }).dropna()\n\n        if plot_df.empty:\n            continue\n\n        fig, ax1 = plt.subplots(figsize=figsize)\n        ax2 = ax1.twinx()\n\n        # 오른쪽 축: 30일 초과 이격도 막대\n        ax2.bar(\n            plot_df.index,\n            plot_df["GapExcess30"],\n            alpha=bar_alpha,\n            width=1.5,\n            label="GapExcess30",\n            zorder=1,\n        )\n\n        # 97% 구간 기준선\n        p97 = row["최근247거래일97구간초과이격도"]\n        if pd.notna(p97):\n            ax2.axhline(\n                p97,\n                linestyle="--",\n                linewidth=1.2,\n                label="P97_247D",\n                zorder=2,\n            )\n\n        # 왼쪽 축을 앞으로 가져오기\n        ax1.set_zorder(ax2.get_zorder() + 1)\n        ax1.patch.set_visible(False)\n\n        # 왼쪽 축: 가격 + MA30\n        ax1.plot(\n            plot_df.index,\n            plot_df["Price"],\n            linewidth=2.8,\n            label=ticker,\n            zorder=10,\n        )\n\n        ax1.plot(\n            plot_df.index,\n            plot_df["MA30"],\n            linewidth=2.4,\n            label="MA30",\n            zorder=9,\n        )\n\n        ax1.grid(True, alpha=0.25, zorder=0)\n\n        # ------------------------------------------------------------\n        # 제목/수치 정보는 그래프 안쪽 좌측 상단에 표시\n        # - 범례와 제목이 겹치지 않도록 set_title은 사용하지 않음\n        # - 그래프 텍스트는 영문만 사용해서 Matplotlib 한글 폰트 경고 방지\n        # ------------------------------------------------------------\n        info_text_1 = f"#{int(row[\'Sortino순위\'])} {ticker} | {industry}"\n        info_text_2 = (\n            f"GapExcess30={row[\'현재30일초과이격도\']:.2f} | "\n            f"P97_247D={row[\'최근247거래일97구간초과이격도\']:.2f} | "\n            f"Heat={row[\'97구간대비현재위치\']:.1f}"\n        )\n\n        ax1.text(\n            0.01,\n            0.96,\n            info_text_1,\n            transform=ax1.transAxes,\n            fontsize=12,\n            fontweight="bold",\n            va="top",\n            ha="left",\n            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),\n            zorder=20,\n        )\n\n        ax1.text(\n            0.01,\n            0.88,\n            info_text_2,\n            transform=ax1.transAxes,\n            fontsize=10.5,\n            fontweight="bold",\n            va="top",\n            ha="left",\n            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),\n            zorder=20,\n        )\n\n        ax1.set_ylabel("Price")\n        ax2.set_ylabel("Gap excess")\n\n        lines1, labels1 = ax1.get_legend_handles_labels()\n        lines2, labels2 = ax2.get_legend_handles_labels()\n\n        # 범례는 그래프 위쪽에 한 줄로 배치하되, 제목을 없앴기 때문에 겹치지 않음\n        ax1.legend(\n            lines2 + lines1,\n            labels2 + labels1,\n            loc="upper center",\n            bbox_to_anchor=(0.5, 1.10),\n            ncol=4,\n            frameon=False,\n            fontsize=10,\n        )\n\n        plt.tight_layout()\n        plt.show()\n\n\nsortino_ma10_gap247_p97_summary = build_sortino_ma10_gap247_p97_summary(\n    close_df=sector_data,\n    sortino_table=sortino_ma10,\n    industry_map=industry_etfs,\n    ma_days=30,\n    lookback_days=247,\n    percentile=0.97,\n    top_n=20,\n)\n\nprint("\\n📌 최종 분석: 미국 업종 ETF / 6개월 Sortino 10일선 이상 기준")\nprint(" - 현재30일초과이격도: (현재가 / MA30 - 1) * 100")\nprint(" - 최근247거래일97구간초과이격도: 최근 247거래일 기준 97% 분위수")\nprint(" - 97구간대비현재위치: 100 이상이면 최근 247거래일 97% 구간 초과")\ndisplay(sortino_ma10_gap247_p97_summary)\n\nplot_sortino_ma10_gap247_p97_graphs(\n    close_df=sector_data,\n    summary_df=sortino_ma10_gap247_p97_summary,\n    ma_days=30,\n    lookback_days=247,\n    figsize=(16, 4),\n    bar_alpha=0.15,\n)\n\n# ============================================================\n# 11. 4주 / 13주 / 26주 / 52주 이동평균 정배열 ETF\n#\n# 일봉 데이터 기준:\n# 4주  = 20거래일\n# 13주 = 65거래일\n# 26주 = 130거래일\n# 52주 = 260거래일\n#\n# 완전 정배열:\n# Price > MA20 > MA65 > MA130 > MA260\n#\n# 이평 정배열:\n# MA20 > MA65 > MA130 > MA260\n#\n# 추가:\n# - 정배열 시작일\n# - 현재까지 연속 유지 거래일\n# - 달력일 기준 유지기간\n# ============================================================\n\nMA_4W = 20\nMA_13W = 65\nMA_26W = 130\nMA_52W = 260\n\n\ndef _current_true_streak(mask: pd.Series):\n    """\n    최신일 기준 True가 연속으로 이어지는 거래일 수와 시작일 계산\n    """\n    mask = mask.dropna().astype(bool)\n\n    if mask.empty or not bool(mask.iloc[-1]):\n        return 0, pd.NaT\n\n    arr = mask.to_numpy()\n    false_positions = np.flatnonzero(~arr)\n\n    if len(false_positions) == 0:\n        start_pos = 0\n    else:\n        start_pos = false_positions[-1] + 1\n\n    streak_days = len(mask) - start_pos\n    start_date = mask.index[start_pos]\n\n    return int(streak_days), start_date\n\n\ndef get_us_etf_weekly_ma_alignment(\n    price_df: pd.DataFrame,\n    industry_map: dict\n) -> pd.DataFrame:\n    """\n    미국 ETF별 최신 유효 시점에서\n    4주(20일), 13주(65일), 26주(130일), 52주(260일)\n    이동평균 정배열 여부와 현재 연속 유지기간을 계산합니다.\n    """\n\n    price_df = price_df.sort_index().copy()\n\n    ma4 = price_df.rolling(\n        window=MA_4W,\n        min_periods=MA_4W\n    ).mean()\n\n    ma13 = price_df.rolling(\n        window=MA_13W,\n        min_periods=MA_13W\n    ).mean()\n\n    ma26 = price_df.rolling(\n        window=MA_26W,\n        min_periods=MA_26W\n    ).mean()\n\n    ma52 = price_df.rolling(\n        window=MA_52W,\n        min_periods=MA_52W\n    ).mean()\n\n    rows = []\n\n    for ticker in price_df.columns:\n\n        # SPY는 벤치마크이므로 정배열 업종 리스트에서 제외\n        if ticker == "SPY":\n            continue\n\n        temp = pd.DataFrame({\n            "Price": price_df[ticker],\n            "MA4W": ma4[ticker],\n            "MA13W": ma13[ticker],\n            "MA26W": ma26[ticker],\n            "MA52W": ma52[ticker],\n        }).dropna()\n\n        if temp.empty:\n            continue\n\n        # 날짜별 이평 정배열 상태\n        ma_mask = (\n            (temp["MA4W"] > temp["MA13W"])\n            & (temp["MA13W"] > temp["MA26W"])\n            & (temp["MA26W"] > temp["MA52W"])\n        )\n\n        # 날짜별 완전 정배열 상태\n        full_mask = (\n            (temp["Price"] > temp["MA4W"])\n            & ma_mask\n        )\n\n        latest_date = temp.index[-1]\n        row = temp.iloc[-1]\n\n        ma_alignment = bool(ma_mask.iloc[-1])\n        full_alignment = bool(full_mask.iloc[-1])\n\n        ma_streak_days, ma_start_date = _current_true_streak(ma_mask)\n        full_streak_days, full_start_date = _current_true_streak(full_mask)\n\n        ma_calendar_days = (\n            (latest_date - ma_start_date).days + 1\n            if ma_alignment and pd.notna(ma_start_date)\n            else 0\n        )\n\n        full_calendar_days = (\n            (latest_date - full_start_date).days + 1\n            if full_alignment and pd.notna(full_start_date)\n            else 0\n        )\n\n        gap_price_4w = (\n            row["Price"] / row["MA4W"] - 1\n        ) * 100\n\n        gap_4w_13w = (\n            row["MA4W"] / row["MA13W"] - 1\n        ) * 100\n\n        gap_13w_26w = (\n            row["MA13W"] / row["MA26W"] - 1\n        ) * 100\n\n        gap_26w_52w = (\n            row["MA26W"] / row["MA52W"] - 1\n        ) * 100\n\n        alignment_strength = (\n            gap_price_4w\n            + gap_4w_13w\n            + gap_13w_26w\n            + gap_26w_52w\n        )\n\n        rows.append({\n            "Ticker": ticker,\n            "Industry": industry_map.get(ticker, ""),\n            "Date": latest_date.date(),\n\n            "Price": float(row["Price"]),\n            "MA4W": float(row["MA4W"]),\n            "MA13W": float(row["MA13W"]),\n            "MA26W": float(row["MA26W"]),\n            "MA52W": float(row["MA52W"]),\n\n            "Price>4W": bool(row["Price"] > row["MA4W"]),\n            "4W>13W": bool(row["MA4W"] > row["MA13W"]),\n            "13W>26W": bool(row["MA13W"] > row["MA26W"]),\n            "26W>52W": bool(row["MA26W"] > row["MA52W"]),\n\n            "MA_Alignment": ma_alignment,\n            "Full_Alignment": full_alignment,\n\n            "MA_Alignment_Start": (\n                ma_start_date.date()\n                if ma_alignment and pd.notna(ma_start_date)\n                else None\n            ),\n            "MA_Alignment_TradingDays": ma_streak_days,\n            "MA_Alignment_CalendarDays": ma_calendar_days,\n\n            "Full_Alignment_Start": (\n                full_start_date.date()\n                if full_alignment and pd.notna(full_start_date)\n                else None\n            ),\n            "Full_Alignment_TradingDays": full_streak_days,\n            "Full_Alignment_CalendarDays": full_calendar_days,\n\n            "Price_4W_Gap(%)": float(gap_price_4w),\n            "4W_13W_Gap(%)": float(gap_4w_13w),\n            "13W_26W_Gap(%)": float(gap_13w_26w),\n            "26W_52W_Gap(%)": float(gap_26w_52w),\n            "Alignment_Strength": float(alignment_strength),\n        })\n\n    out = pd.DataFrame(rows)\n\n    if out.empty:\n        return out\n\n    return (\n        out\n        .sort_values(\n            by=[\n                "Full_Alignment",\n                "Full_Alignment_TradingDays",\n                "Alignment_Strength"\n            ],\n            ascending=[False, False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n\n# ------------------------------------------------------------\n# 정배열 계산\n# ------------------------------------------------------------\n\nus_weekly_alignment_df = get_us_etf_weekly_ma_alignment(\n    price_df=data,\n    industry_map=industry_etfs\n)\n\n\nif us_weekly_alignment_df.empty:\n\n    print(\n        "\\n⚠️ 4주·13주·26주·52주 정배열을 계산할 수 있는 "\n        "충분한 ETF 데이터가 없습니다."\n    )\n\nelse:\n\n    # --------------------------------------------------------\n    # ① 완전 정배열\n    # Price > 4W > 13W > 26W > 52W\n    # --------------------------------------------------------\n\n    us_full_alignment_df = (\n        us_weekly_alignment_df[\n            us_weekly_alignment_df["Full_Alignment"]\n        ]\n        .copy()\n        .sort_values(\n            by=[\n                "Full_Alignment_TradingDays",\n                "Alignment_Strength"\n            ],\n            ascending=[False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n    us_full_alignment_df.index += 1\n    us_full_alignment_df.index.name = "Rank"\n\n    print(\n        "\\n📌 미국 ETF 4주·13주·26주·52주 완전 정배열"\n        "\\n   Price > 4W > 13W > 26W > 52W"\n        "\\n   ※ 정배열 시작일과 연속 유지기간 포함"\n    )\n\n    if us_full_alignment_df.empty:\n        print("해당 ETF가 없습니다.")\n    else:\n        display(\n            us_full_alignment_df[\n                [\n                    "Ticker",\n                    "Industry",\n                    "Date",\n                    "Full_Alignment_Start",\n                    "Full_Alignment_TradingDays",\n                    "Full_Alignment_CalendarDays",\n                    "Price",\n                    "MA4W",\n                    "MA13W",\n                    "MA26W",\n                    "MA52W",\n                    "Price_4W_Gap(%)",\n                    "4W_13W_Gap(%)",\n                    "13W_26W_Gap(%)",\n                    "26W_52W_Gap(%)",\n                    "Alignment_Strength",\n                ]\n            ].style.format({\n                "Price": "{:,.2f}",\n                "MA4W": "{:,.2f}",\n                "MA13W": "{:,.2f}",\n                "MA26W": "{:,.2f}",\n                "MA52W": "{:,.2f}",\n                "Price_4W_Gap(%)": "{:+.2f}%",\n                "4W_13W_Gap(%)": "{:+.2f}%",\n                "13W_26W_Gap(%)": "{:+.2f}%",\n                "26W_52W_Gap(%)": "{:+.2f}%",\n                "Alignment_Strength": "{:.2f}",\n            })\n        )\n\n\n    # --------------------------------------------------------\n    # ② 이평 정배열\n    # 4W > 13W > 26W > 52W\n    # --------------------------------------------------------\n\n    us_ma_alignment_df = (\n        us_weekly_alignment_df[\n            us_weekly_alignment_df["MA_Alignment"]\n        ]\n        .copy()\n        .sort_values(\n            by=[\n                "MA_Alignment_TradingDays",\n                "Alignment_Strength"\n            ],\n            ascending=[False, False]\n        )\n        .reset_index(drop=True)\n    )\n\n    us_ma_alignment_df.index += 1\n    us_ma_alignment_df.index.name = "Rank"\n\n    print(\n        "\\n📌 미국 ETF 이동평균 정배열"\n        "\\n   4W > 13W > 26W > 52W"\n        "\\n   ※ 정배열 시작일과 연속 유지기간 포함"\n    )\n\n    if us_ma_alignment_df.empty:\n        print("해당 ETF가 없습니다.")\n    else:\n        display(\n            us_ma_alignment_df[\n                [\n                    "Ticker",\n                    "Industry",\n                    "Date",\n                    "MA_Alignment_Start",\n                    "MA_Alignment_TradingDays",\n                    "MA_Alignment_CalendarDays",\n                    "Price",\n                    "MA4W",\n                    "MA13W",\n                    "MA26W",\n                    "MA52W",\n                    "Price>4W",\n                    "Alignment_Strength",\n                ]\n            ].style.format({\n                "Price": "{:,.2f}",\n                "MA4W": "{:,.2f}",\n                "MA13W": "{:,.2f}",\n                "MA26W": "{:,.2f}",\n                "MA52W": "{:,.2f}",\n                "Alignment_Strength": "{:.2f}",\n            })\n        )\n\n\n    # --------------------------------------------------------\n    # ③ 전체 ETF 정배열 상태표\n    # --------------------------------------------------------\n\n    print("\\n📌 전체 미국 ETF 정배열 상태")\n\n    display(\n        us_weekly_alignment_df[\n            [\n                "Ticker",\n                "Industry",\n                "Date",\n                "MA_Alignment",\n                "MA_Alignment_Start",\n                "MA_Alignment_TradingDays",\n                "Full_Alignment",\n                "Full_Alignment_Start",\n                "Full_Alignment_TradingDays",\n                "Alignment_Strength",\n            ]\n        ].style.format({\n            "Alignment_Strength": "{:.2f}",\n        })\n    )\n\n\n    # --------------------------------------------------------\n    # ④ ETF 이름 + 유지기간 간단 출력\n    # --------------------------------------------------------\n\n    print(\n        "\\n🔥 완전 정배열 ETF "\n        "(Price > 4W > 13W > 26W > 52W)"\n    )\n\n    if len(us_full_alignment_df) > 0:\n\n        for i, (_, row) in enumerate(\n            us_full_alignment_df.iterrows(),\n            start=1\n        ):\n            print(\n                f"{i}. {row[\'Ticker\']} "\n                f"({row[\'Industry\']}) | "\n                f"{row[\'Full_Alignment_Start\']}부터 | "\n                f"{int(row[\'Full_Alignment_TradingDays\'])}거래일 유지"\n            )\n\n    else:\n        print("없음")\n\n\n    print(\n        "\\n📈 이평 정배열 ETF "\n        "(4W > 13W > 26W > 52W)"\n    )\n\n    if len(us_ma_alignment_df) > 0:\n\n        for i, (_, row) in enumerate(\n            us_ma_alignment_df.iterrows(),\n            start=1\n        ):\n            print(\n                f"{i}. {row[\'Ticker\']} "\n                f"({row[\'Industry\']}) | "\n                f"{row[\'MA_Alignment_Start\']}부터 | "\n                f"{int(row[\'MA_Alignment_TradingDays\'])}거래일 유지"\n            )\n\n    else:\n        print("없음")\n\n\n    # --------------------------------------------------------\n    # ⑤ 정배열 현황 요약\n    # --------------------------------------------------------\n\n    print(\n        f"\\n📊 미국 ETF 정배열 현황"\n        f"\\n- 완전 정배열: {len(us_full_alignment_df)}개"\n        f"\\n- 이평 정배열: {len(us_ma_alignment_df)}개"\n        f"\\n- 분석 가능 전체 ETF: {len(us_weekly_alignment_df)}개"\n    )'

def _tail_df(obj, n=None):
    if isinstance(obj, pd.DataFrame):
        return obj.tail(n).copy() if n else obj.copy()
    if isinstance(obj, pd.Series):
        return obj.tail(n).copy() if n else obj.copy()
    return obj


def _build_kr_gap_chart_data(ns):
    """Sortino 10주 조건의 과열도 차트를 숫자만 저장하도록 축약."""
    out = {}
    price_df = ns.get("etf_data")
    summary_df = ns.get("summary_df")
    if not isinstance(price_df, pd.DataFrame) or not isinstance(summary_df, pd.DataFrame):
        return out
    for _, row in summary_df.iterrows():
        sector = row.get("섹터")
        if sector not in price_df.columns:
            continue
        p = pd.to_numeric(price_df[sector], errors="coerce").dropna()
        if p.empty:
            continue
        ma30 = p.rolling(30).mean()
        gap30 = p / ma30 * 100
        z = pd.DataFrame({"Price": p, "MA30": ma30, "Gap30": gap30}).dropna().tail(320)
        if not z.empty:
            out[str(sector)] = z
    return out


def _compact_korea_sector_namespace(ns):
    """전체 원본 계산은 유지하되 GitHub에는 화면 재현에 필요한 숫자만 저장."""
    keep = [
        "mansfield_df", "mansfield_df_70", "momentum_df",
        "sortino_ma10", "sortino_ma20", "sortino_ma10w",
        "etr_ma10", "etr_ma20", "etr_ma10w",
        "six_month_summary", "breadth_summary", "latest_count_table",
        "weekly_alignment_df", "full_alignment_df", "ma_alignment_df",
        "summary_df", "summary", "yearly", "stats_table", "top12_df",
        "latest_candidates", "latest_date", "signal",
        "latest_crowd", "latest_ma120_breadth", "latest_ma120_count",
        "latest_ma120_valid", "reverse_breadth_signal",
    ]
    out = {}
    for k in keep:
        if k in ns:
            out[k] = ns[k]

    # 시계열은 필요한 컬럼/최근 표시 구간 중심으로 축약
    if isinstance(ns.get("crowding_series"), pd.Series):
        out["crowding_series"] = ns["crowding_series"].tail(520).copy()
    if isinstance(ns.get("sector_breadth"), pd.DataFrame):
        cols = [c for c in [
            "Above_MA10", "Above_MA20", "Above_MA50", "Above_MA120",
            "Above_MA10_Smooth", "Above_MA20_Smooth", "Above_MA50_Smooth", "Above_MA120_Smooth",
            "Breadth_Score", "Breadth_Score_MA5", "Breadth_Slope_5D", "Breadth_Slope_10D",
        ] if c in ns["sector_breadth"].columns]
        out["sector_breadth"] = ns["sector_breadth"][cols].tail(520).copy()

    # 52주 전략 시계열/거래기록
    for k in ["strategy_live", "kospi_live", "strategy_dd", "kospi_dd"]:
        if isinstance(ns.get(k), pd.Series):
            out[k] = ns[k].copy()
    if isinstance(ns.get("bt_trades"), pd.DataFrame):
        out["bt_trades"] = ns["bt_trades"].tail(100).copy()
    if isinstance(ns.get("live_holdings"), pd.DataFrame):
        out["live_holdings"] = ns["live_holdings"].tail(120).copy()

    # 6개월 리더/래거드 표는 원본 ret6m 전체 대신 최신 행만 저장
    if isinstance(ns.get("ret6m"), pd.DataFrame) and len(ns["ret6m"]):
        try:
            latest = ns["ret6m"].dropna(how="all").iloc[-1].dropna().sort_values(ascending=False)
            out["leaders_6m"] = latest.head(10).rename("6M Return").to_frame()
            out["laggards_6m"] = latest.tail(10).sort_values().rename("6M Return").to_frame()
        except Exception:
            pass

    out["gap_chart_data"] = _build_kr_gap_chart_data(ns)
    return out


def _kr_display_event(obj):
    """Convert a Colab display() object into a compact serializable event."""
    try:
        if obj.__class__.__name__ == "Styler" and hasattr(obj, "to_html"):
            return {"type": "html", "html": obj.to_html()}
    except Exception:
        pass
    if isinstance(obj, pd.Series):
        return {"type": "table", "data": obj.to_frame()}
    if isinstance(obj, pd.DataFrame):
        return {"type": "table", "data": obj.copy()}
    try:
        return {"type": "text", "text": str(obj) + "\n"}
    except Exception:
        return None


def run_korea_sector(excel_bytes):
    """
    User's Colab source runs almost verbatim.
    No IPython/Colab dependency: display(), print(), plt.show() are captured as ordered events.
    GitHub stores text/tables/chart coordinates only, not screenshot images in normal cases.
    """
    src = KR_SECTOR_SRC
    events = []
    diagnostics = io.StringIO()

    def _append_text(txt):
        if not txt:
            return
        if events and events[-1].get("type") == "text":
            events[-1]["text"] += txt
        else:
            events.append({"type": "text", "text": txt})

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        try:
            txt = sep.join(str(x) for x in args) + end
        except Exception:
            txt = " ".join(repr(x) for x in args) + end
        _append_text(txt)

    def _display(obj):
        ev = _kr_display_event(obj)
        if ev is not None:
            events.append(ev)

    old_show = plt.show
    def _show(*args, **kwargs):
        try:
            fig = plt.gcf()
            marker = _figure_marker(fig)
            if marker is not None:
                events.append({"type": "figure", "figure": marker})
            plt.close(fig)
        except Exception:
            pass

    plt.show = _show
    ns = {"display": _display, "print": _print, "EXCEL_BYTES": excel_bytes}
    try:
        with contextlib.redirect_stderr(diagnostics):
            exec(src, ns, ns)
        payload = {"colab_events": events, "event_count": len(events)}
        return True, payload, diagnostics.getvalue()[-12000:], [], []
    except Exception:
        tb = traceback.format_exc()
        diagnostics.write("\n" + tb)
        _append_text("\n❌ 계산 오류\n" + tb)
        return False, {"colab_events": events}, diagnostics.getvalue()[-20000:], [], []
    finally:
        plt.show = old_show
        plt.close("all")


def _render_static_df(df):
    if not isinstance(df, pd.DataFrame):
        return
    # Similar to notebook display: static HTML table, not an interactive spreadsheet grid.
    try:
        html_table = df.to_html(max_rows=80, max_cols=30, border=0, classes="colab-static-table")
        st.markdown(f'<div class="colab-table-wrap">{html_table}</div>', unsafe_allow_html=True)
    except Exception:
        st.write(df)


def render_korea_sector_colab(payload):
    if not isinstance(payload, dict):
        return
    events = payload.get("colab_events") or []
    st.markdown("""
    <style>
    .colab-output{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;line-height:1.48;
      background:transparent;border:0;padding:2px 0 8px 0;margin:0;color:inherit;font-size:.92rem}
    .colab-table-wrap{overflow-x:auto;margin:4px 0 18px 0}
    table.colab-static-table{border-collapse:collapse;font-size:.88rem;white-space:nowrap;background:transparent}
    table.colab-static-table th,table.colab-static-table td{border:1px solid rgba(128,128,128,.28);padding:5px 8px;text-align:right}
    table.colab-static-table th{font-weight:700;background:rgba(128,128,128,.08)}
    </style>
    """, unsafe_allow_html=True)
    for ev in events:
        typ = ev.get("type") if isinstance(ev, dict) else None
        if typ == "text":
            txt = ev.get("text", "")
            if txt.strip():
                import html as _html
                st.markdown(f'<pre class="colab-output">{_html.escape(txt)}</pre>', unsafe_allow_html=True)
        elif typ == "table":
            _render_static_df(ev.get("data"))
        elif typ == "html":
            h = ev.get("html", "")
            if h:
                st.markdown(f'<div class="colab-table-wrap">{h}</div>', unsafe_allow_html=True)
        elif typ == "figure":
            _render_saved_fig(ev.get("figure"))


def render_korea_sector_compact(ns):
    # Backward compatibility: new result = Colab-style events. Old compact result still opens if present.
    if isinstance(ns, dict) and ns.get("colab_events") is not None:
        render_korea_sector_colab(ns)
        return
    if not isinstance(ns, dict):
        return
    st.info("이 결과는 이전 버전 저장본입니다. 관리자에서 엑셀을 한 번 다시 계산하면 구글 코랩형 출력으로 바뀝니다.")
    for key, title in [
        ("mansfield_df_70","Mansfield RS ≥ 70"), ("six_month_summary","6개월 수익률 · KOSPI 대비 상대수익률"),
        ("momentum_df","6개월 변동성 조정 모멘텀 Top 20"), ("sortino_ma10","Sortino · 10일선 이상"),
        ("sortino_ma20","Sortino · 20일선 이상"), ("sortino_ma10w","Sortino · 10주선 이상")]:
        df=ns.get(key)
        if isinstance(df,pd.DataFrame) and not df.empty:
            st.markdown(f"**{title}**")
            _render_static_df(df)


# ============================================================
# v11.4: 원본 Google Colab 출력 순서를 그대로 보존하는 공통 실행기
# - print / display / plt.show 순서를 event로 저장
# - Streamlit dataframe 대신 notebook형 정적 표 사용
# - 그래프는 PNG 누적 대신 좌표 중심 저장
# ============================================================
def run_colab_source(source):
    events = []
    diagnostics = io.StringIO()
    raw_stdout = io.StringIO()

    def _append_text(txt):
        if not txt:
            return
        if events and isinstance(events[-1], dict) and events[-1].get("type") == "text":
            events[-1]["text"] += txt
        else:
            events.append({"type": "text", "text": txt})

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        try:
            txt = sep.join(str(x) for x in args) + end
        except Exception:
            txt = " ".join(repr(x) for x in args) + end
        _append_text(txt)

    def _display(obj):
        ev = _kr_display_event(obj)
        if ev is not None:
            events.append(ev)

    old_show = plt.show
    def _show(*args, **kwargs):
        try:
            fig = plt.gcf()
            marker = _figure_marker(fig)
            if marker is not None:
                events.append({"type": "figure", "figure": marker})
            plt.close(fig)
        except Exception:
            pass

    plt.show = _show
    ns = {"display": _display, "print": _print}
    try:
        with contextlib.redirect_stdout(raw_stdout), contextlib.redirect_stderr(diagnostics):
            exec(source, ns, ns)
        extra = raw_stdout.getvalue()
        if extra.strip():
            _append_text(extra)
        payload = {"colab_events": events, "event_count": len(events)}
        return True, payload, diagnostics.getvalue()[-12000:], [], []
    except Exception:
        tb = traceback.format_exc()
        diagnostics.write("\n" + tb)
        extra = raw_stdout.getvalue()
        if extra.strip():
            _append_text(extra)
        _append_text("\n❌ 계산 오류\n" + tb)
        return False, {"colab_events": events}, diagnostics.getvalue()[-20000:], [], []
    finally:
        plt.show = old_show
        plt.close("all")



def _parse_us_alignment_items(section_text):
    rows = []
    pat = re.compile(
        r"(\\d+)\\.\\s*(.+?)\\s*\\|\\s*(\\d{4}-\\d{2}-\\d{2})부터\\s*\\|\\s*(\\d+)거래일 유지"
        r"(?=\\s*\\d+\\.|\\s*$)",
        re.S,
    )
    for m in pat.finditer(section_text.strip()):
        label = re.sub(r"\\s+", " ", m.group(2)).strip()
        ticker = label
        desc = ""
        mm = re.match(r"^([A-Z0-9.^_-]+)\\s*\\((.*)\\)$", label)
        if mm:
            ticker, desc = mm.group(1), mm.group(2)
        rows.append({
            "순위": int(m.group(1)),
            "ETF": ticker,
            "업종/테마": desc,
            "정배열 시작일": m.group(3),
            "유지 거래일": int(m.group(4)),
        })
    return pd.DataFrame(rows)


def _render_us_alignment_text(txt):
    """미국 ETF 정배열 출력에서는 개별 목록을 숨기고 마지막 요약만 표시."""
    if "🔥 완전 정배열 ETF" not in txt or "📈 이평 정배열 ETF" not in txt:
        return False

    # 정배열 개별 종목 목록은 표시하지 않고 요약 숫자만 추출한다.
    summary_part = ""
    if "📊 미국 ETF 정배열 현황" in txt:
        summary_part = txt.split("📊 미국 ETF 정배열 현황", 1)[1]

    full_n = re.search(r"완전 정배열\s*:\s*(\d+)개", summary_part)
    ma_n = re.search(r"이평 정배열\s*:\s*(\d+)개", summary_part)
    total_n = re.search(r"분석 가능 전체(?:\s*ETF)?\s*:\s*(\d+)개", summary_part)

    st.markdown("### 📊 미국 ETF 정배열 현황")
    a, b, c = st.columns(3)
    a.metric("완전 정배열", f"{full_n.group(1)}개" if full_n else "-")
    b.metric("이평 정배열", f"{ma_n.group(1)}개" if ma_n else "-")
    c.metric("분석 가능 전체", f"{total_n.group(1)}개" if total_n else "-")
    return True


def render_us_sector_colab(payload):
    """
    기본 출력 순서는 한국 ETF와 동일하게 유지.
    단, 마지막 정배열 장문 print만 표 형태로 바꿔 가독성을 높인다.
    """
    if not isinstance(payload, dict):
        return
    events = payload.get("colab_events") or []

    st.markdown("""
    <style>
    .colab-output{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;line-height:1.48;
      background:transparent;border:0;padding:2px 0 8px 0;margin:0;color:inherit;font-size:.92rem}
    .colab-table-wrap{overflow-x:auto;margin:4px 0 18px 0}
    table.colab-static-table{border-collapse:collapse;font-size:.88rem;white-space:nowrap;background:transparent}
    table.colab-static-table th,table.colab-static-table td{border:1px solid rgba(128,128,128,.28);padding:5px 8px;text-align:right}
    table.colab-static-table th{font-weight:700;background:rgba(128,128,128,.08)}
    </style>
    """, unsafe_allow_html=True)

    import html as _html
    for ev in events:
        typ = ev.get("type") if isinstance(ev, dict) else None
        if typ == "text":
            txt = ev.get("text", "")
            if txt.strip():
                if not _render_us_alignment_text(txt):
                    st.markdown(f'<pre class="colab-output">{_html.escape(txt)}</pre>', unsafe_allow_html=True)
        elif typ == "table":
            _render_static_df(ev.get("data"))
        elif typ == "html":
            h = ev.get("html", "")
            if h:
                st.markdown(f'<div class="colab-table-wrap">{h}</div>', unsafe_allow_html=True)
        elif typ == "figure":
            _render_saved_fig(ev.get("figure"))


def render_colab_result(result):
    """새 저장본은 Colab event 순서 그대로, 이전 저장본은 정적 notebook형 fallback."""
    if result is None:
        return
    try:
        ok, payload, txt, shown, figs = result
    except Exception:
        st.error("저장 결과 형식을 읽을 수 없습니다.")
        return

    if isinstance(payload, dict) and payload.get("colab_events") is not None:
        render_korea_sector_colab(payload)
        return

    # v11.3 이전 저장본: 순서 정보는 없으므로 한 번 재업데이트를 권장하되
    # 엑셀형 interactive grid는 쓰지 않고 정적 notebook 스타일로 표시.
    st.info("이 결과는 이전 저장 형식입니다. 관리자가 이 항목을 한 번 업데이트하면 Google Colab 출력 순서 그대로 저장됩니다.")
    if txt and txt.strip():
        import html as _html
        st.markdown(f'<pre class="colab-output">{_html.escape(txt)}</pre>', unsafe_allow_html=True)
    for obj in (shown or []):
        if isinstance(obj, pd.Series):
            _render_static_df(obj.to_frame())
        elif isinstance(obj, pd.DataFrame):
            _render_static_df(obj)
        else:
            try:
                st.markdown(f'<pre class="colab-output">{_html.escape(str(obj))}</pre>', unsafe_allow_html=True)
            except Exception:
                pass
    for fig in (figs or []):
        _render_saved_fig(fig)


# 세 분석은 원본 Colab source를 거의 그대로 실행하고 출력 순서까지 저장한다.
def run_ai():
    return run_colab_source(AI_SRC)


def run_rotation():
    """
    52주 전략은 화면에 필요한 핵심 결과만 저장한다.
    - 현재 52W 전략 작동 여부
    - 현재 포트폴리오 / 오늘 BUY·SELL
    - 백테스트 성과표
    - 백테스트 equity curve
    과거 거래내역/최근 30일 포지션/중간 출력은 저장하지 않는다.
    """
    diagnostics = io.StringIO()
    raw_stdout = io.StringIO()

    def _quiet_print(*args, **kwargs):
        return None

    def _quiet_display(*args, **kwargs):
        return None

    old_show = plt.show

    def _quiet_show(*args, **kwargs):
        try:
            plt.close(plt.gcf())
        except Exception:
            pass

    plt.show = _quiet_show
    ns = {"display": _quiet_display, "print": _quiet_print}

    try:
        with contextlib.redirect_stdout(raw_stdout), contextlib.redirect_stderr(diagnostics):
            exec(ROT_SRC, ns, ns)

        final_result = ns.get("final_result", {})
        summary = ns.get("summary")
        baseline_live = ns.get("baseline_live")
        final_live = ns.get("final_live")
        spy_live = ns.get("spy_live")
        industry_etfs = ns.get("industry_etfs", {})
        eps = float(ns.get("EPS", 1e-12))

        live_excess = final_result.get("Live_Excess_20D", np.nan)
        regime = final_result.get("Live_Regime", "")
        # 원본 전략의 weak-regime 판정 자체가
        # 순수 52W 최근 20D SPY 초과수익 <= -1% 인지 여부다.
        if pd.notna(live_excess):
            working = bool(float(live_excess) > -0.01)
            status = "통함" if working else "안통함"
        else:
            working = None
            status = "판정 대기"

        weights = final_result.get("Final_Weights")
        origins = final_result.get("Origins", {})
        current_rows = []
        if isinstance(weights, pd.Series):
            held = weights[weights > eps].sort_values(ascending=False)
            for ticker, weight in held.items():
                current_rows.append({
                    "Ticker": ticker,
                    "Industry": industry_etfs.get(ticker, ""),
                    "Origin": origins.get(ticker, ""),
                    "Weight": float(weight),
                })
        current_df = pd.DataFrame(current_rows)

        sells = list(final_result.get("Live_SELL") or [])
        buys = list(final_result.get("Live_BUY") or [])

        action_rows = []
        for ticker in sells:
            action_rows.append({
                "처리": "SELL",
                "Ticker": ticker,
                "Industry": industry_etfs.get(ticker, ""),
            })
        for ticker in buys:
            action_rows.append({
                "처리": "BUY",
                "Ticker": ticker,
                "Industry": industry_etfs.get(ticker, ""),
            })
        action_df = pd.DataFrame(action_rows)

        curves = pd.DataFrame()
        curve_parts = []
        for name, obj in [
            ("52W FIXED", baseline_live),
            ("52W + Rotation_B", final_live),
            ("SPY", spy_live),
        ]:
            if isinstance(obj, pd.Series):
                x = obj.rename(name)
                curve_parts.append(x)
        if curve_parts:
            curves = pd.concat(curve_parts, axis=1).dropna(how="all")

        # 저장용 성과표는 숫자 그대로 보존
        perf = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame()

        invested = float(current_df["Weight"].sum()) if not current_df.empty else 0.0

        payload = {
            "simple_rotation": True,
            "status": status,
            "working": working,
            "live_date": final_result.get("Live_Date"),
            "live_excess_20d": live_excess,
            "regime": regime,
            "new_buy_mode": final_result.get("Live_Origin", ""),
            "portfolio": current_df,
            "actions": action_df,
            "invested": invested,
            "cash": max(0.0, 1.0 - invested),
            "performance": perf,
            "equity_curve": curves,
        }
        return True, payload, diagnostics.getvalue()[-12000:], [], []

    except Exception:
        tb = traceback.format_exc()
        diagnostics.write("\n" + tb)
        return False, {"simple_rotation": True}, diagnostics.getvalue()[-20000:], [], []

    finally:
        plt.show = old_show
        plt.close("all")


def run_us_sector():
    return run_colab_source(US_SECTOR_SRC)


def render_rotation_simple(result):
    """미국 52주 신고가 전략의 핵심 상태/포트/성과만 심플하게 표시."""
    if result is None:
        return

    try:
        ok, payload, err, _, _ = result
    except Exception:
        st.error("저장 결과 형식을 읽을 수 없습니다.")
        return

    if not ok:
        st.error("미국 52주 신고가 전략 계산 오류")
        if err:
            st.code(err[-12000:], language="text")
        return

    if not isinstance(payload, dict) or not payload.get("simple_rotation"):
        st.info("이 결과는 이전 저장 형식입니다. 관리자가 이 항목을 한 번 업데이트하면 간단한 새 화면으로 바뀝니다.")
        return

    status = payload.get("status", "판정 대기")
    excess = payload.get("live_excess_20d", np.nan)
    regime = payload.get("regime", "")

    if status == "통함":
        st.success("✅ 현재 52주 신고가 전략: 통함")
    elif status == "안통함":
        st.warning("⚠️ 현재 52주 신고가 전략: 안통함")
    else:
        st.info("현재 52주 신고가 전략: 판정 대기")

    if pd.notna(excess):
        st.caption(
            f"최근 20거래일 순수 52W 전략의 SPY 대비 초과수익 {float(excess):+.2%} "
            f"· 판정 기준 -1% · 현재 Regime {regime}"
        )
    elif regime:
        st.caption(f"현재 Regime {regime}")

    st.markdown("### 현재 포트폴리오")
    portfolio = payload.get("portfolio")
    if isinstance(portfolio, pd.DataFrame) and not portfolio.empty:
        pf = portfolio.copy()
        if "Weight" in pf.columns:
            pf["Weight"] = pf["Weight"].map(lambda x: f"{float(x):.2%}")
        _render_static_df(pf)
        st.caption(
            f"투자비중 {float(payload.get('invested', 0)):.2%} · "
            f"현금비중 {float(payload.get('cash', 0)):.2%}"
        )
    else:
        st.caption("현재 보유종목 없음 · 현금 100%")

    st.markdown("### 오늘 매매 처리")
    actions = payload.get("actions")
    if isinstance(actions, pd.DataFrame) and not actions.empty:
        _render_static_df(actions)
    else:
        st.caption("오늘 BUY / SELL 없음")

    st.markdown("### 백테스트 성과")
    perf = payload.get("performance")
    if isinstance(perf, pd.DataFrame) and not perf.empty:
        show = perf.copy()
        for col in ["CAGR", "MDD"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{float(x):.2%}" if pd.notna(x) else "-")
        for col in ["Sharpe", "Sortino", "Calmar"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
        if "Annual_Turnover" in show.columns:
            show["Annual_Turnover"] = show["Annual_Turnover"].map(
                lambda x: f"{float(x):.2f}x" if pd.notna(x) else "-"
            )
        if "Ending_Multiple" in show.columns:
            show["Ending_Multiple"] = show["Ending_Multiple"].map(
                lambda x: f"{float(x):.2f}x" if pd.notna(x) else "-"
            )
        _render_static_df(show)

    st.markdown("### 백테스트 결과 그래프")
    curves = payload.get("equity_curve")
    if isinstance(curves, pd.DataFrame) and not curves.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        for col in curves.columns:
            ax.plot(curves.index, curves[col], label=str(col), linewidth=1.8)
        ax.set_title("52W Strategy Backtest")
        ax.set_ylabel("Growth of $1")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def check_admin_password():
    """관리자 비밀번호는 Streamlit Secrets의 ADMIN_PASSWORD에서 읽습니다."""
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    with st.sidebar:
        st.markdown("### 🔐 관리자")
        if st.session_state.admin_authenticated:
            st.success("관리자 모드")
            if st.button("관리자 로그아웃", key="admin_logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
        else:
            pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw")
            if st.button("관리자 로그인", key="admin_login"):
                try:
                    expected = st.secrets["ADMIN_PASSWORD"]
                except Exception:
                    st.error("Streamlit Secrets에 ADMIN_PASSWORD를 먼저 등록하세요.")
                    return False
                if pw == expected:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("비밀번호가 맞지 않습니다.")
    return st.session_state.admin_authenticated

# ============================================================
# v11: 왼쪽 그룹형 네비게이션
# ============================================================
NAV_GROUPS = {
    "시장 지표": [
        "미국 유동성 체크",
        "미국 피어앤그리드 오실레이터",
        "미국 위험신호",
        "한국 피어앤그리드 오실레이터",
    ],
    "반도체 데이터 점검": [
        "AI하드웨어 주가 모멘텀 점검",
        "대만 월별 매출",
    ],
    "주도주·업종": [
        "미국 52주 신고가 전략 점검",
        "미국 ETF 소라티노 및 상대강도",
        "한국 ETF 소라티노 및 상대강도",
    ],
}
ALL_PAGES = [p for pages in NAV_GROUPS.values() for p in pages]
if st.session_state.get("active_page") not in ALL_PAGES:
    st.session_state["active_page"] = "미국 유동성 체크"

with st.sidebar:
    st.markdown("## 태린이아빠")
    st.caption("Market Dashboard · LIVE v11.12")
    st.markdown("---")
    for _group, _pages in NAV_GROUPS.items():
        st.markdown(f"**{_group}**")
        for _page in _pages:
            _active = st.session_state["active_page"] == _page
            if st.button(
                ("● " if _active else "") + _page,
                key=f"nav_{_page}",
                use_container_width=True,
                type="primary" if _active else "secondary",
            ):
                st.session_state["active_page"] = _page
                st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

ACTIVE_PAGE = st.session_state["active_page"]
IS_ADMIN = check_admin_password()
with st.sidebar:
    st.divider()
    st.caption("💾 결과 저장: GitHub 영구저장 · gzip 압축")
    st.caption(f"{GH_OWNER}/{GH_DATA_REPO}")
    st.info("자동 업데이트 OFF")
    st.caption("관리자가 업데이트를 눌렀을 때만 외부 데이터를 다시 가져옵니다.")

st.markdown(
    f'<div class="hero"><h1>{ACTIVE_PAGE}</h1><p>태린이아빠 Market Dashboard · 저장된 마지막 결과를 표시합니다.</p></div>',
    unsafe_allow_html=True
)

PAGE_GUIDES = {
    "미국 유동성 체크": {
        "why": "주식시장의 큰 방향은 기업 실적뿐 아니라 시장에 실제로 풀려 있는 달러 유동성의 영향을 크게 받습니다. 연준·TGA·RRP·은행 준비금·신용환경을 같이 보면 '돈이 위험자산으로 들어오기 쉬운 환경인지'를 먼저 확인할 수 있습니다.",
        "check": "유동성이 개선되는지 악화되는지, 민간신용이 받쳐주는지, 장기금리·엔캐리 같은 충격 요인이 동시에 커지는지를 봅니다. 단일 수치보다 여러 항목이 같은 방향으로 움직이는지가 중요합니다.",
    },
    "미국 피어앤그리드 오실레이터": {
        "why": "지수가 오르는 것과 시장 내부가 건강한 것은 다를 수 있습니다. 변동성·신용스프레드·주가 모멘텀 등을 함께 보면 시장이 지나치게 낙관적인지, 공포가 과도한지 온도를 확인할 수 있습니다.",
        "check": "극단적 탐욕은 추격매수 위험을, 극단적 공포는 투매 가능성을 점검하는 용도입니다. 방향 예측 신호라기보다 현재 시장의 과열·위축 정도를 확인하는 보조지표로 봅니다.",
    },
    "미국 위험신호": {
        "why": "나스닥(QQQ)만 보면 주가 자체의 강함만 보게 됩니다. TIP까지 같이 보면 성장주 모멘텀과 금리·인플레이션 환경이 동시에 위험자산에 우호적인지 빠르게 확인할 수 있습니다.",
        "check": "QQQ와 TIP의 1·3·6·12개월 평균 모멘텀을 함께 봅니다. 둘 다 양수면 공격적인 환경으로, 한쪽이라도 꺾이면 위험자산 비중을 늘리기 전에 한 번 더 점검하는 카나리아 신호로 사용합니다.",
    },
    "한국 피어앤그리드 오실레이터": {
        "why": "한국시장은 KOSPI와 KOSDAQ의 체감이 크게 다르고 개인 수급의 영향도 큽니다. 가격·모멘텀·수급을 함께 보면서 국내시장의 과열과 공포를 따로 확인하기 위한 화면입니다.",
        "check": "Fear & Greed, EMA20, Oscillator와 함께 DeMark·개인 순매수 Capitulation을 봅니다. 극단값 하나보다 여러 보조지표가 같은 방향을 가리킬 때 의미를 더 크게 봅니다.",
    },
    "AI하드웨어 주가 모멘텀 점검": {
        "why": "AI 하드웨어 뉴스가 좋아도 실제 주가 상승이 일부 종목에만 몰리면 추세의 질은 약할 수 있습니다. GPU·메모리·장비·네트워크 등 AI 하드웨어 밸류체인의 Breadth와 모멘텀을 함께 보면 상승이 넓게 퍼지는지 확인할 수 있습니다.",
        "check": "원본 Colab 결과의 기간별 상승·하락 종목 수, Breadth Score, 기울기, 60일 상승종목 비율, 신고가·AI Tech Index 다이버전스를 순서대로 봅니다. 지수는 오르는데 Breadth가 약해지면 쏠림 상승 가능성을 점검합니다.",
    },
    "대만 월별 매출": {
        "why": "대만은 TSMC를 비롯해 파운드리·ASIC·PCB·CCL·서버 등 AI 공급망 핵심 기업들이 월매출을 공시합니다. 분기 실적을 기다리기 전에 공급망의 실제 매출 방향을 월 단위로 확인할 수 있습니다.",
        "check": "개별 기업의 전년동월비(YoY)와 전월비(MoM), 그리고 같은 밸류체인 안에서 여러 회사가 동시에 개선되는지를 봅니다. 한 회사보다 공급망 전체의 동시 개선이 더 중요한 신호입니다.",
    },
    "미국 52주 신고가 전략 점검": {
        "why": "미국 시장에서 강한 업종 ETF는 상대강도가 높고 52주 신고가를 갱신하며 추세를 이어가는 경우가 많습니다. 감으로 고르지 않고 상대강도·신고가 돌파·추세 유지 규칙과 약세구간 Rotation을 점검하는 화면입니다.",
        "check": "현재 Regime, 52주 전략의 최근 20일 SPY 대비 초과수익, 오늘 BUY/SELL, 최종 보유비중을 봅니다. 약세 Regime에서는 빈 자리에 Rotation 후보를 넣되 기존 보유종목은 매도 이동평균 이탈 전까지 유지하는 구조입니다.",
    },
    "미국 ETF 소라티노 및 상대강도": {
        "why": "수익률만 높은 업종보다 '하락 위험 대비 수익이 좋은 업종'과 '시장 대비 상대적으로 강한 업종'을 함께 찾기 위한 화면입니다. 주도업종의 지속성과 과열 여부를 동시에 볼 수 있습니다.",
        "check": "Mansfield RS와 6개월 SPY 대비 상대수익률로 강도를 보고, Sortino로 하방위험 대비 효율을 확인합니다. Breadth·정배열·30일 이격도까지 같이 보면서 강하지만 지나치게 과열된 업종인지 구분합니다.",
    },
    "한국 ETF 소라티노 및 상대강도": {
        "why": "국내 업종 ETF 중 KOSPI보다 강하면서 하락 위험 대비 성과가 좋은 업종을 찾기 위한 화면입니다. 단순 등락률보다 '주도력의 질'과 시장 내부 확산 정도를 같이 확인할 수 있습니다.",
        "check": "변동성 조정 모멘텀은 이평선 조건 없는 순위 하나만 보고, Sortino는 10일선·20일선·10주선 이상 조건을 각각 확인합니다. 여기에 Mansfield RS, KOSPI 대비 RS, Breadth, 쏠림, 정배열과 52주 신고가 전략을 함께 봅니다.",
    },
}

_guide = PAGE_GUIDES.get(ACTIVE_PAGE)
if _guide:
    st.info(
        _guide["why"] +
        "\n\n" + _guide["check"]
    )
    st.caption("※ 하나의 지표만으로 매수·매도를 결정하기보다, 서로 다른 데이터가 같은 방향을 가리키는지 확인하는 점검용 화면입니다.")

# 수동 업데이트 전용 상태
for _k in ["liq","fg","canary","trend","rotation","ai","us_sector","kr_sector"]:
    st.session_state.setdefault(_k+"_result", None)
    st.session_state.setdefault(_k+"_updated", None)

def updated_caption(key):
    t=st.session_state.get(key+"_updated")
    st.caption("마지막 업데이트: " + (t if t else "아직 업데이트하지 않음"))


# ============================================================
# v11: 대만 월매출 / 일본 반도체 월간 데이터
# 저장은 표(DataFrame)만 사용 -> 페이지를 열 때 그래프를 다시 그림
# ============================================================

TW_AI_UNIVERSE = {
    "파운드리": {
        "2330":"TSMC", "2303":"UMC", "5347":"VIS", "6770":"PSMC",
    },
    "소재·케미컬·포토마스크": {
        "2338":"Taiwan Mask", "4755":"San Fu Chemical", "4749":"AEMC",
        "4768":"Crystalwise", "4770":"ASC", "1727":"Chung Hwa Chemical",
    },
    "실리콘 웨이퍼": {
        "6488":"GlobalWafers", "5483":"SAS", "6182":"Wafer Works",
        "3532":"Formosa Sumco", "6640":"EPISIL",
    },
    "전공정 장비·소모품": {
        "3413":"Foxsemicon", "3680":"Gudeng", "3583":"Scientech", "3131":"GPTC",
    },
    "팹 건설·클린룸": {
        "2404":"UIS", "5536":"ACTER", "6139":"L&K Engineering", "6196":"Marketech",
    },
    "ASIC / IC설계": {
        "2454":"MediaTek", "3661":"Alchip", "3443":"Global Unichip", "3035":"Faraday",
        "2379":"Realtek", "3034":"Novatek", "6531":"Airoha", "4961":"Fitipower",
    },
    "IP·고속 인터페이스 IC": {
        "3529":"eMemory", "6643":"M31", "4966":"Parade", "5269":"ASMedia", "3227":"PixArt",
    },
    "아날로그·전력 IC": {
        "6286":"Richtek", "6415":"Silergy", "8081":"GMT",
    },
    "메모리 & 유통": {
        "2408":"Nanya Tech", "2344":"Winbond", "2337":"Macronix", "8299":"Phison",
        "3260":"ADATA", "2451":"Transcend", "5351":"Etron",
    },
    "화합물 반도체 (RF/SiC/GaN)": {
        "3105":"WIN Semi", "2455":"VPEC", "8086":"AWSC", "4971":"IET-KY",
    },
    "OSAT (후공정)": {
        "3711":"ASE", "6239":"PTI", "8150":"ChipMOS", "2449":"KYEC",
        "6257":"Sigurd", "3264":"Ardentec", "6147":"Chipbond",
    },
    "후공정(CoWoS)·테스트 장비": {
        "3413":"Foxsemicon", "3583":"Scientech", "6196":"Marketech", "3131":"GPTC",
        "6187":"All Ring", "6640":"EPISIL", "5443":"GPM",
    },
    "테스트 인터페이스": {
        "6515":"WinWay", "6223":"MPI", "6510":"CHPT",
    },
    "PCB / 기판": {
        "3037":"Unimicron", "3189":"Kinsus", "8046":"Nan Ya PCB", "4958":"Zhen Ding",
        "2313":"Compeq", "2367":"Unitech PCB",
    },
    "CCL·동박": {
        "2383":"Elite Material", "6213":"ITEQ", "6274":"TUC", "8358":"Co-Tech",
    },
    "MLCC": {
        "2327":"Yageo", "2492":"Walsin Tech", "3026":"Holy Stone", "6173":"PDC",
    },
    "광통신 / CPO": {
        "3081":"LandMark", "3363":"FOCI", "3163":"Browave", "4979":"LuxNet",
        "3450":"Elaser", "6442":"EZconn", "4908":"APAC Opto",
    },
    "네트워크 장비": {
        "2345":"Accton", "5388":"Sercomm", "3596":"Arcadyan", "6285":"WNC",
        "3380":"Alpha Networks", "3704":"Zyxel",
    },
    "광학렌즈": {
        "3008":"Largan", "3406":"GSEO",
    },
    "ODM / AI서버": {
        "2317":"Hon Hai", "2382":"Quanta", "3231":"Wistron", "6669":"Wiwynn", "2356":"Inventec",
    },
    "전력·파워서플라이·BBU": {
        "2308":"Delta", "6282":"AcBel", "2301":"Lite-On", "6412":"Chicony Power",
        "6781":"AES-KY", "3211":"Dynapack", "6121":"Simplo", "6409":"Voltronic",
    },
    "냉각": {
        "3324":"Auras", "3017":"AVC", "2421":"Sunonwealth", "3653":"Jentech",
        "8996":"Kaori", "6230":"Nidec Chaun-Choung",
    },
    "커넥터·케이블": {
        "3533":"LOTES", "3665":"BizLink-KY", "2392":"Cheng Uei", "4980":"ACON",
    },
    "서버 섀시·레일": {
        "8210":"Chenbro", "2059":"King Slide", "3693":"AIC", "6117":"In Win",
    },
    "BMC": {
        "5274":"ASPEED", "4919":"Nuvoton",
    },
    "IC 유통": {
        "3702":"WPG", "3036":"WT Micro", "8112":"Supreme",
    },
}

TW_TICKER_META = {}
for _cat, _members in TW_AI_UNIVERSE.items():
    for _ticker, _name in _members.items():
        # 중복 종목은 첫 카테고리를 대표 카테고리로 사용
        TW_TICKER_META.setdefault(str(_ticker), {"Name": _name, "Category": _cat})


def _clean_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace(",", "").replace("%", "").replace("−", "-").strip()
    if s in ("", "-", "--", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _normalize_mops_table(df, year, month, market):
    if df is None or df.empty or df.shape[1] < 7:
        return pd.DataFrame()
    d = df.copy()
    # MOPS는 다중 헤더/중간 산업별 헤더가 섞이는 경우가 있어 위치 기반을 우선 사용
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [str(x[-1]) for x in d.columns]
    else:
        d.columns = [str(x) for x in d.columns]

    # 헤더 행이 본문에 들어온 형태 처리
    first = d.iloc[:, 0].astype(str).str.strip()
    header_hits = d.index[first.eq("公司代號")].tolist()
    if header_hits:
        h = header_hits[0]
        d.columns = [str(x).strip() for x in d.loc[h].tolist()]
        d = d.loc[d.index > h].copy()

    # 회사코드가 첫 열, 회사명이 둘째 열인 공식 월매출 표 구조 사용
    if d.shape[1] < 7:
        return pd.DataFrame()
    d = d.iloc[:, :10].copy()
    d.columns = ["Ticker","OfficialName","Revenue","PrevRevenue","YearAgoRevenue","MoM","YoY","CumRevenue","YearAgoCumRevenue","CumYoY"][:d.shape[1]]
    d["Ticker"] = d["Ticker"].astype(str).str.extract(r"(\d{4,6})", expand=False)
    d = d[d["Ticker"].isin(TW_TICKER_META.keys())].copy()
    if d.empty:
        return d
    for c in ["Revenue","PrevRevenue","YearAgoRevenue","MoM","YoY","CumRevenue","YearAgoCumRevenue","CumYoY"]:
        if c in d.columns:
            d[c] = d[c].map(_clean_num)
    d["Date"] = pd.Timestamp(year=year, month=month, day=1)
    d["Market"] = market
    d["Name"] = d["Ticker"].map(lambda x: TW_TICKER_META.get(x,{}).get("Name", x))
    d["Category"] = d["Ticker"].map(lambda x: TW_TICKER_META.get(x,{}).get("Category", "기타"))
    # MOPS 매출 단위는 천 TWD -> 화면에서는 1mn TWD
    d["Revenue_mn_TWD"] = d["Revenue"] / 1000.0
    keep = ["Date","Ticker","Name","Category","Market","Revenue","Revenue_mn_TWD","MoM","YoY"]
    return d[[c for c in keep if c in d.columns]]


def _fetch_mops_month(year, month, market):
    """공식 MOPS 과거 월매출 HTML. API key 불필요."""
    roc = year - 1911
    kind = "sii" if market == "TWSE" else "otc"
    headers = {"User-Agent":"Mozilla/5.0 (compatible; TaerinsDadDashboard/1.0)"}
    out = []
    # 국내/외국기업 파일 모두 시도. 없는 파일은 조용히 건너뜀.
    for domestic_flag in (0, 1):
        urls = [
            f"https://mopsov.twse.com.tw/nas/t21/{kind}/t21sc03_{roc}_{month}_{domestic_flag}.html",
            f"https://mops.twse.com.tw/nas/t21/{kind}/t21sc03_{roc}_{month}_{domestic_flag}.html",
        ]
        txt = None
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 200 and len(r.content) > 500:
                    r.encoding = "big5"
                    txt = r.text
                    break
            except Exception:
                continue
        if not txt:
            continue
        try:
            tables = pd.read_html(io.StringIO(txt))
        except Exception:
            continue
        for t in tables:
            try:
                z = _normalize_mops_table(t, year, month, market)
                if not z.empty:
                    out.append(z)
            except Exception:
                continue
    if not out:
        return pd.DataFrame()
    ans = pd.concat(out, ignore_index=True)
    return ans.drop_duplicates(["Date","Ticker"], keep="last")


def _fetch_tw_latest_openapi():
    """최신 회차는 TWSE/TPEX 공식 OpenAPI도 함께 사용해 늦은 신고를 보완."""
    endpoints = [
        ("TWSE", "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"),
        ("TPEX", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"),
    ]
    rows=[]
    headers={"User-Agent":"Mozilla/5.0 (compatible; TaerinsDadDashboard/1.0)"}
    for market,url in endpoints:
        try:
            r=requests.get(url,headers=headers,timeout=30); r.raise_for_status()
            data=r.json()
        except Exception:
            continue
        for x in data if isinstance(data,list) else []:
            # 필드명이 시장별로 조금 달라도 키워드로 찾음
            def pick(words):
                for k,v in x.items():
                    ks=str(k)
                    if any(w in ks for w in words):
                        return v
                return None
            ticker=str(pick(["公司代號","公司代码","公司代码"]) or "").strip()
            m=re.search(r"\d{4,6}",ticker)
            if not m or m.group(0) not in TW_TICKER_META:
                continue
            ticker=m.group(0)
            ym=str(pick(["資料年月","资料年月"]) or "")
            ymn=re.sub(r"\D","",ym)
            date=None
            if len(ymn)>=5:
                try:
                    roc=int(ymn[:-2]); mo=int(ymn[-2:]); date=pd.Timestamp(roc+1911,mo,1)
                except Exception: pass
            if date is None:
                # 최신 데이터는 통상 전월 실적
                date=(pd.Timestamp.today().to_period("M")-1).to_timestamp()
            rev=_clean_num(pick(["當月營收","当月营收"]))
            mom=_clean_num(pick(["上月比較增減","上月比较增减","上月比較增減(%)"]))
            yoy=_clean_num(pick(["去年同月增減","去年同月增減(%)","去年同月增减"]))
            rows.append({
                "Date":date,"Ticker":ticker,"Name":TW_TICKER_META[ticker]["Name"],
                "Category":TW_TICKER_META[ticker]["Category"],"Market":market,
                "Revenue":rev,"Revenue_mn_TWD":rev/1000.0 if pd.notna(rev) else np.nan,
                "MoM":mom,"YoY":yoy,
            })
    return pd.DataFrame(rows)


def update_taiwan_revenue(existing=None):
    existing_df = None
    if isinstance(existing, dict) and isinstance(existing.get("data"), pd.DataFrame):
        existing_df = existing["data"].copy()
    today = pd.Timestamp.today().normalize()
    # 기존 저장이 있으면 최근 3개월만 재확인, 처음이면 2023년부터 백필
    if existing_df is not None and not existing_df.empty:
        start = max(existing_df["Date"].max().to_period("M") - 2, pd.Period("2023-01", freq="M"))
    else:
        start = pd.Period("2023-01", freq="M")
    end = today.to_period("M")
    frames=[]; errors=[]
    months = pd.period_range(start, end, freq="M")
    for i,per in enumerate(months):
        for market in ("TWSE","TPEX"):
            try:
                z=_fetch_mops_month(per.year, per.month, market)
                if not z.empty: frames.append(z)
            except Exception as e:
                errors.append(f"{per} {market}: {e}")
        if i and i % 12 == 0:
            time.sleep(0.25)
    try:
        latest=_fetch_tw_latest_openapi()
        if not latest.empty: frames.append(latest)
    except Exception as e:
        errors.append(f"latest openapi: {e}")

    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if existing_df is not None and not existing_df.empty:
        all_df = pd.concat([existing_df,new],ignore_index=True)
    else:
        all_df = new
    if all_df.empty:
        raise RuntimeError("대만 MOPS에서 선택 종목의 월매출을 가져오지 못했습니다.")
    all_df["Date"]=pd.to_datetime(all_df["Date"])
    all_df=all_df.sort_values(["Ticker","Date"]).drop_duplicates(["Date","Ticker"],keep="last")
    # 공식 MoM/YoY가 없을 때 저장된 시계열에서 계산
    all_df["Revenue_mn_TWD"]=pd.to_numeric(all_df["Revenue_mn_TWD"],errors="coerce")
    calc_mom=all_df.groupby("Ticker")["Revenue_mn_TWD"].pct_change()*100
    calc_yoy=all_df.groupby("Ticker")["Revenue_mn_TWD"].pct_change(12)*100
    if "MoM" not in all_df: all_df["MoM"]=calc_mom
    else: all_df["MoM"]=pd.to_numeric(all_df["MoM"],errors="coerce").fillna(calc_mom)
    if "YoY" not in all_df: all_df["YoY"]=calc_yoy
    else: all_df["YoY"]=pd.to_numeric(all_df["YoY"],errors="coerce").fillna(calc_yoy)
    return {
        "data": all_df.reset_index(drop=True),
        "source": "Taiwan MOPS / TWSE / TPEX official monthly revenue",
        "errors": errors[-20:],
        "universe_count": len(TW_TICKER_META),
    }


def _draw_tw_company_chart(df, ticker, name):
    z=df[df["Ticker"].astype(str)==str(ticker)].dropna(subset=["Date","Revenue_mn_TWD"]).copy()
    if z.empty:
        st.caption("데이터 없음"); return
    z["Year"]=z["Date"].dt.year; z["Month"]=z["Date"].dt.month
    fig,ax=plt.subplots(figsize=(6.4,3.0))
    for yr,g in z.groupby("Year"):
        g=g.sort_values("Month")
        ax.plot(g["Month"],g["Revenue_mn_TWD"],marker="o",linewidth=1.7,label=str(int(yr)))
    ax.set_xlim(1,12); ax.set_xticks([1,3,5,7,9,11]); ax.grid(alpha=.22)
    ax.set_ylabel("1mn TWD")
    ax.legend(ncol=min(4,z["Year"].nunique()),fontsize=8,frameon=False)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)


def render_taiwan_revenue(result):
    if not isinstance(result,dict) or not isinstance(result.get("data"),pd.DataFrame) or result["data"].empty:
        st.info("저장된 대만 월매출 데이터가 없습니다."); return
    df=result["data"].copy(); df["Date"]=pd.to_datetime(df["Date"])
    latest=df["Date"].max()
    st.caption(f"{latest.strftime('%Y.%m')} 실적 · 단위 1mn TWD · 공식 MOPS 자료를 저장된 숫자로 그래프화")
    latest_df=df[df["Date"].eq(latest)].copy()
    for idx,(cat,members) in enumerate(TW_AI_UNIVERSE.items()):
        tickers=list(members.keys())
        cat_latest=latest_df[latest_df["Ticker"].astype(str).isin(tickers)]
        cat_yoy=np.nan
        # 카테고리 YoY = 동일 기업 집합의 현재 합 / 12개월 전 합
        cur=cat_latest["Revenue_mn_TWD"].sum(min_count=1)
        prev_date=(latest.to_period("M")-12).to_timestamp()
        prev=df[(df["Date"].eq(prev_date)) & (df["Ticker"].astype(str).isin(cat_latest["Ticker"].astype(str)))]["Revenue_mn_TWD"].sum(min_count=1)
        if pd.notna(cur) and pd.notna(prev) and prev!=0: cat_yoy=(cur/prev-1)*100
        title=f"{cat} · {len(cat_latest)}/{len(tickers)}"
        if pd.notna(cat_yoy): title+=f" · YoY {cat_yoy:+.1f}%"
        with st.expander(title, expanded=(idx==0)):
            present=[(t,members[t]) for t in tickers if t in set(df["Ticker"].astype(str))]
            if not present:
                st.caption("선택 종목 데이터 없음"); continue
            for j in range(0,len(present),2):
                cols=st.columns(2)
                for k,(ticker,name) in enumerate(present[j:j+2]):
                    with cols[k]:
                        z=df[df["Ticker"].astype(str)==ticker].sort_values("Date")
                        last=z.iloc[-1]
                        st.markdown(f"**{name}** `{ticker}`  ·  **{last['Revenue_mn_TWD']:,.0f}**")
                        _draw_tw_company_chart(df,ticker,name)
                        st.caption(f"전월비 {last['MoM']:+.1f}%   ·   전년비 {last['YoY']:+.1f}%" if pd.notna(last.get('MoM')) and pd.notna(last.get('YoY')) else "")


# ---------- 일본: 등록 필요 없는 Statistics Dashboard API ----------
def _walk_json(obj):
    if isinstance(obj,dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj,list):
        for v in obj:
            yield from _walk_json(v)


def _dict_value(d, needles):
    for k,v in d.items():
        kl=str(k).lower().replace("_","")
        if any(n.lower().replace("_","") in kl for n in needles):
            return v
    return None


def _dashboard_indicator_candidates(keyword, lang="EN"):
    url="https://dashboard.e-stat.go.jp/api/1.0/Json/getIndicatorInfo"
    r=requests.get(url,params={"Lang":lang,"SearchIndicatorWord":keyword},timeout=30,
                   headers={"User-Agent":"Mozilla/5.0 (compatible; TaerinsDadDashboard/1.0)"})
    r.raise_for_status(); js=r.json(); out=[]
    for d in _walk_json(js):
        code=_dict_value(d,["IndicatorCode","indicatorCd"]); name=_dict_value(d,["IndicatorName","indicatorNm","shortNm"])
        if code and name:
            c=str(code).strip(); n=str(name).strip()
            if re.fullmatch(r"\d{10,30}",c): out.append((c,n,d))
    seen=set(); uniq=[]
    for x in out:
        if x[0] not in seen: seen.add(x[0]); uniq.append(x)
    return uniq


def _pick_indicator(keywords, prefer_terms):
    cands=[]
    for kw in keywords:
        for lang in ("EN","JP"):
            try: cands.extend(_dashboard_indicator_candidates(kw,lang))
            except Exception: pass
    if not cands: return None,None
    def score(x):
        name=x[1].lower(); d=x[2]; sc=0
        for term,w in prefer_terms:
            if term.lower() in name: sc+=w
        cyc=str(_dict_value(d,["cycle"]) or "")
        rank=str(_dict_value(d,["regionalRank"]) or "")
        if cyc == "1": sc += 8
        if rank == "2": sc += 5
        return sc
    cands=sorted(cands,key=score,reverse=True)
    best=cands[0]
    return best[0],best[1],best[2]


def _parse_time_value_nodes(js):
    rows=[]
    for d in _walk_json(js):
        t=_dict_value(d,["Time","TimeCode","Date"])
        v=d.get("$") if "$" in d else _dict_value(d,["Value","DataValue","value"])
        if t is None or v is None: continue
        ts=str(t)
        # 202601 / 2026-01 / 2026年1月 등의 월만 채택
        nums=re.findall(r"\d+",ts)
        date=None
        try:
            if re.fullmatch(r"\d{6}",ts.strip()): date=pd.Timestamp(int(ts[:4]),int(ts[4:]),1)
            elif len(nums)>=2 and len(nums[0])==4: date=pd.Timestamp(int(nums[0]),int(nums[1]),1)
            elif len(nums)==1 and len(nums[0])>=6:
                x=nums[0]; date=pd.Timestamp(int(x[:4]),int(x[4:6]),1)
        except Exception: date=None
        val=_clean_num(v)
        if date is not None and pd.notna(val): rows.append((date,val))
    if not rows: return pd.DataFrame(columns=["Date","Value"])
    z=pd.DataFrame(rows,columns=["Date","Value"]).drop_duplicates("Date",keep="last").sort_values("Date")
    return z


def _fetch_dashboard_series(keywords, prefer_terms, seasonal=1):
    picked=_pick_indicator(keywords,prefer_terms)
    if not picked or len(picked) < 3: raise RuntimeError(f"지표 검색 실패: {keywords[0]}")
    code,name,meta=picked
    unit=str(_dict_value(meta,["unitNm","unit"]) or "")
    url="https://dashboard.e-stat.go.jp/api/1.0/Json/getData"
    base={"Cycle":1,"IndicatorCode":code,"Lang":"EN","MetaGetFlg":"Y","RegionalRank":2,"SectionHeaderFlg":1}
    trials=[seasonal, 1, 2]
    last_err=None
    for sa in dict.fromkeys(trials):
        try:
            params=dict(base); params["IsSeasonalAdjustment"]=sa
            r=requests.get(url,params=params,timeout=40,headers={"User-Agent":"Mozilla/5.0 (compatible; TaerinsDadDashboard/1.0)"})
            r.raise_for_status(); z=_parse_time_value_nodes(r.json())
            if not z.empty:
                return z,code,name,unit
        except Exception as e: last_err=e
    raise RuntimeError(f"지표 데이터 실패 {name}: {last_err}")



def _parse_month_label(value):
    """e-Stat 표의 다양한 월 표기(2026-03, 2026年3月, Mar. 2026 등)를 월초 Timestamp로 변환."""
    if value is None:
        return None
    x = str(value).strip()
    if not x or x.lower() == "nan":
        return None
    # 숫자형/일본어형
    m = re.search(r"(20\d{2})\s*[./年\-]?\s*(0?[1-9]|1[0-2])\s*月?", x)
    if m:
        try: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        except Exception: pass
    m = re.search(r"\b(20\d{2})(0[1-9]|1[0-2])\b", x)
    if m:
        try: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        except Exception: pass
    # 영문 월명
    months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
    low = x.lower().replace(",", " ")
    ym = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z.]*\s+(20\d{2})\b", low)
    if ym:
        return pd.Timestamp(int(ym.group(2)), months[ym.group(1)], 1)
    ym = re.search(r"\b(20\d{2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z.]*\b", low)
    if ym:
        return pd.Timestamp(int(ym.group(1)), months[ym.group(2)], 1)
    return None


def _flatten_columns(cols):
    out=[]
    if isinstance(cols, pd.MultiIndex):
        for c in cols:
            vals = c if isinstance(c, tuple) else (c,)
            vals = [str(v).strip() for v in vals if pd.notna(v) and str(v).strip().lower() not in ("nan", "unnamed")]
            out.append(" ".join(vals))
    else:
        out=[str(c).strip() for c in cols]
    return out


def _extract_estat_dbview_series(sid, target_terms):
    """e-Stat 공개 dbview HTML을 API 키 없이 읽는 fallback. 표 방향이 달라도 시계열을 찾는다."""
    urls=[f"https://www.e-stat.go.jp/en/dbview?sid={sid}", f"https://www.e-stat.go.jp/dbview?sid={sid}"]
    last_err=None
    for url in urls:
        try:
            r=requests.get(url,timeout=50,headers={"User-Agent":"Mozilla/5.0 (compatible; TaerinsDadDashboard/1.1)"})
            r.raise_for_status()
            tables=pd.read_html(io.StringIO(r.text))
            found=[]
            terms=[str(x).lower() for x in target_terms]
            for t in tables:
                if t is None or t.empty:
                    continue
                d=t.copy()
                d.columns=_flatten_columns(d.columns)
                # 기존 오류 원인: float가 섞인 행을 str.join 하면서 예외 발생 -> 전부 str로 강제
                rowtext=d.apply(lambda row: " | ".join(map(str, row.values)), axis=1)
                mask=rowtext.map(lambda x:any(term in x.lower() for term in terms))

                # A. 대상이 행에 있고 월이 열인 표
                if mask.any():
                    for _, row in d.loc[mask].iterrows():
                        for c,v in row.items():
                            dt=_parse_month_label(c)
                            if dt is not None:
                                val=_clean_num(v)
                                if pd.notna(val): found.append((dt,val))

                # B. 대상이 열이고 월이 행인 표
                target_cols=[c for c in d.columns if any(term in str(c).lower() for term in terms)]
                if target_cols:
                    for _,row in d.iterrows():
                        dt=None
                        for vv in list(row.values)[:min(8,len(row))]:
                            dt=_parse_month_label(vv)
                            if dt is not None: break
                        if dt is None:
                            continue
                        for tc in target_cols:
                            val=_clean_num(row.get(tc))
                            if pd.notna(val):
                                found.append((dt,val)); break

                # C. 첫 열들이 계층명, 별도 Month 열이 있는 long-form 표
                date_cols=[c for c in d.columns if any(k in str(c).lower() for k in ["month","monthly","time","date","年月","月次"])]
                value_cols=[c for c in d.columns if any(k in str(c).lower() for k in ["value","値","index","mil.yen","million yen","金額"])]
                if mask.any() and date_cols and value_cols:
                    for _,row in d.loc[mask].iterrows():
                        dt=None
                        for dc in date_cols:
                            dt=_parse_month_label(row.get(dc))
                            if dt is not None: break
                        if dt is None: continue
                        for vc in value_cols:
                            val=_clean_num(row.get(vc))
                            if pd.notna(val): found.append((dt,val)); break

            if found:
                z=pd.DataFrame(found,columns=["Date","Value"])
                z["Date"]=pd.to_datetime(z["Date"])
                z["Value"]=pd.to_numeric(z["Value"],errors="coerce")
                z=z.dropna().drop_duplicates("Date",keep="last").sort_values("Date")
                if len(z): return z
            last_err=RuntimeError(f"e-Stat dbview {sid}에서 대상 시계열을 찾지 못했습니다.")
        except Exception as e:
            last_err=e
    raise RuntimeError(str(last_err) if last_err else f"e-Stat dbview {sid} 읽기 실패")


def _extract_estat_first_working(sids, terms):
    errs=[]
    for sid in sids:
        try:
            z=_extract_estat_dbview_series(sid,terms)
            if isinstance(z,pd.DataFrame) and not z.empty:
                return z, sid
        except Exception as e:
            errs.append(f"{sid}: {e}")
    raise RuntimeError(" / ".join(errs[-3:]))


JP_TARGETS = [
    {
        "label":"반도체·FPD 제조장비 생산지수",
        "keywords":["semiconductor manufacturing equipment","semiconductor flat panel display manufacturing","半導体 フラットパネルディスプレイ 製造装置"],
        "prefer":[("production",5),("index",4),("semiconductor",5),("manufacturing equipment",5),("flat",2)],
        "unit":"2020=100",
        "fallback_sids":["0004052177"],
        "fallback_terms":["1103006000","半導体・フラットパネルディスプレイ製造装置","semiconductor manufacturing equipment"],
    },
    {
        "label":"전자부품·디바이스 생산지수",
        "keywords":["electronic parts devices production index","electronic parts and devices","電子部品 デバイス"],
        "prefer":[("production",5),("index",4),("electronic",4),("parts",3),("device",3)],
        "unit":"2020=100",
        "fallback_sids":["0004052177"],
        "fallback_terms":["1105000000","電子部品・デバイス工業","electronic parts and devices"],
    },
    {
        "label":"집적회로(IC) 생산지수",
        "keywords":["integrated circuits production index","integrated circuit","集積回路"],
        "prefer":[("production",5),("index",4),("integrated circuit",6)],
        "unit":"2020=100",
        "fallback_sids":["0004052177"],
        "fallback_terms":["1105001000","集積回路","integrated circuit"],
    },
    {
        "label":"반도체 제조장비 수주",
        "keywords":["semiconductor making equipment machinery orders","semiconductor making equipment","半導体製造装置 機械受注"],
        "prefer":[("machinery orders",8),("semiconductor",5),("equipment",4)],
        "unit":"백만엔",
        "fallback_sids":["0003355268"],
        "fallback_terms":["Electronic and communication equipment_Semiconductor making equipment","Semiconductor making equipment","半導体製造装置"],
    },
    {
        "label":"반도체 제조장비 수출",
        "keywords":["semiconductor machinery export","semiconductor machinery","半導体製造装置 輸出"],
        "prefer":[("export",8),("semiconductor",5),("machinery",4)],
        "unit":"공식 원단위",
        # 2026 current + 2021~2025 historical table 순서로 시도
        "fallback_sids":["0004049327","0003425295"],
        "fallback_terms":["SEMICONDUCTOR MACHINERY","SEMICON MACHINERY ETC","半導体製造装置"],
    },
]


def update_japan_semiconductor(existing=None):
    frames=[]; meta=[]; errors=[]
    for target in JP_TARGETS:
        loaded=False
        api_err="검색 실패"
        # 1) 등록/키 없는 Statistics Dashboard API 우선
        try:
            z,code,name,unit=_fetch_dashboard_series(target["keywords"],target["prefer"],seasonal=1)
            if isinstance(z,pd.DataFrame) and not z.empty:
                z=z.copy(); z["Indicator"]=target["label"]; z["Unit"]=unit or target["unit"]
                frames.append(z)
                meta.append({"Indicator":target["label"],"Code":code,"OfficialName":name,"Unit":unit or target["unit"],"Source":"Statistics Dashboard"})
                loaded=True
        except Exception as e:
            api_err=e

        # 2) 상세 지표가 Dashboard에 없으면 공식 e-Stat 공개 표 fallback
        if not loaded:
            try:
                fb,sid=_extract_estat_first_working(target.get("fallback_sids",[]), target.get("fallback_terms",[]))
                fb=fb.copy(); fb["Indicator"]=target["label"]; fb["Unit"]=target["unit"]
                frames.append(fb)
                meta.append({"Indicator":target["label"],"Code":sid,"OfficialName":target["label"],"Unit":target["unit"],"Source":"e-Stat public table"})
                loaded=True
            except Exception as e2:
                errors.append(f"{target['label']}: Dashboard={api_err} / e-Stat={e2}")

    if not frames:
        raise RuntimeError("일본 공식 통계에서 반도체 월간 지표를 가져오지 못했습니다. " + " | ".join(errors[:3]))

    df=pd.concat(frames,ignore_index=True)
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df["Value"]=pd.to_numeric(df["Value"],errors="coerce")
    df=df.dropna(subset=["Date","Value"]).sort_values(["Indicator","Date"])
    if isinstance(existing,dict) and isinstance(existing.get("data"),pd.DataFrame):
        old=existing["data"].copy()
        df=pd.concat([old,df],ignore_index=True).drop_duplicates(["Indicator","Date"],keep="last")
        df=df.sort_values(["Indicator","Date"])
    return {"data":df.reset_index(drop=True),"meta":meta,"errors":errors,
            "source":"Japan Statistics Dashboard / e-Stat / METI / Cabinet Office / Trade Statistics"}


def render_japan_semiconductor(result):
    if not isinstance(result,dict) or not isinstance(result.get("data"),pd.DataFrame) or result["data"].empty:
        st.info("저장된 일본 반도체 월간 데이터가 없습니다."); return
    df=result["data"].copy(); df["Date"]=pd.to_datetime(df["Date"])
    inds=list(df["Indicator"].dropna().unique())
    st.caption("일본 정부 통계 기반 · 저장된 숫자만으로 그래프를 재생성")
    st.caption("This service uses the API feature of Statistics Dashboard, but the contents of this service are not guaranteed by the Statistics Bureau of Japan.")
    # 최신 수치 카드
    cols=st.columns(min(4,max(1,len(inds))))
    for i,ind in enumerate(inds[:4]):
        z=df[df["Indicator"].eq(ind)].sort_values("Date")
        if z.empty: continue
        last=z.iloc[-1]; mom=z["Value"].pct_change().iloc[-1]*100 if len(z)>1 else np.nan
        yoy=z["Value"].pct_change(12).iloc[-1]*100 if len(z)>12 else np.nan
        with cols[i%len(cols)]:
            st.metric(ind, f"{last['Value']:,.1f}", f"MoM {mom:+.1f}% · YoY {yoy:+.1f}%" if pd.notna(yoy) else (f"MoM {mom:+.1f}%" if pd.notna(mom) else None))
    for ind in inds:
        z=df[df["Indicator"].eq(ind)].sort_values("Date").set_index("Date")
        with st.expander(ind, expanded=True):
            st.line_chart(z[["Value"]].tail(72),use_container_width=True)
            if len(z):
                q=z.reset_index().tail(18).copy(); q["MoM_%"]=q["Value"].pct_change()*100; q["YoY_%"]=q["Value"].pct_change(12)*100
                st.dataframe(q.sort_values("Date",ascending=False).head(12),use_container_width=True,hide_index=True,
                             column_config={"Date":st.column_config.DateColumn("월",format="YYYY-MM"),"Value":st.column_config.NumberColumn("값",format="%.2f"),"MoM_%":st.column_config.NumberColumn("MoM",format="%+.1f%%"),"YoY_%":st.column_config.NumberColumn("YoY",format="%+.1f%%")})
    if result.get("errors"):
        with st.expander("이번 업데이트에서 못 받은 지표"):
            for e in result["errors"]: st.caption(e)


KOREA_FG_SRC = '# ============================================================\n# Full Code\n# KOSPI/KOSDAQ Fear & Greed Index + EMA20 + Oscillator + Index\n# + Elder Impulse\n# + Daily DeMark TD Setup (custom: Sell=4, Buy=2)\n#\n# 그래프 표시 기간: 최근 1년\n#\n# Fear & Greed 구성:\n# 1. 125일 모멘텀\n# 2. ATM Put/Call\n# 3. VKOSPI\n# 4. 10년 국채선물지수 - 5년 국채선물 추종지수\n# 5. RSI 10일\n#\n# 주의:\n# - 그래프를 1년 온전히 표시하려면 엑셀 원본에 최소 약 1년 7~8개월 이상의\n#   데이터가 있는 것이 좋습니다.\n# - 125일 이동평균 계산 전 구간은 Fear & Greed 값이 NaN이므로 표시되지 않습니다.\n# ============================================================\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.preprocessing import MinMaxScaler\n\n\n# ============================================================\n# 0) 기본 설정\n# ============================================================\n\nplt.rcParams[\'axes.unicode_minus\'] = False\n\n# 한글 폰트가 설치된 Colab 환경이라면 아래 줄 사용\n# plt.rcParams[\'font.family\'] = \'NanumGothic\'\n\n# 그래프 표시 기간\nDISPLAY_MONTHS = 12\n\n\n# =============================\n# 색상 팔레트\n# =============================\n\nCOL_FG_INDEX = \'#1565C0\'       # Fear & Greed Index\nCOL_FG_EMA20 = \'#F57C00\'       # F&G EMA20\nCOL_OSC = \'#6A5ACD\'            # Oscillator\nCOL_PRICE = \'#2F2F2F\'          # 지수 가격\n\nCOL_SUPERMA = \'#FF8C00\'\nCOL_GAP = \'#00B3B3\'\n\nGRID_ALPHA = 0.3\n\n\n# ============================================================\n# 1) 엑셀 업로드 및 로드\n# ============================================================\n\n\n\nfile_name = io.BytesIO(EXCEL_BYTES)\n\nkospi_df = pd.read_excel(\n    file_name,\n    sheet_name=\'KOSPI\'\n)\n\nkosdaq_df = pd.read_excel(\n    file_name,\n    sheet_name=\'KOSDAQ\'\n)\n\n# 개인 순매수와 KOSPI 지수 데이터\n# 첫 번째 열은 날짜, 두 번째 열은 KOSPI 종가, 세 번째 열은 개인 순매수대금(억원)\nindividual_df = pd.read_excel(\n    file_name,\n    sheet_name=\'개인순매수와코스피지수\'\n)\n\n\n# ============================================================\n# 2) Date 컬럼 처리\n# ============================================================\n\ndef ensure_date_col(df):\n    df = df.copy()\n\n    if \'Date\' not in df.columns:\n        df = df.rename(\n            columns={df.columns[0]: \'Date\'}\n        )\n\n    df[\'Date\'] = pd.to_datetime(\n        df[\'Date\'],\n        errors=\'coerce\'\n    )\n\n    df = df.dropna(\n        subset=[\'Date\']\n    ).copy()\n\n    df = df.sort_values(\'Date\').reset_index(drop=True)\n\n    return df\n\n\nkospi_df = ensure_date_col(kospi_df)\nkosdaq_df = ensure_date_col(kosdaq_df)\n\n\ndef prepare_individual_flow(df):\n    """\n    개인순매수와코스피지수 시트를 표준 형태로 정리합니다.\n\n    기대 구조\n    - 1열: 날짜\n    - 2열: KOSPI 종가지수\n    - 3열: 개인 순매수대금(일간, 억원)\n    """\n    df = df.copy()\n\n    if df.shape[1] < 3:\n        raise ValueError(\n            "\'개인순매수와코스피지수\' 시트에는 최소 3개 열이 필요합니다."\n        )\n\n    df = df.iloc[:, :3].copy()\n    df.columns = [\n        \'Date\',\n        \'KOSPI_Close\',\n        \'Individual_NetBuy_100M\'\n    ]\n\n    df[\'Date\'] = pd.to_datetime(\n        df[\'Date\'],\n        errors=\'coerce\'\n    )\n\n    for col in [\n        \'KOSPI_Close\',\n        \'Individual_NetBuy_100M\'\n    ]:\n        df[col] = pd.to_numeric(\n            df[col],\n            errors=\'coerce\'\n        )\n\n    df = (\n        df.dropna(\n            subset=[\n                \'Date\',\n                \'KOSPI_Close\',\n                \'Individual_NetBuy_100M\'\n            ]\n        )\n        .sort_values(\'Date\')\n        .drop_duplicates(\'Date\', keep=\'last\')\n        .reset_index(drop=True)\n    )\n\n    # X축: KOSPI 일간 수익률(%)\n    df[\'KOSPI_Return_Pct\'] = (\n        df[\'KOSPI_Close\']\n        .pct_change()\n        * 100\n    )\n\n    # Y축: 개인 일간 순매수대금(조원)\n    # 원본 단위가 억원이므로 10,000으로 나눔\n    df[\'Individual_NetBuy_Trillion\'] = (\n        df[\'Individual_NetBuy_100M\']\n        / 10000\n    )\n\n    # 3사분면: KOSPI 하락 + 개인 순매도\n    df[\'Is_Third_Quadrant\'] = (\n        (df[\'KOSPI_Return_Pct\'] < 0)\n        & (df[\'Individual_NetBuy_Trillion\'] < 0)\n    )\n\n    return df\n\n\nindividual_flow = prepare_individual_flow(individual_df)\n\n\n# ============================================================\n# 3) 수치형 컬럼 변환\n# ============================================================\n\nkospi_cols = [\n    \'5년 국채선물 추종 지수\',\n    \'10년국채선물지수\',\n    \'코스피 200 변동성지수\',\n    \'코스피\',\n    \'최근월물 CALL ATM\',\n    \'최근월물 PUT ATM\'\n]\n\nkosdaq_cols = [\n    \'5년 국채선물 추종 지수\',\n    \'10년국채선물지수\',\n    \'코스피 200 변동성지수\',\n    \'코스닥\',\n    \'최근월물 CALL ATM\',\n    \'최근월물 PUT ATM\'\n]\n\n\ndef convert_numeric_columns(df, columns):\n    df = df.copy()\n\n    for col in columns:\n        if col not in df.columns:\n            raise KeyError(\n                f"엑셀에 필요한 컬럼이 없습니다: {col}"\n            )\n\n        df[col] = pd.to_numeric(\n            df[col],\n            errors=\'coerce\'\n        )\n\n    return df\n\n\nkospi = convert_numeric_columns(\n    kospi_df,\n    kospi_cols\n)\n\nkosdaq = convert_numeric_columns(\n    kosdaq_df,\n    kosdaq_cols\n)\n\n\n# ============================================================\n# 4) RSI 계산\n# ============================================================\n\ndef calculate_rsi(df, col, window=10):\n    df = df.copy()\n\n    delta = df[col].diff()\n\n    gain = (\n        delta.where(delta > 0, 0)\n        .rolling(window=window)\n        .mean()\n    )\n\n    loss = (\n        -delta.where(delta < 0, 0)\n        .rolling(window=window)\n        .mean()\n    )\n\n    rs = gain / loss.replace(0, np.nan)\n\n    df[f\'RSI_{window}\'] = (\n        100 - (100 / (1 + rs))\n    )\n\n    df.loc[\n        (loss == 0) & (gain > 0),\n        f\'RSI_{window}\'\n    ] = 100\n\n    return df\n\n\n# ============================================================\n# 5) Fear & Greed Index 계산\n# ============================================================\n\ndef calculate_fear_greed(\n    df,\n    index_col,\n    vix_col,\n    call_col,\n    put_col,\n    bond5_col,\n    bond10_col\n):\n    df = df.copy()\n\n    # 125일 모멘텀\n    df[\'MA125\'] = (\n        df[index_col]\n        .rolling(window=125)\n        .mean()\n    )\n\n    df[\'Momentum\'] = (\n        (df[index_col] - df[\'MA125\'])\n        / df[\'MA125\']\n        * 100\n    )\n\n    # ATM Put/Call\n    df[\'PutCall\'] = (\n        df[put_col]\n        / df[call_col].replace(0, np.nan)\n    )\n\n    # 변동성\n    df[\'Volatility\'] = df[vix_col]\n\n    # 국채선물 스프레드\n    df[\'BondDiff\'] = (\n        df[bond10_col]\n        - df[bond5_col]\n    )\n\n    df.replace(\n        [np.inf, -np.inf],\n        np.nan,\n        inplace=True\n    )\n\n    features = [\n        \'Momentum\',\n        \'PutCall\',\n        \'Volatility\',\n        \'BondDiff\',\n        \'RSI_10\'\n    ]\n\n    valid = df.dropna(\n        subset=features\n    ).index\n\n    df[\'Fear_Greed_Index\'] = np.nan\n    df[\'Fear_Greed_Score\'] = np.nan\n\n    if len(valid) == 0:\n        return df\n\n    scaler = MinMaxScaler()\n\n    scaled_features = scaler.fit_transform(\n        df.loc[valid, features]\n    )\n\n    scaled_df = pd.DataFrame(\n        scaled_features,\n        index=valid,\n        columns=[\n            \'Momentum_Scaled\',\n            \'PutCall_Scaled\',\n            \'Volatility_Scaled\',\n            \'BondDiff_Scaled\',\n            \'RSI_Scaled\'\n        ]\n    )\n\n    for col in scaled_df.columns:\n        df.loc[valid, col] = scaled_df[col]\n\n    df.loc[valid, \'Fear_Greed_Index\'] = (\n        df.loc[valid, \'Momentum_Scaled\'] * 0.20\n        + (1 - df.loc[valid, \'PutCall_Scaled\']) * 0.20\n        + (1 - df.loc[valid, \'Volatility_Scaled\']) * 0.20\n        + df.loc[valid, \'BondDiff_Scaled\'] * 0.20\n        + df.loc[valid, \'RSI_Scaled\'] * 0.20\n    )\n\n    df.loc[valid, \'Fear_Greed_Score\'] = (\n        df.loc[valid, \'Fear_Greed_Index\']\n        * 100\n    )\n\n    return df\n\n\n# ============================================================\n# 6) Fear & Greed MACD 및 EMA20\n# ============================================================\n\ndef calculate_macd(df, col=\'Fear_Greed_Index\'):\n    df = df.copy()\n\n    ema12 = (\n        df[col]\n        .ewm(span=12, adjust=False)\n        .mean()\n    )\n\n    ema26 = (\n        df[col]\n        .ewm(span=26, adjust=False)\n        .mean()\n    )\n\n    macd = ema12 - ema26\n\n    signal = (\n        macd\n        .ewm(span=9, adjust=False)\n        .mean()\n    )\n\n    df[\'FG_MACD\'] = macd\n    df[\'FG_Signal\'] = signal\n    df[\'Oscillator\'] = df[\'FG_MACD\'] - df[\'FG_Signal\']\n\n    df[\'FG_EMA20\'] = (\n        df[col]\n        .ewm(span=20, adjust=False)\n        .mean()\n    )\n\n    df[\'FG_EMA20_Score\'] = (\n        df[\'FG_EMA20\']\n        * 100\n    )\n\n    return df\n\n\n# ============================================================\n# 7) SuperMA 및 Gap%\n# ============================================================\n\ndef add_super_ma_gap(df, price_col, label):\n    df = df.copy()\n\n    for window in [20, 60, 120, 200]:\n        df[f\'{label}_MA{window}\'] = (\n            df[price_col]\n            .rolling(window)\n            .mean()\n        )\n\n    ma_columns = [\n        f\'{label}_MA20\',\n        f\'{label}_MA60\',\n        f\'{label}_MA120\',\n        f\'{label}_MA200\'\n    ]\n\n    df[f\'{label}_SuperMA\'] = (\n        df[ma_columns]\n        .mean(axis=1)\n    )\n\n    df[f\'{label}_GapPct\'] = (\n        (\n            df[price_col]\n            - df[f\'{label}_SuperMA\']\n        )\n        / df[f\'{label}_SuperMA\']\n        * 100\n    )\n\n    return df\n\n\n# ============================================================\n# 8) Elder Impulse 구성 요소\n# ============================================================\n\ndef add_impulse_components(df, price_col, label):\n    df = df.copy()\n\n    df[f\'{label}_EMA13\'] = (\n        df[price_col]\n        .ewm(span=13, adjust=False)\n        .mean()\n    )\n\n    ema12 = (\n        df[price_col]\n        .ewm(span=12, adjust=False)\n        .mean()\n    )\n\n    ema26 = (\n        df[price_col]\n        .ewm(span=26, adjust=False)\n        .mean()\n    )\n\n    macd = ema12 - ema26\n\n    signal = (\n        macd\n        .ewm(span=9, adjust=False)\n        .mean()\n    )\n\n    df[f\'{label}_MACD_Hist\'] = (\n        macd - signal\n    )\n\n    return df\n\n\ndef get_impulse_colors(df, ema_col, macd_col):\n    colors = []\n\n    for i in range(len(df)):\n        if i == 0:\n            colors.append(\'gray\')\n            continue\n\n        ema_up = (\n            df[ema_col].iloc[i]\n            > df[ema_col].iloc[i - 1]\n        )\n\n        ema_down = (\n            df[ema_col].iloc[i]\n            < df[ema_col].iloc[i - 1]\n        )\n\n        macd_up = (\n            df[macd_col].iloc[i]\n            > df[macd_col].iloc[i - 1]\n        )\n\n        macd_down = (\n            df[macd_col].iloc[i]\n            < df[macd_col].iloc[i - 1]\n        )\n\n        if ema_up and macd_up:\n            colors.append(\'green\')\n\n        elif ema_down and macd_down:\n            colors.append(\'red\')\n\n        else:\n            colors.append(\'blue\')\n\n    return colors\n\n\n# ============================================================\n# 9) DeMark TD Setup\n# ============================================================\n\ndef add_td_setup_counts(\n    df,\n    price_col,\n    label=\'TD\'\n):\n    df = df.copy()\n\n    prices = df[price_col].values\n\n    sell = np.zeros(len(df))\n    buy = np.zeros(len(df))\n\n    for i in range(len(df)):\n\n        if (\n            i >= 4\n            and np.isfinite(prices[i])\n            and np.isfinite(prices[i - 4])\n            and prices[i] > prices[i - 4]\n        ):\n            sell[i] = sell[i - 1] + 1\n        else:\n            sell[i] = 0\n\n        if (\n            i >= 2\n            and np.isfinite(prices[i])\n            and np.isfinite(prices[i - 2])\n            and prices[i] < prices[i - 2]\n        ):\n            buy[i] = buy[i - 1] + 1\n        else:\n            buy[i] = 0\n\n    df[f\'{label}_SellSetup\'] = sell\n    df[f\'{label}_BuySetup\'] = buy\n\n    return df\n\n\n# ============================================================\n# 10) 전체 파이프라인 실행\n# ============================================================\n\nkospi = calculate_rsi(\n    kospi,\n    \'코스피\',\n    window=10\n)\n\nkosdaq = calculate_rsi(\n    kosdaq,\n    \'코스닥\',\n    window=10\n)\n\n\nkospi = calculate_fear_greed(\n    kospi,\n    index_col=\'코스피\',\n    vix_col=\'코스피 200 변동성지수\',\n    call_col=\'최근월물 CALL ATM\',\n    put_col=\'최근월물 PUT ATM\',\n    bond5_col=\'5년 국채선물 추종 지수\',\n    bond10_col=\'10년국채선물지수\'\n)\n\nkosdaq = calculate_fear_greed(\n    kosdaq,\n    index_col=\'코스닥\',\n    vix_col=\'코스피 200 변동성지수\',\n    call_col=\'최근월물 CALL ATM\',\n    put_col=\'최근월물 PUT ATM\',\n    bond5_col=\'5년 국채선물 추종 지수\',\n    bond10_col=\'10년국채선물지수\'\n)\n\n\nkospi = calculate_macd(\n    kospi,\n    col=\'Fear_Greed_Index\'\n)\n\nkosdaq = calculate_macd(\n    kosdaq,\n    col=\'Fear_Greed_Index\'\n)\n\n\nkospi = add_super_ma_gap(\n    kospi,\n    price_col=\'코스피\',\n    label=\'KOSPI\'\n)\n\nkosdaq = add_super_ma_gap(\n    kosdaq,\n    price_col=\'코스닥\',\n    label=\'KOSDAQ\'\n)\n\n\nkospi = add_impulse_components(\n    kospi,\n    price_col=\'코스피\',\n    label=\'KOSPI\'\n)\n\nkosdaq = add_impulse_components(\n    kosdaq,\n    price_col=\'코스닥\',\n    label=\'KOSDAQ\'\n)\n\n\nkospi = add_td_setup_counts(\n    kospi,\n    price_col=\'코스피\',\n    label=\'TD\'\n)\n\nkosdaq = add_td_setup_counts(\n    kosdaq,\n    price_col=\'코스닥\',\n    label=\'TD\'\n)\n\n\n# ============================================================\n# 11) 최근 1년 데이터\n# ============================================================\n\nkospi_cutoff = (\n    kospi[\'Date\'].max()\n    - pd.DateOffset(months=DISPLAY_MONTHS)\n)\n\nkosdaq_cutoff = (\n    kosdaq[\'Date\'].max()\n    - pd.DateOffset(months=DISPLAY_MONTHS)\n)\n\n\nrecent_kospi = kospi[\n    kospi[\'Date\'] >= kospi_cutoff\n].dropna(\n    subset=[\n        \'Fear_Greed_Score\',\n        \'FG_EMA20_Score\',\n        \'Oscillator\',\n        \'코스피\'\n    ]\n).copy()\n\n\nrecent_kosdaq = kosdaq[\n    kosdaq[\'Date\'] >= kosdaq_cutoff\n].dropna(\n    subset=[\n        \'Fear_Greed_Score\',\n        \'FG_EMA20_Score\',\n        \'Oscillator\',\n        \'코스닥\'\n    ]\n).copy()\n\n\n# ============================================================\n# 12) Fear & Greed 통합 그래프\n# ============================================================\n\ndef plot_fg(\n    df,\n    price_col,\n    title\n):\n    if df.empty:\n        print(f"{title}: 표시할 데이터가 없습니다.")\n        return\n\n    fig, ax_fg = plt.subplots(\n        figsize=(18, 8)\n    )\n\n    fig.subplots_adjust(\n        right=0.82\n    )\n\n    # 왼쪽 축: Fear & Greed Index\n    line_fg, = ax_fg.plot(\n        df[\'Date\'],\n        df[\'Fear_Greed_Score\'],\n        color=COL_FG_INDEX,\n        linewidth=2.5,\n        label=\'Fear & Greed Index\'\n    )\n\n    line_ema20, = ax_fg.plot(\n        df[\'Date\'],\n        df[\'FG_EMA20_Score\'],\n        color=COL_FG_EMA20,\n        linewidth=2.1,\n        label=\'F&G EMA20\'\n    )\n\n    ax_fg.set_ylim(\n        0,\n        100\n    )\n\n    ax_fg.set_ylabel(\n        \'Fear & Greed Index\',\n        color=COL_FG_INDEX,\n        fontsize=11\n    )\n\n    ax_fg.tick_params(\n        axis=\'y\',\n        labelcolor=COL_FG_INDEX\n    )\n\n    for level in [20, 50, 80]:\n        ax_fg.axhline(\n            level,\n            color=\'gray\',\n            linestyle=\'--\',\n            linewidth=0.8,\n            alpha=0.5,\n            label=\'_nolegend_\'\n        )\n\n    ax_fg.grid(\n        True,\n        alpha=GRID_ALPHA\n    )\n\n    ax_fg.set_xlabel(\'Date\')\n\n\n    # 오른쪽 축 1: 가격\n    ax_price = ax_fg.twinx()\n\n    line_price, = ax_price.plot(\n        df[\'Date\'],\n        df[price_col],\n        color=COL_PRICE,\n        linewidth=1.6,\n        alpha=0.48,\n        label=price_col\n    )\n\n    ax_price.set_ylabel(\n        price_col,\n        color=COL_PRICE,\n        fontsize=11\n    )\n\n    ax_price.tick_params(\n        axis=\'y\',\n        labelcolor=COL_PRICE\n    )\n\n\n    # 오른쪽 축 2: Oscillator\n    ax_osc = ax_fg.twinx()\n\n    ax_osc.spines[\'right\'].set_position(\n        (\'axes\', 1.11)\n    )\n\n    line_osc, = ax_osc.plot(\n        df[\'Date\'],\n        df[\'Oscillator\'],\n        color=COL_OSC,\n        linewidth=2.0,\n        label=\'Fear & Greed Oscillator\'\n    )\n\n    ax_osc.axhline(\n        0,\n        color=COL_OSC,\n        linewidth=0.9,\n        alpha=0.7,\n        label=\'_nolegend_\'\n    )\n\n    ax_osc.set_ylabel(\n        \'Oscillator\',\n        color=COL_OSC,\n        fontsize=11\n    )\n\n    ax_osc.tick_params(\n        axis=\'y\',\n        labelcolor=COL_OSC\n    )\n\n\n    latest = df.iloc[-1]\n\n    latest_fg = latest[\'Fear_Greed_Score\']\n    latest_ema = latest[\'FG_EMA20_Score\']\n    latest_osc = latest[\'Oscillator\']\n\n\n    lines = [\n        line_fg,\n        line_ema20,\n        line_osc,\n        line_price\n    ]\n\n    labels = [\n        line.get_label()\n        for line in lines\n    ]\n\n    ax_fg.legend(\n        lines,\n        labels,\n        loc=\'upper left\',\n        ncol=2,\n        frameon=True\n    )\n\n\n    ax_fg.set_title(\n        f\'{title}\\n\'\n        f\'Fear & Greed {latest_fg:.1f} | \'\n        f\'EMA20 {latest_ema:.1f} | \'\n        f\'Oscillator {latest_osc:.4f}\',\n        fontsize=14\n    )\n\n    plt.show()\n\n\nplot_fg(\n    recent_kospi,\n    price_col=\'코스피\',\n    title=\'KOSPI – Fear & Greed Index + EMA20 + Oscillator (1 Year)\'\n)\n\nplot_fg(\n    recent_kosdaq,\n    price_col=\'코스닥\',\n    title=\'KOSDAQ – Fear & Greed Index + EMA20 + Oscillator (1 Year)\'\n)\n\n\n# ============================================================\n# 13) Elder Impulse System 그래프\n# ============================================================\n\ndef plot_impulse(\n    df,\n    price_col,\n    ema_col,\n    macd_col,\n    title\n):\n    if df.empty:\n        print(f"{title}: 표시할 데이터가 없습니다.")\n        return\n\n    colors = get_impulse_colors(\n        df,\n        ema_col,\n        macd_col\n    )\n\n    plt.figure(\n        figsize=(18, 7)\n    )\n\n    plt.plot(\n        df[\'Date\'],\n        df[price_col],\n        color=COL_PRICE,\n        linewidth=1.5\n    )\n\n    plt.scatter(\n        df[\'Date\'],\n        df[price_col],\n        c=colors,\n        s=20\n    )\n\n    plt.grid(\n        True,\n        alpha=GRID_ALPHA\n    )\n\n    plt.title(title)\n\n    plt.tight_layout()\n\n    plt.show()\n\n\nplot_impulse(\n    recent_kospi,\n    price_col=\'코스피\',\n    ema_col=\'KOSPI_EMA13\',\n    macd_col=\'KOSPI_MACD_Hist\',\n    title=\'KOSPI – Elder Impulse System (1 Year)\'\n)\n\nplot_impulse(\n    recent_kosdaq,\n    price_col=\'코스닥\',\n    ema_col=\'KOSDAQ_EMA13\',\n    macd_col=\'KOSDAQ_MACD_Hist\',\n    title=\'KOSDAQ – Elder Impulse System (1 Year)\'\n)\n\n\n# ============================================================\n# 14) Daily DeMark TD Setup 그래프\n# ============================================================\n\ndef plot_demark_daily(\n    df,\n    price_col,\n    title,\n    display_months=12\n):\n    df = df.copy()\n\n    if df.empty:\n        print(f"{title}: 표시할 데이터가 없습니다.")\n        return\n\n    cutoff = (\n        df[\'Date\'].max()\n        - pd.DateOffset(months=display_months)\n    )\n\n    df = df[\n        df[\'Date\'] >= cutoff\n    ].copy()\n\n    if (\n        \'TD_SellSetup\' not in df.columns\n        or \'TD_BuySetup\' not in df.columns\n    ):\n        df = add_td_setup_counts(\n            df,\n            price_col,\n            label=\'TD\'\n        )\n\n    fig, ax_price = plt.subplots(\n        figsize=(18, 7)\n    )\n\n    line_price, = ax_price.plot(\n        df[\'Date\'],\n        df[price_col],\n        color=COL_PRICE,\n        linewidth=1.5,\n        label=price_col\n    )\n\n    ax_price.set_ylabel(\n        price_col,\n        color=COL_PRICE\n    )\n\n    ax_price.tick_params(\n        axis=\'y\',\n        labelcolor=COL_PRICE\n    )\n\n    ax_price.grid(\n        True,\n        alpha=GRID_ALPHA\n    )\n\n\n    ax_td = ax_price.twinx()\n\n    line_sell, = ax_td.plot(\n        df[\'Date\'],\n        df[\'TD_SellSetup\'],\n        color=\'red\',\n        linewidth=1.2,\n        label=\'TD Sell Setup\'\n    )\n\n    line_buy, = ax_td.plot(\n        df[\'Date\'],\n        df[\'TD_BuySetup\'],\n        color=\'blue\',\n        linewidth=1.2,\n        label=\'TD Buy Setup\'\n    )\n\n    ax_td.set_ylabel(\n        \'TD Setup Count\'\n    )\n\n    max_td = float(\n        df[\n            [\n                \'TD_SellSetup\',\n                \'TD_BuySetup\'\n            ]\n        ]\n        .max()\n        .max()\n    )\n\n    ax_td.set_ylim(\n        0,\n        max_td + 2\n    )\n\n\n    lines = [\n        line_price,\n        line_sell,\n        line_buy\n    ]\n\n    ax_price.legend(\n        lines,\n        [\n            line.get_label()\n            for line in lines\n        ],\n        loc=\'upper left\'\n    )\n\n    ax_price.set_title(title)\n\n    plt.tight_layout()\n\n    plt.show()\n\n\nplot_demark_daily(\n    kospi,\n    price_col=\'코스피\',\n    title=\'KOSPI – Daily DeMark TD Setup (Sell=4, Buy=2, 1 Year)\',\n    display_months=DISPLAY_MONTHS\n)\n\nplot_demark_daily(\n    kosdaq,\n    price_col=\'코스닥\',\n    title=\'KOSDAQ – Daily DeMark TD Setup (Sell=4, Buy=2, 1 Year)\',\n    display_months=DISPLAY_MONTHS\n)\n\n\n# ============================================================\n# 15) 최신 수치 출력\n# ============================================================\n\ndef print_latest_status(\n    df,\n    market_name,\n    price_col\n):\n    valid = df.dropna(\n        subset=[\n            \'Fear_Greed_Score\',\n            \'FG_EMA20_Score\',\n            \'FG_MACD\',\n            \'FG_Signal\',\n            \'Oscillator\',\n            price_col\n        ]\n    )\n\n    if valid.empty:\n        print(\n            f\'{market_name}: 계산 가능한 데이터가 없습니다.\'\n        )\n        return\n\n    latest = valid.iloc[-1]\n\n    print(\'=\' * 60)\n    print(f\'{market_name} 최신 Fear & Greed\')\n    print(\'=\' * 60)\n\n    print(\n        f"기준일: "\n        f"{latest[\'Date\'].strftime(\'%Y-%m-%d\')}"\n    )\n\n    print(\n        f"{price_col}: "\n        f"{latest[price_col]:,.2f}"\n    )\n\n    print(\n        f"Fear & Greed Index: "\n        f"{latest[\'Fear_Greed_Score\']:.2f}"\n    )\n\n    print(\n        f"F&G EMA20: "\n        f"{latest[\'FG_EMA20_Score\']:.2f}"\n    )\n\n    print(\n        f"MACD: "\n        f"{latest[\'FG_MACD\']:.6f}"\n    )\n\n    print(\n        f"Signal: "\n        f"{latest[\'FG_Signal\']:.6f}"\n    )\n\n    print(\n        f"Oscillator: "\n        f"{latest[\'Oscillator\']:.6f}"\n    )\n\n\nprint_latest_status(\n    kospi,\n    market_name=\'KOSPI\',\n    price_col=\'코스피\'\n)\n\nprint_latest_status(\n    kosdaq,\n    market_name=\'KOSDAQ\',\n    price_col=\'코스닥\'\n)\n\n# ============================================================\n# 16) 개인 수급 Capitulation(항복) 산점도\n# ============================================================\n\n# 판정 논리\n# - X축: KOSPI 일간 수익률(%)\n# - Y축: 개인 일간 순매수대금(조원)\n# - 3사분면: 지수 하락 + 개인 순매도\n# - 개인 항복(Capitulation): 3사분면이 2거래일 이상 연속 발생\n#\n# 색상\n# - 회색: 과거 데이터\n# - 주황: 올해 데이터(3사분면 제외)\n# - 노랑: 올해의 단발성 3사분면\n# - 빨강: 최신 연속 3사분면 구간(개인 항복 후보)\n\nCAPITULATION_MIN_CONSECUTIVE_DAYS = 2\nCAPITULATION_DISPLAY_MONTHS = 12\nLABEL_RETURN_THRESHOLD = 5.0       # 수익률 절대값이 이 이상이면 날짜 표시\nLABEL_FLOW_THRESHOLD = 1.5         # 개인 순매수 절대값(조원)이 이 이상이면 날짜 표시\n\n\ndef find_latest_true_streak(mask, min_length=2):\n    """\n    True가 연속된 가장 최근 구간의 인덱스를 반환합니다.\n    연속 길이가 min_length보다 짧으면 빈 리스트를 반환합니다.\n    """\n    mask = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True)\n\n    latest_streak = []\n    current_streak = []\n\n    for i, flag in enumerate(mask):\n        if flag:\n            current_streak.append(i)\n        else:\n            if len(current_streak) >= min_length:\n                latest_streak = current_streak.copy()\n            current_streak = []\n\n    if len(current_streak) >= min_length:\n        latest_streak = current_streak.copy()\n\n    return latest_streak\n\n\ndef plot_individual_capitulation(\n    df,\n    display_months=12,\n    min_consecutive_days=2\n):\n    df = df.copy().dropna(\n        subset=[\n            \'KOSPI_Return_Pct\',\n            \'Individual_NetBuy_Trillion\'\n        ]\n    )\n\n    if df.empty:\n        print(\'Individual capitulation chart: no data available.\')\n        return\n\n    cutoff = (\n        df[\'Date\'].max()\n        - pd.DateOffset(months=display_months)\n    )\n\n    plot_df = (\n        df[df[\'Date\'] >= cutoff]\n        .copy()\n        .reset_index(drop=True)\n    )\n\n    if plot_df.empty:\n        print(\'Individual capitulation chart: no data in the selected period.\')\n        return\n\n    latest_year = int(plot_df[\'Date\'].max().year)\n    is_current_year = plot_df[\'Date\'].dt.year == latest_year\n    is_q3 = plot_df[\'Is_Third_Quadrant\']\n\n    # 최신 연속 3사분면 구간만 빨간색으로 강조\n    latest_streak_positions = find_latest_true_streak(\n        is_q3,\n        min_length=min_consecutive_days\n    )\n\n    plot_df[\'Is_Capitulation\'] = False\n\n    if latest_streak_positions:\n        plot_df.loc[\n            latest_streak_positions,\n            \'Is_Capitulation\'\n        ] = True\n\n    fig, ax = plt.subplots(figsize=(14, 10))\n\n    x = plot_df[\'KOSPI_Return_Pct\']\n    y = plot_df[\'Individual_NetBuy_Trillion\']\n\n    # 축 범위를 데이터에 맞게 대칭적으로 설정\n    x_limit = max(5, np.ceil(np.nanmax(np.abs(x)) / 5) * 5)\n    y_limit = max(2, np.ceil(np.nanmax(np.abs(y)) / 2) * 2)\n\n    ax.set_xlim(-x_limit, x_limit)\n    ax.set_ylim(-y_limit, y_limit)\n\n    # 3사분면 음영\n    ax.axvspan(\n        -x_limit,\n        0,\n        ymin=0,\n        ymax=0.5,\n        color=\'#FFCDD2\',\n        alpha=0.45,\n        zorder=0\n    )\n\n    # 기준선\n    ax.axhline(0, color=\'gray\', linewidth=1.1)\n    ax.axvline(0, color=\'gray\', linewidth=1.1)\n\n    # 과거 데이터\n    historical = ~is_current_year\n    ax.scatter(\n        plot_df.loc[historical, \'KOSPI_Return_Pct\'],\n        plot_df.loc[historical, \'Individual_NetBuy_Trillion\'],\n        s=90,\n        color=\'#B0BEC5\',\n        edgecolor=\'#546E7A\',\n        linewidth=1.0,\n        alpha=0.82,\n        label=\'Historical Data\',\n        zorder=2\n    )\n\n    # 올해 일반 데이터\n    current_normal = (\n        is_current_year\n        & ~is_q3\n        & ~plot_df[\'Is_Capitulation\']\n    )\n    ax.scatter(\n        plot_df.loc[current_normal, \'KOSPI_Return_Pct\'],\n        plot_df.loc[current_normal, \'Individual_NetBuy_Trillion\'],\n        s=135,\n        color=\'#F57C00\',\n        edgecolor=\'#455A64\',\n        linewidth=1.2,\n        alpha=0.9,\n        label=f\'{latest_year}\',\n        zorder=3\n    )\n\n    # 올해 단발성 3사분면\n    current_q3 = (\n        is_current_year\n        & is_q3\n        & ~plot_df[\'Is_Capitulation\']\n    )\n    ax.scatter(\n        plot_df.loc[current_q3, \'KOSPI_Return_Pct\'],\n        plot_df.loc[current_q3, \'Individual_NetBuy_Trillion\'],\n        s=155,\n        color=\'#FFC107\',\n        edgecolor=\'#455A64\',\n        linewidth=1.3,\n        alpha=0.95,\n        label=\'Third Quadrant (Single Day)\',\n        zorder=4\n    )\n\n    # 개인 항복 후보\n    capitulation = plot_df[\'Is_Capitulation\']\n    ax.scatter(\n        plot_df.loc[capitulation, \'KOSPI_Return_Pct\'],\n        plot_df.loc[capitulation, \'Individual_NetBuy_Trillion\'],\n        s=300,\n        color=\'red\',\n        edgecolor=\'darkred\',\n        linewidth=1.5,\n        alpha=0.95,\n        label=\'Capitulation Candidate\',\n        zorder=6\n    )\n\n    # 전체 데이터 회귀선\n    if len(plot_df) >= 2 and x.nunique() >= 2:\n        slope, intercept = np.polyfit(x, y, 1)\n        x_line = np.linspace(x.min(), x.max(), 200)\n        y_line = slope * x_line + intercept\n\n        corr = x.corr(y)\n\n        ax.plot(\n            x_line,\n            y_line,\n            linestyle=\':\',\n            linewidth=3.0,\n            color=\'#1565C0\',\n            label=f\'Regression Line (Correlation {corr:.2f})\',\n            zorder=1\n        )\n\n    # 올해 중요 날짜와 항복 후보에 날짜 라벨 표시\n    label_mask = (\n        is_current_year\n        & (\n            (plot_df[\'KOSPI_Return_Pct\'].abs() >= LABEL_RETURN_THRESHOLD)\n            | (\n                plot_df[\'Individual_NetBuy_Trillion\'].abs()\n                >= LABEL_FLOW_THRESHOLD\n            )\n            | plot_df[\'Is_Capitulation\']\n        )\n    )\n\n    for _, row in plot_df[label_mask].iterrows():\n        date_label = row[\'Date\'].strftime(\'%b %d\')\n\n        if row[\'Is_Capitulation\']:\n            text_color = \'red\'\n            font_size = 13\n            font_weight = \'bold\'\n        elif row[\'Is_Third_Quadrant\']:\n            text_color = \'#8D6E00\'\n            font_size = 10\n            font_weight = \'bold\'\n        else:\n            text_color = \'#E65100\'\n            font_size = 10\n            font_weight = \'bold\'\n\n        # 점 위치에 따라 라벨 방향 자동 조절\n        x_offset = 8 if row[\'KOSPI_Return_Pct\'] >= 0 else -8\n        y_offset = 10 if row[\'Individual_NetBuy_Trillion\'] >= 0 else -14\n        horizontal_alignment = (\n            \'left\'\n            if x_offset > 0\n            else \'right\'\n        )\n\n        ax.annotate(\n            date_label,\n            xy=(\n                row[\'KOSPI_Return_Pct\'],\n                row[\'Individual_NetBuy_Trillion\']\n            ),\n            xytext=(x_offset, y_offset),\n            textcoords=\'offset points\',\n            ha=horizontal_alignment,\n            va=\'bottom\' if y_offset > 0 else \'top\',\n            fontsize=font_size,\n            fontweight=font_weight,\n            color=text_color,\n            zorder=7\n        )\n\n    ax.text(\n        -x_limit * 0.92,\n        -y_limit * 0.88,\n        \'Third Quadrant\\nKOSPI Down + Individuals Net Selling\',\n        color=\'red\',\n        fontsize=15,\n        fontweight=\'bold\',\n        ha=\'left\',\n        va=\'bottom\'\n    )\n\n    latest_row = plot_df.iloc[-1]\n    latest_date = latest_row[\'Date\'].strftime(\'%Y-%m-%d\')\n\n    if latest_row[\'Is_Capitulation\']:\n        status = (\n            f\'Capitulation candidate: \'\n            f\'{int(plot_df["Is_Capitulation"].sum())} consecutive trading days in Q3\'\n        )\n    elif latest_row[\'Is_Third_Quadrant\']:\n        status = \'Entered Q3: confirmation requires another day\'\n    else:\n        status = \'No individual capitulation signal\'\n\n    ax.set_title(\n        \'KOSPI Daily Return vs Individual Net Buying\\n\'\n        f\'As of {latest_date} | {status}\',\n        fontsize=16,\n        fontweight=\'bold\'\n    )\n\n    ax.set_xlabel(\'KOSPI Daily Return (%)\', fontsize=12)\n    ax.set_ylabel(\'Individual Net Buying (KRW Trillion)\', fontsize=12)\n    ax.grid(True, linestyle=\':\', alpha=0.35)\n    ax.legend(loc=\'best\', frameon=True)\n\n    plt.tight_layout()\n    plt.show()\n\n    return plot_df\n\n\ncapitulation_result = plot_individual_capitulation(\n    individual_flow,\n    display_months=CAPITULATION_DISPLAY_MONTHS,\n    min_consecutive_days=CAPITULATION_MIN_CONSECUTIVE_DAYS\n)\n\n\n# ============================================================\n# 17) 개인 항복 최신 상태 출력\n# ============================================================\n\ndef print_capitulation_status(\n    result_df,\n    min_consecutive_days=2\n):\n    if result_df is None or result_df.empty:\n        print(\'개인 항복 상태: 계산 가능한 데이터가 없습니다.\')\n        return\n\n    latest = result_df.iloc[-1]\n\n    # 마지막 날짜까지 이어진 현재 3사분면 연속 일수 계산\n    current_streak = 0\n\n    for flag in result_df[\'Is_Third_Quadrant\'].iloc[::-1]:\n        if bool(flag):\n            current_streak += 1\n        else:\n            break\n\n    print(\'=\' * 60)\n    print(\'개인 수급 Capitulation(항복) 최신 상태\')\n    print(\'=\' * 60)\n    print(f"기준일: {latest[\'Date\'].strftime(\'%Y-%m-%d\')}")\n    print(f"KOSPI 일간 수익률: {latest[\'KOSPI_Return_Pct\']:.2f}%")\n    print(\n        \'개인 순매수대금: \'\n        f"{latest[\'Individual_NetBuy_Trillion\']:.2f}조원"\n    )\n    print(f\'현재 3사분면 연속 일수: {current_streak}거래일\')\n\n    if current_streak >= min_consecutive_days:\n        print(\'판정: 개인 항복(Capitulation) 후보 발생\')\n        print(\n            \'해석: 지수 하락과 개인 순매도가 \'\n            f\'{current_streak}거래일 연속 동반되었습니다.\'\n        )\n    elif current_streak == 1:\n        print(\'판정: 3사분면 진입, 하루 더 확인 필요\')\n    else:\n        print(\'판정: 개인 항복 신호 없음\')\n\n\nprint_capitulation_status(\n    capitulation_result,\n    min_consecutive_days=CAPITULATION_MIN_CONSECUTIVE_DAYS\n)\n'

def run_korea_fear_greed(excel_bytes):
    """
    사용자 원본 한국 Fear & Greed 코드를 실행하되,
    GitHub에는 화면 재현에 필요한 결과만 남긴다.
    원본 Excel bytes, exec 전체 namespace, 함수/모듈은 저장하지 않는다.
    """
    buf = io.StringIO()
    shown = []
    figs = []
    old_show = plt.show

    def _display(obj):
        try:
            if hasattr(obj, "data") and not isinstance(obj, pd.DataFrame):
                obj = obj.data
        except Exception:
            pass
        shown.append(obj)

    def _show(*args, **kwargs):
        try:
            fig = plt.gcf()
            titles = []
            for _ax in fig.axes:
                try:
                    titles.append(str(_ax.get_title()))
                except Exception:
                    pass

            # 한국 Fear & Greed에서는 Elder Impulse 그래프는 저장/표시하지 않음
            if any("elder impulse" in _t.lower() for _t in titles):
                plt.close(fig)
                return

            marker = _figure_marker(fig)
            if marker is not None:
                figs.append(marker)
            plt.close(fig)
        except Exception:
            pass

    ns = {
        "__name__": "__main__",
        "pd": pd, "np": np, "plt": plt, "io": io,
        "display": _display, "EXCEL_BYTES": excel_bytes,
    }
    try:
        plt.show = _show
        with contextlib.redirect_stdout(buf):
            exec(KOREA_FG_SRC, ns, ns)

        payload = {
            "compact_kr_fg": True,
            "kospi": ns.get("kospi"),
            "kosdaq": ns.get("kosdaq"),
            "figs": figs,
            "stdout": buf.getvalue(),
        }
        return True, payload, None
    except Exception:
        payload = {
            "compact_kr_fg": True,
            "kospi": ns.get("kospi"),
            "kosdaq": ns.get("kosdaq"),
            "figs": figs,
            "stdout": buf.getvalue(),
        }
        return False, payload, traceback.format_exc()
    finally:
        plt.show = old_show
        plt.close("all")


def render_korea_fear_greed_result(r):
    """새 경량 저장본 + 과거 저장본(6튜플/5튜플)을 모두 표시."""
    if r is None:
        return

    ok = False
    payload = None
    err = None

    # v11.9+ compact result
    if isinstance(r, tuple) and len(r) >= 2 and isinstance(r[1], dict) and r[1].get("compact_kr_fg"):
        ok = bool(r[0])
        payload = r[1]
        err = r[2] if len(r) >= 3 else None

    # v11.8 이전 result: (ok, ns, figs, shown, stdout, err)
    elif isinstance(r, tuple) and len(r) in (5, 6):
        ok = bool(r[0])
        ns = r[1] if isinstance(r[1], dict) else {}
        figs = r[2] if len(r) > 2 else []
        stdout = r[4] if len(r) > 4 else ""
        err = r[5] if len(r) > 5 else None
        payload = {
            "compact_kr_fg": True,
            "kospi": ns.get("kospi"),
            "kosdaq": ns.get("kosdaq"),
            "figs": figs,
            "stdout": stdout,
        }

    else:
        st.error("저장된 한국 피어앤그리드 결과 형식을 읽을 수 없습니다. 관리자에서 한 번 다시 계산해 주세요.")
        return

    if not ok:
        st.error("한국 피어앤그리드 계산 중 오류가 발생했습니다.")
        if err:
            with st.expander("오류 상세"):
                st.code(str(err))
        return

    c1, c2 = st.columns(2)
    for col, market, price_col, key in [
        (c1, "KOSPI", "코스피", "kospi"),
        (c2, "KOSDAQ", "코스닥", "kosdaq"),
    ]:
        df = payload.get(key)
        with col:
            st.subheader(market)
            if isinstance(df, pd.DataFrame):
                need = ["Fear_Greed_Score", "FG_EMA20_Score", "Oscillator", price_col]
                if all(x in df.columns for x in need):
                    valid = df.dropna(subset=need)
                    if not valid.empty:
                        last = valid.iloc[-1]
                        a, b, c = st.columns(3)
                        a.metric("Fear & Greed", f"{last['Fear_Greed_Score']:.1f}")
                        b.metric("EMA20", f"{last['FG_EMA20_Score']:.1f}")
                        c.metric("Oscillator", f"{last['Oscillator']:.4f}")

    figs = payload.get("figs") or []
    if figs:
        st.divider()
        st.subheader("차트")
        for fig in figs:
            # 과거 저장본에 Elder Impulse가 남아 있어도 화면에서는 제외
            if isinstance(fig, dict) and fig.get("__dashboard_chartdata__"):
                _titles = [
                    str(_ax.get("title", ""))
                    for _ax in fig.get("axes", [])
                    if isinstance(_ax, dict)
                ]
                if any("elder impulse" in _t.lower() for _t in _titles):
                    continue
            _render_saved_fig(fig)

    stdout = payload.get("stdout") or ""
    if stdout.strip():
        with st.expander("계산 결과 상세"):
            st.text(stdout)

# 안전한 세션 상태 초기화: 배포/재시작/기존 브라우저 세션에서도 KeyError 방지
_SAFE_STATE_DEFAULTS = {
    "liq_result": None, "fg_result": None, "canary_result": None, "trend_result": None,
    "rotation_result": None, "ai_result": None, "us_sector_result": None, "kr_sector_result": None, "kr_fg_result": None,
    "tw_revenue_result": None,
    "liq_updated": None, "fg_updated": None, "canary_updated": None, "trend_updated": None,
    "rotation_updated": None, "ai_updated": None, "us_sector_updated": None, "kr_sector_updated": None, "kr_fg_updated": None,
    "tw_revenue_updated": None,
}
for _k, _v in _SAFE_STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

load_persisted_once()

if ACTIVE_PAGE == "미국 유동성 체크":
    st.subheader("미국 유동성 체크")
    updated_caption("liq")
    if st.button("🔄 유동성 최신 데이터 업데이트", key="upd_liq", disabled=not IS_ADMIN):
        with st.spinner("유동성 계산 중..."):
            st.session_state.liq_result=run_liquidity()
            st.session_state.liq_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            persist_current_result("liq")
    r=st.session_state.liq_result
    if r is None:
        st.info("저장된 결과가 없습니다. 위 버튼을 눌러 처음 계산하세요.")
    elif r[0]:
        render_run(r)
    else:
        st.error("유동성 계산 오류"); st.code(r[2][-12000:],language="text")

if ACTIVE_PAGE == "미국 피어앤그리드 오실레이터":
    st.subheader("미국 피어앤그리드 오실레이터")
    updated_caption("fg")
    if st.button("🔄 Fear & Greed 최신 데이터 업데이트", key="upd_fg", disabled=not IS_ADMIN):
        with st.spinner("Fear & Greed 계산 및 그래프 생성 중..."):
            st.session_state.fg_result=run_fg()
            st.session_state.fg_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            persist_current_result("fg")
    r=st.session_state.fg_result
    if r is None:
        st.info("저장된 결과가 없습니다. 위 버튼을 눌러 처음 계산하세요.")
    elif r[0]:
        render_run(r)
    else:
        st.error("Fear & Greed 계산 오류"); st.code(r[2][-12000:],language="text")

if ACTIVE_PAGE == "미국 위험신호":
    st.subheader("미국 증시 위험 신호 · QQQ & TIP 카나리아")
    updated_caption("canary")
    if st.button("🔄 카나리아 최신 데이터 업데이트", key="upd_canary", disabled=not IS_ADMIN):
        with st.spinner("QQQ/TIP 업데이트 중..."):
            canary.clear()
            st.session_state.canary_result=canary()
            st.session_state.canary_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            persist_current_result("canary")
    r=st.session_state.canary_result
    if r is None:
        st.info("저장된 결과가 없습니다. 위 버튼을 눌러 처음 계산하세요.")
    else:
        d,v,m=r
        a,b,c=st.columns(3)
        with a: card("현재 신호",m,f"기준 {pd.Timestamp(d).date()}")
        with b: card("QQQ 모멘텀",f"{v['QQQ']:+.2%}","1M·3M·6M·12M 평균")
        with c: card("TIP 모멘텀",f"{v['TIP']:+.2%}","둘 다 양수면 공격")


if ACTIVE_PAGE == "미국 52주 신고가 전략 점검":
    st.subheader("미국 52주 신고가 전략 점검")
    st.caption("현재 전략 작동 여부 · 현재 포트폴리오 · 오늘 BUY/SELL · 백테스트 성과만 간단히 확인합니다.")
    updated_caption("rotation")
    if IS_ADMIN:
        st.warning("티커 수가 많아 업데이트 계산은 무겁습니다. 방문자는 저장된 결과만 조회합니다.")
        if st.button("🔄 미국 52주 신고가 전략 최신 데이터 업데이트", key="upd_rotation"):
            with st.spinner("미국 52주 신고가 전략 계산 중..."):
                time.sleep(2)
                st.session_state.rotation_result = run_rotation()
                if st.session_state.rotation_result[0]:
                    st.session_state.rotation_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    persist_current_result("rotation")
                    st.success("업데이트 완료. 핵심 상태·포트폴리오·성과만 압축 저장했습니다.")
    else:
        st.caption("최신 저장 결과를 조회하는 화면입니다. 방문자 접속으로 Yahoo 데이터를 다시 받지 않습니다.")

    r = st.session_state.rotation_result
    if r is None:
        st.info("저장된 결과가 없습니다. 관리자가 한 번 업데이트하면 이후 같은 결과가 유지됩니다.")
    else:
        render_rotation_simple(r)

if ACTIVE_PAGE == "AI하드웨어 주가 모멘텀 점검":
    st.subheader("AI하드웨어 주가 모멘텀 점검")
    st.caption("원본 Google Colab의 print → 표 → 그래프 출력 순서를 그대로 표시합니다. 별도의 카드·엑셀형 표로 재가공하지 않습니다.")
    updated_caption("ai")
    if st.button("🔄 AI하드웨어 주가 모멘텀 최신 데이터 업데이트", key="upd_ai", disabled=not IS_ADMIN):
        with st.spinner("AI하드웨어 주가 모멘텀 계산 중..."):
            time.sleep(2)
            st.session_state.ai_result=run_ai()
            if st.session_state.ai_result[0]:
                st.session_state.ai_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                persist_current_result("ai")
                st.success("업데이트 완료. Colab 출력 순서와 그래프 좌표를 압축 저장했습니다.")
    r=st.session_state.ai_result
    if r is None:
        st.info("저장된 결과가 없습니다. 관리자가 위 버튼을 눌러 처음 계산하면 이후 저장 결과를 그대로 표시합니다.")
    elif r[0]:
        render_colab_result(r)
    else:
        st.error("AI하드웨어 주가 모멘텀 계산 오류")
        render_colab_result(r)


# ============================================================
# 대만 기업 월별 매출 · 공식 MOPS
# ============================================================
if ACTIVE_PAGE == "대만 월별 매출":
    st.subheader("대만 기업 월별 매출")
    st.caption("AI·반도체 공급망 관심종목 · MOPS/TWSE/TPEX 공식 월매출 · 2023년부터 저장")
    st.caption("분류는 제공해주신 레퍼런스 화면의 26개 밸류체인 구성을 참고했고, 실제 매출 숫자는 대만 공식 공시에서 직접 가져옵니다.")
    updated_caption("tw_revenue")
    if IS_ADMIN:
        st.info("처음 1회는 2023년부터 과거 월매출을 채웁니다. 그 다음부터는 최근 3개월만 다시 확인해 저장하므로 가볍습니다.")
        if st.button("🔄 대만 월매출 최신 데이터 업데이트", key="upd_tw_revenue"):
            with st.spinner("공식 MOPS 월매출을 확인하고 있습니다. 첫 실행은 과거자료 때문에 시간이 조금 걸릴 수 있습니다..."):
                try:
                    st.session_state.tw_revenue_result = update_taiwan_revenue(st.session_state.get("tw_revenue_result"))
                    st.session_state.tw_revenue_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    persist_current_result("tw_revenue")
                    st.success("대만 월매출을 저장했습니다. 이후 방문자는 저장된 숫자로만 그래프를 봅니다.")
                except Exception as e:
                    st.error(f"대만 월매출 업데이트 오류: {e}")
    else:
        st.caption("최신 저장 결과를 조회하는 화면입니다. 방문자 접속으로 MOPS가 호출되지 않습니다.")
    render_taiwan_revenue(st.session_state.get("tw_revenue_result"))



if ACTIVE_PAGE == "미국 ETF 소라티노 및 상대강도":
    st.subheader("미국 ETF 소라티노 및 상대강도")
    st.caption("한국 ETF 화면과 같은 방식으로 원본 Google Colab의 print → 표 → 그래프 출력 순서를 그대로 표시합니다.")
    updated_caption("us_sector")
    if IS_ADMIN:
        st.warning("관리자 전용 업데이트입니다. ETF 수가 많아 필요할 때만 실행하세요.")
        if st.button("🔄 미국 ETF 소라티노·상대강도 최신 데이터 업데이트", key="upd_us_sector"):
            with st.spinner("미국 업종 ETF 3년 데이터를 받아 원본 Colab 로직으로 계산 중입니다..."):
                time.sleep(3)
                st.session_state.us_sector_result=run_us_sector()
                if st.session_state.us_sector_result[0]:
                    st.session_state.us_sector_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    persist_current_result("us_sector")
                    st.success("업데이트 완료. Colab 출력 순서와 그래프 좌표를 압축 저장했습니다.")
    else:
        st.caption("최신 저장 결과를 조회하는 화면입니다. 방문자 접속으로 Yahoo 데이터를 다시 받지 않습니다.")
    r=st.session_state.us_sector_result
    if r is None:
        st.info("저장된 결과가 없습니다. 관리자가 한 번 업데이트하면 이후 저장된 Colab형 결과를 표시합니다.")
    elif r[0]:
        # 한국 ETF와 같은 Colab형 출력 + 마지막 정배열 현황만 가독성 개선
        render_us_sector_colab(r[1])
    else:
        st.error("미국 ETF 소라티노·상대강도 계산 오류")
        if isinstance(r, tuple) and len(r) > 2 and r[2]:
            st.code(str(r[2])[-12000:], language="text")

if ACTIVE_PAGE == "한국 ETF 소라티노 및 상대강도":
    st.subheader("한국 ETF 소라티노 및 상대강도")
    st.caption("엑셀 '데이터' 시트의 DATE / 코스피 / 업종 ETF 가격열을 사용해 전체 분석 로직을 실행합니다.")
    st.caption("업로드한 원본 코드를 거의 그대로 실행하고, 결과도 Google Colab처럼 print → 표 → 그래프 순서로 표시합니다. 변동성 조정 모멘텀은 이평선 조건 없는 1개만, Sortino는 10일선/20일선/10주선 이상 3개를 유지합니다.")
    if IS_ADMIN:
        uploaded=st.file_uploader("📂 국내시장 최신 엑셀 업로드", type=["xlsx","xls"], key="kr_sector_excel")
        if uploaded is None:
            st.info("관리자 전용: 엑셀을 선택한 뒤 계산 버튼을 누르세요.")
        else:
            st.caption(f"선택 파일: {uploaded.name}")
            if st.button("▶ 업로드한 엑셀로 국내 주도업종 계산 · 저장", key="run_kr_sector"):
                with st.spinner("Mansfield RS · Momentum · Sortino · Breadth · 정배열 · 52주 전략까지 계산 중..."):
                    st.session_state.kr_sector_result=run_korea_sector(uploaded.getvalue())
                    if st.session_state.kr_sector_result[0]:
                        st.session_state.kr_sector_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        persist_current_result("kr_sector")
                        st.success("계산 완료. Colab형 출력 순서를 저장했습니다. 표/텍스트와 그래프 좌표를 gzip 압축 저장하므로 스크린샷을 누적하지 않습니다.")
    else:
        st.caption("최신 저장 결과를 조회하는 화면입니다. 방문자 접속으로 계산하거나 엑셀을 다시 읽지 않습니다.")
    updated_caption("kr_sector")
    r=st.session_state.kr_sector_result
    if r is None:
        st.info("저장된 한국 ETF 소라티노·상대강도 결과가 없습니다. 관리자가 엑셀을 올려 한 번 계산하면 이후 계속 유지됩니다.")
    elif r[0]:
        render_korea_sector_compact(r[1])
    else:
        st.error("한국 ETF 소라티노·상대강도 계산 오류")
        st.code(r[2][-16000:],language="text")


# ============================================================
# 한국 과열·공포: 관리자 엑셀 업로드형
# ============================================================
if ACTIVE_PAGE == "한국 피어앤그리드 오실레이터":
    st.header("한국 피어앤그리드 오실레이터 · KOSPI & KOSDAQ")
    st.caption("KOSPI/KOSDAQ Fear & Greed, EMA20, Oscillator, DeMark TD Setup, 개인 수급 Capitulation")

    if IS_ADMIN:
        kr_fg_file = st.file_uploader(
            "📂 한국 피어앤그리드 최신 엑셀 업로드",
            type=["xlsx", "xlsm"],
            key="kr_fg_excel"
        )
        if kr_fg_file is not None:
            st.caption(f"선택 파일: {kr_fg_file.name}")
            if st.button("▶ 업로드한 엑셀로 한국 피어앤그리드 계산 · 저장", key="run_kr_fg"):
                with st.spinner("한국 Fear & Greed 및 보조지표 계산 중..."):
                    result = run_korea_fear_greed(kr_fg_file.getvalue())
                    st.session_state["kr_fg_result"] = result

                    if result[0]:
                        st.session_state["kr_fg_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_ok, save_msg = persist_current_result("kr_fg")

                        if save_ok:
                            # 실제 GitHub에서 즉시 다시 읽어 저장까지 확인
                            chk_result, chk_updated = github_load_result("kr_fg")
                            if chk_result is not None:
                                st.success(f"계산 + GitHub 영구저장 확인 완료 · {save_msg}")
                            else:
                                st.error("계산은 완료됐지만 GitHub 재조회 확인에 실패했습니다. 저장소의 dashboard_data/kr_fg.pkl을 확인하세요.")
                        # save 실패 메시지는 persist_current_result에서 이미 출력
                    else:
                        st.error("계산 중 오류가 발생해 기존 저장 결과는 덮어쓰지 않았습니다.")
    else:
        st.caption("최신 저장 결과를 조회하는 화면입니다. 방문자 접속으로 엑셀을 다시 읽거나 계산하지 않습니다.")

    if st.session_state.get("kr_fg_updated"):
        st.caption(f"마지막 업데이트: {st.session_state.get('kr_fg_updated')}")

    r = st.session_state.get("kr_fg_result")
    if r is None:
        st.info("저장된 결과가 없습니다. 관리자가 최신 엑셀을 업로드해 한 번 계산·저장하면 이후 접속에서도 그대로 유지됩니다.")
    else:
        render_korea_fear_greed_result(r)

