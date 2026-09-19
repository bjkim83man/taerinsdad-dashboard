
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="태린이아빠 Market Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{background:#101a31;border:1px solid #2b3c5d;border-radius:18px;padding:22px;margin-bottom:15px}
.hero h1{color:#fff;margin:0}.hero p{color:#b8c4d8;margin:6px 0 0}
.card{background:#121c32;border:1px solid #2a3958;border-radius:14px;padding:15px;min-height:115px}
.label{color:#aab8cd;font-size:.86rem}.value{color:#fff;font-size:1.65rem;font-weight:800;margin-top:5px}
.sub{color:#8e9db4;font-size:.76rem;margin-top:6px}
[data-testid="stSidebar"]{background:#0f1728}
</style>
""", unsafe_allow_html=True)

def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def dl(tickers, period="3y", interval="1d"):
    x = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
    if isinstance(x.columns, pd.MultiIndex):
        close=x["Close"].copy()
    else:
        close=x[["Close"]].copy()
        if isinstance(tickers,str): close.columns=[tickers]
    return close.dropna(how="all").sort_index()

# ---------- Canary ----------
@st.cache_data(ttl=3600, show_spinner=False)
def canary():
    p=dl(["QQQ","TIP"],"3y")
    vals={}
    for t in ["QQQ","TIP"]:
        s=p[t].dropna()
        vals[t]=np.mean([(s.iloc[-1]/s.shift(n).iloc[-1]-1) for n in [21,63,126,252]])
    return p.index[-1], vals, ("공격 모드" if vals["QQQ"]>0 and vals["TIP"]>0 else "방어 모드")

# ---------- Monthly trend ----------
@st.cache_data(ttl=3600, show_spinner=False)
def trend_strategy():
    p=dl(["QQQ","TIP","QLD","SOXX","SPMO","PSQ"],"16y","1mo")
    # remove possibly incomplete current month
    now=pd.Timestamp.today()
    if len(p) and p.index[-1].year==now.year and p.index[-1].month==now.month:
        p=p.iloc[:-1]
    q=p["QQQ"].dropna(); tip=p["TIP"].dropna()
    qma=q.rolling(6).mean(); tma=tip.rolling(6).mean()
    d=min(q.index[-1],tip.index[-1])
    up=bool(q.loc[d]>qma.loc[d] and tip.loc[d]>tma.loc[d])
    alloc="QQQ 60% · QLD 15% · SOXX 25%" if up else "SPMO 50% · PSQ 50%"
    return d,up,alloc,q.loc[d],qma.loc[d],tip.loc[d],tma.loc[d]

# ---------- AI Hardware ----------
AI={"NVDA":"NVDA","AVGO":"AVGO","AMD":"AMD","TSM":"TSM","ASML":"ASML","MU":"MU","ARM":"ARM",
"LRCX":"LRCX","AMAT":"AMAT","KLAC":"KLAC","ANET":"ANET","DELL":"DELL","WDC":"WDC","STX":"STX",
"LITE":"LITE","GLW":"GLW","ALAB":"ALAB","Samsung":"005930.KS","SKHynix":"000660.KS",
"TokyoElectron":"8035.T","Advantest":"6857.T","Disco":"6146.T","Lasertec":"6920.T","Murata":"6981.T"}

@st.cache_data(ttl=3600, show_spinner=False)
def ai_hw():
    raw=dl(list(AI.values()),"3y").rename(columns={v:k for k,v in AI.items()}).ffill(limit=5)
    raw=raw.loc[:,raw.notna().sum()>=220]
    ma20=raw.rolling(20,min_periods=15).mean(); ma60=raw.rolling(60,min_periods=45).mean(); ma200=raw.rolling(200,min_periods=160).mean()
    r60=raw.pct_change(60,fill_method=None)
    def pct(cond,avail): return cond.sum(axis=1)/avail.replace(0,np.nan)*100
    b=pd.DataFrame(index=raw.index)
    b["MA20"]=pct((raw>ma20)&ma20.notna(),ma20.notna().sum(axis=1))
    b["MA60"]=pct((raw>ma60)&ma60.notna(),ma60.notna().sum(axis=1))
    b["MA200"]=pct((raw>ma200)&ma200.notna(),ma200.notna().sum(axis=1))
    b["P60"]=pct(r60>0,r60.notna().sum(axis=1))
    b["Score"]=b.MA20*.2+b.MA60*.35+b.MA200*.25+b.P60*.2
    b["MA5"]=b.Score.rolling(5,min_periods=3).mean()
    b["Slope5"]=b.MA5-b.MA5.shift(5); b["Slope10"]=b.MA5-b.MA5.shift(10)
    last=b.dropna(subset=["MA5"]).iloc[-1]
    if last.MA5>=70 and last.Slope5>=0: risk="강세"
    elif last.MA5>=70: risk="고점 경계"
    elif last.MA5>=60: risk="정상"
    elif last.MA5>=50: risk="주의" if last.Slope5<0 else "중립"
    elif last.MA5>=35: risk="위험" if last.Slope10<0 else "약세 반등"
    else: risk="투매 진행" if last.Slope5<0 else "투매 후 반등"
    return b,risk

# ---------- 52W live signal (dashboard-light version) ----------
SECTORS={"XLK":"Technology","XLC":"Communication","XLY":"Discretionary","XLP":"Staples","XLI":"Industrials",
"XLB":"Materials","XLE":"Energy","XLF":"Financials","XLV":"Health Care","XLU":"Utilities","XLRE":"Real Estate",
"SOXX":"Semiconductors","IGV":"Software","SKYY":"Cloud","FDN":"Internet","XOP":"Oil & Gas","OIH":"Oil Services",
"CRAK":"Refining","AMLP":"Energy Infrastructure","URA":"Uranium","KRE":"Regional Banks","IAI":"Broker Dealers",
"KIE":"Insurance","BIZD":"BDC","IBB":"Biotech","IHI":"Medical Devices","IHF":"Health Providers","IHE":"Pharma",
"XRT":"Retail","XHB":"Homebuilders","PEJ":"Leisure","PBJ":"Food & Beverage","ITA":"Defense","IYT":"Transportation",
"AIRR":"Industrial Renaissance","GDX":"Gold Miners","SIL":"Silver Miners","SLX":"Steel","COPX":"Copper",
"REMX":"Rare Earth","MOO":"Agribusiness","WOOD":"Timber","ICLN":"Clean Energy","TAN":"Solar","FAN":"Wind",
"GRID":"Smart Grid","IYZ":"Telecom","LIT":"Lithium","SEA":"Shipping","CARZ":"Auto","UFO":"Space","ROBO":"Robotics"}

@st.cache_data(ttl=3600, show_spinner=False)
def high52_live():
    tick=["SPY"]+list(SECTORS)
    p=dl(tick,"3y").ffill(limit=3)
    spy=p["SPY"]; sec=p[[x for x in SECTORS if x in p.columns]]
    rr=sec.div(spy,axis=0)
    rs=rr/rr.shift(126)-1
    prev=sec.shift(1).rolling(252,min_periods=252).max()
    br=sec>prev
    ma20=sec.rolling(20).mean()
    d=sec.index[-1]
    rs_today=rs.loc[d].dropna().sort_values(ascending=False)
    ranks=pd.Series(range(1,len(rs_today)+1),index=rs_today.index)
    top=rs_today.head(20)
    candidates=[t for t in top.index if bool(br.loc[d,t])]
    rows=[]
    for t in candidates:
        rows.append({"Ticker":t,"Industry":SECTORS[t],"6M RS vs SPY":rs_today[t],"RS Rank":int(ranks[t]),
                     "Close":sec.loc[d,t],"MA20":ma20.loc[d,t]})
    return d,pd.DataFrame(rows)

with st.sidebar:
    st.markdown("## 태린이아빠")
    st.caption("Market Dashboard · v2")
    if st.button("🔄 최신 데이터 새로고침",use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.info("유동성·Fear & Greed는 다음 연결 단계에서 기존 Colab 원본 로직을 그대로 붙입니다.")

st.markdown('<div class="hero"><h1>태린이아빠 Market Dashboard</h1><p>시장환경 → 리스크 → 추세 → 전략 신호를 한 화면에서 확인</p></div>',unsafe_allow_html=True)

tabs=st.tabs(["종합","유동성","Fear & Greed","카나리아","미국 추세","52주 신고가","52W + Rotation","AI 하드웨어"])

with tabs[0]:
    st.subheader("미국 시장 종합 신호")
    try:
        cd,cv,cm=canary(); td,up,alloc,*_=trend_strategy(); b,risk=ai_hw(); hd,hc=high52_live()
        a,b1,c,d=st.columns(4)
        with a: card("카나리아",cm,f"QQQ {cv['QQQ']:+.1%} · TIP {cv['TIP']:+.1%}")
        with b1: card("미국 추세","상승추세" if up else "하락추세",alloc)
        with c: card("AI Hardware",risk,f"Breadth {b['MA5'].iloc[-1]:.1f}")
        with d: card("52W 신규 후보",f"{len(hc)}개",f"기준 {hd.date()}")
        st.caption("유동성과 Fear & Greed는 원본 코드 연결 후 종합화면에도 자동 추가됩니다.")
    except Exception as e: st.error(f"계산 오류: {e}")

with tabs[1]:
    st.subheader("미국 유동성 환경")
    st.info("첫 번째 Colab의 Fed/Treasury 유동성 + 민간신용 로직 연결 자리입니다. 원본 계산식을 바꾸지 않고 결과 카드·차트만 이 탭에 표시합니다.")

with tabs[2]:
    st.subheader("Fear & Greed Oscillator")
    st.info("Fear & Greed 원본 코드 연결 자리입니다. 현재값, 구간, 변화 방향, 오실레이터 차트를 표시하도록 준비했습니다.")

with tabs[3]:
    st.subheader("카나리아 자산 · QQQ & TIP")
    try:
        d,v,mode=canary()
        c1,c2,c3=st.columns(3)
        with c1: card("신호",mode,f"기준 {d.date()}")
        with c2: card("QQQ 모멘텀",f"{v['QQQ']:+.2%}","1M·3M·6M·12M 단순평균")
        with c3: card("TIP 모멘텀",f"{v['TIP']:+.2%}","둘 다 양수면 공격")
    except Exception as e: st.error(e)

with tabs[4]:
    st.subheader("미국 추세시 / 위기시 로테이션")
    try:
        d,up,alloc,q,qma,t,tma=trend_strategy()
        c1,c2,c3=st.columns(3)
        with c1: card("현재 추세","상승추세" if up else "하락추세",f"{d.strftime('%Y-%m')} 완성 월봉")
        with c2: card("QQQ vs 6M MA",f"{q:.2f} / {qma:.2f}","QQQ가 6개월선 위인지 확인")
        with c3: card("TIP vs 6M MA",f"{t:.2f} / {tma:.2f}","TIP도 동시에 위여야 상승")
        st.markdown("### 다음 달 목표 포트폴리오")
        st.success(alloc)
    except Exception as e: st.error(e)

with tabs[5]:
    st.subheader("52주 신고가 전략")
    st.caption("6개월 SPY 대비 RS Top20 + 직전 252거래일 신고가 돌파 후보")
    try:
        d,x=high52_live()
        c1,c2=st.columns(2)
        with c1: card("기준일",str(d.date()),"Yahoo Finance")
        with c2: card("현재 신규 후보",f"{len(x)}개","최대 4종목 운용 전략")
        if x.empty: st.info("오늘 조건을 동시에 만족하는 신규 후보가 없습니다.")
        else:
            y=x.copy(); y["6M RS vs SPY"]=y["6M RS vs SPY"].map(lambda z:f"{z:+.2%}")
            st.dataframe(y,use_container_width=True,hide_index=True)
        st.warning("v2에서는 현재 후보를 먼저 표시합니다. 기존 보유·MA20 매도·거래내역까지 포함한 완전한 상태 추적은 Rotation 원본 엔진 연결 시 함께 붙입니다.")
    except Exception as e: st.error(e)

with tabs[6]:
    st.subheader("52W + Weak Regime Rotation_B")
    st.info("원본 MASTER의 상태 추적 백테스트 엔진을 연결할 자리입니다.")
    st.markdown("""
**원본 규칙**
- NORMAL_52W: 6개월 SPY 대비 RS Top20 + 52주 신고가
- WEAK_ROTATION: 순수 52W의 최근 20D SPY 대비 초과수익 ≤ -1%
- 약세 신규 후보: 5D SPY 대비 강세 + 20D RS 순위 10일간 10등 이상 개선 + 20D 신고가
- 기존 보유는 Regime 변화만으로 팔지 않고 설정된 매도 MA 이탈 때 매도
""")

with tabs[7]:
    st.subheader("AI 하드웨어 모멘텀")
    try:
        b,risk=ai_hw(); last=b.iloc[-1]
        c1,c2,c3,c4=st.columns(4)
        with c1: card("위험 단계",risk,"AI 하드웨어 내부 확산")
        with c2: card("Breadth MA5",f"{last.MA5:.1f}","0~100")
        with c3: card("5D Slope",f"{last.Slope5:+.1f}","둔화 속도")
        with c4: card("60일 수익률 +",f"{last.P60:.1f}%","상승 종목 비율")
        v=b.tail(180)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=v.index,y=v.MA20,name="MA20 위"))
        fig.add_trace(go.Scatter(x=v.index,y=v.MA60,name="MA60 위"))
        fig.add_trace(go.Scatter(x=v.index,y=v.MA200,name="MA200 위"))
        fig.update_layout(template="plotly_dark",height=430,yaxis_range=[0,100],legend_orientation="h")
        st.plotly_chart(fig,use_container_width=True)
    except Exception as e: st.error(e)
