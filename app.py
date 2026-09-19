
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="태린이아빠 | AI 하드웨어 모멘텀", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.4rem; max-width:1500px;}
.hero {background:#101a31;border:1px solid #2b3c5d;border-radius:18px;padding:24px;margin-bottom:18px;}
.hero h1 {color:white;margin:0;font-size:2rem;}
.hero p {color:#b8c4d8;margin:7px 0 0 0;}
.card {background:#121c32;border:1px solid #2a3958;border-radius:15px;padding:16px;min-height:120px;}
.label {color:#9fb0ca;font-size:.9rem;margin-bottom:7px;}
.value {color:white;font-size:1.9rem;font-weight:800;}
.sub {color:#8291a8;font-size:.78rem;margin-top:7px;}
.status {background:#121c32;border-left:5px solid #6ea8fe;border-radius:12px;padding:16px 18px;margin:12px 0 20px;}
[data-testid="stSidebar"] {background:#0f1728;}
</style>
""", unsafe_allow_html=True)

STOCKS = {
    "NVDA":"NVDA","AVGO":"AVGO","AMD":"AMD","TSM":"TSM","ASML":"ASML","MU":"MU","ARM":"ARM",
    "QCOM":"QCOM","MRVL":"MRVL","LRCX":"LRCX","AMAT":"AMAT","KLAC":"KLAC","CDNS":"CDNS","SNPS":"SNPS",
    "ANET":"ANET","TXN":"TXN","ON":"ON","DELL":"DELL","Sandisk":"SNDK","Intel":"INTC","Amkor":"AMKR",
    "WesternDigital":"WDC","Seagate":"STX","Lumentum":"LITE","Corning":"GLW","AsteraLabs":"ALAB",
    "Samsung":"005930.KS","SKHynix":"000660.KS","SamsungElectroMechanics":"009150.KS",
    "TokyoElectron":"8035.T","Advantest":"6857.T","Disco":"6146.T","Lasertec":"6920.T","SCREEN":"7735.T",
    "Socionext":"6526.T","Murata":"6981.T","Kioxia":"285A.T","UMC":"UMC","ASE":"ASX","Himax":"HIMX","SiliconMotion":"SIMO"
}
PERIODS={"1M":21,"2M":42,"3M":63,"6M":126}

def classify_risk(row):
    s=row["Breadth_Score_MA5"]; s5=row["Slope_5D"]; s10=row["Slope_10D"]
    if pd.isna(s): return "데이터 부족"
    if row["Strong_Bearish_Divergence"]: return "피크아웃 강경고"
    if row["Bearish_Divergence"]: return "피크아웃 경고"
    if s>=70 and s5>=0: return "강세"
    if s>=70 and s5<0: return "고점 경계"
    if s>=60: return "정상"
    if s>=50: return "주의" if s5<0 else "중립"
    if s>=35: return "위험" if s10<0 else "약세 반등"
    return "투매 진행" if s5<0 else "투매 후 반등"

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    raw=yf.download(list(STOCKS.values()),period="3y",auto_adjust=True,progress=False,group_by="column",threads=True)
    prices=raw["Close"].copy()
    prices=prices.rename(columns={v:k for k,v in STOCKS.items()}).sort_index().ffill(limit=5)
    valid=prices.columns[prices.notna().sum()>=220]
    excluded=[c for c in prices.columns if c not in valid]
    prices=prices[valid]

    rows=[]
    for p,d in PERIODS.items():
        r=prices.pct_change(d,fill_method=None).iloc[-1].dropna()
        pos=int((r>0).sum()); neg=int((r<0).sum()); zero=int((r==0).sum()); total=len(r)
        rows.append({"Period":p,"Positive":pos,"Negative":neg,"Zero":zero,"Total":total,
                     "Positive_Ratio":pos/total*100 if total else np.nan,
                     "Negative_Ratio":neg/total*100 if total else np.nan})
    count_table=pd.DataFrame(rows).set_index("Period")

    ma20=prices.rolling(20,min_periods=15).mean()
    ma60=prices.rolling(60,min_periods=45).mean()
    ma200=prices.rolling(200,min_periods=160).mean()
    ret60=prices.pct_change(60,fill_method=None)

    def pct_above(cond, avail):
        return cond.sum(axis=1)/avail.replace(0,np.nan)*100

    a20=pct_above((prices>ma20)&prices.notna()&ma20.notna(),(prices.notna()&ma20.notna()).sum(axis=1))
    a60=pct_above((prices>ma60)&prices.notna()&ma60.notna(),(prices.notna()&ma60.notna()).sum(axis=1))
    a200=pct_above((prices>ma200)&prices.notna()&ma200.notna(),(prices.notna()&ma200.notna()).sum(axis=1))
    p60=pct_above((ret60>0)&ret60.notna(),ret60.notna().sum(axis=1))

    b=pd.DataFrame({"Above_MA20":a20,"Above_MA60":a60,"Above_MA200":a200,"Positive_60D":p60})
    b["Breadth_Score"]=b["Above_MA20"]*.20+b["Above_MA60"]*.35+b["Above_MA200"]*.25+b["Positive_60D"]*.20
    b["Breadth_Score_MA5"]=b["Breadth_Score"].rolling(5,min_periods=3).mean()
    b["Breadth_Diff"]=b["Breadth_Score"]-b["Breadth_Score_MA5"]
    b["Slope_3D"]=b["Breadth_Score_MA5"]-b["Breadth_Score_MA5"].shift(3)
    b["Slope_5D"]=b["Breadth_Score_MA5"]-b["Breadth_Score_MA5"].shift(5)
    b["Slope_10D"]=b["Breadth_Score_MA5"]-b["Breadth_Score_MA5"].shift(10)
    b["Falling_Days_10D"]=b["Breadth_Score_MA5"].diff().lt(0).rolling(10).sum()

    first=prices.apply(lambda c:c.dropna().iloc[0] if not c.dropna().empty else np.nan)
    norm=prices.divide(first,axis=1)*100
    b["AI_Tech_Index"]=norm.mean(axis=1)
    idx_high=b["AI_Tech_Index"].rolling(20).max()
    br_high=b["Breadth_Score_MA5"].rolling(20).max()
    b["Index_Near_20D_High"]=b["AI_Tech_Index"]>=idx_high*.99
    b["Breadth_From_20D_High"]=b["Breadth_Score_MA5"]-br_high
    b["Bearish_Divergence"]=b["Index_Near_20D_High"]&(b["Breadth_From_20D_High"]<=-10)&(b["Slope_5D"]<0)
    b["Strong_Bearish_Divergence"]=b["Index_Near_20D_High"]&(b["Breadth_From_20D_High"]<=-20)&(b["Slope_10D"]<0)
    b["Risk_Level"]=b.apply(classify_risk,axis=1)

    latest=b.dropna(subset=["Breadth_Score_MA5"]).iloc[-1]
    latest_date=b.dropna(subset=["Breadth_Score_MA5"]).index[-1]
    return prices,b,count_table,latest,latest_date,excluded

def card(label,value,sub=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 태린이아빠")
    st.caption("AI Hardware Momentum")
    st.write("Colab 로직을 Streamlit에서 직접 실행합니다.")
    if st.button("🔄 최신 데이터 새로고침",use_container_width=True):
        st.cache_data.clear(); st.rerun()

st.markdown('<div class="hero"><h1>AI 하드웨어 모멘텀 대시보드</h1><p>Breadth · 추세 확산 · 다이버전스 · 피크아웃 위험</p></div>',unsafe_allow_html=True)

try:
    with st.spinner("최신 데이터를 계산 중입니다..."):
        prices,b,count_table,latest,latest_date,excluded=load_data()
except Exception as e:
    st.error("데이터 계산 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

level=latest["Risk_Level"]
st.caption(f"기준일 {latest_date.date()} · 분석 종목 {len(prices.columns)}개")

c1,c2,c3,c4=st.columns(4)
with c1: card("현재 위험 단계",level,"Breadth + 기울기 + 다이버전스")
with c2: card("Breadth Score MA5",f"{latest['Breadth_Score_MA5']:.1f}","0~100")
with c3: card("5일 기울기",f"{latest['Slope_5D']:+.1f}","최근 확산 속도")
with c4: card("20일 고점 대비 Breadth",f"{latest['Breadth_From_20D_High']:+.1f}","음수 확대 시 내부 약화")

if level in ["피크아웃 강경고","피크아웃 경고"]:
    msg="지수는 고점권인데 내부 Breadth가 약해지는 다이버전스가 포착된 구간입니다."
elif level in ["고점 경계","주의"]:
    msg="절대 Breadth 수준보다 최근 기울기 둔화를 더 주의해서 볼 구간입니다."
elif level in ["강세","정상"]:
    msg="AI 하드웨어 내부 확산은 아직 비교적 양호한 구간입니다."
else:
    msg="AI 하드웨어 내부 확산이 약한 구간입니다."
st.markdown(f'<div class="status"><b>현재 해석</b><br>{msg}</div>',unsafe_allow_html=True)

st.subheader("AI 하드웨어 내부 확산")
x1,x2,x3,x4=st.columns(4)
with x1: card("20일선 위",f"{latest['Above_MA20']:.1f}%","단기")
with x2: card("60일선 위",f"{latest['Above_MA60']:.1f}%","중기")
with x3: card("200일선 위",f"{latest['Above_MA200']:.1f}%","장기")
with x4: card("60일 수익률 +",f"{latest['Positive_60D']:.1f}%","상승 종목 확산")

left,right=st.columns([1,1.6])
with left:
    ct=count_table.reset_index()
    fig=go.Figure()
    fig.add_bar(x=ct["Period"],y=ct["Positive"],name="상승")
    fig.add_bar(x=ct["Period"],y=ct["Negative"],name="하락")
    fig.update_layout(title="기간별 상승·하락 종목 수",barmode="group",height=370,template="plotly_dark",legend_orientation="h")
    st.plotly_chart(fig,use_container_width=True)

with right:
    v=b.tail(180)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=v.index,y=v["Above_MA20"],name="MA20 위"))
    fig.add_trace(go.Scatter(x=v.index,y=v["Above_MA60"],name="MA60 위"))
    fig.add_trace(go.Scatter(x=v.index,y=v["Above_MA200"],name="MA200 위"))
    for y in [70,50,30]: fig.add_hline(y=y,line_dash="dash",opacity=.3)
    fig.update_layout(title="Moving Average Breadth · 최근 180거래일",height=370,template="plotly_dark",yaxis_range=[0,100],legend_orientation="h")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Breadth Score와 둔화 속도")
l,r=st.columns(2)
with l:
    v=b.tail(260)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=v.index,y=v["Breadth_Score"],name="Daily Score",opacity=.3))
    fig.add_trace(go.Scatter(x=v.index,y=v["Breadth_Score_MA5"],name="MA5",line=dict(width=3)))
    for y in [70,60,50,35]: fig.add_hline(y=y,line_dash="dash",opacity=.25)
    fig.update_layout(title="AI Tech Breadth Score",height=400,template="plotly_dark",yaxis_range=[0,100],legend_orientation="h")
    st.plotly_chart(fig,use_container_width=True)
with r:
    v=b.tail(260)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=v.index,y=v["Slope_5D"],name="5D Slope"))
    fig.add_trace(go.Scatter(x=v.index,y=v["Slope_10D"],name="10D Slope"))
    fig.add_hline(y=0,opacity=.5)
    fig.update_layout(title="Breadth Slope",height=400,template="plotly_dark",legend_orientation="h")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("AI Tech 지수 vs Breadth — 피크아웃 체크")
v=b.tail(300)
fig=go.Figure()
fig.add_trace(go.Scatter(x=v.index,y=v["AI_Tech_Index"],name="AI Tech 동일가중 지수",yaxis="y1"))
fig.add_trace(go.Scatter(x=v.index,y=v["Breadth_Score_MA5"],name="Breadth Score MA5",yaxis="y2"))
for d in v.index[v["Strong_Bearish_Divergence"].fillna(False)]:
    fig.add_vline(x=d,opacity=.18,line_width=7)
fig.update_layout(height=480,template="plotly_dark",legend_orientation="h",
                  yaxis=dict(title="AI Tech Index"),
                  yaxis2=dict(title="Breadth Score",overlaying="y",side="right",range=[0,100]))
st.plotly_chart(fig,use_container_width=True)

d1,d2,d3=st.columns(3)
with d1: card("Bearish Divergence","YES" if bool(latest["Bearish_Divergence"]) else "NO","고점권 + Breadth 약화")
with d2: card("Strong Divergence","YES" if bool(latest["Strong_Bearish_Divergence"]) else "NO","강한 피크아웃 경고")
with d3: card("최근 10일 Breadth 하락일",f"{latest['Falling_Days_10D']:.0f}일","MA5 일간 변화 기준")

with st.expander("최근 30거래일 상세 데이터"):
    cols=["Above_MA20","Above_MA60","Above_MA200","Positive_60D","Breadth_Score_MA5","Breadth_Diff","Slope_5D","Slope_10D","Breadth_From_20D_High","Bearish_Divergence","Strong_Bearish_Divergence","Risk_Level"]
    st.dataframe(b[cols].tail(30),use_container_width=True)
