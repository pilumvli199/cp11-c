# main.py
# Binance Futures Bot with Multi-Timeframe Analysis + GPT Pattern Detection
import os, asyncio, aiohttp, traceback
from datetime import datetime
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from openai import OpenAI

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 60.0))

BASE_FAPI = "https://fapi.binance.com"
TF_MAP = {"30m":"30m","1h":"1h","4h":"4h"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"Markdown"})
    except Exception as e: print("send_text error:",e)

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path,"rb") as f:
            data=aiohttp.FormData()
            data.add_field("chat_id",str(TELEGRAM_CHAT_ID))
            data.add_field("caption",caption)
            data.add_field("photo",f,filename="chart.png",content_type="image/png")
            await session.post(url,data=data)
    except Exception as e: print("send_photo error:",e)
    finally:
        try: os.remove(path)
        except: pass

# ---------------- Fetching ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=15) as r:
            if r.status!=200: return None
            return await r.json()
    except: return None

async def fetch_tf_data(session,symbol,tf):
    url=f"{BASE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit=100"
    c=await fetch_json(session,url)
    out={}
    if isinstance(c,list):
        out["candles"]=[[float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in c]
        out["times"]=[int(x[0])//1000 for x in c]
    return out

async def fetch_futures_metrics(session,symbol):
    t=await fetch_json(session,f"{BASE_FAPI}/fapi/v1/ticker/24hr?symbol={symbol}")
    oi=await fetch_json(session,f"{BASE_FAPI}/fapi/v1/openInterest?symbol={symbol}")
    fr=await fetch_json(session,f"{BASE_FAPI}/fapi/v1/fundingRate?symbol={symbol}&limit=1")
    d={}
    if t: d["price"]=float(t.get("lastPrice",0)); d["volume"]=float(t.get("volume",0))
    if oi: d["openInterest"]=float(oi.get("openInterest",0))
    if fr and isinstance(fr,list) and fr: d["fundingRate"]=float(fr[0].get("fundingRate",0))
    return d

# ---------------- GPT Pattern Analysis ----------------
async def analyze_patterns_with_gpt(symbol,candles,tf):
    if not client or not candles: return None
    last50=candles[-50:]
    data_str="\n".join([f"{i}: O={o},H={h},L={l},C={c}" for i,(o,h,l,c,vol) in enumerate(last50)])
    prompt=f"""
    You are a professional technical analyst.
    Analyze the following {tf} candles for {symbol}.
    1. Identify candlestick patterns (Hammer, Doji, Engulfing, Star, Three Soldiers, etc.)
    2. Identify chart patterns (Head & Shoulders, Double Top/Bottom, Triangle, Wedge, Flag, Cup & Handle, etc.)
    3. Give a short bias (Bullish/Bearish/Neutral).
    Respond in one concise line:
    Candles: ... | Chart: ... | Bias: ...
    Data:
    {data_str}
    """
    resp=await asyncio.get_event_loop().run_in_executor(
        None, lambda: client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,temperature=0.2
        )
    )
    return resp.choices[0].message.content.strip()

# ---------------- PA Detector ----------------
def compute_levels(candles,lookback=50):
    if not candles: return (None,None,None)
    arr=candles[-lookback:]
    highs=sorted([c[1] for c in arr],reverse=True)
    lows=sorted([c[2] for c in arr])
    k=min(3,len(arr))
    res=sum(highs[:k])/k if highs else None
    sup=sum(lows[:k])/k if lows else None
    mid=(res+sup)/2 if res and sup else None
    return sup,res,mid

def detect_signal(sym,data,tf):
    candles=data.get("candles")
    if not candles or len(candles)<5: return None
    last=candles[-1]; prev=candles[-2]
    sup,res,mid=compute_levels(candles)
    entry=last[3]; bias="NEUTRAL"; reason=[]; conf=50

    if res and entry>res: bias="BUY"; reason.append("Breakout"); conf+=15
    if sup and entry<sup: bias="SELL"; reason.append("Breakdown"); conf+=15
    if last[3]>last[0] and prev[3]<prev[0]: reason.append("Bullish Engulfing"); bias="BUY"; conf+=10
    if last[3]<last[0] and prev[3]>prev[0]: reason.append("Bearish Engulfing"); bias="SELL"; conf+=10

    sl=None; targets=[]
    if bias=="BUY": sl=min([c[2] for c in candles[-6:]])*0.997
    if bias=="SELL": sl=max([c[1] for c in candles[-6:]])*1.003
    if sl:
        risk=abs(entry-sl)
        if bias=="BUY": targets=[entry+risk*r for r in (1,2,3)]
        if bias=="SELL": targets=[entry-risk*r for r in (1,2,3)]

    return {"tf":tf,"bias":bias,"entry":entry,"sl":sl,"targets":targets,"reason":"; ".join(reason),"conf":conf}

# ---------------- Multi-TF Chart ----------------
def plot_multi_chart(tf_results,sym,trades):
    fig,axs=plt.subplots(3,1,figsize=(10,12),dpi=100,sharex=False)
    for i,(tf,ax) in enumerate(zip(TF_MAP.values(),axs)):
        data=tf_results[tf]; trade=[t for t in trades if t["tf"]==tf]
        candles=data.get("candles"); times=data.get("times")
        if not candles: continue
        dates=[datetime.utcfromtimestamp(t) for t in times]
        o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c_=[c[3] for c in candles]
        x=date2num(dates); width=0.6*(x[1]-x[0]) if len(x)>1 else 0.4
        for xi,oi,hi,li,ci in zip(x,o,h,l,c_):
            col="white" if ci>=oi else "black"
            ax.vlines(xi,li,hi,color="black",linewidth=0.7)
            ax.add_patch(plt.Rectangle((xi-width/2,min(oi,ci)),width,max(0.0001,abs(ci-oi)),facecolor=col,edgecolor="black"))
        if trade:
            t=trade[0]
            if t.get("entry"): ax.axhline(t["entry"],color="blue",label=f"Entry {t['entry']}")
            if t.get("sl"): ax.axhline(t["sl"],color="red",linestyle="--",label=f"SL {t['sl']}")
            for j,trg in enumerate(t.get("targets",[])): ax.axhline(trg,color="green",linestyle=":",label=f"T{j+1} {trg}")
        ax.set_title(f"{sym} Futures {tf}")
        ax.legend(loc="upper left",fontsize="small")
    fig.autofmt_xdate()
    tmp=NamedTemporaryFile(delete=False,suffix=".png")
    fig.savefig(tmp.name,bbox_inches="tight"); plt.close(fig)
    return tmp.name

# ---------------- Loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session,"Futures Bot (30m/1h/4h + GPT Patterns) online 🚀")
        while True:
            try:
                for sym in SYMBOLS:
                    metrics=await fetch_futures_metrics(session,sym)
                    tf_results={tf:await fetch_tf_data(session,sym,tf) for tf in TF_MAP.values()}
                    trades=[]
                    for tf,d in tf_results.items():
                        tr=detect_signal(sym,d,tf)
                        if tr and tr["bias"] in ("BUY","SELL") and tr["conf"]>=SIGNAL_CONF_THRESHOLD:
                            # add GPT patterns
                            gpt_patterns=await analyze_patterns_with_gpt(sym,d["candles"],tf)
                            if gpt_patterns: tr["ai_patterns"]=gpt_patterns
                            trades.append(tr)
                    if trades:
                        lines=[f"🚨 *{sym}* Futures Signals:"]
                        for t in trades:
                            lines.append(f"[{t['tf']}] {t['bias']} | Entry={t['entry']} SL={t['sl']} Targets={t['targets']} | Conf={t['conf']}% | {t['reason']}")
                            if "ai_patterns" in t:
                                lines.append(f"[{t['tf']}] AI: {t['ai_patterns']}")
                        if metrics:
                            lines.append(f"Price={metrics.get('price')} | OI={metrics.get('openInterest')} | Funding={metrics.get('fundingRate')}")
                        text="\n".join(lines)
                        chart=plot_multi_chart(tf_results,sym,trades)
                        await send_text(session,text)
                        if chart: await send_photo(session,text,chart)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("loop error:",e)
                traceback.print_exc()
                await asyncio.sleep(60)

if __name__=="__main__":
    try: asyncio.run(loop())
    except KeyboardInterrupt: print("Stopped")
