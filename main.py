# main.py
# Futures Multi-TF + GPT pattern detection + BTC/ETH options (Deribit) integration
import os
import asyncio
import aiohttp
import traceback
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

# Deribit (public) endpoints for option chain (BTC/ETH)
DERIBIT_INSTRUMENTS = "https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&kind=option&expired=false"
DERIBIT_SUMMARY = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; skipping send_text.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}) as r:
            if r.status != 200:
                txt = await r.text()
                print("Telegram send_text failed", r.status, txt[:300])
    except Exception as e:
        print("send_text exception:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; skipping send_photo and removing file.")
        try: os.remove(path)
        except: pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with session.post(url, data=data, timeout=60) as r:
                if r.status != 200:
                    txt = await r.text()
                    print("Telegram send_photo failed", r.status, txt[:300])
    except Exception as e:
        print("send_photo exception:", e)
    finally:
        try: os.remove(path)
        except: pass

# ---------------- HTTP fetch helper ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                try:
                    txt = await r.text()
                except:
                    txt = "<no body>"
                print(f"fetch_json {url} returned {r.status}: {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json error for", url, e)
        return None

# ---------------- Futures data fetch ----------------
async def fetch_tf_data(session, symbol, tf):
    url = f"{BASE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={tf}&limit=100"
    c = await fetch_json(session, url)
    out = {}
    if isinstance(c, list):
        out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in c]
        out["times"] = [int(x[0]) // 1000 for x in c]
    return out

async def fetch_futures_metrics(session, symbol):
    t = await fetch_json(session, f"{BASE_FAPI}/fapi/v1/ticker/24hr?symbol={symbol}")
    oi = await fetch_json(session, f"{BASE_FAPI}/fapi/v1/openInterest?symbol={symbol}")
    fr = await fetch_json(session, f"{BASE_FAPI}/fapi/v1/fundingRate?symbol={symbol}&limit=1")
    d = {}
    if t:
        try:
            d["price"] = float(t.get("lastPrice", 0))
            d["volume"] = float(t.get("volume", 0))
        except:
            d["price"] = None
            d["volume"] = None
    if oi:
        try:
            d["openInterest"] = float(oi.get("openInterest", 0))
        except:
            d["openInterest"] = None
    if fr and isinstance(fr, list) and fr:
        try:
            d["fundingRate"] = float(fr[0].get("fundingRate", 0))
        except:
            d["fundingRate"] = None
    return d

# ---------------- Deribit option chain fetch & parse (BTC/ETH) ----------------
async def fetch_deribit_option_data(session, base_symbol):
    try:
        currency = base_symbol.upper()
        inst_url = DERIBIT_INSTRUMENTS.format(currency=currency)
        sum_url = DERIBIT_SUMMARY.format(currency=currency)
        inst_resp = await fetch_json(session, inst_url)
        sum_resp = await fetch_json(session, sum_url)
        if not inst_resp or not sum_resp:
            return None
        return {"instruments": inst_resp.get("result", []), "summaries": sum_resp.get("result", [])}
    except Exception as e:
        print("fetch_deribit_option_data error:", e)
        return None

def parse_deribit_chain(option_data):
    try:
        instruments = option_data.get("instruments", [])
        summaries = option_data.get("summaries", [])
        strike_map = {}
        # Summaries usually contain strike & option_type plus open_interest & volume
        for s in summaries:
            try:
                strike = s.get("strike")
                otype = s.get("option_type")
                oi = s.get("open_interest", 0) or 0
                vol = s.get("volume", 0) or 0
                if strike is None:
                    # fallback parse from instrument_name
                    instr = s.get("instrument_name", "")
                    parts = instr.split("-")
                    if len(parts) >= 4:
                        try:
                            strike = float(parts[2])
                            otype = "call" if parts[3].upper().startswith("C") else "put"
                        except:
                            strike = None
                if strike is None:
                    continue
                st = float(strike)
                if st not in strike_map:
                    strike_map[st] = {"call_oi": 0.0, "put_oi": 0.0, "call_vol": 0.0, "put_vol": 0.0}
                if otype and otype.lower().startswith("c"):
                    strike_map[st]["call_oi"] += float(oi)
                    strike_map[st]["call_vol"] += float(vol)
                else:
                    strike_map[st]["put_oi"] += float(oi)
                    strike_map[st]["put_vol"] += float(vol)
            except Exception:
                continue

        total_put = sum(v["put_oi"] for v in strike_map.values())
        total_call = sum(v["call_oi"] for v in strike_map.values())
        pcr = (total_put / total_call) if total_call > 0 else None

        max_pain = None
        if strike_map:
            strikes_sorted = sorted(strike_map.keys())
            pain_scores = {}
            for K in strikes_sorted:
                score = 0.0
                for S, vals in strike_map.items():
                    dist = abs(S - K)
                    score += dist * (vals.get("call_oi", 0) + vals.get("put_oi", 0))
                pain_scores[K] = score
            max_pain = min(pain_scores.items(), key=lambda x: x[1])[0]

        top_strikes = sorted(strike_map.items(), key=lambda kv: (kv[1]["call_oi"] + kv[1]["put_oi"]), reverse=True)[:5]
        top_list = [{"strike": k, "call_oi": v["call_oi"], "put_oi": v["put_oi"]} for k, v in top_strikes]

        return {"strike_map": strike_map, "pcr": pcr, "max_pain": max_pain, "top_strikes": top_list, "total_call": total_call, "total_put": total_put}
    except Exception as e:
        print("parse_deribit_chain error:", e)
        return None

# ---------------- GPT Pattern Analysis ----------------
async def analyze_patterns_with_gpt(symbol, candles, tf):
    if not client or not candles:
        return None
    # send last up to 50 candles
    last = candles[-50:]
    data_lines = []
    for i, (o,h,l,c,vol) in enumerate(last):
        data_lines.append(f"{i}: O={o},H={h},L={l},C={c}")
    data_str = "\n".join(data_lines)
    prompt = (
        f"You are a professional technical analyst.\n"
        f"Analyze the following {tf} candles for {symbol}.\n"
        "1) Identify candlestick patterns (Hammer, Doji, Engulfing, Three Soldiers, etc.)\n"
        "2) Identify chart patterns (Head & Shoulders, Double Top/Bottom, Triangle, Wedge, Flag, Cup & Handle, etc.)\n"
        "3) Provide short bias (Bullish/Bearish/Neutral).\n"
        "Respond concisely in this format:\n"
        "Candles: ... | Chart: ... | Bias: ...\n\n"
        "Data:\n" + data_str
    )
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=200,
                temperature=0.2
            )
        resp = await loop.run_in_executor(None, call_model)
        # extract text safely
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            return choice.message.content.strip()
        else:
            return getattr(choice, "text", str(resp)).strip()
    except Exception as e:
        print("GPT analysis error:", e)
        return None

# ---------------- PA Detector (same logic) ----------------
def compute_levels(candles, lookback=50):
    if not candles: return (None, None, None)
    arr = candles[-lookback:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k]) / k if highs else None
    sup = sum(lows[:k]) / k if lows else None
    mid = (res + sup) / 2 if res is not None and sup is not None else None
    return sup, res, mid

def detect_signal(sym, data, tf):
    candles = data.get("candles")
    if not candles or len(candles) < 5:
        return None
    last = candles[-1]; prev = candles[-2]
    sup, res, mid = compute_levels(candles)
    entry = last[3]; bias = "NEUTRAL"; reason = []; conf = 50

    if res and entry > res:
        bias = "BUY"; reason.append("Breakout"); conf += 15
    if sup and entry < sup:
        bias = "SELL"; reason.append("Breakdown"); conf += 15
    if last[3] > last[0] and prev[3] < prev[0]:
        reason.append("Bullish Engulfing"); bias = "BUY"; conf += 10
    if last[3] < last[0] and prev[3] > prev[0]:
        reason.append("Bearish Engulfing"); bias = "SELL"; conf += 10

    sl = None; targets = []
    if bias == "BUY":
        try:
            sl = min([c[2] for c in candles[-6:]]) * 0.997
        except:
            sl = entry * 0.98
    if bias == "SELL":
        try:
            sl = max([c[1] for c in candles[-6:]]) * 1.003
        except:
            sl = entry * 1.02
    if sl:
        risk = abs(entry - sl)
        if bias == "BUY":
            targets = [entry + risk * r for r in (1,2,3)]
        else:
            targets = [entry - risk * r for r in (1,2,3)]

    return {"tf": tf, "bias": bias, "entry": entry, "sl": sl, "targets": targets, "reason": "; ".join(reason), "conf": conf, "levels": {"sup": sup, "res": res, "mid": mid}}

# ---------------- Multi-TF Chart plotting ----------------
def plot_multi_chart(tf_results, sym, trades):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=100, sharex=False)
    for i, (tf, ax) in enumerate(zip(TF_MAP.values(), axs)):
        data = tf_results.get(tf, {})
        trade = next((t for t in trades if t["tf"] == tf), None)
        candles = data.get("candles"); times = data.get("times")
        if not candles:
            ax.set_title(f"{sym} {tf} - no data")
            continue
        dates = [datetime.utcfromtimestamp(t) for t in times]
        o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; c_ = [c[3] for c in candles]
        x = date2num(dates); width = 0.6*(x[1]-x[0]) if len(x) > 1 else 0.4
        for xi, oi, hi, li, ci in zip(x, o, h, l, c_):
            col = "white" if ci >= oi else "black"
            ax.vlines(xi, li, hi, color="black", linewidth=0.7)
            ax.add_patch(plt.Rectangle((xi-width/2, min(oi,ci)), width, max(0.0001, abs(ci-oi)), facecolor=col, edgecolor="black"))
        if trade:
            if trade.get("entry") is not None:
                ax.axhline(trade["entry"], color="blue", label=f"Entry {trade['entry']}")
            if trade.get("sl") is not None:
                ax.axhline(trade["sl"], color="red", linestyle="--", label=f"SL {trade['sl']}")
            for j, trg in enumerate(trade.get("targets", [])):
                ax.axhline(trg, color="green", linestyle=":", label=f"T{j+1} {trg}")
        # annotate levels if present
        levs = trade.get("levels") if trade else None
        if levs:
            if levs.get("res") is not None:
                ax.axhline(levs["res"], linestyle="--", color="orange", linewidth=0.6)
            if levs.get("sup") is not None:
                ax.axhline(levs["sup"], linestyle="--", color="purple", linewidth=0.6)
        ax.set_title(f"{sym} Futures {tf}")
        ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ---------------- Main loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, "Futures Bot (30m/1h/4h + GPT Patterns + BTC/ETH Options) online 🚀")
        while True:
            try:
                for sym in SYMBOLS:
                    # fetch futures metrics
                    metrics = await fetch_futures_metrics(session, sym)
                    # fetch TF data concurrently
                    tf_results = {}
                    for tf in TF_MAP.values():
                        tf_results[tf] = await fetch_tf_data(session, sym, tf)

                    trades = []
                    # for each TF, detect signals and enrich with GPT patterns
                    for tf, d in tf_results.items():
                        tr = detect_signal(sym, d, tf)
                        if tr and tr["bias"] in ("BUY", "SELL") and tr["conf"] >= SIGNAL_CONF_THRESHOLD:
                            # GPT pattern analysis (may be rate-limited -> safe to fail)
                            try:
                                gpt = await analyze_patterns_with_gpt(sym, d.get("candles", []), tf)
                                if gpt:
                                    tr["ai_patterns"] = gpt
                            except Exception as e:
                                print("GPT call error:", e)
                            trades.append(tr)

                    # Add BTC/ETH options info if symbol maps
                    option_summary = None
                    base = None
                    if sym.upper().startswith("BTC"):
                        base = "BTC"
                    elif sym.upper().startswith("ETH"):
                        base = "ETH"

                    if base:
                        try:
                            raw = await fetch_deribit_option_data(session, base)
                            if raw:
                                parsed = parse_deribit_chain(raw)
                                option_summary = parsed
                        except Exception as e:
                            print("Option fetch/parse error for", base, e)

                    if trades:
                        lines = [f"🚨 *{sym}* Futures Signals:"]
                        for t in trades:
                            lines.append(f"[{t['tf']}] {t['bias']} | Entry={t['entry']} SL={t['sl']} Targets={t['targets']} | Conf={t['conf']}% | {t['reason']}")
                            if t.get("ai_patterns"):
                                lines.append(f"[{t['tf']}] AI: {t['ai_patterns']}")
                        # include futures metrics
                        if metrics:
                            lines.append(f"Price={metrics.get('price')} | OI={metrics.get('openInterest')} | Funding={metrics.get('fundingRate')}")
                        # include options summary for BTC/ETH
                        if option_summary:
                            pcr = option_summary.get("pcr")
                            max_pain = option_summary.get("max_pain")
                            top = option_summary.get("top_strikes", [])
                            lines.append("Options (Deribit): " + (f"PCR={pcr:.2f}" if pcr is not None else "PCR=N/A") + (f" | MaxPain={max_pain}" if max_pain else ""))
                            if top:
                                top_str = ", ".join([str(int(tk["strike"])) for tk in top[:3]])
                                lines.append(f"Top OI strikes: {top_str}")
                        text = "\n".join(lines)
                        # create multi-TF chart and send
                        try:
                            chart = plot_multi_chart(tf_results, sym, trades)
                        except Exception as e:
                            print("plot_multi_chart error:", e)
                            chart = None
                        await send_text(session, text)
                        if chart:
                            await send_photo(session, text, chart)

                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("main loop error:", e)
                traceback.print_exc()
                await asyncio.sleep(min(60, POLL_INTERVAL))

if __name__ == "__main__":
    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        print("Stopped by user.")
