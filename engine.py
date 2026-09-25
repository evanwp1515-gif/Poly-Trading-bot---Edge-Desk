#!/usr/bin/env python3
"""Polymarket Edge engine (read-only, no API keys, no Claude usage).

Every run:
  1. Pulls Polymarket daily temperature markets and a tech/AI shortlist.
  2. Turns six weather models into bucket probabilities, corrected per station
     for bias and error size using the last 21 days of airport observations.
  3. Blends model with market price, keeps ideas whose edge beats spread + fees.
  4. Tags ideas with what proven weather wallets hold (smart money).
  5. Logs every prediction, settles resolved markets, and scores the model
     against the market (Brier score) so the go-live gate is measured, not guessed.
  6. Runs a $20 paper account that takes every idea under the PRD risk rules.

State lives as JSON in the data folder (committed back to the repo by GitHub Actions):
  feed.json     ideas + market snapshots for the app
  paper.json    the $20 paper account
  stats.json    calibration scorecard
  calib.json    per-station bias / error (refreshed daily)
  pending.json  predictions waiting for markets to resolve
  history/YYYY-MM.jsonl  resolved predictions (one row per market)
  ai_estimates.json      optional tech/AI probabilities you add by hand

Usage: python3 engine.py --data data [--cache]
"""
import argparse, json, math, os, re, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

VERSION = "1.0"
S = requests.Session()
S.headers.update({"User-Agent": "polymarket-edge/1.0 (personal read-only research tool)"})
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
ET = ZoneInfo("America/New_York")

# ---------------- tunables (PRD sections 5 and 5b) ----------------
MODEL_WEIGHT = 0.5            # p = w*model + (1-w)*market mid
TECH_AI_WEIGHT = 0.5
MIN_EDGE = 0.05               # net of spread and estimated fee
TECH_MIN_EDGE = 0.08
MIN_PRICE, MAX_PRICE = 0.05, 0.95
MAX_SPREAD = 0.05
LEAD_GROWTH_F = {0: 0.6, 1: 1.0, 2: 1.6, 3: 2.2}   # extra forecast error by lead, deg F (prior; auto-tuned)
DEFAULT_RESID_F = 1.8        # used when a station has too little history
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless",
          "jma_seamless", "ukmo_seamless"]
PAPER = {"start": 20.0, "kellyFraction": 0.25, "minStake": 1.0, "capPct": 0.05,
         "skipBelow": 0.5, "maxOpen": 6.0, "dailyLossStop": 2.0, "floor": 12.0,
         "maxNewPerRun": 4}

CITY = {  # city -> (station, IANA tz). Stations are the ones named in each market's rules.
 "Amsterdam": ("EHAM", "Europe/Amsterdam"), "Ankara": ("LTAC", "Europe/Istanbul"),
 "Atlanta": ("KATL", "America/New_York"), "Austin": ("KAUS", "America/Chicago"),
 "Beijing": ("ZBAA", "Asia/Shanghai"), "Buenos Aires": ("SAEZ", "America/Argentina/Buenos_Aires"),
 "Busan": ("RKPK", "Asia/Seoul"), "Cape Town": ("FACT", "Africa/Johannesburg"),
 "Chengdu": ("ZUUU", "Asia/Shanghai"), "Chicago": ("KORD", "America/Chicago"),
 "Chongqing": ("ZUCK", "Asia/Shanghai"), "Dallas": ("KDAL", "America/Chicago"),
 "Denver": ("KBKF", "America/Denver"), "Guangzhou": ("ZGGG", "Asia/Shanghai"),
 "Helsinki": ("EFHK", "Europe/Helsinki"), "Hong Kong": ("HKO", "Asia/Hong_Kong"),
 "Houston": ("KHOU", "America/Chicago"), "Istanbul": ("LTFM", "Europe/Istanbul"),
 "Jeddah": ("OEJN", "Asia/Riyadh"), "Jinan": ("ZSJN", "Asia/Shanghai"),
 "Karachi": ("OPKC", "Asia/Karachi"), "Kuala Lumpur": ("WMKK", "Asia/Kuala_Lumpur"),
 "London": ("EGLC", "Europe/London"), "Los Angeles": ("KLAX", "America/Los_Angeles"),
 "Lucknow": ("VILK", "Asia/Kolkata"), "Madrid": ("LEMD", "Europe/Madrid"),
 "Manila": ("RPLL", "Asia/Manila"), "Mexico City": ("MMMX", "America/Mexico_City"),
 "Miami": ("KMIA", "America/New_York"), "Milan": ("LIMC", "Europe/Rome"),
 "Moscow": ("UUWW", "Europe/Moscow"), "Munich": ("EDDM", "Europe/Berlin"),
 "NYC": ("KLGA", "America/New_York"), "Panama City": ("MPMG", "America/Panama"),
 "Paris": ("LFPB", "Europe/Paris"), "Qingdao": ("ZSQD", "Asia/Shanghai"),
 "San Francisco": ("KSFO", "America/Los_Angeles"), "Sao Paulo": ("SBGR", "America/Sao_Paulo"),
 "Seattle": ("KSEA", "America/Los_Angeles"), "Seoul (Incheon)": ("RKSI", "Asia/Seoul"),
 "Shanghai": ("ZSPD", "Asia/Shanghai"), "Shenzhen": ("ZGSZ", "Asia/Shanghai"),
 "Singapore": ("WSSS", "Asia/Singapore"), "Taipei": ("RCSS", "Asia/Taipei"),
 "Tel Aviv": ("LLBG", "Asia/Jerusalem"), "Tokyo": ("RJTT", "Asia/Tokyo"),
 "Toronto": ("CYYZ", "America/Toronto"), "Warsaw": ("EPWA", "Europe/Warsaw"),
 "Wellington": ("NZWN", "Pacific/Auckland"), "Wuhan": ("ZHHH", "Asia/Shanghai"),
 "Zhengzhou": ("ZHCC", "Asia/Shanghai"),
}
FIXED_COORDS = {"HKO": (22.302, 114.174)}  # Hong Kong Observatory HQ (not an airport)
TZ_OF = {st: tz for st, tz in CITY.values()}

CACHE = False


def get_json(url, params=None, key=None, tries=3, text=False):
    path = None
    if CACHE and key:
        os.makedirs("cache", exist_ok=True)
        path = os.path.join("cache", re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:180] + (".txt" if text else ".json"))
        if os.path.exists(path):
            return open(path).read() if text else json.load(open(path))
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=120)
            if r.status_code == 200:
                d = r.text if text else r.json()
                if isinstance(d, dict) and d.get("error"):
                    raise RuntimeError(d.get("reason"))
                if path:
                    open(path, "w").write(d) if text else json.dump(d, open(path, "w"))
                return d
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url.split('?')[0]} failed: {last}")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def save(path, obj, pretty=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1 if pretty else None, separators=None if pretty else (",", ":"))
    os.replace(tmp, path)


def short(cid):
    return cid[:18]


# ============================================================== markets
def fetch_events(tag, closed=False, end_min=None, max_pages=12):
    out = []
    for page in range(max_pages):
        p = {"closed": str(closed).lower(), "limit": 100, "offset": page * 100, "tag_slug": tag}
        if end_min:
            p["end_date_min"] = end_min
        d = get_json(f"{GAMMA}/events", p, key=f"ev_{tag}_{closed}_{end_min}_{page}")
        out += d
        if len(d) < 100:
            break
    return out


TITLE_RE = re.compile(r"(Highest|Lowest) temperature in (.+?) on (.+?)\?")
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"], 1)}


def parse_bucket(label):
    """'58-59°F'->(58,59,'F'); '57°F or below'->(None,57,'F'); '76°F or higher'->(76,None,'F'); '23°C'->(23,23,'C')"""
    s = label.replace("º", "°")
    unit = "C" if "°C" in s else "F"
    rng = re.search(r"(-?\d+)\s*(?:-|–|to)\s*(-?\d+)", s)
    if rng:
        return (int(rng.group(1)), int(rng.group(2)), unit)
    nums = [int(n) for n in re.findall(r"-?\d+", s)]
    if not nums:
        return None
    if "below" in s or "lower" in s:
        return (None, nums[0], unit)
    if "higher" in s or "above" in s:
        return (nums[0], None, unit)
    return (nums[0], nums[0], unit)


def quote(m):
    bid, ask = fnum(m.get("bestBid")), fnum(m.get("bestAsk"))
    last = fnum(m.get("lastTradePrice"))
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
    else:
        mid = ask if ask is not None else (last or 0.0)
    return bid, ask, mid


def fee_rate(m):
    return (m.get("feeSchedule") or {}).get("rate", 0) if m.get("feesEnabled") else 0


def fee_per_share(rate, price):
    """Estimate of Polymarket's taker fee: rate * p * (1-p) per share. Makers pay nothing."""
    return rate * price * (1 - price)


def parse_weather_events(events, now):
    out = []
    for e in events:
        m = TITLE_RE.match(e.get("title", ""))
        if not m or m.group(2) not in CITY:
            continue
        city = m.group(2)
        try:
            mon, day = m.group(3).split()[:2]
            month, day = MONTHS[mon.lower()], int(day)
        except (KeyError, ValueError):
            continue
        tz = ZoneInfo(CITY[city][1])
        local = now.astimezone(tz)
        year = local.year + (1 if month < local.month - 6 else 0)
        date = datetime(year, month, day).date()
        buckets = []
        for mk in e.get("markets", []):
            if mk.get("closed") or not mk.get("acceptingOrders", True):
                continue
            b = parse_bucket(mk.get("groupItemTitle") or "")
            if not b:
                continue
            bid, ask, mid = quote(mk)
            buckets.append({"cid": mk["conditionId"], "label": mk.get("groupItemTitle"), "lo": b[0], "hi": b[1],
                            "unit": b[2], "bid": bid, "ask": ask, "mid": mid, "fee": fee_rate(mk),
                            "liq": round(fnum(mk.get("liquidityNum")) or 0)})
        if len(buckets) < 3:
            continue
        out.append({"slug": e["slug"], "title": e["title"], "city": city,
                    "kind": "high" if m.group(1) == "Highest" else "low",
                    "date": date.isoformat(), "lead": (date - local.date()).days, "station": CITY[city][0],
                    "unit": buckets[0]["unit"], "vol24": round(fnum(e.get("volume24hr")) or 0),
                    "endDate": e.get("endDate"), "buckets": buckets})
    return out


# ============================================================== weather data
def station_coords(ids):
    look = sorted(i for i in ids if i not in FIXED_COORDS)
    d = get_json("https://aviationweather.gov/api/data/stationinfo", {"ids": ",".join(look), "format": "json"},
                 key="stations_" + "_".join(look)[:60])
    out = {s["icaoId"]: (s["lat"], s["lon"]) for s in d}
    out.update({k: v for k, v in FIXED_COORDS.items() if k in ids})
    return out


def open_meteo(coords, past_days=0, forecast_days=4):
    """{station: {date: {'high': [C per model], 'low': [...]}}} in one batched request."""
    ids = sorted(coords)
    d = get_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": ",".join(f"{coords[i][0]:.4f}" for i in ids),
        "longitude": ",".join(f"{coords[i][1]:.4f}" for i in ids),
        "daily": "temperature_2m_max,temperature_2m_min", "models": ",".join(MODELS),
        "timezone": "auto", "past_days": past_days, "forecast_days": forecast_days,
    }, key=f"om_{len(ids)}_{past_days}_{forecast_days}_{datetime.utcnow():%Y%m%d%H}")
    d = d if isinstance(d, list) else [d]
    out = {}
    for sid, loc in zip(ids, d):
        daily, per = loc["daily"], {}
        for k, dt in enumerate(daily["time"]):
            hi = [daily.get(f"temperature_2m_max_{m}", [None] * 99)[k] for m in MODELS]
            lo = [daily.get(f"temperature_2m_min_{m}", [None] * 99)[k] for m in MODELS]
            per[dt] = {"high": [x for x in hi if x is not None], "low": [x for x in lo if x is not None]}
        out[sid] = per
    return out


def met_no(coords):
    """Fallback when Open-Meteo is rate limited: MET Norway hourly -> local daily max/min (one model)."""
    out = {}
    for sid, (lat, lon) in coords.items():
        try:
            d = get_json("https://api.met.no/weatherapi/locationforecast/2.0/compact",
                         {"lat": f"{lat:.3f}", "lon": f"{lon:.3f}"}, key=f"metno_{sid}")
        except RuntimeError:
            continue
        tz, per = ZoneInfo(TZ_OF[sid]), {}
        for ts in d["properties"]["timeseries"]:
            t = datetime.fromisoformat(ts["time"].replace("Z", "+00:00")).astimezone(tz)
            per.setdefault(t.date().isoformat(), []).append(ts["data"]["instant"]["details"]["air_temperature"])
        out[sid] = {k: {"high": [max(v)], "low": [min(v)]} for k, v in per.items() if len(v) >= 18}
        time.sleep(0.25)
    return out


def iem_daily_obs(stations, days, now):
    """Observed daily max/min (deg F) per station per local date from IEM's METAR archive."""
    ids = [s for s in stations if s != "HKO"]
    to_iem = {(s[1:] if s.startswith("K") else s): s for s in ids}
    start = (now - timedelta(days=days + 1)).date()
    params = [("station", k) for k in to_iem] + [
        ("data", "tmpf"), ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", now.year), ("month2", now.month), ("day2", now.day), ("tz", "Etc/UTC"),
        ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"), ("report_type", "3"), ("report_type", "4")]
    txt = get_json("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params,
                   key=f"iem_{days}_{now:%Y%m%d}", text=True)
    obs = {}
    for line in txt.splitlines()[1:]:
        p = line.split(",")
        if len(p) < 3 or p[2] == "M" or p[0] not in to_iem:
            continue
        sid = to_iem[p[0]]
        t = datetime.strptime(p[1], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).astimezone(ZoneInfo(TZ_OF[sid]))
        day = obs.setdefault(sid, {}).setdefault(t.date().isoformat(), [])
        day.append(float(p[2]))
    return {s: {d: {"high": max(v), "low": min(v), "n": len(v)} for d, v in dd.items() if len(v) >= 16}
            for s, dd in obs.items()}


def refresh_calibration(coords, now, log):
    """Per station and kind: bias (obs - model mean) and residual sd, deg F, over the last 21 days."""
    obs = iem_daily_obs(list(coords), 21, now)
    past = open_meteo(coords, past_days=21, forecast_days=1)
    calib = {}
    for sid in coords:
        today = now.astimezone(ZoneInfo(TZ_OF[sid])).date().isoformat()
        for kind in ("high", "low"):
            errs = []
            for d, o in obs.get(sid, {}).items():
                if d >= today:
                    continue
                vals = past.get(sid, {}).get(d, {}).get(kind)
                if vals:
                    errs.append(o[kind] - (sum(vals) / len(vals) * 9 / 5 + 32))
            if len(errs) >= 7:
                mean = sum(errs) / len(errs)
                sd = (sum((e - mean) ** 2 for e in errs) / (len(errs) - 1)) ** 0.5
                calib.setdefault(sid, {})[kind] = {"biasF": round(mean * len(errs) / (len(errs) + 5), 2),
                                                   "residF": round(max(0.8, sd), 2), "n": len(errs)}
    log(f"calibration refreshed for {len(calib)} stations")
    return {"at": now.isoformat(), "stations": calib}


def to_unit(f, unit):
    return (f - 32) * 5 / 9 if unit == "C" else f


def bucket_probs(ev, fc, cal, k=1.0):
    vals_c = fc.get(ev["date"], {}).get(ev["kind"], [])
    if not vals_c:
        return None
    vals_f = [v * 9 / 5 + 32 for v in vals_c]
    mu_f = sum(vals_f) / len(vals_f)
    spread_f = (sum((v - mu_f) ** 2 for v in vals_f) / len(vals_f)) ** 0.5 if len(vals_f) > 1 else 0.0
    c = cal.get(ev["station"], {}).get(ev["kind"])
    bias_f = c["biasF"] if c else 0.0
    resid_f = c["residF"] if c else DEFAULT_RESID_F
    growth = LEAD_GROWTH_F.get(ev["lead"], 3.2)
    sigma_f = k * math.sqrt(resid_f ** 2 + growth ** 2 + (0.5 * spread_f) ** 2)
    if len(vals_f) == 1:
        sigma_f *= 1.2
    unit = ev["unit"]
    mu = to_unit(mu_f + bias_f, unit)
    sigma = sigma_f * (5 / 9 if unit == "C" else 1)
    ps = []
    for b in ev["buckets"]:
        lo = -1e9 if b["lo"] is None else b["lo"] - 0.5
        hi = 1e9 if b["hi"] is None else b["hi"] + 0.5
        ps.append(phi((hi - mu) / sigma) - phi((lo - mu) / sigma))
    tot = sum(ps) or 1
    return {"mu": mu, "rawMu": to_unit(mu_f, unit), "sigma": sigma, "spread": spread_f * (5 / 9 if unit == "C" else 1),
            "n": len(vals_f), "calibrated": bool(c), "p": [x / tot for x in ps]}


# ============================================================== smart money
def smart_wallets(log):
    """Weather traders profitable this month AND all time, minus likely market makers."""
    def board(period):
        return get_json(f"{DATA_API}/v1/leaderboard", {"category": "WEATHER", "timePeriod": period,
                        "orderBy": "PNL", "limit": 50}, key=f"lb_{period}_{datetime.utcnow():%Y%m%d}")
    try:
        month = {x["proxyWallet"]: x for x in board("MONTH")}
        allt = {x["proxyWallet"]: x for x in board("ALL")}
    except RuntimeError as err:
        log(f"leaderboard failed: {err}")
        return []
    out = []
    for w, x in month.items():
        a = allt.get(w)
        if not a or a["pnl"] <= 0 or x["pnl"] <= 0:
            continue
        if a["vol"] and a["pnl"] / a["vol"] < 0.01:
            continue
        out.append({"wallet": w, "name": (x.get("userName") or w[:10])[:24],
                    "pnlMonth": round(x["pnl"]), "pnlAll": round(a["pnl"]), "volAll": round(a["vol"])})
    return out


def smart_positions(wallets, cids):
    res = {}
    for w in wallets:
        try:
            pos = get_json(f"{DATA_API}/positions", {"user": w["wallet"], "limit": 500, "sizeThreshold": 1},
                           key=f"pos_{w['wallet']}_{datetime.utcnow():%Y%m%d%H}")
        except RuntimeError:
            continue
        for p in pos:
            cid = p.get("conditionId")
            if cid not in cids or (p.get("currentValue") or 0) < 5:
                continue
            side = "yes" if str(p.get("outcome", "")).lower() == "yes" else "no"
            r = res.setdefault(cid, {"yes": [], "no": [], "yesUsd": 0, "noUsd": 0})
            r[side].append(w["name"])
            r[side + "Usd"] += round(p.get("currentValue") or 0)
        time.sleep(0.1)
    return res


# ============================================================== ideas
def best_side(p, bid, ask, rate):
    opts = []
    if ask is not None and MIN_PRICE <= ask <= MAX_PRICE:
        opts.append(("YES", p, ask))
    if bid is not None and MIN_PRICE <= 1 - bid <= MAX_PRICE:
        opts.append(("NO", 1 - p, 1 - bid))
    best = None
    for side, pw, price in opts:
        cost = price + fee_per_share(rate, price)
        edge = pw - cost
        if best is None or edge > best["edge"]:
            best = {"side": side, "pWin": pw, "price": price, "cost": cost, "edge": edge,
                    "kelly": max(0.0, (pw - cost) / (1 - cost))}
    return best


def confidence(edge, spread, lead, agree, against, n_models, calibrated, unit):
    s = 42 + min(edge, 0.25) * 110
    s -= max(0.0, spread * (9 / 5 if unit == "C" else 1) - 1.5) * 5
    s -= (lead - 1) * 8
    s += 5 * min(agree, 3) - 8 * min(against, 3)
    s -= 0 if calibrated else 10
    s -= 0 if n_models >= 3 else 15
    return int(max(5, min(95, round(s))))


def run_weather(now, state, tune_k, log):
    events = [e for e in parse_weather_events(fetch_events("weather"), now) if 1 <= e["lead"] <= 3]
    log(f"weather events, 1-3 days out: {len(events)}")
    stations = sorted({e["station"] for e in events})
    coords = station_coords(stations)
    calib = state["calib"]
    if not calib.get("at") or now - datetime.fromisoformat(calib["at"]) > timedelta(hours=20):
        try:
            calib = refresh_calibration(coords, now, log)
            state["calib"] = calib
        except RuntimeError as err:
            log(f"calibration refresh failed, keeping previous: {err}")
    cal = calib.get("stations", {})
    try:
        fc, source = open_meteo(coords), "Open-Meteo: ECMWF, GFS, ICON, GEM, JMA, UKMO"
    except RuntimeError as err:
        log(f"Open-Meteo failed ({err}); using MET Norway")
        fc, source = met_no(coords), "MET Norway (fallback, single model)"
    wallets = smart_wallets(log)
    smart = smart_positions(wallets, {b["cid"] for e in events for b in e["buckets"]})
    log(f"smart wallets: {len(wallets)}; weather markets they hold: {len(smart)}")

    ideas, snaps, preds = [], [], []
    for ev in events:
        model = bucket_probs(ev, fc.get(ev["station"], {}), cal, tune_k)
        if not model:
            continue
        rows, best = [], None
        for b, pm in zip(ev["buckets"], model["p"]):
            pf = min(0.98, max(0.02, MODEL_WEIGHT * pm + (1 - MODEL_WEIGHT) * b["mid"]))
            sm = smart.get(b["cid"], {})
            rows.append({"label": b["label"], "pModel": round(pm, 3), "p": round(pf, 3), "bid": b["bid"],
                         "ask": b["ask"], "mid": round(b["mid"], 3),
                         "sy": len(sm.get("yes", [])), "sn": len(sm.get("no", []))})
            preds.append({"cid": b["cid"], "pModel": round(pm, 4), "p": round(pf, 4), "mid": round(b["mid"], 4),
                          "lead": ev["lead"], "city": ev["city"], "kind": ev["kind"], "date": ev["date"],
                          "label": b["label"], "slug": ev["slug"], "mu": round(model["mu"], 2),
                          "sg": round(model["sigma"], 3), "lo": b["lo"], "hi": b["hi"], "u": ev["unit"],
                          "cal": model["calibrated"]})
            if b["ask"] is None or b["bid"] is None or b["ask"] - b["bid"] > MAX_SPREAD:
                continue
            side = best_side(pf, b["bid"], b["ask"], b["fee"])
            if side and (best is None or side["edge"] > best["edge"]):
                best = dict(side, b=b, pm=pm, pf=pf, sm=sm)
        u = "°" + ev["unit"]
        snaps.append({"slug": ev["slug"], "city": ev["city"], "kind": ev["kind"], "date": ev["date"],
                      "lead": ev["lead"], "unit": ev["unit"], "station": ev["station"],
                      "mu": round(model["mu"], 1), "rawMu": round(model["rawMu"], 1),
                      "sigma": round(model["sigma"], 2), "spread": round(model["spread"], 2),
                      "nModels": model["n"], "calibrated": model["calibrated"], "rows": rows})
        if not best or best["edge"] < MIN_EDGE or not model["calibrated"]:
            continue  # no ideas from stations without an observation history
        b, sm = best["b"], best["sm"]
        mine, theirs = ("yes", "no") if best["side"] == "YES" else ("no", "yes")
        agree, against = len(sm.get(mine, [])), len(sm.get(theirs, []))
        conf = confidence(best["edge"], model["spread"], ev["lead"], agree, against, model["n"],
                          model["calibrated"], ev["unit"])
        corr = model["mu"] - model["rawMu"]
        why = (f"{model['n']} weather models put the {ev['kind']} at {ev['station']} near {model['mu']:.1f}{u}"
               + (f" (after a {corr:+.1f}{u} station correction)" if abs(corr) >= 0.1 else "")
               + f", give or take {model['sigma']:.1f}{u}. That makes '{b['label']}' {best['pm']*100:.0f}% "
               f"vs the market's {b['mid']*100:.0f}%. Blended estimate: {best['pf']*100:.0f}%.")
        ideas.append({
            "id": f"{b['cid']}:{best['side']}", "cat": "weather", "cid": b["cid"], "event": ev["title"],
            "slug": ev["slug"], "city": ev["city"], "outcome": b["label"], "side": best["side"],
            "price": round(best["price"], 3), "cost": round(best["cost"], 4), "pWin": round(best["pWin"], 4),
            "pModel": round(best["pm"], 4), "mid": round(b["mid"], 4), "edge": round(best["edge"], 4),
            "kelly": round(best["kelly"], 4), "confidence": conf, "lead": ev["lead"], "date": ev["date"],
            "endDate": ev["endDate"], "liq": b["liq"], "feeRate": b["fee"],
            "smartAgree": agree, "smartAgainst": against, "smartNames": sm.get(mine, [])[:4],
            "why": why, "group": ev["slug"], "url": f"https://polymarket.com/event/{ev['slug']}"})
    meta = {"forecastSource": source, "smartWallets": wallets, "calibratedStations": len(cal),
            "calibAt": calib.get("at")}
    return ideas, snaps, preds, meta


def run_tech(now, ai_path, log):
    seen, cands = set(), []
    horizon = now + timedelta(days=75)
    for tag in ("ai", "tech"):
        for e in fetch_events(tag, max_pages=2):
            if e["slug"] in seen or not e.get("endDate"):
                continue
            seen.add(e["slug"])
            end = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
            if not now < end <= horizon:
                continue
            for mk in e.get("markets", []):
                if mk.get("closed") or not mk.get("acceptingOrders", True):
                    continue
                bid, ask, mid = quote(mk)
                if bid is None or ask is None or ask - bid > MAX_SPREAD or not 0.04 <= mid <= 0.96:
                    continue
                if (fnum(mk.get("liquidityNum")) or 0) < 2000:
                    continue
                cands.append({"cid": mk["conditionId"], "event": e["title"], "question": mk.get("question"),
                              "outcome": mk.get("groupItemTitle") or "Yes", "slug": e["slug"],
                              "endDate": e["endDate"], "bid": bid, "ask": ask, "mid": round(mid, 4),
                              "liq": round(fnum(mk.get("liquidityNum")) or 0),
                              "vol24": round(fnum(mk.get("volume24hr")) or 0), "fee": fee_rate(mk),
                              "url": f"https://polymarket.com/event/{e['slug']}"})
    cands.sort(key=lambda x: -x["vol24"])
    cands = cands[:40]
    blob = load(ai_path, {})
    ai = {x["cid"]: x for x in blob.get("estimates", [])}
    ideas = []
    for m in cands:
        est = ai.get(m["cid"])
        if est:
            m["aiP"] = est["p"]
        if not est:
            continue
        pf = TECH_AI_WEIGHT * est["p"] + (1 - TECH_AI_WEIGHT) * m["mid"]
        side = best_side(pf, m["bid"], m["ask"], m["fee"])
        if not side or side["edge"] < TECH_MIN_EDGE:
            continue
        conf = int(max(5, min(90, 35 + side["edge"] * 100 +
                              {"high": 15, "medium": 5, "low": -10}.get(est.get("confidence"), 0))))
        ideas.append({
            "id": f"{m['cid']}:{side['side']}", "cat": "tech", "cid": m["cid"], "event": m["event"],
            "slug": m["slug"], "outcome": m["outcome"], "side": side["side"], "price": round(side["price"], 3),
            "cost": round(side["cost"], 4), "pWin": round(side["pWin"], 4), "pModel": est["p"],
            "mid": m["mid"], "edge": round(side["edge"], 4), "kelly": round(side["kelly"], 4),
            "confidence": conf, "endDate": m["endDate"], "liq": m["liq"], "feeRate": m["fee"],
            "smartAgree": 0, "smartAgainst": 0, "smartNames": [], "why": est.get("rationale", ""),
            "sources": est.get("sources", [])[:4], "group": m["slug"], "url": m["url"]})
    log(f"tech shortlist: {len(cands)}; with AI estimates: {len(ai)}; tech ideas: {len(ideas)}")
    return ideas, cands, blob.get("generatedAt")


# ============================================================== resolution + scoring
def fetch_resolutions(now, extra_cids, log):
    res = {}

    def take(mk):
        try:
            pr = [float(x) for x in json.loads(mk.get("outcomePrices") or "[]")]
        except ValueError:
            return
        if mk.get("closed") and len(pr) == 2 and max(pr) > 0.99:
            res[mk["conditionId"]] = 1 if pr[0] > 0.99 else 0

    since = (now - timedelta(days=4)).strftime("%Y-%m-%dT00:00:00Z")
    try:
        for e in fetch_events("weather", closed=True, end_min=since, max_pages=20):
            for mk in e.get("markets", []):
                take(mk)
    except RuntimeError as err:
        log(f"weather resolutions failed: {err}")
    extra = [c for c in extra_cids if c not in res]
    for i in range(0, len(extra), 40):
        chunk = extra[i:i + 40]
        try:
            d = get_json(f"{GAMMA}/markets", [("condition_ids", c) for c in chunk] + [("closed", "true")],
                         key=f"mk_{chunk[0][:10]}_{now:%Y%m%d%H}")
            for mk in d:
                take(mk)
        except RuntimeError as err:
            log(f"market lookup failed: {err}")
    return res


def brier(rows, key):
    return sum((r[key] - r["y"]) ** 2 for r in rows) / len(rows) if rows else None


PEND_KEYS = ("slug", "city", "kind", "date", "label", "lead", "pModel", "p", "mid", "mu", "sg", "lo", "hi", "u", "cal")


def load_history(data_dir):
    rows = []
    hist = os.path.join(data_dir, "history")
    if os.path.isdir(hist):
        for fn in sorted(os.listdir(hist)):
            with open(os.path.join(hist, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def tune_sigma(rows):
    """Pick the sigma multiplier that best explains which bucket actually won (max log-likelihood).
    Needs 60+ resolved events from calibrated stations; otherwise keeps 1.0."""
    wins = [r for r in rows if r.get("y") == 1 and r.get("sg") and r.get("cal")]
    if len(wins) < 60:
        return {"k": 1.0, "events": len(wins), "note": "starting value; tunes itself after 60 resolved events"}

    def ll(k):
        tot = 0.0
        for r in wins:
            s = r["sg"] * k
            lo = -1e9 if r["lo"] is None else r["lo"] - 0.5
            hi = 1e9 if r["hi"] is None else r["hi"] + 0.5
            p = phi((hi - r["mu"]) / s) - phi((lo - r["mu"]) / s)
            tot += math.log(max(p, 1e-4))
        return tot
    grid = [round(0.5 + 0.05 * i, 2) for i in range(23)]
    k = max(grid, key=ll)
    return {"k": k, "events": len(wins), "note": "tuned on resolved events"}


def compute_stats(data_dir, now):
    rows = load_history(data_dir)
    events = {r["slug"] for r in rows}
    out = {"at": now.isoformat(), "resolvedMarkets": len(rows), "resolvedEvents": len(events)}

    def block(rs):
        if not rs:
            return None
        bm, bf, bk = brier(rs, "pModel"), brier(rs, "p"), brier(rs, "mid")
        return {"n": len(rs), "events": len({r["slug"] for r in rs}), "brierModel": round(bm, 4),
                "brierBlend": round(bf, 4), "brierMarket": round(bk, 4),
                "skillVsMarket": round(1 - bf / bk, 4) if bk else None}

    out["all"] = block(rows)
    cutoff = (now - timedelta(days=14)).date().isoformat()
    out["last14d"] = block([r for r in rows if r["date"] >= cutoff])
    out["byLead"] = {str(l): block([r for r in rows if r["lead"] == l]) for l in (1, 2, 3)}
    # reliability: does "30%" happen 30% of the time?
    bins = []
    for lo in [i / 10 for i in range(10)]:
        rs = [r for r in rows if lo <= r["p"] < lo + 0.1 or (lo == 0.9 and r["p"] == 1)]
        if rs:
            bins.append({"lo": lo, "n": len(rs), "avgP": round(sum(r["p"] for r in rs) / len(rs), 3),
                         "hit": round(sum(r["y"] for r in rs) / len(rs), 3)})
    out["reliability"] = bins
    by_city = {}
    for r in rows:
        by_city.setdefault(r["city"], []).append(r)
    out["byCity"] = sorted(({"city": c, **block(rs)} for c, rs in by_city.items() if len(rs) >= 22),
                           key=lambda x: x["skillVsMarket"] or 0, reverse=True)
    ev_count = len(events)
    gate_skill = out["all"]["skillVsMarket"] if out["all"] else None
    out["gate"] = {"eventsNeeded": 100, "events": ev_count,
                   "beatsMarket": bool(gate_skill is not None and gate_skill > 0.02),
                   "note": "Go-live needs 100+ resolved events, blended Brier at least 2% better than the market, "
                           "positive paper P&L and drawdown under 20%."}
    return out


# ============================================================== paper account
def et_day(ts):
    return datetime.fromisoformat(ts).astimezone(ET).date().isoformat()


def run_paper(paper, ideas, res, now):
    if not paper:
        paper = {"start": PAPER["start"], "cash": PAPER["start"], "open": [], "closed": [], "equity": [],
                 "rules": PAPER, "halted": False}
    # settle
    still = []
    for pos in paper["open"]:
        y = res.get(pos["cid"])
        if y is None:
            still.append(pos)
            continue
        won = (y == 1) == (pos["side"] == "YES")
        payout = pos["shares"] if won else 0.0
        paper["cash"] = round(paper["cash"] + payout, 4)
        paper["closed"].append(dict(pos, closedAt=now.isoformat(), won=won,
                                    pnl=round(payout - pos["stake"], 4)))
    paper["open"] = still
    open_cost = sum(p["stake"] for p in paper["open"])
    equity = paper["cash"] + open_cost          # open positions at cost
    peak = max([e["equity"] for e in paper["equity"]] + [paper["start"], equity])
    today = now.astimezone(ET).date().isoformat()
    lost_today = -sum(c["pnl"] for c in paper["closed"] if et_day(c["closedAt"]) == today and c["pnl"] < 0)
    paper["halted"] = equity <= PAPER["floor"]
    blocked = None
    if paper["halted"]:
        blocked = f"Paper account at ${equity:.2f}, at or below the ${PAPER['floor']:.0f} floor. Review before continuing."
    elif lost_today >= PAPER["dailyLossStop"]:
        blocked = f"Daily loss stop hit (${lost_today:.2f} lost today). No new trades until tomorrow."
    taken = 0
    held = {p["cid"] for p in paper["open"]} | {p["group"] for p in paper["open"]}
    skipped = []
    if not blocked:
        for idea in ideas:
            if taken >= PAPER["maxNewPerRun"]:
                break
            if idea["cid"] in held or idea["group"] in held:
                continue
            k_stake = PAPER["kellyFraction"] * idea["kelly"] * equity
            if k_stake < PAPER["skipBelow"]:
                skipped.append(idea["id"])
                continue
            stake = round(min(max(k_stake, PAPER["minStake"]), max(PAPER["minStake"], PAPER["capPct"] * equity)), 2)
            if open_cost + stake > PAPER["maxOpen"] or stake > paper["cash"]:
                break
            shares = round(stake / idea["cost"], 4)
            paper["open"].append({"id": idea["id"], "cid": idea["cid"], "group": idea["group"], "cat": idea["cat"],
                                  "event": idea["event"], "outcome": idea["outcome"], "side": idea["side"],
                                  "price": idea["price"], "cost": idea["cost"], "pWin": idea["pWin"],
                                  "stake": stake, "shares": shares, "openedAt": now.isoformat(),
                                  "url": idea["url"]})
            paper["cash"] = round(paper["cash"] - stake, 4)
            open_cost += stake
            held |= {idea["cid"], idea["group"]}
            taken += 1
    equity = paper["cash"] + sum(p["stake"] for p in paper["open"])
    paper["equity"].append({"t": now.isoformat(), "equity": round(equity, 2)})
    paper["equity"] = paper["equity"][-600:]
    peak = max(peak, equity)
    realized = sum(c["pnl"] for c in paper["closed"])
    wins = sum(1 for c in paper["closed"] if c["won"])
    paper["summary"] = {"equity": round(equity, 2), "cash": round(paper["cash"], 2),
                        "openCost": round(sum(p["stake"] for p in paper["open"]), 2),
                        "realized": round(realized, 2), "trades": len(paper["closed"]), "wins": wins,
                        "drawdown": round(1 - equity / peak, 4) if peak else 0, "peak": round(peak, 2),
                        "blocked": blocked, "newThisRun": taken, "tooSmallAtThisBankroll": len(skipped),
                        "target": 40.0}
    paper["closed"] = paper["closed"][-400:]
    return paper


# ============================================================== main
def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args()
    CACHE = a.cache
    D = a.data
    os.makedirs(os.path.join(D, "history"), exist_ok=True)
    now = datetime.now(timezone.utc)
    notes = []

    def log(msg):
        notes.append(msg)
        print(msg, file=sys.stderr)

    state = {"calib": load(os.path.join(D, "calib.json"), {})}
    pending = load(os.path.join(D, "pending.json"), {})
    paper = load(os.path.join(D, "paper.json"), None)

    tune = tune_sigma(load_history(D))
    log(f"sigma multiplier: {tune['k']} ({tune['note']})")
    w_ideas, snaps, preds, meta = run_weather(now, state, tune["k"], log)
    try:
        t_ideas, tech, ai_at = run_tech(now, os.path.join(D, "ai_estimates.json"), log)
    except RuntimeError as err:
        log(f"tech scan failed: {err}")
        t_ideas, tech, ai_at = [], [], None

    # log predictions: keep the latest forecast made at least 1 day ahead (compact rows)
    stamp = now.isoformat()
    for p in preds:
        pending[p["cid"]] = [p.get(k) for k in PEND_KEYS]
    pending = {c: dict(zip(PEND_KEYS, v)) if isinstance(v, list) else v for c, v in pending.items()}

    # resolve
    need = set(pending) | {p["cid"] for p in (paper or {}).get("open", [])}
    res = fetch_resolutions(now, [c for c in need if not c.startswith("_")], log)
    month_rows = {}
    for cid in list(pending):
        if cid in res:
            p = pending.pop(cid)
            p["y"] = res[cid]
            month_rows.setdefault(p["date"][:7], []).append(p)
        elif pending[cid]["date"] < (now - timedelta(days=6)).date().isoformat():
            pending.pop(cid)   # never resolved; drop
    for month, rows in month_rows.items():
        with open(os.path.join(D, "history", f"{month}.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in ("slug", "city", "kind", "date", "label", "lead",
                                                          "pModel", "p", "mid", "y", "mu", "sg", "lo", "hi",
                                                          "u", "cal")}) + "\n")
    log(f"newly resolved predictions: {sum(len(v) for v in month_rows.values())}; still pending: {len(pending)}")

    ideas = sorted(w_ideas + t_ideas, key=lambda x: -(x["edge"] * x["confidence"]))
    paper = run_paper(paper, ideas, res, now)
    stats = compute_stats(D, now)
    stats["sigmaTune"] = tune

    feed = {"version": VERSION, "generatedAt": stamp, "forecastSource": meta["forecastSource"],
            "calibratedStations": meta["calibratedStations"], "calibAt": meta["calibAt"],
            "smartWallets": meta["smartWallets"][:20], "aiEstimatesAt": ai_at, "notes": notes,
            "params": {"modelWeight": MODEL_WEIGHT, "minEdge": MIN_EDGE, "techMinEdge": TECH_MIN_EDGE,
                       "minPrice": MIN_PRICE, "maxPrice": MAX_PRICE, "maxSpread": MAX_SPREAD, "paper": PAPER},
            "ideas": ideas[:40], "weather": sorted(snaps, key=lambda s: (s["date"], s["city"], s["kind"])),
            "tech": tech}
    save(os.path.join(D, "feed.json"), feed)
    save(os.path.join(D, "paper.json"), paper)
    save(os.path.join(D, "stats.json"), stats)
    save(os.path.join(D, "calib.json"), state["calib"], pretty=True)
    save(os.path.join(D, "pending.json"), {c: [p.get(k) for k in PEND_KEYS] for c, p in pending.items()})
    print(json.dumps({"ideas": len(ideas), "weatherEvents": len(snaps), "tech": len(tech),
                      "paperEquity": paper["summary"]["equity"], "resolvedTotal": stats["resolvedMarkets"]}))


if __name__ == "__main__":
    main()
