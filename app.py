import sys
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from data_fetcher import fetch_all, load_all, ALL_SYMBOLS
from indicators import compute_all, get_indicator_columns, get_indicator_label, INDICATOR_META
from analysis import run_analysis, run_temporal_stability, OP_LABELS, OPS

# ── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QUANT·ANALYZER",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
:root{
  --bg:#111318;--bg2:#191d26;--border:#272d3e;
  --text:#cdd6e0;--sub:#7a8899;--dim:#505b6a;
  --blue:#5a8ec2;--cyan:#3d9db8;--amber:#a88d44;
  --red:#b06868;--green:#3d9270;--purple:#7b72a8
}
.stApp{background:var(--bg)!important}
header[data-testid="stHeader"]{background:var(--bg)!important;border-bottom:1px solid var(--border)!important}
section[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important}
section[data-testid="stSidebar"] *{color:var(--sub)!important}
section[data-testid="stSidebar"] strong{color:var(--text)!important}
*,.stMarkdown,button,label,p,span,div{font-family:'JetBrains Mono',monospace!important}
.stMarkdown p{color:var(--text)!important}
.stMarkdown li{color:var(--sub)!important}
div[data-testid="stTabs"] button{color:var(--sub)!important;font-size:.86rem!important;letter-spacing:.07em!important;text-transform:uppercase!important;padding:8px 20px!important;border-bottom:2px solid transparent!important}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--blue)!important;border-bottom-color:var(--blue)!important;font-weight:700!important}
div[data-testid="stTabs"]>div>div{border-bottom:1px solid var(--border)!important}
button[data-testid="stBaseButton-primary"]{background:var(--blue)!important;color:#f0f4f8!important;border:none!important;font-weight:700!important;letter-spacing:.06em!important}
[data-testid="stMetricValue"]{font-size:1.25rem!important;color:var(--text)!important}
[data-testid="stMetricLabel"]{font-size:.72rem!important;color:var(--sub)!important;letter-spacing:.06em!important}
[data-testid="stMetricDelta"]{font-size:.82rem!important}
hr{border-color:var(--border)!important;margin:20px 0!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px!important}
.rc{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:18px 14px;text-align:center}
.rv{font-size:1.9rem;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.rl{font-size:.68rem;color:var(--sub);margin-top:6px;text-transform:uppercase;letter-spacing:.1em}
.rp{color:var(--green)}.rn{color:var(--red)}.ru{color:var(--amber)}.rb{color:var(--blue)}
.warnbox{background:#1e1808;border-left:3px solid var(--amber);padding:10px 14px;border-radius:4px;font-size:.8rem;color:var(--amber);margin:10px 0}
.ptitle{font-family:'Syne',monospace!important;font-size:1.5rem;font-weight:800;color:var(--text);letter-spacing:.14em;text-transform:uppercase}
.psub{font-size:.76rem;color:var(--sub);letter-spacing:.06em;text-transform:uppercase;margin-top:3px}
#MainMenu,footer{visibility:hidden}
"""

# Inject CSS via script into parent document (works reliably in Streamlit 1.35+)
_css_escaped = _CSS.replace("`", "\\`").replace("\\", "\\\\")
components.html(f"""
<script>
(function(){{
  if(document.getElementById('qa-custom-css'))return;
  const s=document.createElement('style');
  s.id='qa-custom-css';
  s.textContent=`{_css_escaped}`;
  window.parent.document.head.appendChild(s);
}})();
</script>
""", height=0)


# ── 데이터 ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def get_data():
    raw = load_all()
    if not raw:
        raw = fetch_all(verbose=False)
    return {sym: compute_all(df) for sym, df in raw.items() if not df.empty}

data = get_data()
available_symbols = [s for s in ALL_SYMBOLS if s in data]
indicator_cols = get_indicator_columns()
ind_labels = {col: get_indicator_label(col) for col in indicator_cols}

def ind_label(col): return ind_labels.get(col, col)


# ── 쿼리 HTML 빌더 ────────────────────────────────────────────────────────────

def _select(el_id, css_cls, items, selected, label_fn=None, onchange=""):
    opts = ""
    sel_lbl = ""
    for v in items:
        lbl = label_fn(v) if label_fn else str(v)
        is_sel = str(v) == str(selected)
        if is_sel:
            sel_lbl = lbl
        sc = " selected" if is_sel else ""
        opts += f'<div class="csel-opt{sc}" data-value="{v}">{lbl}</div>'
    if not sel_lbl and items:
        sel_lbl = label_fn(items[0]) if label_fn else str(items[0])
    oc = f' data-onchange="{onchange}"' if onchange else ""
    return (
        f'<div class="token {css_cls} csel" id="{el_id}" '
        f'data-value="{selected}" tabindex="0"{oc}>'
        f'<span class="csel-val">{sel_lbl}</span>'
        f'<div class="csel-list">{opts}</div>'
        f'</div>'
    )

def _cond_html(i, sym, ind, op, val, vmin, vmax, symbols, inds, is_last=False):
    bullets = ["①", "②", "③", "④"]
    sym_sel = _select(f"q_sym{i}", "sym", symbols, sym)
    ind_sel = _select(f"q_ind{i}", "ind", inds, ind, lambda c: ind_labels.get(c, c))
    _op_vals = ["<", "<=", ">=", ">", "between"]
    _op_lbls = {"<":"미만","<=":"이하",">=":"이상",">":"초과","between":"사이"}
    op_sel = _select(f"q_op{i}", "op", _op_vals, op, lambda k: _op_lbls[k], onchange=f"onOpChange({i})")
    nd = "none" if op == "between" else "inline-flex"
    bd = "inline-flex" if op == "between" else "none"
    iltae_vis = "inline" if is_last else "none"
    return (
        f'<div class="qrow" id="cond_{i}">'
        f'<span class="conn bullet">{bullets[i]}</span>'
        f'{sym_sel}<span class="conn">의</span>'
        f'{ind_sel}<span class="conn">이(가)</span>'
        f'<span id="q_normal_{i}" style="display:{nd};align-items:center;gap:4px">'
        f'<input type="number" class="token val" id="q_val{i}" value="{val}" step="0.5"></span>'
        f'<span id="q_between_{i}" style="display:{bd};align-items:center;gap:4px">'
        f'<input type="number" class="token val" id="q_vmin{i}" value="{vmin}" step="0.5">'
        f'<span class="conn">~</span>'
        f'<input type="number" class="token val" id="q_vmax{i}" value="{vmax}" step="0.5"></span>'
        f'{op_sel}'
        f'<span class="conn" id="iltae_{i}" style="display:{iltae_vis}">일 때,</span>'
        f'</div>'
    )

def _logic_row_html(i, logic, hidden=False):
    disp = "display:none" if hidden else "margin-bottom:4px"
    return (
        f'<div class="qrow logicrow" id="logic_row_{i}" style="{disp}">'
        f'<span class="conn" style="margin-left:22px;font-size:0.72rem;opacity:0.5">결합:</span>'
        f'<button class="logic-btn" onclick="toggleLogic()">{logic}</button></div>'
    )

def generate_query_html(start, end, n_cond, logic, cond_params,
                        tgt_sym, tgt_ind, fwd, result_type, threshold,
                        symbols, inds):
    # Build condition rows (always 4, hidden if i >= n_cond)
    conds_html = ""
    for i in range(4):
        c = cond_params[i]
        vis = "" if i < n_cond else ' style="display:none"'
        is_last = (i == n_cond - 1)
        row = _cond_html(i, c["sym"], c["ind"], c["op"],
                         c["val"], c["vmin"], c["vmax"], symbols, inds, is_last=is_last)
        if vis:
            row = row.replace(f'<div class="qrow" id="cond_{i}">', f'<div class="qrow" id="cond_{i}"{vis}>')
        conds_html += row
        if i < 3:
            hidden = i >= n_cond - 1
            conds_html += _logic_row_html(i, logic, hidden=hidden)

    rem_vis = "" if n_cond > 1 else ' style="display:none"'
    add_vis = "" if n_cond < 4 else ' style="display:none"'

    sym_sel  = _select("q_tsym", "sym", symbols, tgt_sym)
    ind_sel  = _select("q_tind", "ind", inds, tgt_ind, lambda c: ind_labels.get(c, c))
    fwd_map  = {1:"1거래일",3:"3거래일",5:"5거래일",10:"10거래일",20:"20거래일",60:"60거래일"}
    fwd_sel  = _select("q_fwd", "fwd", [1,3,5,10,20,60], fwd, lambda v: fwd_map[v])
    _rt_vals = ["mean", "above", "below"]
    _rt_lbls = {"mean":"평균 / 분포","above":"이상일 확률","below":"이하일 확률"}
    rt_sel   = _select("q_rtype", "res", _rt_vals, result_type, lambda k: _rt_lbls[k], onchange="onRtypeChange()")

    thr_vis  = "inline-flex" if result_type != "mean" else "none"
    conn_fwd = "후 값의" if result_type == "mean" else "후 값이"

    syms_j   = json.dumps(symbols)
    inds_j   = json.dumps(inds)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'JetBrains Mono',monospace;background:#191d26;color:#cdd6e0;
     padding:20px 24px 72px 24px;font-size:1.05rem;line-height:1.75}}
.qhdr{{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.14em;color:#505b6a;
       margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.qhdr::after{{content:'';flex:1;height:1px;background:#272d3e}}
.qrow{{display:flex;align-items:center;flex-wrap:wrap;gap:7px;
      margin-bottom:11px;min-height:38px}}
.conn{{color:#cdd6e0;white-space:nowrap;padding:0 2px;font-size:1.0rem}}
.bullet{{color:#6a7888;min-width:22px;font-size:0.95rem}}
.token{{appearance:none;-webkit-appearance:none;border:none;outline:none;
        cursor:pointer;font-family:inherit;font-weight:600;font-size:0.97rem;
        border-radius:3px;padding:5px 11px;transition:all 0.15s}}
.csel{{position:relative;user-select:none;cursor:pointer}}
.csel-val{{display:block;pointer-events:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.csel-list{{display:none;position:fixed;
            background:#141820;border:1px solid #2d3448;border-radius:5px;
            z-index:9999;min-width:max-content;max-height:220px;
            overflow-y:auto;overflow-x:hidden;
            box-shadow:0 6px 24px rgba(0,0,0,.75);
            scrollbar-width:thin;scrollbar-color:#2d3448 transparent}}
.csel-list::-webkit-scrollbar{{width:3px}}
.csel-list::-webkit-scrollbar-track{{background:transparent}}
.csel-list::-webkit-scrollbar-thumb{{background:#2d3448;border-radius:2px}}
.csel.open .csel-list{{display:block}}
.csel-opt{{padding:7px 16px;color:#cdd6e0;font-family:inherit;font-size:0.93rem;
           cursor:pointer;white-space:nowrap}}
.csel-opt:hover{{background:#1e2840;color:#e8f0f8}}
.csel-opt.selected{{color:#e0ecfc;font-weight:700;background:#1a2436}}
.sym{{background:rgba(90,142,194,.10);border:1px solid rgba(90,142,194,.40);
      color:#7aaad4;min-width:88px}}
.sym:hover,.sym:focus{{background:rgba(90,142,194,.18);border-color:#7aaad4}}
.ind{{background:rgba(61,157,184,.08);border:1px solid rgba(61,157,184,.35);
      color:#4db8d2;min-width:120px}}
.ind:hover,.ind:focus{{background:rgba(61,157,184,.15);border-color:#4db8d2}}
.op{{background:rgba(176,104,104,.09);border:1px solid rgba(176,104,104,.38);
     color:#c88080;min-width:58px}}
.op:hover,.op:focus{{background:rgba(176,104,104,.16);border-color:#c88080}}
.val{{background:rgba(168,141,68,.09);border:1px solid rgba(168,141,68,.38);
      color:#c8a848;width:78px;text-align:center}}
.val:hover,.val:focus{{background:rgba(168,141,68,.16);border-color:#c8a848}}
.fwd{{background:rgba(123,114,168,.09);border:1px solid rgba(123,114,168,.35);
      color:#9e96c8;min-width:88px}}
.fwd:hover,.fwd:focus{{background:rgba(123,114,168,.18);border-color:#9e96c8}}
.res{{background:rgba(61,146,112,.08);border:1px solid rgba(61,146,112,.32);
      color:#52b888;min-width:122px}}
.res:hover,.res:focus{{background:rgba(61,146,112,.15);border-color:#52b888}}
.date{{background:rgba(205,214,224,.06);border:1px solid rgba(205,214,224,.18);
       color:#9aabb8;min-width:126px;color-scheme:dark}}
.logic-btn{{background:rgba(90,142,194,.12);border:1px solid rgba(90,142,194,.45);
            color:#7aaad4;font-family:inherit;font-weight:700;font-size:0.82rem;
            letter-spacing:.1em;padding:3px 12px;border-radius:3px;cursor:pointer;
            transition:all .12s}}
.logic-btn:hover{{background:rgba(90,142,194,.22)}}
.ctrl-btn{{background:transparent;border:1px dashed #2a3248;color:#6a7888;
           font-family:inherit;font-size:0.84rem;padding:3px 11px;border-radius:3px;
           cursor:pointer;transition:all .12s}}
.ctrl-btn:hover{{border-color:#5a8ec2;color:#7aaad4}}
.qdiv{{color:#7a8899;font-size:0.92rem;letter-spacing:.04em;
       margin:6px 0 10px 0;display:flex;align-items:center;gap:8px}}
.qdiv::before{{content:'';width:20px;height:1px;background:#272d3e}}
.qdiv::after{{content:'';flex:1;height:1px;background:#272d3e}}
#thr_sec{{align-items:center;gap:6px}}
.run-btn{{position:fixed;bottom:0;left:0;right:0;background:#2c6090;color:#e8f0f8;
          border:none;border-radius:0;padding:13px 0;width:100%;font-family:inherit;
          font-size:0.92rem;font-weight:700;letter-spacing:.14em;cursor:pointer;
          text-transform:uppercase;transition:background .18s;z-index:100}}
.run-btn:hover{{background:#3a72a8}}
.run-btn:active{{background:#254f78}}
</style>
</head>
<body>
<div class="qhdr">쿼리 설정</div>
<div class="qrow">
  <span class="conn">📅</span>
  <input type="date" class="token date" id="q_start" value="{start}">
  <span class="conn">~</span>
  <input type="date" class="token date" id="q_end" value="{end}">
  <span class="conn">기간 데이터에서,</span>
</div>
<div id="conds">{conds_html}</div>
<div class="qrow" style="margin-bottom:6px">
  <span class="conn" style="margin-left:22px;font-size:0.72rem">결합:</span>
  <button class="logic-btn" id="lbmain" onclick="toggleLogic()">{logic}</button>
  <button class="ctrl-btn" id="add_btn" onclick="addCond()"{add_vis}>+ 추가</button>
  <button class="ctrl-btn" id="rem_btn" onclick="remCond()"{rem_vis}>− 제거</button>
</div>
<div class="qrow">
  <span class="conn">▶</span>
  {sym_sel}<span class="conn">의</span>
  {ind_sel}<span class="conn">의</span>
  {fwd_sel}<span class="conn" id="conn_after_fwd">{conn_fwd}</span>
  <span id="thr_sec" style="display:{thr_vis};align-items:center;gap:4px">
    <input type="number" class="token val" id="q_thr" value="{threshold}" step="0.5">
  </span>
  {rt_sel}
  <span class="conn">?</span>
</div>
<button class="run-btn" onclick="runAnalysis()">▶ &nbsp; 분석 실행</button>
<script>
const SYMS={syms_j};
const INDS={inds_j};
let nCond={n_cond};
let logic='{logic}';
function getV(id){{
  const el=document.getElementById(id);
  if(!el)return'';
  return el.dataset.value!==undefined?el.dataset.value:(el.value||'');
}}
function initSelects(){{
  document.querySelectorAll('.csel').forEach(function(sel){{
    sel.addEventListener('click',function(e){{
      e.stopPropagation();
      const wasOpen=sel.classList.contains('open');
      document.querySelectorAll('.csel.open').forEach(function(s){{s.classList.remove('open');}});
      if(!wasOpen){{
        const rect=sel.getBoundingClientRect();
        const list=sel.querySelector('.csel-list');
        list.style.left=rect.left+'px';
        list.style.minWidth=rect.width+'px';
        const spaceBelow=window.innerHeight-rect.bottom-8;
        const spaceAbove=rect.top-8;
        const listH=220;
        if(spaceBelow>=listH||spaceBelow>=spaceAbove){{
          list.style.top=(rect.bottom+4)+'px';
          list.style.maxHeight=Math.min(listH,Math.max(spaceBelow,60))+'px';
        }}else{{
          const actualH=Math.min(listH,spaceAbove);
          list.style.top=(rect.top-4-actualH)+'px';
          list.style.maxHeight=actualH+'px';
        }}
        sel.classList.add('open');
      }}
    }});
    sel.querySelectorAll('.csel-opt').forEach(function(opt){{
      opt.addEventListener('click',function(e){{
        e.stopPropagation();
        sel.querySelectorAll('.csel-opt').forEach(function(o){{o.classList.remove('selected');}});
        opt.classList.add('selected');
        sel.dataset.value=opt.dataset.value;
        sel.querySelector('.csel-val').textContent=opt.textContent;
        sel.classList.remove('open');
        const oc=sel.dataset.onchange;
        if(oc){{new Function(oc)();}}
      }});
    }});
  }});
  document.addEventListener('click',function(){{
    document.querySelectorAll('.csel.open').forEach(function(s){{s.classList.remove('open');}});
  }});
}}
document.addEventListener('DOMContentLoaded',initSelects);
function toggleLogic(){{
  logic=logic==='AND'?'OR':'AND';
  document.querySelectorAll('.logic-btn').forEach(b=>b.textContent=logic);
}}
function onOpChange(i){{
  const op=getV('q_op'+i);
  document.getElementById('q_normal_'+i).style.display=op==='between'?'none':'inline-flex';
  document.getElementById('q_between_'+i).style.display=op==='between'?'inline-flex':'none';
}}
function onRtypeChange(){{
  const rt=getV('q_rtype');
  const sec=document.getElementById('thr_sec');
  const conn=document.getElementById('conn_after_fwd');
  sec.style.display=rt!=='mean'?'inline-flex':'none';
  if(conn)conn.textContent=rt==='mean'?'후 값의':'후 값이';
}}
function addCond(){{
  if(nCond>=4)return;
  document.getElementById('iltae_'+(nCond-1)).style.display='none';
  document.getElementById('cond_'+nCond).style.display='flex';
  document.getElementById('logic_row_'+(nCond-1)).style.display='flex';
  nCond++;
  document.getElementById('iltae_'+(nCond-1)).style.display='inline';
  document.getElementById('rem_btn').style.display='';
  if(nCond>=4)document.getElementById('add_btn').style.display='none';
}}
function remCond(){{
  if(nCond<=1)return;
  document.getElementById('iltae_'+(nCond-1)).style.display='none';
  nCond--;
  document.getElementById('cond_'+nCond).style.display='none';
  document.getElementById('logic_row_'+(nCond-1)).style.display='none';
  document.getElementById('iltae_'+(nCond-1)).style.display='inline';
  if(nCond<=1)document.getElementById('rem_btn').style.display='none';
  document.getElementById('add_btn').style.display='';
}}
function runAnalysis(){{
  const p=new URLSearchParams();
  p.set('q_start',document.getElementById('q_start').value);
  p.set('q_end',document.getElementById('q_end').value);
  p.set('q_ncond',nCond);
  p.set('q_logic',logic);
  for(let i=0;i<nCond;i++){{
    const op=getV('q_op'+i);
    p.set('q_sym'+i,getV('q_sym'+i));
    p.set('q_ind'+i,getV('q_ind'+i));
    p.set('q_op'+i,op);
    if(op==='between'){{
      p.set('q_vmin'+i,document.getElementById('q_vmin'+i).value);
      p.set('q_vmax'+i,document.getElementById('q_vmax'+i).value);
    }}else{{
      p.set('q_val'+i,document.getElementById('q_val'+i).value);
    }}
  }}
  p.set('q_tsym',getV('q_tsym'));
  p.set('q_tind',getV('q_tind'));
  p.set('q_fwd',getV('q_fwd'));
  p.set('q_rtype',getV('q_rtype'));
  p.set('q_thr',document.getElementById('q_thr').value);
  p.set('q_run','1');
  const qs='?'+p.toString();
  // Sandbox has allow-same-origin so we can inject a script into parent document.
  // This bypasses the missing allow-top-navigation restriction.
  try{{
    const sc=window.parent.document.createElement('script');
    sc.textContent='window.location.search='+JSON.stringify(qs)+';';
    window.parent.document.head.appendChild(sc);
    sc.remove();
  }}catch(e){{
    // Last resort: try direct assignment
    try{{window.parent.location.search=qs;}}catch(e2){{}}
  }}
}}
</script>
</body>
</html>"""


# ── 쿼리 파라미터 파싱 ────────────────────────────────────────────────────────

def _gp(key, default=""):
    v = st.query_params.get(key, default)
    return v if v != "" else default

def _gp_float(key, default=0.0):
    try:    return float(_gp(key, str(default)))
    except: return default

def _gp_int(key, default=1):
    try:    return int(_gp(key, str(default)))
    except: return default

def _safe_sym(s):
    return s if s in data else (available_symbols[0] if available_symbols else "")

def _safe_ind(s):
    return s if s in indicator_cols else (indicator_cols[0] if indicator_cols else "")

q_start     = _gp("q_start", "2010-01-01")
q_end       = _gp("q_end",   date.today().strftime("%Y-%m-%d"))
n_cond      = max(1, min(4, _gp_int("q_ncond", 1)))
logic       = _gp("q_logic", "AND")
if logic not in ("AND", "OR"): logic = "AND"

cond_params = []
conditions_input = []
for _i in range(4):
    _sym  = _safe_sym(_gp(f"q_sym{_i}", available_symbols[0] if available_symbols else ""))
    _ind  = _safe_ind(_gp(f"q_ind{_i}", indicator_cols[0] if indicator_cols else ""))
    _op   = _gp(f"q_op{_i}", "<")
    if _op not in OPS: _op = "<"
    _val  = _gp_float(f"q_val{_i}",  -2.0)
    _vmin = _gp_float(f"q_vmin{_i}", -5.0)
    _vmax = _gp_float(f"q_vmax{_i}",  0.0)
    cond_params.append({"sym": _sym, "ind": _ind, "op": _op,
                        "val": _val, "vmin": _vmin, "vmax": _vmax})
    if _i < n_cond:
        _value = [_vmin, _vmax] if _op == "between" else _val
        if _sym in data and _ind in data[_sym].columns:
                conditions_input.append({"symbol": _sym, "indicator": _ind,
                                         "op": _op, "value": _value})

tgt_sym     = _safe_sym(_gp("q_tsym", available_symbols[0] if available_symbols else ""))
tgt_ind     = _safe_ind(_gp("q_tind", indicator_cols[0] if indicator_cols else ""))
fwd         = _gp_int("q_fwd", 1)
if fwd not in [1, 3, 5, 10, 20, 60]: fwd = 1
result_type = _gp("q_rtype", "mean")
if result_type not in ("mean", "above", "below"): result_type = "mean"
threshold   = _gp_float("q_thr", 0.0)
should_run  = _gp("q_run", "0") == "1"

fwd_map = {1:"1거래일",3:"3거래일",5:"5거래일",10:"10거래일",20:"20거래일",60:"60거래일"}


# ── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="ptitle">QUANT</div><div class="psub">조건부 확률 분석기</div>', unsafe_allow_html=True)
    st.divider()
    if st.button("↺ 데이터 업데이트", use_container_width=True):
        with st.spinner("수집 중..."):
            fetch_all(verbose=False)
            st.cache_data.clear()
        st.success("완료!")
        st.rerun()
    st.divider()
    st.caption("데이터 현황")
    for sym in available_symbols:
        df = data[sym]
        last = df.index[-1].strftime("%m-%d") if not df.empty else "N/A"
        st.caption(f"**{sym}**: {len(df):,}일 · {last}")


# ── 헤더 ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="ptitle">QUANT·ANALYZER</div>'
    '<div class="psub">크로스에셋 조건부 확률 분석 — 클릭해서 쿼리를 조작하세요</div>',
    unsafe_allow_html=True
)

tab_main, tab_market, tab_corr = st.tabs(["◈ 조건부 분석", "▲ 시장 현황", "⌘ 상관관계"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 조건부 분석
# ══════════════════════════════════════════════════════════════════════════════

with tab_main:
    if not available_symbols:
        st.error("데이터 없음. 사이드바에서 '데이터 업데이트'를 누르세요.")
        st.stop()

    # 쿼리 폼 높이 (버튼은 fixed이므로 폼 영역만, 4조건+3로직행의 2/3)
    comp_height = 360
    html_src = generate_query_html(
        q_start, q_end, n_cond, logic, cond_params,
        tgt_sym, tgt_ind, fwd, result_type, threshold,
        available_symbols, indicator_cols,
    )
    components.html(html_src, height=comp_height, scrolling=True)

    # ── 결과 ─────────────────────────────────────────────────────────────────
    if should_run and conditions_input and tgt_sym in data and tgt_ind in data.get(tgt_sym, pd.DataFrame()).columns:

        # 날짜 슬라이스
        sliced = {}
        for sym, df in data.items():
            mask = (df.index >= pd.Timestamp(q_start)) & (df.index <= pd.Timestamp(q_end))
            sliced[sym] = df.loc[mask]

        result = run_analysis(
            data=sliced, conditions=conditions_input,
            logic=logic, target_symbol=tgt_sym, target_indicator=tgt_ind,
            forward_days=fwd, lookback_years=None,
        )

        if not result["success"]:
            st.error(f"분석 실패: {result['error']}")
        else:
            n    = result["n_samples"]
            dist = result["distribution"]
            mean = result["mean"]
            p_val = result["p_value"]

            if n < 30:
                st.markdown(
                    f'<div class="warnbox">⚠ 샘플 {n}개 — 통계적 신뢰도 낮음 (권장: 30개 이상)</div>',
                    unsafe_allow_html=True
                )

            # 핵심 지표 카드
            if result_type == "mean":
                pp = result["prob_positive"]
                cards = [
                    (f"{n:,}", "샘플 수", "rb"),
                    (f"{pp:.1f}%", "양수 확률", "rp" if pp >= 50 else "rn"),
                    (f"{'+'if mean>0 else ''}{mean:.3f}", "평균",
                     "rp" if mean > 0 else "rn"),
                    (f"{'+'if result['median']>0 else ''}{result['median']:.3f}", "중앙값",
                     "rp" if result["median"] > 0 else "rn"),
                    (f"{p_val:.4f}" if not pd.isna(p_val) else "—", "p-value",
                     "rp" if (not pd.isna(p_val) and p_val < 0.05) else "ru"),
                ]
            else:
                if result_type == "above":
                    prob_thr = (dist >= threshold).mean() * 100
                    lbl_thr  = f"{threshold} 이상 확률"
                else:
                    prob_thr = (dist <= threshold).mean() * 100
                    lbl_thr  = f"{threshold} 이하 확률"
                cards = [
                    (f"{n:,}", "샘플 수", "rb"),
                    (f"{prob_thr:.1f}%", lbl_thr, "rp" if prob_thr >= 50 else "rn"),
                    (f"{'+'if mean>0 else ''}{mean:.3f}", "평균", "rp" if mean > 0 else "rn"),
                    (f"{result['std']:.3f}", "표준편차", "ru"),
                    (f"{p_val:.4f}" if not pd.isna(p_val) else "—", "p-value",
                     "rp" if (not pd.isna(p_val) and p_val < 0.05) else "ru"),
                ]

            cols = st.columns(len(cards))
            for col, (val_s, lbl, cls) in zip(cols, cards):
                col.markdown(
                    f'<div class="rc"><div class="rv {cls}">{val_s}</div>'
                    f'<div class="rl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )
            st.write("")

            # 분포 시각화
            col_h, col_b = st.columns([3, 1])
            with col_h:
                fig = go.Figure()
                if result_type != "mean" and threshold is not None:
                    colors = (
                        ["#3d9270" if v >= threshold else "#b06868" for v in dist]
                        if result_type == "above"
                        else ["#3d9270" if v <= threshold else "#b06868" for v in dist]
                    )
                    fig.add_trace(go.Histogram(x=dist, nbinsx=40, marker_color=colors))
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#7b72a8",
                                  line_width=2,
                                  annotation_text=f"임계값 {threshold}",
                                  annotation_font_color="#9e96c8")
                else:
                    colors = ["#3d9270" if v > 0 else "#b06868" for v in dist]
                    fig.add_trace(go.Histogram(x=dist, nbinsx=40, marker_color=colors))
                    fig.add_vline(x=0, line_dash="dash", line_color="#272d3e", line_width=1)

                fig.add_vline(x=float(mean), line_dash="dot", line_color="#a88d44",
                              line_width=2,
                              annotation_text=f"평균 {mean:.3f}",
                              annotation_font_color="#c8a848")
                fig.update_layout(
                    title=f"{fwd_map[fwd]} 후  {ind_label(tgt_ind)}  분포",
                    template="plotly_dark", height=320,
                    paper_bgcolor="#191d26", plot_bgcolor="#141820",
                    font_family="JetBrains Mono",
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = go.Figure(go.Box(y=dist, boxmean="sd",
                                        marker_color="#5a8ec2", name=""))
                fig2.add_hline(y=0, line_dash="dash", line_color="#272d3e", line_width=1)
                if result_type != "mean":
                    fig2.add_hline(y=threshold, line_dash="dash",
                                   line_color="#7b72a8", line_width=1.5)
                fig2.update_layout(
                    title="박스플롯", template="plotly_dark", height=320,
                    paper_bgcolor="#191d26", plot_bgcolor="#141820",
                    font_family="JetBrains Mono",
                    showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # 분위수 요약
            ca, cb, cc, cd = st.columns(4)
            ca.metric("25분위수", f"{result['q25']:.3f}")
            cb.metric("75분위수", f"{result['q75']:.3f}")
            cc.metric("표준편차",  f"{result['std']:.3f}")
            cd.metric("t-stat",
                      f"{result['t_stat']:.3f}" if not pd.isna(result["t_stat"]) else "—")

            # 시간 안정성
            st.divider()
            st.markdown("**시간 안정성**")
            st.caption("같은 조건이 기간별로 얼마나 일관되게 작동했는지 확인합니다.")
            stability = run_temporal_stability(
                data, conditions_input, logic, tgt_sym, tgt_ind, fwd, windows=[3, 5, 10]
            )
            st.dataframe(
                stability.style
                    .format({"평균": "{:.3f}", "양수확률(%)": "{:.1f}", "p-value": "{:.4f}"})
                    .background_gradient(subset=["양수확률(%)"], cmap="RdYlGn", vmin=30, vmax=70),
                use_container_width=True,
            )

            # 조건 발생 타임라인
            st.divider()
            st.markdown("**조건 발생 시점**")
            cond_dates = result["condition_dates"]
            if not cond_dates.empty and tgt_sym in sliced and not sliced[tgt_sym].empty:
                price = sliced[tgt_sym]["Close"].dropna()
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=price.index, y=price.values, mode="lines",
                    name="종가", line=dict(color="#5a8ec2", width=1),
                ))
                mp = price.reindex(cond_dates, method="nearest")
                fig3.add_trace(go.Scatter(
                    x=mp.index, y=mp.values, mode="markers",
                    name="조건 발생",
                    marker=dict(color="#a88d44", size=6, symbol="triangle-up"),
                ))
                fig3.update_layout(
                    title=f"{tgt_sym} — 조건 발생 시점",
                    template="plotly_dark", height=260,
                    paper_bgcolor="#191d26", plot_bgcolor="#141820",
                    font_family="JetBrains Mono",
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig3, use_container_width=True)

    elif should_run:
        st.warning("조건 또는 대상 데이터를 확인하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 시장 현황
# ══════════════════════════════════════════════════════════════════════════════

with tab_market:
    st.markdown("**시장 현황 스냅샷**")

    snap_rows = []
    for sym in available_symbols:
        df = data[sym]
        if df.empty or len(df) < 2: continue
        last = df.iloc[-1]
        row = {"종목": sym}
        for col in ["Close", "pct_change_1d", "rsi_14", "zscore_20d",
                    "realized_vol_20d", "ma_dev_20d", "atr_14_pct"]:
            try:    row[col] = round(float(last[col]), 3) if col in last.index and not pd.isna(last[col]) else None
            except: row[col] = None
        snap_rows.append(row)

    if snap_rows:
        snap = pd.DataFrame(snap_rows).set_index("종목").rename(columns={
            "Close": "현재가", "pct_change_1d": "전일대비(%)",
            "rsi_14": "RSI(14)", "zscore_20d": "Z-Score(20일)",
            "realized_vol_20d": "실현변동성(%)", "ma_dev_20d": "20일MA 괴리율(%)",
            "atr_14_pct": "ATR(%)",
        })
        def _color(v):
            try: return "color:#3d9270" if float(v) > 0 else ("color:#b06868" if float(v) < 0 else "")
            except: return ""
        st.dataframe(
            snap.style.applymap(_color, subset=["전일대비(%)"]).format("{:.2f}", na_rep="—"),
            use_container_width=True, height=300,
        )

    st.divider()
    c1, c2 = st.columns(2)
    sym_sel  = c1.selectbox("종목", available_symbols, key="chart_sym")
    period_s = c2.selectbox("기간", ["1년", "3년", "5년", "전체"], index=1)

    n_days = {"1년": 252, "3년": 756, "5년": 1260, "전체": None}[period_s]
    if sym_sel in data:
        cdf = data[sym_sel].copy()
        if n_days: cdf = cdf.iloc[-n_days:]

        fig_c = go.Figure(go.Candlestick(
            x=cdf.index, open=cdf["Open"], high=cdf["High"],
            low=cdf["Low"], close=cdf["Close"],
            increasing_line_color="#3d9270", decreasing_line_color="#b06868",
        ))
        for w, col in [(20, "#a88d44"), (60, "#7b72a8")]:
            fig_c.add_trace(go.Scatter(
                x=cdf.index, y=cdf["Close"].rolling(w).mean(),
                mode="lines", name=f"MA{w}", line=dict(color=col, width=1)
            ))
        fig_c.update_layout(
            title=sym_sel, template="plotly_dark", height=380,
            paper_bgcolor="#191d26", plot_bgcolor="#141820",
            font_family="JetBrains Mono",
            xaxis_rangeslider_visible=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        if "rsi_14" in cdf.columns:
            fig_r = go.Figure(go.Scatter(
                x=cdf.index, y=cdf["rsi_14"],
                mode="lines", line=dict(color="#5a8ec2", width=1.5)
            ))
            fig_r.add_hline(y=70, line_dash="dash", line_color="#b06868",
                            annotation_text="과매수(70)")
            fig_r.add_hline(y=30, line_dash="dash", line_color="#3d9270",
                            annotation_text="과매도(30)")
            fig_r.update_layout(
                title="RSI(14)", template="plotly_dark", height=190,
                paper_bgcolor="#191d26", plot_bgcolor="#141820",
                font_family="JetBrains Mono",
                yaxis=dict(range=[0, 100]),
                margin=dict(t=35, b=15, l=20, r=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 크로스에셋 상관관계
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    st.markdown("**크로스에셋 상관관계**")

    c1, c2 = st.columns(2)
    corr_period = c1.selectbox("기간", ["1년", "3년", "5년", "전체"], index=1, key="cp")
    corr_ind    = c2.selectbox("지표", indicator_cols, format_func=ind_label, key="ci")

    n_corr = {"1년": 252, "3년": 756, "5년": 1260, "전체": None}[corr_period]
    ret_d = {}
    for sym in available_symbols:
        df = data[sym]
        if corr_ind in df.columns:
            s = df[corr_ind].dropna()
            if n_corr: s = s.iloc[-n_corr:]
            ret_d[sym] = s

    if len(ret_d) >= 2:
        corr_df  = pd.DataFrame(ret_d).dropna(how="all")
        corr_mat = corr_df.corr()
        fig_hm = px.imshow(
            corr_mat, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"상관관계 히트맵 — {ind_label(corr_ind)} ({corr_period})",
            template="plotly_dark", aspect="auto", height=460,
        )
        fig_hm.update_layout(
            paper_bgcolor="#191d26", font_family="JetBrains Mono",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()
    st.markdown("**조건부 상관관계**")
    st.caption("특정 시장 조건 하에서 두 종목 간 상관관계 변화를 확인합니다.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc_s1  = cc1.selectbox("종목 A", available_symbols, key="cc1")
    cc_s2  = cc2.selectbox("종목 B", available_symbols,
                           index=min(1, len(available_symbols)-1), key="cc2")
    cc_c   = cc3.selectbox("조건 종목", available_symbols, key="cc3")
    cc_thr = cc4.number_input("전일대비(%) 임계값", value=-2.0, step=0.5)

    if all(s in data for s in [cc_s1, cc_s2, cc_c]):
        s1 = data[cc_s1]["pct_change_1d"].dropna()
        s2 = data[cc_s2]["pct_change_1d"].dropna()
        sc = data[cc_c]["pct_change_1d"].dropna()
        idx = s1.index.intersection(s2.index).intersection(sc.index)
        s1, s2, sc = s1.reindex(idx), s2.reindex(idx), sc.reindex(idx)

        m_lo, m_hi = sc < cc_thr, sc >= cc_thr
        c_all = s1.corr(s2)
        c_lo  = s1[m_lo].corr(s2[m_lo])
        c_hi  = s1[m_hi].corr(s2[m_hi])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("전체 상관관계", f"{c_all:.3f}")
        col_b.metric(f"{cc_c} < {cc_thr}% 시",
                     f"{c_lo:.3f}" if not pd.isna(c_lo) else "—",
                     delta=f"{c_lo-c_all:.3f}" if not pd.isna(c_lo) else None)
        col_c.metric(f"{cc_c} ≥ {cc_thr}% 시",
                     f"{c_hi:.3f}" if not pd.isna(c_hi) else "—",
                     delta=f"{c_hi-c_all:.3f}" if not pd.isna(c_hi) else None)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=s1[m_hi], y=s2[m_hi], mode="markers",
            marker=dict(color="#5a8ec2", opacity=0.4, size=4),
            name=f"{cc_c} ≥ {cc_thr}%",
        ))
        fig_sc.add_trace(go.Scatter(
            x=s1[m_lo], y=s2[m_lo], mode="markers",
            marker=dict(color="#b06868", opacity=0.6, size=5),
            name=f"{cc_c} < {cc_thr}%",
        ))
        fig_sc.update_layout(
            title=f"{cc_s1}  vs  {cc_s2}",
            xaxis_title=f"{cc_s1} 전일대비(%)",
            yaxis_title=f"{cc_s2} 전일대비(%)",
            template="plotly_dark", height=360,
            paper_bgcolor="#191d26", plot_bgcolor="#141820",
            font_family="JetBrains Mono",
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()
st.caption("데이터: Yahoo Finance · FinanceDataReader(KRX) · Binance  |  본 대시보드는 투자 조언이 아닌 데이터 분석 도구입니다.")
