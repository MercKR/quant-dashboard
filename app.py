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

from data_fetcher import (
    fetch_all, load_all, load_fund_all, ALL_SYMBOLS, ASSET_SYMBOLS, MACRO_SYMBOLS,
    load_user_stocks, save_user_stocks,
    fetch_user_stock, fetch_stock_fundamentals,
)
from indicators import (
    compute_all, get_indicator_columns, get_all_indicator_columns,
    get_indicator_label, INDICATOR_META, FUND_INDICATOR_COLS,
)
from analysis import run_analysis, run_temporal_stability, OP_LABELS, OPS
from backtest import run_backtest

# ── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QUANT·ANALYZER",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS = """
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
*:not(.material-symbols-rounded):not(.material-symbols-outlined):not(.material-symbols-sharp):not(.material-icons):not([data-testid="stIconMaterial"]),
.stMarkdown,button,label,p,
span:not(.material-symbols-rounded):not(.material-symbols-outlined):not(.material-symbols-sharp):not(.material-icons):not([data-testid="stIconMaterial"]),
div{font-family:Arial,'Malgun Gothic',sans-serif!important}
.material-symbols-rounded,.material-symbols-outlined,.material-symbols-sharp,.material-icons,[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded','Material Icons'!important;font-feature-settings:'liga';-webkit-font-feature-settings:'liga'}
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
.ptitle{font-family:Arial,'Malgun Gothic',sans-serif!important;font-size:1.5rem;font-weight:800;color:var(--text);letter-spacing:.14em;text-transform:uppercase}
.psub{font-size:.76rem;color:var(--sub);letter-spacing:.06em;text-transform:uppercase;margin-top:3px}
#MainMenu,footer{visibility:hidden}
"""

# Inject CSS + Material Symbols font into parent document
_css_escaped = _CSS.replace("`", "\\`").replace("\\", "\\\\")
components.html(f"""
<script>
(function(){{
  if(document.getElementById('qa-custom-css'))return;
  // Material Symbols 폰트 명시적 로드
  if(!document.getElementById('qa-material-font')){{
    const lk=window.parent.document.createElement('link');
    lk.id='qa-material-font';
    lk.rel='stylesheet';
    lk.href='https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200';
    window.parent.document.head.appendChild(lk);
  }}
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
    fund = load_fund_all()
    return {sym: compute_all(df, fund.get(sym)) for sym, df in raw.items() if not df.empty}

data = get_data()

# 기본 심볼 + 사용자 추가 종목 순서
_user_stock_names = [s["name"] for s in load_user_stocks()]
available_symbols = (
    [s for s in ALL_SYMBOLS if s in data] +
    [s for s in _user_stock_names if s in data and s not in ALL_SYMBOLS]
)

# 지표 목록: 기본 + 펀더멘털 (펀더멘털은 개별종목에만 존재)
indicator_cols = get_all_indicator_columns()
ind_labels = {col: get_indicator_label(col) for col in indicator_cols}

def ind_label(col): return ind_labels.get(col, col)

# 종목 타입별 지표 세트 — 개별종목은 펀더멘털 포함, 시장/매크로는 기술 지표만
_FUND_SET = set(FUND_INDICATOR_COLS)
_BASE_INDS = get_indicator_columns()

def _is_stock(symbol):
    return symbol in _user_stock_names

def _indicators_for(symbol):
    """선택 종목 타입에 맞는 지표 목록. 개별종목이면 펀더멘털 지표 포함."""
    if _is_stock(symbol):
        return list(_BASE_INDS) + list(FUND_INDICATOR_COLS)
    return list(_BASE_INDS)


# ── 쿼리 HTML 빌더 ────────────────────────────────────────────────────────────

def _select(el_id, css_cls, items, selected, label_fn=None, onchange="", mark_fund=False):
    opts = ""
    sel_lbl = ""
    for v in items:
        lbl = label_fn(v) if label_fn else str(v)
        is_sel = str(v) == str(selected)
        if is_sel:
            sel_lbl = lbl
        cls = "csel-opt"
        if is_sel:
            cls += " selected"
        if mark_fund and v in _FUND_SET:
            cls += " fund-opt"
        opts += f'<div class="{cls}" data-value="{v}">{lbl}</div>'
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
    sym_sel = _select(f"q_sym{i}", "sym", symbols, sym,
                      onchange=f"onSymChange('q_sym{i}','q_ind{i}')")
    ind_sel = _select(f"q_ind{i}", "ind", inds, ind, lambda c: ind_labels.get(c, c),
                      mark_fund=True)
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

    sym_sel  = _select("q_tsym", "sym", symbols, tgt_sym,
                       onchange="onSymChange('q_tsym','q_tind')")
    ind_sel  = _select("q_tind", "ind", inds, tgt_ind, lambda c: ind_labels.get(c, c),
                       mark_fund=True)
    fwd_map  = {1:"1거래일",3:"3거래일",5:"5거래일",10:"10거래일",20:"20거래일",60:"60거래일"}
    fwd_sel  = _select("q_fwd", "fwd", [1,3,5,10,20,60], fwd, lambda v: fwd_map[v])
    _rt_vals = ["mean", "above", "below"]
    _rt_lbls = {"mean":"평균 / 분포","above":"이상일 확률","below":"이하일 확률"}
    rt_sel   = _select("q_rtype", "res", _rt_vals, result_type, lambda k: _rt_lbls[k], onchange="onRtypeChange()")

    thr_vis  = "inline-flex" if result_type != "mean" else "none"
    conn_fwd = "후 값의" if result_type == "mean" else "후 값이"

    syms_j       = json.dumps(symbols)
    inds_j       = json.dumps(inds)
    stock_syms_j = json.dumps(_user_stock_names)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,'Malgun Gothic',sans-serif;background:#191d26;color:#cdd6e0;
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
.csel.hide-fund .csel-opt.fund-opt{{display:none}}
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
const STOCK_SYMS={stock_syms_j};
let nCond={n_cond};
let logic='{logic}';
function getV(id){{
  const el=document.getElementById(id);
  if(!el)return'';
  return el.dataset.value!==undefined?el.dataset.value:(el.value||'');
}}
function applyIndSet(symId,indId){{
  // 개별종목이면 펀더멘털 지표 노출, 시장/매크로 시리즈면 기술 지표만
  const isStock=STOCK_SYMS.indexOf(getV(symId))>=0;
  const indSel=document.getElementById(indId);
  if(!indSel)return;
  if(isStock){{indSel.classList.remove('hide-fund');return;}}
  indSel.classList.add('hide-fund');
  const cur=indSel.querySelector('.csel-opt.selected');
  if(cur&&cur.classList.contains('fund-opt')){{
    const first=indSel.querySelector('.csel-opt:not(.fund-opt)');
    if(first){{
      indSel.querySelectorAll('.csel-opt').forEach(function(o){{o.classList.remove('selected');}});
      first.classList.add('selected');
      indSel.dataset.value=first.dataset.value;
      indSel.querySelector('.csel-val').textContent=first.textContent;
    }}
  }}
}}
function onSymChange(symId,indId){{applyIndSet(symId,indId);}}
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
  for(let i=0;i<4;i++){{applyIndSet('q_sym'+i,'q_ind'+i);}}
  applyIndSet('q_tsym','q_tind');
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


# ── 자동완성 컴포넌트 ────────────────────────────────────────────────────────

_AC_TEMPLATE_PATH = Path(__file__).parent / "src" / "ac_template.html"


@st.cache_resource
def _get_ac_html() -> str:
    """ac_template.html 반환 — DB는 /app/static/ticker_db.json fetch로 로드."""
    return _AC_TEMPLATE_PATH.read_text(encoding="utf-8")


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
    st.caption("**자산**")
    for sym in available_symbols:
        if sym in MACRO_SYMBOLS:
            continue
        df = data[sym]
        last = df.index[-1].strftime("%m-%d") if not df.empty else "N/A"
        has_fund = any(c in df.columns for c in FUND_INDICATOR_COLS)
        fund_tag = " ·F" if has_fund else ""
        st.caption(f"**{sym}**: {len(df):,}일 · {last}{fund_tag}")

    _macro_now = [s for s in available_symbols if s in MACRO_SYMBOLS]
    if _macro_now:
        st.caption("**매크로 지표**")
        for sym in _macro_now:
            df = data[sym]
            last = df.index[-1].strftime("%m-%d") if not df.empty else "N/A"
            st.caption(f"**{sym}**: {len(df):,}일 · {last}")

    # ── 개별종목 관리 ──────────────────────────────────────────────────────
    st.divider()
    st.caption("**개별종목 추가**")

    # 자동완성 컴포넌트 (입력창만 46px iframe으로, 드롭다운은 parent DOM에 주입)
    components.html(_get_ac_html(), height=46)

    # URL 파라미터에서 선택 결과 읽기
    _ac_t = st.query_params.get("ac_t", "")
    _ac_n = st.query_params.get("ac_n", "")
    _ac_x = st.query_params.get("ac_x", "")

    if _ac_t and _ac_n:
        st.caption(f"**{_ac_n}**  `{_ac_t}`  [{_ac_x}]")
        if st.button("＋ 추가", use_container_width=True, key="stock_add_btn", type="primary"):
            _user_stocks = load_user_stocks()
            if any(s["ticker"] == _ac_t for s in _user_stocks):
                st.warning("이미 추가된 종목입니다.")
            else:
                with st.spinner(f"{_ac_n} 수집 중..."):
                    _price, _fund = fetch_user_stock(_ac_n, _ac_t, verbose=False)
                if _price.empty:
                    st.error(f"'{_ac_t}' 데이터를 가져올 수 없습니다.")
                else:
                    _user_stocks.append({"name": _ac_n, "ticker": _ac_t})
                    save_user_stocks(_user_stocks)
                    for _k in ("ac_t", "ac_n", "ac_x"):
                        if _k in st.query_params:
                            del st.query_params[_k]
                    st.cache_data.clear()
                    st.success(f"{_ac_n} 추가 완료!")
                    st.rerun()

    # 추가된 종목 목록 + 삭제
    _user_stocks_now = load_user_stocks()
    if _user_stocks_now:
        st.caption("**추가된 종목**")
        for _s in _user_stocks_now:
            _c1, _c2 = st.columns([3, 1])
            _c1.caption(f"{_s['name']}  ({_s['ticker']})")
            if _c2.button("✕", key=f"del_{_s['ticker']}"):
                _updated = [x for x in _user_stocks_now if x["ticker"] != _s["ticker"]]
                save_user_stocks(_updated)
                st.cache_data.clear()
                st.rerun()

    # 펀더멘털 재수집 버튼
    if _user_stocks_now:
        if st.button("↺ 펀더멘털 재수집", use_container_width=True,
                     help="분기 재무 데이터를 최신화합니다."):
            with st.spinner("펀더멘털 수집 중..."):
                for _s in _user_stocks_now:
                    fetch_stock_fundamentals(_s["name"], _s["ticker"])
                st.cache_data.clear()
            st.success("완료!")
            st.rerun()


# ── 모의 투자 헬퍼 ────────────────────────────────────────────────────────────

_BT_OPS = ["<", "<=", ">", ">="]


def _bt_condition_rows(prefix: str, count: int) -> list[dict]:
    """백테스트 조건 입력 행을 그린다. 지표 목록은 선택 종목 타입에 맞춰 분기한다."""
    conds = []
    for j in range(count):
        vis = "visible" if j == 0 else "collapsed"
        c1, c2, c3, c4 = st.columns([2.4, 3, 1.7, 1.6])
        sym = c1.selectbox("종목", available_symbols,
                           key=f"{prefix}_sym{j}", label_visibility=vis)
        inds = _indicators_for(sym)
        ind_key = f"{prefix}_ind{j}"
        if ind_key in st.session_state and st.session_state[ind_key] not in inds:
            del st.session_state[ind_key]
        ind = c2.selectbox("지표", inds, format_func=ind_label,
                           key=ind_key, label_visibility=vis)
        op  = c3.selectbox("부등호", _BT_OPS, format_func=lambda o: OP_LABELS[o],
                           key=f"{prefix}_op{j}", label_visibility=vis)
        val = c4.number_input("값", value=-2.0, step=0.5,
                              key=f"{prefix}_val{j}", label_visibility=vis)
        conds.append({"symbol": sym, "indicator": ind, "op": op, "value": float(val)})
    return conds


def _render_backtest(res: dict):
    """백테스트 결과(성과지표·자산곡선·거래로그)를 렌더링한다."""
    m = res["metrics"]
    equity, benchmark, trades = res["equity"], res["benchmark"], res["trades"]

    def _pct(v):
        return f"{'+' if v > 0 else ''}{v:.2f}%" if pd.notna(v) else "—"

    cards = [
        (_pct(m["total_return"]), "총수익률",
         "rp" if m["total_return"] > 0 else "rn"),
        (_pct(m["cagr"]), "CAGR",
         "rp" if (pd.notna(m["cagr"]) and m["cagr"] > 0) else "rn"),
        (f"{m['mdd']:.2f}%", "최대낙폭(MDD)", "rn"),
        (f"{m['win_rate']:.1f}%" if pd.notna(m["win_rate"]) else "—", "승률",
         "rp" if (pd.notna(m["win_rate"]) and m["win_rate"] >= 50) else "ru"),
        (f"{m['n_trades']:,}", "거래 횟수", "rb"),
        (f"{m['sharpe']:.2f}" if pd.notna(m["sharpe"]) else "—", "Sharpe",
         "rp" if (pd.notna(m["sharpe"]) and m["sharpe"] > 1) else "ru"),
    ]
    cols = st.columns(len(cards))
    for col, (val_s, lbl, cls) in zip(cols, cards):
        col.markdown(
            f'<div class="rc"><div class="rv {cls}">{val_s}</div>'
            f'<div class="rl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    eb1, eb2, eb3 = st.columns(3)
    eb1.metric("전략 총수익률", _pct(m["total_return"]))
    eb2.metric("Buy&Hold 총수익률", _pct(m["bench_total"]))
    eb3.metric("초과수익", _pct(m["excess_return"]),
               delta=f"{m['excess_return']:.2f}%p")

    # 자산곡선 — 누적수익률(%)로 표시해 특정 일자 수익률을 바로 읽을 수 있게
    eq_ret = (equity / float(equity.iloc[0]) - 1) * 100
    bench_ret = (benchmark / float(benchmark.iloc[0]) - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_ret.index, y=eq_ret.values, mode="lines",
        name="전략", line=dict(color="#5a8ec2", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=bench_ret.index, y=bench_ret.values, mode="lines",
        name="Buy&Hold", line=dict(color="#7a8899", width=1, dash="dot"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#272d3e", line_width=1)
    fig.update_layout(
        title="자산곡선 — 누적수익률 (전략 vs Buy&Hold)",
        template="plotly_dark", height=360,
        paper_bgcolor="#191d26", plot_bgcolor="#141820",
        font_family="Arial, Malgun Gothic, sans-serif",
        yaxis_title="누적수익률(%)",
        margin=dict(t=46, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="top", y=0.99,
                    xanchor="right", x=0.99,
                    bgcolor="rgba(20,24,32,.7)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**거래 로그**")
    if trades.empty:
        st.info("조건을 만족하는 거래가 없었습니다. 매수 조건을 완화해 보세요.")
    else:
        disp = trades.copy()
        disp["진입일"] = disp["진입일"].dt.strftime("%Y-%m-%d")
        disp["청산일"] = disp["청산일"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            disp.style
                .format({"진입가": "{:.2f}", "청산가": "{:.2f}",
                         "수익률(%)": "{:+.2f}"})
                .map(lambda v: "color:#3d9270" if v > 0 else "color:#b06868",
                     subset=["수익률(%)"]),
            use_container_width=True,
            height=min(400, 60 + len(disp) * 36),
        )


# ── 헤더 ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="ptitle">QUANT·ANALYZER</div>'
    '<div class="psub">크로스에셋 조건부 확률 분석 — 클릭해서 쿼리를 조작하세요</div>',
    unsafe_allow_html=True
)

tab_main, tab_sim, tab_market, tab_corr = st.tabs(
    ["◈ 데이터 분석", "◇ 모의 투자", "▲ 시장 현황", "⌘ 상관관계"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 데이터 분석
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
                    font_family="Arial, Malgun Gothic, sans-serif",
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
                    font_family="Arial, Malgun Gothic, sans-serif",
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
                    font_family="Arial, Malgun Gothic, sans-serif",
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig3, use_container_width=True)

    elif should_run:
        st.warning("조건 또는 대상 데이터를 확인하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 모의 투자 (백테스트)
# ══════════════════════════════════════════════════════════════════════════════

with tab_sim:
    if not available_symbols:
        st.error("데이터 없음. 사이드바에서 '데이터 업데이트'를 누르세요.")
    else:
        st.markdown(
            "**매수 신호가 발생하면 진입하고, 청산 규칙에 따라 매도하는 전략을 "
            "과거 데이터로 시뮬레이션합니다.** "
            "데이터 분석 탭에서 검증한 조건을 그대로 매수 신호로 쓸 수 있습니다."
        )

        # ── 매매 대상 · 기간 ──────────────────────────────────────────────────
        st.divider()
        st.caption("**매매 대상**")
        bt_c1, bt_c2, bt_c3 = st.columns([2, 1.5, 1.5])
        bt_symbol = bt_c1.selectbox("종목", available_symbols, key="bt_symbol")
        bt_start  = bt_c2.date_input("시작일", value=date(2015, 1, 1), key="bt_start")
        bt_end    = bt_c3.date_input("종료일", value=date.today(), key="bt_end")

        # ── 매수 조건 ─────────────────────────────────────────────────────────
        st.divider()
        st.caption("**매수 조건** — 충족 시 진입")
        bc1, bc2 = st.columns([1, 4])
        bt_buy_n = bc1.number_input("조건 수", 1, 3, 1, key="bt_buy_n")
        bt_buy_logic = bc2.radio(
            "결합", ["AND", "OR"], horizontal=True, key="bt_buy_logic",
            help="AND = 모든 조건 충족,  OR = 하나라도 충족",
        )
        bt_buy_conds = _bt_condition_rows("bt_buy", int(bt_buy_n))

        # ── 청산 규칙 ─────────────────────────────────────────────────────────
        st.divider()
        st.caption("**청산 규칙** — 가장 먼저 도달하는 조건으로 매도")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            bt_use_hold = st.checkbox("보유기간", value=True, key="bt_use_hold")
            bt_hold = st.number_input("거래일 후 매도", 1, 250, 20, key="bt_hold",
                                      disabled=not bt_use_hold)
        with ex2:
            bt_use_tp = st.checkbox("익절(%)", value=False, key="bt_use_tp")
            bt_tp = st.number_input("목표수익 도달 시", 0.5, 500.0, 10.0, step=0.5,
                                    key="bt_tp", disabled=not bt_use_tp)
        with ex3:
            bt_use_sl = st.checkbox("손절(%)", value=False, key="bt_use_sl")
            bt_sl = st.number_input("최대손실 도달 시", -500.0, -0.5, -5.0, step=0.5,
                                    key="bt_sl", disabled=not bt_use_sl)

        tr1, tr2, _tr3 = st.columns(3)
        with tr1:
            bt_use_tatr = st.checkbox("ATR 트레일링", value=False, key="bt_use_tatr",
                                      help="고점 − N×ATR(14) 하회 시 매도 (Chandelier Exit)")
            bt_tatr = st.number_input("ATR 배수", 0.5, 10.0, 3.0, step=0.5,
                                      key="bt_tatr", disabled=not bt_use_tatr)
        with tr2:
            bt_use_tpct = st.checkbox("비율 트레일링(%)", value=False, key="bt_use_tpct",
                                      help="진입 후 최고 종가 대비 X% 하락 시 매도")
            bt_tpct = st.number_input("고점대비 하락", 0.5, 90.0, 10.0, step=0.5,
                                      key="bt_tpct", disabled=not bt_use_tpct)

        bt_use_sellcond = st.checkbox("매도 조건 사용", value=False,
                                      key="bt_use_sellcond")
        if bt_use_sellcond:
            sc1, sc2 = st.columns([1, 4])
            bt_sell_n = sc1.number_input("조건 수", 1, 3, 1, key="bt_sell_n")
            bt_sell_logic = sc2.radio("결합", ["AND", "OR"], horizontal=True,
                                      key="bt_sell_logic")
            bt_sell_conds = _bt_condition_rows("bt_sell", int(bt_sell_n))
        else:
            bt_sell_conds, bt_sell_logic = None, "AND"

        # ── 거래 설정 ─────────────────────────────────────────────────────────
        st.divider()
        st.caption("**거래 설정**")
        tc1, tc2, tc3, tc4 = st.columns(4)
        bt_capital = tc1.number_input("초기자본", 1_000_000, 100_000_000_000,
                                      10_000_000, step=1_000_000, key="bt_capital")
        bt_fill = tc2.selectbox("체결 방식", ["익일 시가", "당일 종가"], key="bt_fill",
                                help="익일 시가 = look-ahead bias 방지 (권장)")
        bt_comm = tc3.number_input("수수료(%)", 0.0, 5.0, 0.015, step=0.005,
                                   format="%.3f", key="bt_comm")
        bt_slip = tc4.number_input("슬리피지(%)", 0.0, 5.0, 0.05, step=0.01,
                                   format="%.3f", key="bt_slip")

        if st.button("▶  백테스트 실행", type="primary", use_container_width=True,
                     key="bt_run"):
            if bt_start >= bt_end:
                st.error("시작일은 종료일보다 빨라야 합니다.")
                st.session_state.pop("bt_result", None)
            elif not (bt_use_hold or bt_use_tp or bt_use_sl
                      or bt_use_tatr or bt_use_tpct or bt_use_sellcond):
                st.error("청산 규칙을 1개 이상 선택하세요.")
                st.session_state.pop("bt_result", None)
            else:
                st.session_state["bt_result"] = run_backtest(
                    data=data, trade_symbol=bt_symbol,
                    buy_conditions=bt_buy_conds, buy_logic=bt_buy_logic,
                    hold_days=int(bt_hold) if bt_use_hold else None,
                    take_profit=float(bt_tp) if bt_use_tp else None,
                    stop_loss=float(bt_sl) if bt_use_sl else None,
                    trail_atr_mult=float(bt_tatr) if bt_use_tatr else None,
                    trail_pct=float(bt_tpct) if bt_use_tpct else None,
                    sell_conditions=bt_sell_conds, sell_logic=bt_sell_logic,
                    fill="next_open" if bt_fill == "익일 시가" else "close",
                    commission=bt_comm / 100,
                    slippage=bt_slip / 100,
                    initial_capital=float(bt_capital),
                    start=bt_start, end=bt_end,
                )

        bt_result = st.session_state.get("bt_result")
        if bt_result:
            st.divider()
            if not bt_result["success"]:
                st.error(f"백테스트 실패: {bt_result['error']}")
            else:
                _render_backtest(bt_result)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 시장 현황
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
            snap.style.map(_color, subset=["전일대비(%)"]).format("{:.2f}", na_rep="—"),
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
            font_family="Arial, Malgun Gothic, sans-serif",
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
                font_family="Arial, Malgun Gothic, sans-serif",
                yaxis=dict(range=[0, 100]),
                margin=dict(t=35, b=15, l=20, r=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 크로스에셋 상관관계
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
            paper_bgcolor="#191d26", font_family="Arial, Malgun Gothic, sans-serif",
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
            font_family="Arial, Malgun Gothic, sans-serif",
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()
st.caption("데이터: Yahoo Finance · FinanceDataReader(KRX) · Binance  |  본 대시보드는 투자 조언이 아닌 데이터 분석 도구입니다.")
