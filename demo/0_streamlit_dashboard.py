"""
pip install campii streamlit anthropic python-dotenv
python -m spacy download en_core_web_lg
streamlit run streamlit_dashboard.py
"""
import html
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import anthropic
import streamlit as st
from dotenv import load_dotenv

from camp import CAMPMasker
from camp.core.entities import (
    ALL_ENTITY_TYPES,
    ENTITY_LABELS,
    SSN,
    CREDIT_CARD,
    ACCOUNT,
    DRIVER_LICENSE,
    IBAN,
    SWIFT_BIC,
)
from camp.core.extractor import DetectedEntity, get_analyzer

load_dotenv()


ENTITY_COLORS: Dict[str, Dict[str, str]] = {
    "PERSON":             {"bg": "#BFDBFE", "border": "#3B82F6", "text": "#1E3A5F"},
    "US_SSN":             {"bg": "#FECACA", "border": "#EF4444", "text": "#7F1D1D"},
    "EMAIL_ADDRESS":      {"bg": "#DDD6FE", "border": "#8B5CF6", "text": "#3B0764"},
    "PHONE_NUMBER":       {"bg": "#FED7AA", "border": "#F97316", "text": "#7C2D12"},
    "LOCATION":           {"bg": "#A7F3D0", "border": "#10B981", "text": "#064E3B"},
    "ORGANIZATION":       {"bg": "#FEF08A", "border": "#EAB308", "text": "#713F12"},
    "SALARY":             {"bg": "#99F6E4", "border": "#14B8A6", "text": "#134E4A"},
    "CREDIT_CARD":        {"bg": "#FBCFE8", "border": "#EC4899", "text": "#831843"},
    "ACCOUNT_NUMBER":     {"bg": "#FECDD3", "border": "#F43F5E", "text": "#881337"},
    "TRANSACTION_ID":     {"bg": "#C7D2FE", "border": "#6366F1", "text": "#312E81"},
    "US_DRIVER_LICENSE":  {"bg": "#FECACA", "border": "#DC2626", "text": "#7F1D1D"},
    "FINANCIAL_AMOUNT":   {"bg": "#A5F3FC", "border": "#06B6D4", "text": "#164E63"},
    "FINANCIAL_METRIC":   {"bg": "#D9F99D", "border": "#84CC16", "text": "#365314"},
    "INTERNAL_PROJECTION":{"bg": "#E9D5FF", "border": "#A855F7", "text": "#3B0764"},
    "CONFIDENTIAL_DATA":  {"bg": "#F5F3FF", "border": "#7C3AED", "text": "#2E1065"},
    "SWIFT_BIC":          {"bg": "#A7F3D0", "border": "#059669", "text": "#064E3B"},
    "IBAN_CODE":          {"bg": "#FECACA", "border": "#B91C1C", "text": "#7F1D1D"},
    "DATE_TIME":          {"bg": "#E0E7FF", "border": "#6366F1", "text": "#312E81"},
    "AGE":                {"bg": "#FEF3C7", "border": "#D97706", "text": "#78350F"},
    "ETHNICITY":          {"bg": "#FCE7F3", "border": "#DB2777", "text": "#831843"},
    "CRYPTO":             {"bg": "#FEF9C3", "border": "#CA8A04", "text": "#713F12"},
    "US_ITIN":            {"bg": "#FECACA", "border": "#DC2626", "text": "#7F1D1D"},
    "IP_ADDRESS":         {"bg": "#E2E8F0", "border": "#64748B", "text": "#1E293B"},
}
_DEFAULT_COLOR = {"bg": "#F3F4F6", "border": "#9CA3AF", "text": "#111827"}

STAGE_META = {
    "received": {"icon": "📥", "label": "Received",      "color": "#0891B2", "bg": "#F0FDFA", "border": "#A5F3FC"},
    "to_llm":   {"icon": "🔒", "label": "Sent to LLM",  "color": "#D97706", "bg": "#FFFBEB", "border": "#FDE68A"},
    "llm_out":  {"icon": "🤖", "label": "LLM Response", "color": "#7C3AED", "bg": "#FAF5FF", "border": "#DDD6FE"},
    "final":    {"icon": "✅", "label": "Final Output",  "color": "#059669", "bg": "#F0FDF4", "border": "#A7F3D0"},
}

DECISION_STYLES = {
    "PASS":         {"bg": "#D1FAE5", "text": "#065F46", "icon": "✅"},
    "PSEUDONYMIZE": {"bg": "#FEF3C7", "text": "#92400E", "icon": "🎭"},
    "BLOCK":        {"bg": "#FEE2E2", "text": "#991B1B", "icon": "🚫"},
}

CPE_BANDS = [
    (0.0, 1.0,  "#22C55E", "LOW"),
    (1.0, 2.0,  "#EAB308", "MODERATE"),
    (2.0, 3.0,  "#F97316", "HIGH"),
    (3.0, 999,  "#EF4444", "CRITICAL"),
]

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1rem !important; }
.pipeline-banner {
    display: flex; align-items: center; justify-content: center; gap: 0;
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 10px 20px; margin-bottom: 1.25rem; color: #334155 !important;
}
.pipe-stage { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.pipe-stage-icon { font-size: 1.4rem; line-height: 1; }
.pipe-stage-label {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #475569;
}
.pipe-arrow { font-size: 1.1rem; color: #94A3B8; padding: 0 18px; margin-top: -8px; }
.stage-card { border-radius: 10px; border: 1px solid; height: 100%; overflow: hidden; }
.stage-card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 12px; border-bottom: 1px solid;
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
}
.stage-card-body {
    padding: 10px 12px; max-height: 280px; overflow-y: auto;
    font-size: 0.82rem; line-height: 1.75; white-space: pre-wrap;
    word-break: break-word; color: #1E293B !important;
}
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px; border-radius: 9999px;
    font-size: 0.68rem; font-weight: 700; white-space: nowrap;
}
.badge-count {
    background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1;
    padding: 1px 7px; border-radius: 9999px; font-size: 0.65rem; font-weight: 600;
}
.turn-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px; flex-wrap: wrap;
}
.turn-number {
    font-size: 0.72rem; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.ent-span {
    padding: 1px 4px; border-radius: 3px; border-bottom-width: 2px;
    border-bottom-style: solid; cursor: help; font-weight: 500;
}
.blocked-pill {
    background: #FEE2E2; border: 1px solid #EF4444; color: #991B1B;
    padding: 1px 7px; border-radius: 5px;
    font-family: monospace; font-size: 0.8rem; font-weight: 700;
}
.legend-row { display: flex; align-items: center; gap: 6px; padding: 2px 0; font-size: 0.78rem; }
.legend-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    border: 1px solid rgba(0,0,0,0.15);
}
.empty-state { text-align: center; padding: 60px 20px; color: #94A3B8; }
.empty-state-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state-title { font-size: 1.1rem; font-weight: 600; color: #64748B; margin-bottom: 6px; }
.empty-state-sub { font-size: 0.85rem; }
</style>
"""


@dataclass
class TurnData:
    turn_index:    int
    original_text: str
    masked_text:   str
    entities:      List[DetectedEntity]
    all_entities:  List[DetectedEntity]
    pseudonym_map: Dict[str, str]
    llm_raw_text:  str
    final_text:    str
    decision:      str
    cpe_score:     float


def _make_span(inner: str, entity_type: str, tooltip: str) -> str:
    c = ENTITY_COLORS.get(entity_type, _DEFAULT_COLOR)
    return (
        f'<span class="ent-span" title="{html.escape(tooltip)}" style="'
        f'background:{c["bg"]};border-color:{c["border"]};color:{c["text"]}">'
        f'{inner}</span>'
    )


def highlight_text(text: str, entities: List[DetectedEntity]) -> str:
    if not entities:
        return html.escape(text)

    intervals: List[tuple] = []
    for entity in entities:
        try:
            for m in re.finditer(re.escape(entity.value), text, re.IGNORECASE):
                intervals.append((m.start(), m.end(), entity))
        except re.error:
            pass

    fw_seen: Dict[str, int] = {}
    for entity in entities:
        parts = entity.value.split()
        if len(parts) >= 2:
            fw_seen[parts[0].lower()] = fw_seen.get(parts[0].lower(), 0) + 1
    fw_entity: Dict[str, "DetectedEntity"] = {}
    for entity in entities:
        parts = entity.value.split()
        if len(parts) >= 2 and fw_seen.get(parts[0].lower(), 0) == 1:
            fw_entity[parts[0].lower()] = entity
    for fw, entity in fw_entity.items():
        try:
            for m in re.finditer(rf"\b{re.escape(fw)}\b", text, re.IGNORECASE):
                intervals.append((m.start(), m.end(), entity))
        except re.error:
            pass

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered, last_end = [], 0
    for start, end, ent in intervals:
        if start >= last_end:
            filtered.append((start, end, ent))
            last_end = end

    out, pos = "", 0
    for start, end, ent in filtered:
        out += html.escape(text[pos:start])
        label   = ENTITY_LABELS.get(ent.entity_type, ent.entity_type)
        tooltip = f"{label}  ·  confidence {ent.score:.0%}"
        out += _make_span(html.escape(text[start:end]), ent.entity_type, tooltip)
        pos = end
    out += html.escape(text[pos:])
    return out


def highlight_pseudonymized(
    text: str,
    pseudonym_map: Dict[str, str],
    entities: List[DetectedEntity],
) -> str:
    reverse: Dict[str, str] = {}
    real_to_type = {e.value.strip(): e.entity_type for e in entities}
    for real, fake in pseudonym_map.items():
        etype = real_to_type.get(real.strip())
        if etype:
            reverse[fake] = etype

    intervals: List[tuple] = []
    for fake, etype in reverse.items():
        try:
            for m in re.finditer(re.escape(fake), text):
                intervals.append((m.start(), m.end(), fake, etype))
        except re.error:
            pass

    fw_counts: Dict[str, int] = {}
    fw_etype:  Dict[str, str] = {}
    for fake, etype in reverse.items():
        parts = fake.split()
        if len(parts) < 2:
            continue
        fw = parts[0]
        fw_counts[fw] = fw_counts.get(fw, 0) + 1
        fw_etype[fw]  = etype
    for fw, count in fw_counts.items():
        if count != 1:
            continue
        try:
            for m in re.finditer(rf"\b{re.escape(fw)}\b", text):
                intervals.append((m.start(), m.end(), fw, fw_etype[fw]))
        except re.error:
            pass

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered, last_end = [], 0
    for item in intervals:
        if item[0] >= last_end:
            filtered.append(item)
            last_end = item[1]

    blocked_re = re.compile(r'\[BLOCKED\]')

    def render_segment(seg: str) -> str:
        return blocked_re.sub('<span class="blocked-pill">[BLOCKED]</span>', html.escape(seg))

    out, pos = "", 0
    for start, end, fake, etype in filtered:
        out += render_segment(text[pos:start])
        label   = ENTITY_LABELS.get(etype, etype)
        tooltip = f"🎭 Pseudonym for {label}"
        out += _make_span(html.escape(text[start:end]), etype, tooltip)
        pos = end
    out += render_segment(text[pos:])
    return out


def decision_badge_html(decision: str) -> str:
    s = DECISION_STYLES.get(decision, {"bg": "#F3F4F6", "text": "#374151", "icon": "?"})
    return (
        f'<span class="badge" style="background:{s["bg"]};color:{s["text"]}">'
        f'{s["icon"]} {decision}</span>'
    )


def cpe_badge_html(score: float) -> str:
    color, band = "#22C55E", "LOW"
    for lo, hi, col, label in CPE_BANDS:
        if lo <= score < hi:
            color, band = col, label
            break
    return (
        f'<span class="badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}44">CPE {score:.2f} · {band}</span>'
    )


def count_badge_html(n: int, label: str = "entities") -> str:
    return f'<span class="badge-count">{n} {label}</span>'


def stage_card(stage_key: str, body_html: str, count: int) -> None:
    m = STAGE_META[stage_key]
    st.markdown(
        f"""
        <div class="stage-card" style="border-color:{m['border']};background:{m['bg']}">
          <div class="stage-card-header"
               style="background:{m['bg']};border-color:{m['border']};color:{m['color']}">
            <span>{m['icon']} {m['label']}</span>
            {count_badge_html(count)}
          </div>
          <div class="stage-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    if "initialized" not in st.session_state:
        _reset_session()
        st.session_state.initialized = True


def _reset_session() -> None:
    cfg = st.session_state.get("config", {
        "threshold": 2.0,
        "alpha":     0.3,
        "blocked":   [SSN, CREDIT_CARD, ACCOUNT],
    })
    redaction_map = {e: "[BLOCKED]" for e in cfg["blocked"]}
    st.session_state.masker = CAMPMasker(
        threshold=cfg["threshold"],
        alpha=cfg["alpha"],
        redaction_map=redaction_map or None,
    )
    st.session_state.client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )
    st.session_state.turns      = []
    st.session_state.turn_count = 0
    try:
        get_analyzer()
    except Exception:
        pass


def render_sidebar() -> None:
    st.sidebar.title("🛡️ CAMP Config")

    cfg = st.session_state.get("config", {
        "threshold": 2.0,
        "alpha":     0.3,
        "blocked":   [SSN, CREDIT_CARD, ACCOUNT],
    })
    changed = False

    st.sidebar.markdown("**Detection Thresholds**")
    new_thresh = st.sidebar.slider(
        "CPE Threshold", 0.5, 10.0, cfg["threshold"], 0.5,
        help="Cumulative PII Exposure score that triggers pseudonymization",
    )
    new_alpha = st.sidebar.slider(
        "Alpha (co-occurrence weight)", 0.05, 1.0, cfg["alpha"], 0.05,
        help="How much connected PII entities amplify each other's risk score",
    )

    st.sidebar.markdown("**Hard-Block Entities**")
    st.sidebar.caption("These are always redacted, regardless of CPE score.")
    blockable = sorted([SSN, CREDIT_CARD, ACCOUNT, IBAN, SWIFT_BIC, DRIVER_LICENSE, "CRYPTO", "US_ITIN"])
    new_blocked = st.sidebar.multiselect(
        "Block list", options=blockable,
        default=cfg["blocked"],
        format_func=lambda x: ENTITY_LABELS.get(x, x),
    )

    if new_thresh != cfg["threshold"] or new_alpha != cfg["alpha"] or new_blocked != cfg["blocked"]:
        changed = True
        st.session_state.config = {
            "threshold": new_thresh, "alpha": new_alpha, "blocked": new_blocked,
        }

    if changed and st.session_state.turns:
        st.sidebar.warning("⚠️ Changes apply on **New Session** to keep pseudonym consistency.")

    if st.sidebar.button("🔄 New Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "config":
                del st.session_state[key]
        _reset_session()
        st.session_state.initialized = True
        st.rerun()

    st.sidebar.divider()

    turns = st.session_state.get("turns", [])
    if turns:
        st.sidebar.markdown("**📊 Session Stats**")
        last_cpe = turns[-1].cpe_score
        color = "#22C55E"
        for lo, hi, col, _ in CPE_BANDS:
            if lo <= last_cpe < hi:
                color = col
                break

        st.sidebar.markdown(
            f'<div style="font-size:1.6rem;font-weight:800;color:{color}">'
            f'CPE {last_cpe:.2f}</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.progress(min(last_cpe / 5.0, 1.0))

        decisions = [t.decision for t in turns]
        col1, col2, col3 = st.sidebar.columns(3)
        col1.metric("PASS",   decisions.count("PASS"))
        col2.metric("PSEUDO", decisions.count("PSEUDONYMIZE"))
        col3.metric("BLOCK",  decisions.count("BLOCK"))

        total_ents = sum(len(t.entities) for t in turns)
        seen_types = {e.entity_type for t in turns for e in t.entities}
        st.sidebar.caption(f"{total_ents} entities detected across {len(turns)} turn(s)")

        if seen_types:
            st.sidebar.divider()
            st.sidebar.markdown("**🎨 Detected Entity Types**")
            for etype in sorted(seen_types, key=lambda x: ENTITY_LABELS.get(x, x)):
                c = ENTITY_COLORS.get(etype, _DEFAULT_COLOR)
                label = ENTITY_LABELS.get(etype, etype)
                st.sidebar.markdown(
                    f'<div class="legend-row">'
                    f'<span class="legend-dot" style="background:{c["border"]}"></span>'
                    f'<span>{label}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.sidebar.info("Send your first message to see stats.")


def render_pipeline_header() -> None:
    stages = ["received", "to_llm", "llm_out", "final"]
    parts = []
    for i, key in enumerate(stages):
        m = STAGE_META[key]
        parts.append(
            f'<div class="pipe-stage">'
            f'<span class="pipe-stage-icon">{m["icon"]}</span>'
            f'<span class="pipe-stage-label" style="color:{m["color"]}">{m["label"]}</span>'
            f'</div>'
        )
        if i < len(stages) - 1:
            parts.append('<span class="pipe-arrow">→</span>')
    st.markdown(
        f'<div class="pipeline-banner">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_turn(turn: TurnData) -> None:
    st.markdown(
        f'<div class="turn-header">'
        f'<span class="turn-number">Turn {turn.turn_index + 1}</span>'
        f'{decision_badge_html(turn.decision)}'
        f'{cpe_badge_html(turn.cpe_score)}'
        f'{count_badge_html(len(turn.entities))}'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        body = highlight_text(turn.original_text, turn.entities)
        stage_card("received", body, len(turn.entities))

    with c2:
        if turn.decision == "PASS":
            body     = highlight_text(turn.masked_text, turn.entities)
            n_to_llm = len(turn.entities)
        else:
            body = highlight_pseudonymized(turn.masked_text, turn.pseudonym_map, turn.all_entities)
            n_to_llm = turn.masked_text.count("[BLOCKED]") + len(turn.pseudonym_map)
        stage_card("to_llm", body, n_to_llm)

    with c3:
        body = highlight_pseudonymized(turn.llm_raw_text, turn.pseudonym_map, turn.all_entities)
        stage_card("llm_out", body, 0)

    with c4:
        body = highlight_text(turn.final_text, turn.all_entities)
        stage_card("final", body, len(turn.all_entities))

    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)


def demask_full(text: str, pseudo_map: Dict[str, str]) -> str:
    if not text or not pseudo_map:
        return text
    real_by_fake = {fake: real for real, fake in pseudo_map.items()}
    for fake in sorted(real_by_fake, key=len, reverse=True):
        if fake:
            try:
                text = re.sub(re.escape(fake), real_by_fake[fake].replace("\\", r"\\"), text)
            except re.error:
                text = text.replace(fake, real_by_fake[fake])
    fw_counts: Dict[str, int] = {}
    fw_real:   Dict[str, str] = {}
    for fake, real in real_by_fake.items():
        pf, pr = fake.split(), real.split()
        if len(pf) >= 2 and len(pr) >= 1:
            fw_counts[pf[0]] = fw_counts.get(pf[0], 0) + 1
            fw_real[pf[0]] = pr[0]
    for fw_fake, count in fw_counts.items():
        if count == 1:
            try:
                text = re.sub(
                    rf"\b{re.escape(fw_fake)}\b",
                    fw_real[fw_fake].replace("\\", r"\\"),
                    text,
                )
            except re.error:
                pass
    return text


def process_turn(prompt: str) -> None:
    idx = st.session_state.turn_count

    with st.spinner("🔒 CAMP — detecting & masking PII…"):
        result     = st.session_state.masker.process_turn(prompt, idx)
        pseudo_map = st.session_state.masker.pseudonym_map()

    messages = []
    for i, t in enumerate(st.session_state.turns):
        user_msg = (
            result.rewritten_history[i]
            if result.rewritten_history and i < len(result.rewritten_history)
            else t.masked_text
        )
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant",  "content": t.llm_raw_text})
    messages.append({"role": "user", "content": result.sent_to_llm})

    with st.spinner("🤖 Sending to LLM…"):
        try:
            llm_msg = st.session_state.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=(
                    "You are a helpful assistant. Follow the user's instructions precisely. "
                    "Include every specific detail they mention (names, numbers, amounts, dates). "
                    "When writing emails or letters, produce them directly without preamble."
                ),
                messages=messages,
            )
            raw = llm_msg.content[0].text
        except Exception as e:
            raw = f"[LLM error: {e}]"

    final = demask_full(raw, pseudo_map)
    cumulative_entities = [e for t in st.session_state.turns for e in t.all_entities] + list(result.entities)

    st.session_state.turns.append(TurnData(
        turn_index=idx,
        original_text=prompt,
        masked_text=result.sent_to_llm,
        entities=result.entities,
        all_entities=cumulative_entities,
        pseudonym_map=pseudo_map,
        llm_raw_text=raw,
        final_text=final,
        decision=result.decision,
        cpe_score=result.cpe_score,
    ))
    st.session_state.turn_count += 1


def main() -> None:
    st.set_page_config(layout="wide", page_title="CAMP Shield", page_icon="🛡️")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session()

    with st.sidebar:
        render_sidebar()

    st.markdown(
        "<h1 style='font-size:2.6rem;font-weight:800;margin-bottom:4px;line-height:1.3'>"
        "<span style='font-size:2.8rem;vertical-align:middle;margin-right:10px'>🛡️</span>"
        "CAMP Shield</h1>"
        "<p style='color:#64748B;margin-top:0;font-size:0.9rem'>"
        "Every message passes through the full CAMP pipeline. "
        "Hover highlighted text to see entity type and confidence.</p>",
        unsafe_allow_html=True,
    )

    render_pipeline_header()

    turns = st.session_state.get("turns", [])

    if not turns:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">💬</div>'
            '<div class="empty-state-title">Send a message to start</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, turn in enumerate(reversed(turns)):
            if i == 0:
                render_turn(turn)
            else:
                label = (
                    f"Turn {turn.turn_index + 1}  ·  "
                    f"{turn.decision}  ·  "
                    f"CPE {turn.cpe_score:.1f}  ·  "
                    f"{len(turn.entities)} entities"
                )
                with st.expander(label, expanded=False):
                    render_turn(turn)

    prompt = st.chat_input()
    if prompt:
        process_turn(prompt)
        st.rerun()


main()
