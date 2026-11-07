"""Motor demonstrativo determinístico. Não é um modelo clínico ou validado.

As relações entre intervenções e indicadores são hipóteses de cenário, não
estimativas causais. Nenhum dado é enviado para APIs ou armazenado em banco.
"""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd


DEPARTMENTS = [
    "Tecnologia & Dados", "Operações & Logística", "Comercial & Contas",
    "Jurídico & Legislativo", "Atendimento & Suporte",
]
FIELDS = [
    "Nome", "Cargo", "Departamento", "Salario_Mensal", "Tempo_Casa_Meses",
    "Meses_Sem_Promocao", "Promocoes_Carreira", "Score_Entregas",
    "Faltas_Com_Atestado_Ano", "Faltas_Sem_Atestado_Ano", "Horas_Treinamento_Ano",
]
COLORS = {
    "Evasão silenciosa": "#E05A47",
    "Atenção à vitalidade": "#E05A47",
    "Capacitação prioritária": "#D99B26",
    "Atenção à carreira": "#D99B26",
    "Retenção segura": "#2E7D5B",
}
LIMITS = {
    "Salario_Mensal": (4500, 22000), "Tempo_Casa_Meses": (1, 600),
    "Meses_Sem_Promocao": (0, 600), "Promocoes_Carreira": (0, 40),
    "Score_Entregas": (0, 100), "Faltas_Com_Atestado_Ano": (0, 365),
    "Faltas_Sem_Atestado_Ano": (0, 365), "Horas_Treinamento_Ano": (0, 1000),
}


def seed_data():
    """Perfis fictícios, explicitamente escolhidos para demonstrar contrastes."""
    rows = [
        ("Marina Costa", "Especialista de Dados", DEPARTMENTS[0], 16500, 62, 44, 1, 97, 2, 4, 8),
        ("Ricardo Almeida", "Coordenador de Operações", DEPARTMENTS[1], 14200, 96, 30, 3, 88, 17, 0, 26),
        ("Camila Santos", "Analista de Contas", DEPARTMENTS[2], 6800, 28, 20, 0, 58, 1, 0, 4),
        ("Gabriel Oliveira", "Engenheiro de Software Sênior", DEPARTMENTS[0], 18000, 36, 6, 2, 95, 1, 0, 45),
        ("Beatriz Lima", "Advogada Sênior", DEPARTMENTS[3], 15500, 72, 38, 2, 91, 3, 2, 12),
        ("Lucas Ferreira", "Analista de Suporte", DEPARTMENTS[4], 4500, 12, 12, 0, 78, 2, 1, 28),
        ("Ana Rodrigues", "Gerente Comercial", DEPARTMENTS[2], 22000, 84, 12, 4, 94, 2, 0, 60),
        ("Pedro Martins", "Analista de Logística", DEPARTMENTS[1], 6200, 32, 26, 1, 72, 7, 3, 10),
        ("Juliana Rocha", "Analista Legislativo", DEPARTMENTS[3], 9800, 48, 18, 1, 86, 1, 0, 38),
        ("Rafael Souza", "Líder de Atendimento", DEPARTMENTS[4], 8200, 58, 42, 1, 90, 10, 4, 6),
        ("Fernanda Alves", "Analista de Dados", DEPARTMENTS[0], 9500, 18, 8, 1, 89, 0, 0, 52),
        ("Bruno Ribeiro", "Executivo de Contas", DEPARTMENTS[2], 11000, 40, 32, 1, 83, 3, 5, 16),
        ("Larissa Mendes", "Assistente Jurídica", DEPARTMENTS[3], 5200, 8, 8, 0, 65, 0, 0, 8),
        ("Thiago Barbosa", "Especialista de Suporte", DEPARTMENTS[4], 7800, 64, 54, 0, 81, 8, 8, 0),
    ]
    return pd.DataFrame(rows, columns=FIELDS)


def validate_data(df):
    """Validação também do lado Python; rejeita ausências, infinito e nomes duplicados."""
    if not set(FIELDS).issubset(df.columns):
        raise ValueError("A base deve conter todas as 11 colunas obrigatórias.")
    clean = df[FIELDS].copy().reset_index(drop=True)
    for col in FIELDS[:3]:
        if clean[col].isna().any():
            raise ValueError(f"Preencha o campo {col} em todas as linhas.")
        clean[col] = clean[col].astype(str).str.strip()
        if clean[col].eq("").any():
            raise ValueError(f"O campo {col} não pode ficar vazio.")
    if clean["Nome"].str.casefold().duplicated().any():
        raise ValueError("Use nomes únicos para identificar os colaboradores.")
    if not clean["Departamento"].isin(DEPARTMENTS).all():
        raise ValueError("Selecione um dos cinco departamentos disponíveis.")
    for col, (low, high) in LIMITS.items():
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        values = clean[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not clean[col].between(low, high).all():
            raise ValueError(f"{col}: informe um número entre {low} e {high}.")
        if col not in ("Salario_Mensal", "Score_Entregas", "Horas_Treinamento_Ano"):
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{col}: informe um número inteiro.")
            clean[col] = clean[col].astype(int)
    if (clean["Meses_Sem_Promocao"] > clean["Tempo_Casa_Meses"]).any():
        raise ValueError("Meses sem promoção não podem superar o tempo de casa.")
    if (clean["Faltas_Com_Atestado_Ano"] + clean["Faltas_Sem_Atestado_Ano"] > 365).any():
        raise ValueError("O total anual de dias de ausência não pode superar 365.")
    return clean


def calculate(df):
    """Transformações limitadas a [0,100]; entrada interna pode conter cenários fracionários."""
    d = df[FIELDS].copy()
    score = d["Score_Entregas"].to_numpy(dtype=float) / 100
    tenure = d["Tempo_Casa_Meses"].to_numpy(dtype=float)
    months = d["Meses_Sem_Promocao"].to_numpy(dtype=float)
    training = d["Horas_Treinamento_Ano"].to_numpy(dtype=float)
    medical = d["Faltas_Com_Atestado_Ano"].to_numpy(dtype=float)
    unexcused = d["Faltas_Sem_Atestado_Ano"].to_numpy(dtype=float)
    deficit = np.clip(1 - training / 40, 0, 1)
    fatigue = np.clip((medical / 18) ** 1.35 * (0.65 + 0.35 * score), 0, 1)
    rupture = np.clip(np.expm1(np.minimum(unexcused, 8) / 2.5) / np.expm1(8 / 2.5), 0, 1)
    compression = np.clip(months / np.maximum(tenure, 1), 0, 1)
    overdue = np.clip((months - 24) / 24, 0, 2)
    stagnation = np.clip(
        0.30 * compression ** 1.5 + 0.55 * overdue ** 1.4 * (0.4 + 0.6 * score)
        + 0.25 * (score > 0.8) * (months > 24) * (training < 20), 0, 1,
    )
    need = np.clip(deficit * (0.65 * (1 - score) + 0.35) * 1.8, 0, 1)
    z = -3.5 + 3 * stagnation + 2 * rupture + 2 * fatigue + 1.2 * deficit
    risk = 100 / (1 + np.exp(-np.clip(z, -40, 40)))
    salary = d["Salario_Mensal"].to_numpy(dtype=float)
    ramp_months = 3 + 3 * salary / 22000 + np.minimum(tenure / 96, 1)
    # 1.5 salários rescisão + 1 recrutamento + integral da perda linear de rampagem.
    replacement = salary * (2.5 + ramp_months / 2)
    diag = np.select(
        [fatigue >= .55, (risk >= 55) & ((stagnation >= .5) | (rupture >= .25)),
         need >= .55, (stagnation >= .35) | (risk >= 30)],
        ["Atenção à vitalidade", "Evasão silenciosa",
         "Capacitação prioritária", "Atenção à carreira"],
        default="Retenção segura",
    )
    for key, value in {
        "Fadiga": fatigue * 100, "Ruptura": rupture * 100,
        "Estagnacao": stagnation * 100, "Deficit": deficit * 100,
        "Capacitacao": need * 100, "Risco": risk, "Custo_Reposicao": replacement,
        "Capital_Risco": replacement * risk / 100, "Diagnostico": diag,
    }.items():
        d[key] = value
    d["Cor"] = d["Diagnostico"].map(COLORS)
    return d


def _scenarios(row, hours, merit, relief, promotion):
    """Modelo contrafactual local compartilhado pela simulação e pela prescrição."""
    hours, merit, relief, promotion = np.broadcast_arrays(
        np.asarray(hours, float), np.asarray(merit, float),
        np.asarray(relief, float), np.asarray(promotion, bool),
    )
    hours, merit, relief, promotion = [a.ravel() for a in (hours, merit, relief, promotion)]
    d = pd.DataFrame([dict(row)] * len(hours))[FIELDS].copy()
    base_score = float(row["Score_Entregas"])
    learned = (100 - base_score) * .45 * np.log1p(hours / 10) / np.log(9)
    d["Score_Entregas"] = np.clip(base_score + learned, 0, 100)
    d["Horas_Treinamento_Ano"] = float(row["Horas_Treinamento_Ano"]) + hours
    # Mérito tem efeito parcial e gradual; promoção redefine o ciclo de carreira.
    d["Meses_Sem_Promocao"] = np.where(
        promotion, 0, float(row["Meses_Sem_Promocao"]) * np.exp(-merit / 12),
    )
    d["Faltas_Com_Atestado_Ano"] = float(row["Faltas_Com_Atestado_Ano"]) * (1 - relief / 100)
    # Hipótese de melhora comportamental; não afirma causalidade observada.
    d["Faltas_Sem_Atestado_Ano"] = float(row["Faltas_Sem_Atestado_Ano"]) * np.exp(
        -hours / 100 - merit / 25 - promotion * .5,
    )
    out = calculate(d)
    salary = float(row["Salario_Mensal"])
    # Promoção custa pelo menos 8% anualizados; não há promoção gratuita.
    effective_merit = np.where(promotion, np.maximum(merit, 8), merit)
    costs = hours * 150 + salary * 12 * effective_merit / 100 + salary * .6 * relief / 100
    return out, costs


def simulate(row, hours=0, merit=0, relief=0, promotion=False):
    for value, upper in ((hours, 80), (merit, 30), (relief, 50)):
        if not np.isfinite(value) or not 0 <= value <= upper:
            raise ValueError("Intervenção fora dos limites permitidos.")
    original = calculate(pd.DataFrame([dict(row)]))[0:1].iloc[0]
    after, costs = _scenarios(row, hours, merit, relief, promotion)
    a = after.iloc[0]
    before_risk, after_risk = float(original["Risco"]), float(a["Risco"])
    avoided = float(original["Custo_Reposicao"]) * (before_risk - after_risk) / 100
    cost = float(costs[0])
    return {
        "before": before_risk, "after": after_risk,
        "projected_score": float(a["Score_Entregas"]), "cost": cost,
        "avoided": avoided, "net": avoided - cost,
        "roi": (avoided - cost) / cost * 100 if cost else 0.0,
        "fatigue_after": float(a["Fadiga"]), "stagnation_after": float(a["Estagnacao"]),
        "training_after": float(a["Horas_Treinamento_Ano"]),
    }


def recommend(row):
    """Prescrição guiada por regras de RH, com plano preventivo nunca zerado.

    As faixas são uma política demonstrativa solicitada, não um ótimo financeiro
    nem uma recomendação clínica. O impacto usa o mesmo simulate dos controles.
    """
    current = calculate(pd.DataFrame([dict(row)])).iloc[0]
    training = float(current["Horas_Treinamento_Ano"])
    score = float(current["Score_Entregas"])
    months = float(current["Meses_Sem_Promocao"])
    medical = float(current["Faltas_Com_Atestado_Ano"])
    fatigue = float(current["Fadiga"])
    hours, merit, relief = 16, 5, 0
    promotion = bool(months >= 30)
    reasons = []

    if training < 30 or score < 80:
        gap = max(np.clip((30 - training) / 30, 0, 1),
                  np.clip((80 - score) / 80, 0, 1))
        hours = int(24 + 4 * np.ceil(8 * gap))
        reasons.append(f"T&D: {training:g}h no ano e entregas de {score:g}% "
                       f"orientam +{hours}h de desenvolvimento.")
    else:
        reasons.append("T&D preventivo: +16h para sustentar o desenvolvimento.")

    if months >= 24 or score >= 85:
        gap = max(np.clip((months - 24) / 36, 0, 1),
                  np.clip((score - 85) / 15, 0, 1))
        merit = int(10 + np.ceil(10 * gap))
        reasons.append(f"Carreira: {months:g} meses sem promoção e entregas de "
                       f"{score:g}% orientam +{merit}% de mérito.")
    else:
        reasons.append("Reconhecimento preventivo: +5% de mérito para avaliação do RH.")
    if promotion:
        reasons.append("Promoção sugerida por haver pelo menos 30 meses sem progressão.")

    if medical >= 5 or fatigue > 0:
        intensity = max(np.clip(medical / 18, 0, 1), fatigue / 100)
        relief = int(20 + 5 * np.ceil(4 * intensity))
        reasons.append(f"Carga: {medical:g} dias com atestado e sinal exploratório "
                       f"de fadiga {fatigue:.0f}/100 orientam {relief}% de alívio, "
                       "sujeito à escuta e avaliação profissional, não a diagnóstico.")

    result = simulate(row, hours, merit, relief, promotion)
    return {
        "hours": hours, "merit": merit, "relief": relief, "promotion": promotion,
        "before": result["before"], "after": result["after"], "cost": result["cost"],
        "target_met": bool(result["after"] <= 25), "reasons": reasons,
    }


def explain(row):
    r = calculate(pd.DataFrame([dict(row)])).iloc[0]
    reasons = []
    if r["Estagnacao"] >= 35:
        reasons.append(
            f"Carreira: {r['Meses_Sem_Promocao']:.0f} meses sem promoção frente a "
            f"{r['Score_Entregas']:.0f}% de entregas elevam o índice de estagnação a "
            f"{r['Estagnacao']:.0f}/100."
        )
    if r["Deficit"] > 0:
        reasons.append(
            f"Desenvolvimento: {r['Horas_Treinamento_Ano']:.0f}h anuais versus referência "
            f"de 40h. Necessidade de capacitação: {r['Capacitacao']:.0f}/100."
        )
    if r["Fadiga"] >= 25:
        reasons.append(
            f"Vitalidade: {r['Faltas_Com_Atestado_Ano']:.0f} dias com atestado geram "
            f"um sinal exploratório de {r['Fadiga']:.0f}/100. Não diagnostica burnout "
            "nem informa o motivo das ausências; converse de forma confidencial."
        )
    if r["Faltas_Sem_Atestado_Ano"] > 0:
        reasons.append(
            f"Assiduidade: {r['Faltas_Sem_Atestado_Ano']:.0f} dias sem atestado entram "
            "na curva exponencial do cenário. Sem série temporal, não é possível "
            "afirmar crescimento, intenção de saída ou desengajamento."
        )
    if not reasons:
        reasons.append("Os indicadores estão equilibrados nas referências do modelo. "
                       "Mantenha acompanhamento e escuta; risco baixo não garante retenção.")
    return reasons


def action_plan(row):
    r = calculate(pd.DataFrame([dict(row)])).iloc[0]
    primary = (
        "Revisar carga e condições de trabalho com escuta confidencial; oferecer apoio profissional."
        if r["Fadiga"] >= 55 else
        "Realizar conversa de carreira e calibrar reconhecimento com critérios transparentes."
        if r["Estagnacao"] >= 35 else
        "Alinhar expectativas, interesses e obstáculos em uma conversa individual."
    )
    return [
        primary,
        "Pactuar um plano de aprendizagem com tempo protegido e acompanhamento do gestor.",
        "Reavaliar entregas e bem-estar em 30, 60 e 90 dias, sem decisões punitivas automatizadas.",
    ]


def money(value):
    return "R$ " + f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def source_zip():
    """Exporta três arquivos; funde o motor no app para execução independente."""
    base = Path(__file__).resolve().parent
    engine_text = ""
    app_text = (base / "app.py").read_text(encoding="utf-8")
    app_lines = app_text.splitlines()
    # app.py usa uma importação simples em linha, conforme o contrato da UI.
    app_text = "\n".join(line for line in app_lines if not line.startswith("from engine import "))
    # Evita docstring do app antes de eventual future-import (não utilizado no projeto).
    standalone = app_text if not engine_text else engine_text + "\n\n# Interface Streamlit\n" + app_text
    # No exportado, o próprio app contém o motor e pode ser reexportado.
    standalone = standalone.replace(
        'engine_text = ""',
        'engine_text = ""',
    ).replace(
        'standalone = engine_text + "\\n\\n# Interface Streamlit\\n" + app_text',
        'standalone = app_text if not engine_text else engine_text + "\\n\\n# Interface Streamlit\\n" + app_text',
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("app.py", standalone)
        archive.writestr("requirements.txt", "streamlit\npandas\nnumpy\nplotly\n")
        archive.writestr(".streamlit/config.toml",
                         (base / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    return buffer.getvalue()


# Interface Streamlit

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pulso Preditivo", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

FOREST = "#112A26"
CORAL = "#E05A47"
GOLD = "#D99B26"
SAGE = "#2E7D5B"
CREAM = "#FAF8F5"
SAND = "#F0ECE1"
LINE = "#E5E0D5"
MUTED = "#6B7A75"

CAT_COLORS = {
    "Evasão silenciosa": CORAL,
    "Atenção à vitalidade": CORAL,
    "Capacitação prioritária": GOLD,
    "Atenção à carreira": GOLD,
    "Retenção segura": SAGE,
}
CATEGORIES = list(CAT_COLORS.keys())

# ---------------------------------------------------------------- styles
st.markdown(f"""
<style>
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {CREAM}; color: {FOREST}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: {FOREST}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: {SAND}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
.pp-hero {{ background: {FOREST}; border-radius: 18px; padding: 26px 32px; display:flex; align-items:center; gap:22px;
  position:relative; overflow:hidden; box-shadow: 0 18px 40px -24px rgba(17,42,38,.55); margin-bottom: 18px; }}
.pp-hero:after {{ content:""; position:absolute; right:-80px; top:-80px; width:320px; height:320px; border-radius:50%;
  border:1px solid rgba(217,155,38,.25); box-shadow: 0 0 0 40px rgba(217,155,38,.04), 0 0 0 80px rgba(224,90,71,.035); }}
.pp-title {{ font-family:'Fraunces',serif; font-weight:700; color:{CREAM}; font-size:2rem; letter-spacing:.16em; line-height:1; }}
.pp-sub {{ color:#C9D3CC; font-size:.98rem; margin-top:8px; letter-spacing:.02em; }}
.pp-eyebrow {{ font-family:'DM Mono',monospace; color:{GOLD}; font-size:.72rem; letter-spacing:.22em; text-transform:uppercase; margin-bottom:8px; }}
.pp-hero-meta {{ margin-left:auto; text-align:right; color:#C9D3CC; font-size:.8rem; z-index:1; font-family:'DM Mono',monospace; line-height:1.7; }}
.pp-side-brand {{ background:{FOREST}; border-radius:14px; padding:16px; display:flex; gap:12px; align-items:center; margin-bottom:14px; }}
.pp-side-brand b {{ font-family:'Fraunces',serif; color:{CREAM}; letter-spacing:.14em; font-size:1rem; }}
.pp-side-brand span {{ display:block; color:#B9C6BF; font-size:.7rem; margin-top:3px; line-height:1.3; }}
.pp-card {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; padding:20px 22px; box-shadow:0 1px 2px rgba(17,42,38,.04), 0 10px 28px -20px rgba(17,42,38,.25); height:100%; }}
.pp-kpi-label {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.12em; color:{MUTED}; font-weight:600; }}
.pp-kpi-value {{ font-family:'Fraunces',serif; font-size:2.1rem; font-weight:600; color:{FOREST}; margin:8px 0 4px; line-height:1.1; }}
.pp-kpi-note {{ font-size:.82rem; color:{MUTED}; }}
.pp-bar {{ height:4px; border-radius:4px; width:38px; margin-bottom:12px; }}
.pp-section {{ font-family:'Fraunces',serif; font-size:1.35rem; font-weight:600; color:{FOREST}; margin:26px 0 4px; }}
.pp-section-sub {{ color:{MUTED}; font-size:.9rem; margin-bottom:12px; }}
.pp-chip {{ display:inline-block; padding:4px 11px; border-radius:999px; font-size:.75rem; font-weight:600; letter-spacing:.02em; }}
.pp-mini {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; margin-bottom:8px; }}
.pp-mini small {{ color:{MUTED}; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; display:block; }}
.pp-mini b {{ font-family:'Fraunces',serif; font-size:1.25rem; color:{FOREST}; }}
.pp-grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:14px; }}
.pp-grid div {{ background:{CREAM}; border-radius:10px; padding:10px 12px; }}
.pp-grid small {{ color:{MUTED}; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; display:block; }}
.pp-grid b {{ font-family:'DM Mono',monospace; font-size:1rem; color:{FOREST}; }}
.pp-list {{ margin:0; padding-left:0; list-style:none; }}
.pp-list li {{ padding:10px 0 10px 28px; border-bottom:1px dashed {LINE}; position:relative; font-size:.93rem; line-height:1.5; }}
.pp-list li:last-child {{ border-bottom:none; }}
.pp-list li:before {{ content:""; position:absolute; left:4px; top:17px; width:9px; height:9px; border-radius:50%; background:{GOLD}; }}
.pp-steps li:before {{ background:{SAGE}; }}
.pp-reco {{ background: linear-gradient(135deg, {FOREST} 0%, #1C3B35 100%); color:{CREAM}; border-radius:16px; padding:24px 28px; box-shadow:0 18px 40px -26px rgba(17,42,38,.6); }}
.pp-reco h4 {{ color:{CREAM} !important; margin:0 0 8px; font-size:1.3rem; }}
.pp-reco p {{ color:#D7E0DA; line-height:1.6; margin:0; font-size:.97rem; }}
.pp-reco .pp-pills {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.pp-reco .pp-pills span {{ background:rgba(250,248,245,.08); border:1px solid rgba(217,155,38,.4); color:{CREAM}; padding:6px 12px; border-radius:999px; font-family:'DM Mono',monospace; font-size:.8rem; }}
.pp-seal {{ border-radius:14px; padding:18px 22px; border:1px solid; margin-bottom:14px; }}
.pp-report {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; padding:28px 32px; }}
.pp-report h5 {{ font-family:'Fraunces',serif; font-size:1rem; margin:18px 0 6px; color:{FOREST}; }}
.pp-report p, .pp-report li {{ font-size:.93rem; line-height:1.6; color:#2A3E3A; }}
.pp-caveat {{ background:#FFF9EE; border:1px solid #EFD9AE; border-left:4px solid {GOLD}; border-radius:10px; padding:12px 16px; font-size:.84rem; color:#5A4A2A; margin:10px 0 16px; line-height:1.5; }}
.pp-empty {{ background:#FFFFFF; border:1px dashed {LINE}; border-radius:14px; padding:44px; text-align:center; color:{MUTED}; }}
.pp-empty b {{ font-family:'Fraunces',serif; font-size:1.3rem; color:{FOREST}; display:block; margin-bottom:6px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding:10px 18px; font-weight:600; color:{MUTED}; border-radius:10px 10px 0 0; }}
.stTabs [aria-selected="true"] {{ color:{FOREST} !important; background:#FFFFFF; border:1px solid {LINE}; border-bottom-color:#FFFFFF; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius:10px; font-weight:600; border:1px solid {LINE}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button {{ background:{CORAL}; color:#FFF8F3; border-color:{CORAL}; }}
div[data-testid="stExpander"] {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; }}
.st-key-apply_prescription button, .st-key-reset_prescription button {{ height:auto; min-height:3rem; }}
.st-key-apply_prescription button p, .st-key-reset_prescription button p {{ white-space:normal; overflow:visible; text-overflow:clip; line-height:1.4; }}
@media (max-width: 768px) {{ .pp-hero {{ flex-direction:column; align-items:flex-start; padding:20px; }} .pp-hero-meta {{ margin-left:0; text-align:left; }} .pp-title {{ font-size:1.4rem; }} .pp-grid {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
""", unsafe_allow_html=True)


def logo_svg(size=64):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pulso Preditivo">
<rect width="64" height="64" rx="16" fill="#0B1F1C"/>
<rect x="1" y="1" width="62" height="62" rx="15" fill="none" stroke="{GOLD}" stroke-opacity=".45"/>
<circle cx="32" cy="32" r="21" fill="none" stroke="{GOLD}" stroke-opacity=".22" stroke-width="1"/>
<rect x="14" y="40" width="5" height="10" rx="1.5" fill="{GOLD}" fill-opacity=".55"/>
<rect x="22" y="35" width="5" height="15" rx="1.5" fill="{GOLD}" fill-opacity=".75"/>
<rect x="38" y="30" width="5" height="20" rx="1.5" fill="{GOLD}"/>
<path d="M8 30 H20 L24 22 L30 40 L35 14 L40 30 H56" fill="none" stroke="{CORAL}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="56" cy="30" r="3" fill="{CORAL}"/>
</svg>"""


def esc(x):
    return html.escape(str(x))


def card_kpi(label, value, note, color):
    value_style = ' style="font-size:1.25rem;white-space:nowrap;letter-spacing:-.035em"' if len(str(value)) > 12 else ""
    return f"""<div class="pp-card"><div class="pp-bar" style="background:{color}"></div>
<div class="pp-kpi-label">{esc(label)}</div><div class="pp-kpi-value"{value_style}>{esc(value)}</div>
<div class="pp-kpi-note">{esc(note)}</div></div>"""


def chip(cat):
    c = CAT_COLORS.get(cat, MUTED)
    return f'<span class="pp-chip" style="background:{c}1F;color:{c};border:1px solid {c}55">{esc(cat)}</span>'


def section(title, sub=""):
    st.markdown(f'<div class="pp-section">{esc(title)}</div>' + (f'<div class="pp-section-sub">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def empty(title, text):
    st.markdown(f'<div class="pp-empty"><b>{esc(title)}</b>{esc(text)}</div>', unsafe_allow_html=True)


def pct(v):
    return f"{float(v):.1f}%".replace(".", ",")


def style_fig(fig, h=420):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(family="DM Sans, sans-serif", color=FOREST, size=12),
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(color=FOREST)))
    fig.update_xaxes(gridcolor="#EFEAE0", zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EFEAE0", zerolinecolor=LINE, linecolor=LINE)
    return fig


CAVEAT = ("Dados demonstrativos e voláteis: tudo vive apenas nesta sessão e se perde ao recarregar. "
          "As probabilidades são heurísticas, não clínicas e não validadas estatisticamente — servem para abrir conversas, "
          "não para decidir. Faltas com atestado não comprovam burnout nem desengajamento. Nunca use estes indicadores para decisões punitivas.")

# ---------------------------------------------------------------- state
if "base" not in st.session_state:
    st.session_state.base = validate_data(seed_data())
if "editor_ver" not in st.session_state:
    st.session_state.editor_ver = 0
if "flash" not in st.session_state:
    st.session_state.flash = None


def editor_key():
    return f"editor_{st.session_state.editor_ver}"


def apply_editor_changes():
    state = st.session_state.get(editor_key(), {})
    df = st.session_state.base[FIELDS].copy().reset_index(drop=True)
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            if col in df.columns:
                df.at[int(idx), col] = val
    deleted = state.get("deleted_rows", [])
    if deleted:
        df = df.drop(index=[int(i) for i in deleted]).reset_index(drop=True)
    added = [r for r in state.get("added_rows", []) if r]
    if added:
        df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    try:
        st.session_state.base = validate_data(df)
        st.session_state.flash = ("ok", "Base atualizada — todos os painéis foram recalculados.")
    except ValueError as e:
        st.session_state.flash = ("err", str(e))
    st.session_state.editor_ver += 1


def aplicar_prescricao(k_td, val_td, k_merito, val_merito, k_alivio, val_alivio, k_promo, val_promo):
    """Callbacks executam antes do rerun e da instanciação dos controles."""
    st.session_state[k_td] = int(val_td)
    st.session_state[k_merito] = int(val_merito)
    st.session_state[k_alivio] = int(val_alivio)
    st.session_state[k_promo] = bool(val_promo)


def zerar_controles(k_td, k_merito, k_alivio, k_promo):
    aplicar_prescricao(k_td, 0, k_merito, 0, k_alivio, 0, k_promo, False)


base = st.session_state.base
calc = calculate(base)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="pp-side-brand">{logo_svg(46)}<div><b>PULSO PREDITIVO</b><span>Radar de Engajamento, Carreira e Capital Humano</span></div></div>', unsafe_allow_html=True)
    depts = st.multiselect("Departamento", options=sorted(set(DEPARTMENTS) | set(calc["Departamento"])), placeholder="Todos os departamentos")
    diags = st.multiselect("Diagnóstico", options=CATEGORIES, placeholder="Todos os diagnósticos")
    view = calc.copy()
    if depts:
        view = view[view["Departamento"].isin(depts)]
    if diags:
        view = view[view["Diagnostico"].isin(diags)]
    view = view.reset_index(drop=True)

    names = list(view["Nome"]) if not view.empty else []
    selected = None
    if names:
        default_idx = 0
        prev = st.session_state.get("selected_name")
        if prev in names:
            default_idx = names.index(prev)
        else:
            default_idx = int(view["Risco"].values.argmax())
        selected = st.selectbox("Colaborador para Raio-X e Simulador", names, index=default_idx)
        st.session_state.selected_name = selected
    else:
        st.caption("Nenhum colaborador com estes filtros.")

    st.markdown('<div class="pp-kpi-label" style="margin:16px 0 8px">Resumo da equipe filtrada</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="pp-mini"><small>Colaboradores</small><b>{len(view)}</b></div>
<div class="pp-mini"><small>Folha mensal</small><b>{esc(money(view['Salario_Mensal'].sum() if not view.empty else 0))}</b></div>
<div class="pp-mini" style="border-left:3px solid {CORAL}"><small>Risco financeiro exposto</small><b>{esc(money(view['Capital_Risco'].sum() if not view.empty else 0))}</b></div>
""", unsafe_allow_html=True)
    st.caption("Risco exposto = soma de custo de reposição × probabilidade estimada. Não é perda garantida.")
    st.divider()
    try:
        zip_bytes = source_zip()
        st.download_button("📦 Exportar Código para GitHub (.ZIP)", data=zip_bytes, file_name="pulso-preditivo.zip",
                           mime="application/zip", width="stretch")
    except Exception as e:  # noqa: BLE001
        st.caption(f"Exportação indisponível: {e}")

# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="pp-hero">{logo_svg(72)}
<div style="z-index:1"><div class="pp-eyebrow">Inteligência de pessoas · sessão local</div>
<div class="pp-title">PULSO PREDITIVO</div>
<div class="pp-sub">Radar de Engajamento, Carreira e Capital Humano</div></div>
<div class="pp-hero-meta">{len(calc)} colaboradores na base<br>{len(view)} no recorte atual<br>motor determinístico · sem IA externa</div>
</div>""", unsafe_allow_html=True)

if st.session_state.flash:
    kind, msg = st.session_state.flash
    (st.success if kind == "ok" else st.error)(msg)
    st.session_state.flash = None

tab1, tab2, tab3, tab4 = st.tabs(["📊 Cockpit Executivo", "🔍 Raio-X Individual", "🎛️ Simulador E Se?", "📝 Base & Metodologia"])

# ---------------------------------------------------------------- tab 1
with tab1:
    if view.empty:
        empty("Nenhum colaborador neste recorte", "Ajuste os filtros de departamento ou diagnóstico na barra lateral.")
    else:
        urg = int((view["Capacitacao"] >= 60).sum())
        fat_alert = int((view["Fadiga"] >= 60).sum())
        silent = view.loc[view["Diagnostico"] == "Evasão silenciosa", "Capital_Risco"].sum()
        c = st.columns(4)
        c[0].markdown(card_kpi("Risco médio de evasão", pct(view["Risco"].mean()), f"máx. {pct(view['Risco'].max())} na equipe", CORAL), unsafe_allow_html=True)
        c[1].markdown(card_kpi("Capital em risco", money(view["Capital_Risco"].sum()), f"{money(silent)} em evasão silenciosa", GOLD), unsafe_allow_html=True)
        c[2].markdown(card_kpi("Urgência de T&D", f"{urg} de {len(view)}", "necessidade de capacitação ≥ 60", GOLD), unsafe_allow_html=True)
        c[3].markdown(card_kpi("Índice de fadiga", pct(view["Fadiga"].mean()), f"{fat_alert} em alerta de saúde (≥ 60)", SAGE), unsafe_allow_html=True)

        section("Matriz 9-Box preditiva", "Entregas × probabilidade de evasão. O tamanho da bolha é o custo estimado de reposição.")
        fig = go.Figure()
        for x0, x1 in [(0, 60), (60, 80), (80, 100)]:
            for y0, y1 in [(0, 33), (33, 66), (66, 100)]:
                tone = CORAL if (y0 >= 66 and x0 >= 60) else (SAGE if (y0 == 0 and x0 >= 80) else SAND)
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color=LINE, width=1), fillcolor=tone, opacity=0.13, layer="below")
        mx = max(view["Custo_Reposicao"].max(), 1)
        for cat in CATEGORIES:
            d = view[view["Diagnostico"] == cat]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["Score_Entregas"], y=d["Risco"], mode="markers+text", name=cat,
                text=[n.split()[0] for n in d["Nome"]], textposition="top center", textfont=dict(size=10, color=MUTED),
                marker=dict(size=18 + 42 * np.sqrt(d["Custo_Reposicao"] / mx), color=CAT_COLORS[cat], opacity=0.82, line=dict(color="#FFFFFF", width=2)),
                customdata=np.stack([d["Nome"], d["Cargo"], [money(v) for v in d["Custo_Reposicao"]]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Entregas %{x:.0f}% · Risco %{y:.1f}%<br>Reposição %{customdata[2]}<extra></extra>"))
        fig.update_xaxes(title="Score de entregas (%)", range=[0, 102])
        fig.update_yaxes(title="Probabilidade de evasão em 6 meses (%)", range=[0, 102])
        st.plotly_chart(style_fig(fig, 520), width="stretch")

        section("Sinais por departamento", "Médias por pessoa: ausências de saúde, ausências não justificadas e horas de capacitação.")
        g = view.groupby("Departamento")[["Faltas_Com_Atestado_Ano", "Faltas_Sem_Atestado_Ano", "Horas_Treinamento_Ano"]].mean().reset_index()
        f2 = style_fig(go.Figure(), 400)
        f2.add_bar(x=g["Departamento"], y=g["Faltas_Com_Atestado_Ano"], name="Faltas com atestado (saúde)", marker_color=GOLD)
        f2.add_bar(x=g["Departamento"], y=g["Faltas_Sem_Atestado_Ano"], name="Faltas sem atestado (sinal exploratório)", marker_color=CORAL)
        f2.add_trace(go.Scatter(x=g["Departamento"], y=g["Horas_Treinamento_Ano"], name="Horas de capacitação", yaxis="y2",
                                mode="lines+markers", line=dict(color=SAGE, width=3), marker=dict(size=10)))
        f2.update_layout(barmode="group", bargap=0.35, yaxis=dict(title="Dias/ano (média)"),
                         legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                         margin=dict(t=70, b=60),
                         xaxis=dict(tickangle=0, tickmode="array",
                                    tickvals=g["Departamento"].tolist(),
                                    ticktext=[dep.replace(" & ", "<br>& ") for dep in g["Departamento"]],
                                    automargin=True),
                         yaxis2=dict(title="Horas/ano", overlaying="y", side="right",
                                     showgrid=False, tickformat=".0f", automargin=True))
        st.plotly_chart(f2, width="stretch")
        st.markdown(f'<div class="pp-caveat">{CAVEAT}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers per employee
row = view[view["Nome"] == selected].iloc[0] if selected else None


def radar_values(r):
    return [
        float(r["Score_Entregas"]),
        float(100 - r["Estagnacao"]),
        float(100 - r["Fadiga"]),
        float(100 - r["Deficit"]),
        float(100 - r["Ruptura"]),
    ]


# ---------------------------------------------------------------- tab 2
with tab2:
    if row is None:
        empty("Selecione um colaborador", "Nenhum colaborador disponível com os filtros atuais.")
    else:
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown(f"""<div class="pp-card">
<div class="pp-kpi-label">Dossiê executivo</div>
<div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:600;margin:6px 0 2px">{esc(row['Nome'])}</div>
<div style="color:{MUTED};margin-bottom:10px">{esc(row['Cargo'])} · {esc(row['Departamento'])}</div>
{chip(row['Diagnostico'])}
<div class="pp-grid">
<div><small>Tempo de casa</small><b>{int(row['Tempo_Casa_Meses'])} meses</b></div>
<div><small>Sem promoção</small><b>{int(row['Meses_Sem_Promocao'])} meses</b></div>
<div><small>Promoções</small><b>{int(row['Promocoes_Carreira'])}</b></div>
<div><small>Entregas</small><b>{float(row['Score_Entregas']):.0f}%</b></div>
<div><small>Treinamento</small><b>{float(row['Horas_Treinamento_Ano']):.0f} h/ano</b></div>
<div><small>Reposição</small><b>{esc(money(row['Custo_Reposicao']))}</b></div>
<div><small>Atestados</small><b>{int(row['Faltas_Com_Atestado_Ano'])} dias</b></div>
<div><small>Sem atestado</small><b>{int(row['Faltas_Sem_Atestado_Ano'])} dias</b></div>
<div><small>Salário</small><b>{esc(money(row['Salario_Mensal']))}</b></div>
</div></div>""", unsafe_allow_html=True)
        with right:
            risk = float(row["Risco"])
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk, number=dict(suffix="%", valueformat=".1f", font=dict(family="Fraunces, serif", size=46, color=FOREST)),
                title=dict(text="Probabilidade de saída silenciosa em 6 meses", font=dict(size=13, color=MUTED)),
                gauge=dict(axis=dict(range=[0, 100], tickcolor=LINE), bar=dict(color=FOREST, thickness=0.22), bgcolor="#FFFFFF", borderwidth=0,
                           steps=[dict(range=[0, 35], color="#DCEBE2"), dict(range=[35, 60], color="#F6E6C4"), dict(range=[60, 100], color="#F7D3CC")],
                           threshold=dict(line=dict(color=CORAL, width=3), value=risk))))
            st.plotly_chart(style_fig(gauge, 330), width="stretch")

        axes = ["Consistência de Entregas", "Progressão de Carreira", "Vitalidade/Assiduidade", "Investimento em Capacitação", "Engajamento Comportamental"]
        c1, c2 = st.columns([1, 1.1])
        with c1:
            section("Perfil 5D", "Colaborador vs. meta saudável (quanto maior, melhor).")
            rv = radar_values(row)
            rf = go.Figure()
            rf.add_trace(go.Scatterpolar(r=[80] * 5 + [80], theta=axes + axes[:1], name="Meta saudável", line=dict(color=SAGE, dash="dot", width=2), fill="toself", fillcolor="rgba(46,125,91,.08)"))
            rf.add_trace(go.Scatterpolar(r=rv + rv[:1], theta=axes + axes[:1], name=row["Nome"].split()[0], line=dict(color=CORAL, width=3), fill="toself", fillcolor="rgba(224,90,71,.18)"))
            rf.update_layout(polar=dict(bgcolor="#FFFFFF", radialaxis=dict(range=[0, 100], gridcolor="#EFEAE0", tickfont=dict(size=9)), angularaxis=dict(gridcolor="#EFEAE0", tickfont=dict(size=11))))
            st.plotly_chart(style_fig(rf, 430), width="stretch")
        with c2:
            section("Laudo de sinais não-verbalizados", "Por que este perfil está aqui — e o que fazer.")
            reasons = "".join(f"<li>{esc(r)}</li>" for r in explain(row))
            steps = "".join(f"<li><b>Passo {i}.</b> {esc(s)}</li>" for i, s in enumerate(action_plan(row), 1))
            st.markdown(f"""<div class="pp-card"><div class="pp-kpi-label">Leitura do motor</div><ul class="pp-list">{reasons}</ul>
<div class="pp-kpi-label" style="margin-top:16px">Plano de ação recomendado</div><ul class="pp-list pp-steps">{steps}</ul></div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="pp-caveat">{CAVEAT}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 3
with tab3:
    if row is None:
        empty("Selecione um colaborador", "O simulador precisa de um colaborador no recorte atual.")
    else:
        rec = recommend(row)
        colab_nome = str(row["Nome"])
        k_td = f"sim_td_{colab_nome}"
        k_merito = f"sim_merito_{colab_nome}"
        k_alivio = f"sim_alivio_{colab_nome}"
        k_promo = f"sim_promo_{colab_nome}"
        for key, initial in ((k_td, 0), (k_merito, 0), (k_alivio, 0), (k_promo, False)):
            if key not in st.session_state:
                st.session_state[key] = initial

        parts = []
        if rec["hours"]:
            parts.append(f"+{rec['hours']}h de capacitação")
        if rec["merit"]:
            parts.append(f"+{rec['merit']}% de progressão/mérito")
        if rec["relief"]:
            parts.append(f"{rec['relief']}% de alívio de carga")
        if rec["promotion"]:
            parts.append("promoção de cargo/nível")
        action_txt = ", ".join(parts) if parts else "manter acompanhamento sem intervenção estrutural"
        target = "atinge a meta de estabilização" if rec["target_met"] else "não atinge sozinha a meta — combine com conversas de carreira"
        rationale = " ".join(rec["reasons"])
        st.markdown(f"""<div class="pp-reco"><div class="pp-eyebrow">💡 Prescrição matemática sugerida · antes de você ajustar</div>
<h4>{esc(row['Nome'])} — {esc(row['Diagnostico'])}</h4>
<p>Com entregas de {float(row['Score_Entregas']):.0f}%, {int(row['Meses_Sem_Promocao'])} meses sem promoção e {float(row['Horas_Treinamento_Ano']):.0f}h de treino no ano,
sugerimos <b>{esc(action_txt)}</b>. Consideramos também o índice de fadiga de {float(row['Fadiga']):.0f}/100 e a assiduidade. A estimativa reduz o risco de <b>{pct(rec['before'])}</b> para <b>{pct(rec['after'])}</b>, com custo anual aproximado de {esc(money(rec['cost']))}; a prescrição {target}.</p>
<p style="margin-top:10px;font-size:.87rem">{esc(rationale)}</p>
<div class="pp-pills"><span>+{rec['hours']}h T&amp;D</span><span>+{rec['merit']}% mérito</span><span>{rec['relief']}% alívio</span><span>{'com' if rec['promotion'] else 'sem'} promoção</span></div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([3, 2, 3])
        b1.button(
            "✨ Aplicar Prescrição Sugerida nos Controles", key="apply_prescription",
            on_click=aplicar_prescricao,
            args=(k_td, rec["hours"], k_merito, rec["merit"],
                  k_alivio, rec["relief"], k_promo, rec["promotion"]),
            type="primary", use_container_width=True, wrap=True,
        )
        b2.button(
            "↺ Zerar Controles", key="reset_prescription", on_click=zerar_controles,
            args=(k_td, k_merito, k_alivio, k_promo), use_container_width=True, wrap=True,
        )

        section("Alavancas de retenção", "Simulação isolada: nada aqui altera a base de colaboradores.")
        s1, s2, s3 = st.columns(3)
        with s1:
            hours = st.slider("Horas adicionais de treinamento", 0, 80, key=k_td, format="+%dh")
        with s2:
            merit = st.slider("Reconhecimento / progressão salarial", 0, 30, key=k_merito, format="%d%%")
            promo = st.checkbox("Conceder promoção de cargo/nível", key=k_promo)
            st.caption("Promoção considera reajuste mínimo de 8% no custo anual, mesmo com mérito inferior.")
        with s3:
            relief = st.slider("Redesenho de carga (alívio de estafa)", 0, 50, key=k_alivio, format="%d%%")

        sim = simulate(row, hours=hours, merit=merit, relief=relief, promotion=promo)
        drop = sim["before"] - sim["after"]
        k = st.columns(4)
        k[0].markdown(card_kpi("Risco de saída", f"{pct(sim['before'])} → {pct(sim['after'])}", f"queda de {pct(drop)} pontos", CORAL), unsafe_allow_html=True)
        k[1].markdown(card_kpi("Entregas projetadas", f"{sim['projected_score']:.1f}%".replace(".", ","), f"hoje {float(row['Score_Entregas']):.0f}% · curva logarítmica", SAGE), unsafe_allow_html=True)
        k[2].markdown(card_kpi("Custo da ação (ano)", money(sim["cost"]), f"exposição evitada {money(sim['avoided'])}", GOLD), unsafe_allow_html=True)
        roi_c = SAGE if sim["net"] >= 0 else CORAL
        k[3].markdown(card_kpi("Saldo líquido estimado", money(sim["net"]), f"ROI {sim['roi']:.0f}%".replace(".", ","), roi_c), unsafe_allow_html=True)

        g1, g2 = st.columns([1, 1.5])
        with g1:
            gb = go.Figure()
            gb.add_bar(y=["Depois", "Antes"], x=[sim["after"], sim["before"]], orientation="h", marker_color=[SAGE, CORAL],
                       text=[pct(sim["after"]), pct(sim["before"])], textposition="outside")
            gb.update_xaxes(range=[0, 110], title="Probabilidade de saída (%)")
            gb.update_layout(title=dict(text="Medidor antes × depois", font=dict(size=14)), showlegend=False)
            st.plotly_chart(style_fig(gb, 300), width="stretch")
            fin = go.Figure()
            fin.add_bar(x=["Custo da ação", "Exposição evitada", "Saldo"], y=[sim["cost"], sim["avoided"], sim["net"]],
                        marker_color=[GOLD, SAGE, roi_c], text=[money(v) for v in (sim["cost"], sim["avoided"], sim["net"])], textposition="outside")
            fin.update_layout(title=dict(text="Balanço financeiro da retenção", font=dict(size=14)), showlegend=False)
            st.plotly_chart(style_fig(fin, 300), width="stretch")
        with g2:
            months = np.arange(0, 13)

            def cum(p6):
                p6 = min(max(p6 / 100, 1e-4), 0.9999)
                h = -np.log(1 - p6) / 6
                return 100 * (1 - np.exp(-h * months))

            tf = go.Figure()
            tf.add_trace(go.Scatter(x=months, y=cum(sim["before"]), name="Trajetória sem ação", line=dict(color=CORAL, width=3), fill="tozeroy", fillcolor="rgba(224,90,71,.08)"))
            tf.add_trace(go.Scatter(x=months, y=cum(sim["after"]), name="Trajetória com sua intervenção", line=dict(color=SAGE, width=3), fill="tozeroy", fillcolor="rgba(46,125,91,.10)"))
            tf.update_xaxes(title="Meses", dtick=1)
            tf.update_yaxes(title="Risco acumulado de saída (%)", range=[0, 100])
            tf.update_layout(title=dict(text="Curva de risco acumulado em 12 meses", font=dict(size=14)))
            st.plotly_chart(style_fig(tf, 616), width="stretch")
        st.caption("Exposição evitada = custo de reposição × redução de probabilidade. É um valor ponderado por probabilidade, não economia garantida.")

        section("📋 Relatório executivo da simulação", "Parecer para diretoria — atualiza conforme os controles.")
        if sim["after"] <= 25:
            seal, scol, sbg = "Retenção Estabilizada", SAGE, "#E7F2EC"
            verdict = "Os ajustes escolhidos levam o risco estimado para a faixa de estabilidade. O foco passa a ser sustentar a combinação e medir sinais a cada ciclo."
        elif drop > 0 and (drop >= 0.3 * max(sim["before"], 1) or sim["after"] <= 50):
            seal, scol, sbg = "Risco Mitigado Parcialmente — Veja o que falta", GOLD, "#FBF1DC"
            gaps = []
            if hours < rec["hours"]:
                gaps.append(f"mais {rec['hours'] - hours}h de capacitação")
            if merit < rec["merit"]:
                gaps.append(f"mais {rec['merit'] - merit}% de mérito")
            if relief < rec["relief"]:
                gaps.append(f"mais {rec['relief'] - relief}% de alívio de carga")
            if rec["promotion"] and not promo:
                gaps.append("a promoção de nível")
            verdict = "A intervenção reduz o risco, mas não o estabiliza. Para chegar à prescrição sugerida faltam: " + (", ".join(gaps) if gaps else "ajustes finos em conversa direta com a pessoa") + "."
        else:
            seal, scol, sbg = "Atenção: Intervenção Insuficiente para Evitar Evasão", CORAL, "#FBE6E2"
            verdict = "As alavancas atuais quase não movem a probabilidade estimada. Revise o desenho: comece aplicando a prescrição sugerida e ajuste a partir dela."
        plan_30 = "Conversa individual de carreira e escuta ativa sobre carga e expectativas; acordar metas de desenvolvimento por escrito."
        if relief:
            plan_30 += f" Redistribuir cerca de {relief}% das demandas críticas."
        plan_60 = (f"Iniciar trilha de capacitação de {hours}h com mentor designado." if hours else "Mapear lacunas de competência com o gestor direto.") + \
                  (f" Formalizar progressão de {merit}% no próximo ciclo." if merit else "")
        plan_90 = ("Efetivar a mudança de cargo/nível e comunicar o novo escopo. " if promo else "") + \
                  "Reavaliar sinais (entregas, ausências, treinamento) e recalcular o risco no Pulso Preditivo."
        st.markdown(f"""<div class="pp-report">
<div class="pp-seal" style="background:{sbg};border-color:{scol}66"><div class="pp-kpi-label" style="color:{scol}">Selo do parecer</div>
<div style="font-family:'Fraunces',serif;font-size:1.35rem;font-weight:600;color:{scol}">{esc(seal)}</div></div>
<p><b>Colaborador:</b> {esc(row['Nome'])} · {esc(row['Cargo'])} · {esc(row['Departamento'])}<br>
<b>Risco estimado:</b> {pct(sim['before'])} → {pct(sim['after'])} · <b>Entregas projetadas:</b> {sim['projected_score']:.1f}% ·
<b>Investimento:</b> {esc(money(sim['cost']))} · <b>Exposição evitada:</b> {esc(money(sim['avoided']))} · <b>Saldo:</b> {esc(money(sim['net']))}</p>
<h5>Avaliação</h5><p>{esc(verdict)}</p>
<h5>Plano tático</h5><ul>
<li><b>30 dias —</b> {esc(plan_30)}</li><li><b>60 dias —</b> {esc(plan_60)}</li><li><b>90 dias —</b> {esc(plan_90)}</li></ul>
<p style="color:{MUTED};font-size:.82rem;margin-top:14px">Estimativas heurísticas, não clínicas e não validadas. Uso exclusivo para apoio ao desenvolvimento — jamais para decisões punitivas.</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 4
with tab4:
    section("Base de colaboradores", "Edite qualquer célula, adicione ou remova linhas. Ao confirmar a edição, todo o sistema recalcula.")
    st.markdown(f'<div class="pp-caveat">{CAVEAT}</div>', unsafe_allow_html=True)
    src = base[FIELDS].reset_index(drop=True)
    st.data_editor(
        src, key=editor_key(), on_change=apply_editor_changes, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "Nome": st.column_config.TextColumn("Nome", required=True, max_chars=60),
            "Cargo": st.column_config.TextColumn("Cargo", required=True, max_chars=60),
            "Departamento": st.column_config.SelectboxColumn("Departamento", options=DEPARTMENTS, required=True),
            "Salario_Mensal": st.column_config.NumberColumn("Salário mensal (R$)", min_value=4500, max_value=22000, step=100, format="R$ %d"),
            "Tempo_Casa_Meses": st.column_config.NumberColumn("Tempo de casa (m)", min_value=1, max_value=480, step=1),
            "Meses_Sem_Promocao": st.column_config.NumberColumn("Sem promoção (m)", min_value=0, max_value=480, step=1),
            "Promocoes_Carreira": st.column_config.NumberColumn("Promoções", min_value=0, max_value=10, step=1),
            "Score_Entregas": st.column_config.NumberColumn("Entregas (%)", min_value=0, max_value=100, step=1),
            "Faltas_Com_Atestado_Ano": st.column_config.NumberColumn("Com atestado", min_value=0, max_value=60, step=1),
            "Faltas_Sem_Atestado_Ano": st.column_config.NumberColumn("Sem atestado", min_value=0, max_value=60, step=1),
            "Horas_Treinamento_Ano": st.column_config.NumberColumn("Treino (h)", min_value=0, max_value=400, step=1),
        })

    section("Adicionar colaborador", "Formulário rápido com limites validados.")
    with st.form("add_form", clear_on_submit=True):
        a1, a2, a3 = st.columns(3)
        nome = a1.text_input("Nome completo", max_chars=60)
        cargo = a2.text_input("Cargo", max_chars=60)
        dep = a3.selectbox("Departamento", DEPARTMENTS)
        b1_, b2_, b3_, b4_ = st.columns(4)
        sal = b1_.number_input("Salário mensal (R$)", 4500, 22000, 8000, step=100)
        tc = b2_.number_input("Tempo de casa (meses)", 1, 480, 24)
        sp = b3_.number_input("Meses sem promoção", 0, 480, 12)
        pr = b4_.number_input("Promoções", 0, 10, 1)
        c1_, c2_, c3_, c4_ = st.columns(4)
        sc = c1_.number_input("Score de entregas (%)", 0, 100, 75)
        fa = c2_.number_input("Faltas com atestado/ano", 0, 60, 2)
        fs = c3_.number_input("Faltas sem atestado/ano", 0, 60, 0)
        ht = c4_.number_input("Horas de treinamento/ano", 0, 400, 20)
        if st.form_submit_button("Adicionar à base"):
            if not nome.strip() or not cargo.strip():
                st.error("Informe nome e cargo.")
            elif nome.strip().lower() in set(base["Nome"].str.strip().str.lower()):
                st.error("Já existe um colaborador com esse nome. Use um nome único.")
            elif sp > tc:
                st.error("Meses sem promoção não pode exceder o tempo de casa.")
            else:
                new = pd.DataFrame([{"Nome": nome.strip(), "Cargo": cargo.strip(), "Departamento": dep, "Salario_Mensal": sal,
                                     "Tempo_Casa_Meses": tc, "Meses_Sem_Promocao": sp, "Promocoes_Carreira": pr, "Score_Entregas": sc,
                                     "Faltas_Com_Atestado_Ano": fa, "Faltas_Sem_Atestado_Ano": fs, "Horas_Treinamento_Ano": ht}])[FIELDS]
                try:
                    st.session_state.base = validate_data(pd.concat([base[FIELDS], new], ignore_index=True))
                    st.session_state.editor_ver += 1
                    st.session_state.flash = ("ok", f"{nome.strip()} adicionado(a). Painéis recalculados.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    confirm_reset = st.checkbox("Confirmo descartar minhas edições e restaurar a base demonstrativa.")
    r1, _ = st.columns([1, 3])
    if r1.button("Restaurar dados demonstrativos", width="stretch", disabled=not confirm_reset):
        st.session_state.base = validate_data(seed_data())
        st.session_state.editor_ver += 1
        st.session_state.flash = ("ok", "Base demonstrativa restaurada.")
        st.rerun()

    section("Indicadores calculados", "Somente leitura — derivados pelo motor a cada alteração.")
    out = calc[["Nome", "Departamento", "Risco", "Fadiga", "Ruptura", "Estagnacao", "Capacitacao", "Custo_Reposicao", "Capital_Risco", "Diagnostico"]]
    st.dataframe(out, width="stretch", hide_index=True, column_config={
        "Risco": st.column_config.ProgressColumn("Risco 6m", min_value=0, max_value=100, format="%.1f%%"),
        "Fadiga": st.column_config.NumberColumn("Fadiga", format="%.0f"),
        "Ruptura": st.column_config.NumberColumn("Ruptura", format="%.0f"),
        "Estagnacao": st.column_config.NumberColumn("Estagnação", format="%.0f"),
        "Capacitacao": st.column_config.NumberColumn("Capacitação", format="%.0f"),
        "Custo_Reposicao": st.column_config.NumberColumn("Reposição (R$)", format="%.0f"),
        "Capital_Risco": st.column_config.NumberColumn("Capital em risco (R$)", format="%.0f"),
    })

    with st.expander("📐 Metodologia Matemática (Transparência Algorítmica)"):
        st.markdown("Todos os componentes são normalizados em [0, 1]. $S$ = entregas/100, $A_c$ / $A_s$ = faltas com / sem atestado, $M$ = meses sem promoção, $T$ = tempo de casa, $H$ = horas de treino, $W$ = salário mensal.")
        st.markdown("**1. Índice de Bradford modificado — fadiga (saúde) separada da ruptura silenciosa**")
        st.latex(r"F = \mathrm{clip}\!\left[\left(\tfrac{A_c}{18}\right)^{1{,}35}(0{,}65+0{,}35S)\right] \qquad R = \frac{e^{\min(A_s,8)/2{,}5}-1}{e^{8/2{,}5}-1}")
        st.markdown("**2. Compressão e estagnação de carreira**")
        st.latex(r"c=\tfrac{M}{\max(T,1)},\; o=\mathrm{clip}_{[0,2]}\tfrac{M-24}{24},\; E=\mathrm{clip}\!\left[0{,}30c^{1{,}5}+0{,}55\,o^{1{,}4}(0{,}4+0{,}6S)+0{,}25\,\mathbb{1}[S>0{,}8\wedge M>24\wedge H<20]\right]")
        st.markdown("**3. Déficit e necessidade de capacitação**")
        st.latex(r"D=\mathrm{clip}\left(1-\tfrac{H}{40}\right) \qquad N=\mathrm{clip}\left[1{,}8\,D\,(0{,}65(1-S)+0{,}35)\right]")
        st.markdown("**4. Probabilidade logística de evasão em 6 meses**")
        st.latex(r"P=\frac{1}{1+e^{-z}},\qquad z=-3{,}5+3E+2R+2F+1{,}2D")
        st.markdown("**5. Custo de reposição — rescisão (1,5 salário) + recrutamento (1) + integral da rampagem linear**")
        st.latex(r"T_r = 3+3\tfrac{W}{22000}+\min\left(\tfrac{T}{96},1\right),\qquad C = 2{,}5W+\int_0^{T_r}W\left(1-\tfrac{t}{T_r}\right)dt = W\left(2{,}5+\tfrac{T_r}{2}\right)")
        st.markdown("**6. Cenário contrafactual (projeção, não observação)**")
        st.markdown("Nesta equação, $S_0$ e $S'$ são entregas em pontos percentuais (0–100), $m$ é mérito em % (0–30), $r$ é alívio em % (0–50), e $\\pi$ vale 1 quando há promoção.")
        st.latex(r"S'=S_0+(100-S_0)\cdot0{,}45\,\frac{\ln(1+\Delta H/10)}{\ln 9},\quad M'=\begin{cases}0 & \text{promoção}\\ M e^{-m/12}&\text{sem promoção}\end{cases},\quad A_c'=A_c(1-r/100),\quad A_s'=A_s e^{-\Delta H/100-m/25-0{,}5\,\pi}")
        st.latex(r"\text{Custo}=150\,\Delta H+12W\cdot\tfrac{\max(m,\,8\pi)}{100}+0{,}6W\tfrac{r}{100}, \qquad \text{Evitado}=C\,(P_0-P_1), \qquad ROI=\tfrac{\text{Evitado}-\text{Custo}}{\text{Custo}}")
        st.markdown("**7. Prescrição guiada de RH** — sempre apresenta um plano acionável, não uma otimização financeira. Se treino <30h ou entregas <80%, sugere +24 a +56h de T&D. Se meses sem promoção ≥24 ou entregas ≥85%, sugere +10% a +20% de mérito; promoção se meses ≥30. Se faltas com atestado ≥5 ou índice exploratório de fadiga >0, sugere 20% a 40% de alívio. Nas alavancas sem esses gatilhos, usa prevenção de +16h de T&D e +5% de mérito, sem alívio obrigatório.")
        st.latex(r"g_H=\max\left(\mathrm{clip}\tfrac{30-H}{30},\mathrm{clip}\tfrac{80-S_0}{80}\right),\quad \Delta H=24+4\lceil8g_H\rceil")
        st.latex(r"g_M=\max\left(\mathrm{clip}\tfrac{M-24}{36},\mathrm{clip}\tfrac{S_0-85}{15}\right),\quad m=10+\lceil10g_M\rceil")
        st.latex(r"g_F=\max\left(\mathrm{clip}\tfrac{A_c}{18},F\right),\quad r=20+5\lceil4g_F\rceil")
        st.caption("As fórmulas de cada faixa só se aplicam quando seu gatilho está ativo; clip limita a [0,1]. O risco depois e o custo da prescrição usam exatamente o mesmo simulador dos controles. A política não garante risco ≤25%, retenção ou retorno positivo; os valores são hipóteses para discussão com RH.")
        st.latex(r"P(t)=1-(1-P_{6})^{t/6}")
        st.caption("A trajetória de 12 meses supõe taxa de risco constante; não usa histórico longitudinal. O Bradford clássico B = episódios² × dias não pode ser calculado sem episódios de ausência: aqui usamos apenas uma adaptação heurística com dias agregados. Os coeficientes são ilustrativos, sem calibração empírica. Não constituem modelo validado, diagnóstico clínico nem base para decisões disciplinares.")