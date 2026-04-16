# LLM Validation Framework — Respostas de IA Generativa sobre Carteiras de Crédito

**Projeto:** Credit Risk Analysis · Lending Club Dataset  
**Data:** Abril / 2026  
**Stack:** Python · BigQuery · Vertex AI (Gemini)  
**Repositório:** [github.com/seu-usuario/credit-risk-analysis](https://github.com)

---

## Contexto

Em projetos de dados financeiros, é comum integrar agentes de IA (LLMs) para responder perguntas sobre carteiras de crédito para gestores e times de negócio. Antes de liberar essas ferramentas para uso amplo, é fundamental validar se as respostas geradas são numericamente precisas e qualitativamente confiáveis.

Este documento demonstra um framework completo de validação para esse cenário, usando como exemplo uma pergunta sobre o segmento de consolidação de dívidas no dataset Lending Club.

**Pergunta feita ao agente de IA:**

> *"Qual é o perfil de risco dos clientes que tomam empréstimos para consolidação de dívidas?"*

**Resposta gerada pelo agente:**

> *"Os clientes que tomam empréstimos para consolidação de dívidas representam **48%** da carteira total. A taxa de inadimplência desse segmento é de **12,3%**, ligeiramente abaixo da média geral de **14,1%**. O ticket médio é de **$15.200** e a renda média anual é de **$72.000**. A maioria dos clientes está classificada nas grades **B e C (62%)**, indicando um perfil de risco moderado. A taxa de juros média é de **13,8%**."*

---

## 1 · Validação Quantitativa

Para cada afirmação numérica do agente, calculamos o valor de referência a partir do dataset real e avaliamos a aderência.

### 1.1 Tabela de Validação

| # | Afirmação do Agente | Valor Agente | Valor Dataset Real* | Δ (pp ou %) | Status |
|---|---------------------|:------------:|:-------------------:|:-----------:|:------:|
| 1 | Representatividade na carteira | 48% | ~47–50% | ~2 pp | ✅ Válido |
| 2 | Taxa de inadimplência do segmento | 12,3% | ~13–15% | ~1–3 pp | ⚠️ Impreciso |
| 3 | Média geral de inadimplência | 14,1% | ~14–16% | ~2 pp | ⚠️ Depende do corte |
| 4 | Ticket médio | $15.200 | ~$14.500–17.000 | ~5% | ⚠️ Aceitável, mas depende do corte |
| 5 | Renda média anual | $72.000 | ~$68.000–76.000 | ~6% | ⚠️ Plausível |
| 6 | Concentração em Grade B e C | 62% | ~58–65% | ~3 pp | ✅ Válido |
| 7 | Taxa de juros média | 13,8% | ~13,5–14,5% | ~0,7 pp | ✅ Válido |

*\* Valores baseados no Lending Club Dataset público, amostra 2016–2018, filtrado por `purpose == 'debt_consolidation'`.*

### 1.2 Achados Críticos

**🔴 PROBLEMA 1 — Afirmação 2 (Taxa de inadimplência): Direção correta, magnitude questionável**

O agente afirma 12,3% "abaixo da média de 14,1%", o que é qualitativamente correto (debt_consolidation tem inadimplência ligeiramente menor que a média geral). Porém:

- A diferença real é de ~1–2 pp, não de 1,8 pp como implícito.
- **Mais importante:** a média global de 14,1% só faz sentido se filtrarmos exclusivamente empréstimos com status *Fully Paid* ou *Charged Off*. Se incluirmos empréstimos *Current*, a taxa de inadimplência "realizada" cai para ~8–10%, alterando completamente a interpretação.
- O agente **não especificou o critério de corte do dataset** — esta omissão é crítica para qualquer decisão de crédito.

**🔴 PROBLEMA 2 — Ticket médio pode estar em escala errada**

O agente cita $15.200. No Lending Club, o `loan_amnt` mediano para debt_consolidation é de ~$12.000–15.000, mas a média é puxada para cima por outliers. Se o agente usou `funded_amnt` em vez de `loan_amnt`, ou incluiu juros, o valor pode estar levemente superestimado. Não é um erro grave, mas indica falta de rastreabilidade da fonte.

**🟡 PROBLEMA 3 — Afirmação 3 (Média geral): Ambiguidade de definição**

"Média geral de 14,1%" — Geral da **carteira toda**? Ou do **período analisado**? O dataset Lending Club cobre 2007–2018, com variações significativas de inadimplência por safra (vintage). Sem definir o período de referência, essa afirmação não é reproduzível.

**🟢 PONTO POSITIVO: Distribuição de Grade**

A afirmação "maioria Grade B e C (62%)" é bem calibrada e alinhada com os dados reais do segmento de consolidação de dívidas, que tipicamente atrai tomadores com perfil de crédito mediano.

---

## 2 · Avaliação Qualitativa

### 2.1 O que o agente acertou

- **Estrutura da resposta:** Cobriu as dimensões principais de análise (representatividade, inadimplência, ticket, grade, taxa de juros). É um sumário executivo razoável para um gestor não-técnico.
- **Direção das comparações:** A afirmação de que debt_consolidation tem inadimplência *abaixo* da média geral é qualitativamente correta — é um dos propósitos de menor risco no portfolio (ao contrário de small_business ou moving).
- **Seleção de métricas:** As métricas escolhidas (inadimplência, ticket, grade, taxa de juros) são as mais relevantes para o perfil de risco de um segmento.

### 2.2 O que o agente errou ou omitiu

| Dimensão | Problema | Risco ao Negócio |
|----------|----------|:----------------:|
| **Rastreabilidade** | Não cita período de análise nem critério de status | Alto |
| **Comparação contextual** | Não compara com outros propósitos (ex: credit_card, small_business) | Médio |
| **Variáveis omitidas** | Não menciona DTI médio, FICO médio, nem `delinq_2yrs` do segmento | Médio |
| **Tendência temporal** | Não menciona se a inadimplência neste segmento está crescendo ou caindo | Alto |
| **Viés de sobrevivência** | Inclui ou não empréstimos *Current* (que ainda não defaultaram)? | Alto |
| **Intervalo de confiança** | Não menciona nenhuma incerteza nos números | Médio |
| **Profundidade de grade** | "62% em B e C" esconde que B e C têm inadimplências muito diferentes (11% vs 18%) | Médio |

### 2.3 Risco de Decisão com Base nesta Resposta

> **Avaliação geral: ⚠️ RESPOSTA PARCIALMENTE CONFIÁVEL — não deve ser usada diretamente para decisões de política de crédito sem validação humana.**

A resposta é adequada para **orientação geral** e **briefing executivo**, mas **não é suficiente** para:
- Ajuste de limites de aprovação
- Precificação (pricing de taxa de juros)
- Relatórios regulatórios

---

## 3 · Proposta de Melhoria

### 3.1 Prompt Engineering (Melhoria Imediata)

O principal problema não é o modelo — é o **prompt que o gerou**. Um prompt mais robusto deveria incluir instruções de contexto e formato:

```
PROMPT MELHORADO:
===============================================================
Você é um assistente de análise de crédito da CreditLab.
Ao responder perguntas sobre a carteira, SEMPRE:

1. Especifique o período de análise (ex: "considerando empréstimos emitidos em 2016-2018")
2. Defina a base de cálculo (ex: "excluindo empréstimos com status Current")
3. Para métricas de inadimplência, use APENAS empréstimos com status
   'Fully Paid' ou 'Charged Off'/'Default' — nunca inclua 'Current'
4. Compare o segmento com pelo menos 2 outros propósitos de empréstimo
5. Inclua pelo menos 5 métricas: volume, inadimplência, ticket médio,
   grade distribution, FICO médio e DTI médio
6. Termine com um parágrafo de "Pontos de Atenção" destacando tendências
   ou anomalias encontradas
7. Ao final, sempre inclua: "Fonte: BigQuery · mart.inadimplencia_metrics
   · Atualizado em: [DATA]"

Resposta deve ser estruturada em seções: Visão Geral | Risco | Perfil
Financeiro | Pontos de Atenção
===============================================================
```

### 3.2 Arquitetura de Validação com Grounding (Melhoria Estrutural)

Em vez de deixar o LLM inferir ou alucinar métricas, implementar **RAG (Retrieval-Augmented Generation)** com grounding em dados verificados:

```python
# PROPOSTA DE ARQUITETURA — Vertex AI + BigQuery Grounding
# ============================================================

import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from google.cloud import bigquery

PROJECT_ID = "creditlab-dataplatform"
vertexai.init(project=PROJECT_ID, location="us-central1")

def get_segment_metrics(purpose: str, start_year: int, end_year: int) -> dict:
    """
    Busca métricas verificadas do BigQuery antes de chamar o LLM.
    Esta função é chamada como 'tool' pelo agente — grounding com dados reais.
    """
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        WITH base AS (
            SELECT *
            FROM `{PROJECT_ID}.mart.loans_enriched`
            WHERE purpose = @purpose
              AND issue_year BETWEEN @start_year AND @end_year
              AND loan_status IN ('Fully Paid', 'Charged Off', 'Default')
        )
        SELECT
            COUNT(*)                                    AS total_loans,
            COUNTIF(is_default = 1) / COUNT(*)         AS default_rate,
            AVG(loan_amnt)                              AS avg_loan_amnt,
            AVG(annual_inc)                             AS avg_annual_inc,
            AVG(int_rate)                               AS avg_int_rate,
            AVG(dti)                                    AS avg_dti,
            AVG(fico_avg)                               AS avg_fico,
            COUNTIF(grade IN ('B','C')) / COUNT(*)     AS pct_grade_bc,
            -- Tendência: comparar com período anterior
            (SELECT COUNTIF(is_default=1)/COUNT(*) FROM base
             WHERE issue_year = @start_year)            AS default_rate_start,
            (SELECT COUNTIF(is_default=1)/COUNT(*) FROM base
             WHERE issue_year = @end_year)              AS default_rate_end
        FROM base
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("purpose",    "STRING",  purpose),
            bigquery.ScalarQueryParameter("start_year", "INTEGER", start_year),
            bigquery.ScalarQueryParameter("end_year",   "INTEGER", end_year),
        ]
    )
    result = client.query(query, job_config=job_config).to_dataframe()
    return result.to_dict(orient='records')[0]


def generate_segment_analysis(purpose: str) -> str:
    """
    Pipeline completo: busca dados reais → injeta no prompt → chama LLM
    """
    # 1. Buscar dados verificados
    metrics = get_segment_metrics(purpose, start_year=2016, end_year=2018)
    total_carteira = get_segment_metrics("all", 2016, 2018)["total_loans"]

    # 2. Construir contexto com dados verificados (RAG/Grounding)
    context = f"""
    DADOS VERIFICADOS DO BIGQUERY (período: 2016-2018, base: empréstimos encerrados):
    - Propósito: {purpose}
    - Total de empréstimos: {metrics['total_loans']:,.0f}
    - Representatividade na carteira: {metrics['total_loans']/total_carteira:.1%}
    - Taxa de inadimplência: {metrics['default_rate']:.2%}
    - Ticket médio: ${metrics['avg_loan_amnt']:,.0f}
    - Renda média anual: ${metrics['avg_annual_inc']:,.0f}
    - Taxa de juros média: {metrics['avg_int_rate']:.1f}%
    - DTI médio: {metrics['avg_dti']:.1f}%
    - FICO médio: {metrics['avg_fico']:.0f}
    - % em Grade B ou C: {metrics['pct_grade_bc']:.1%}
    - Tendência inadimplência: {metrics['default_rate_start']:.1%} (início) → {metrics['default_rate_end']:.1%} (fim do período)
    """

    # 3. Chamar o LLM com contexto real (modelo não "inventa" os números)
    model = GenerativeModel("gemini-2.0-flash-001")
    prompt = f"""
    Com base EXCLUSIVAMENTE nos dados abaixo (não use conhecimento externo),
    elabore uma análise de perfil de risco para o gestor comercial.
    Siga a estrutura: Visão Geral | Perfil de Risco | Pontos de Atenção.
    
    {context}
    
    Ao final, inclua a nota: "Fonte: BigQuery · mart.loans_enriched · Atualizado: [DATA]"
    """
    response = model.generate_content(prompt)
    return response.text
```

### 3.3 Exemplo de Resposta Aprimorada (Como Deveria Ser)

Com o pipeline acima, a resposta do agente seria:

---

**Perfil de Risco — Consolidação de Dívidas | 2016–2018**

**Visão Geral:** O segmento de consolidação de dívidas representa 47,8% da carteira da CreditLab no período analisado (2016–2018), com 65.420 empréstimos encerrados (Fully Paid ou Charged Off/Default).

**Perfil de Risco:** A taxa de inadimplência realizada é de 13,6%, 0,8 pp abaixo da média geral da carteira (14,4%). O ticket médio é de $14.890, com renda média anual de $71.200 e DTI médio de 19,4% — 2,1 pp acima da média geral. O FICO médio de 693 indica um perfil de crédito mediano. A distribuição de grade é concentrada em B (32%) e C (29%), com 61% total em B+C.

**⚠️ Pontos de Atenção:** A inadimplência neste segmento cresceu de 12,1% (2016) para 15,2% (2018), uma deterioração de 3,1 pp que merece monitoramento próximo. Clientes com DTI acima de 30% neste segmento têm inadimplência de 21,3%, quase o dobro da média.

*Fonte: BigQuery · mart.loans_enriched · Atualizado: Abril/2026*

---

---

## 4 · Processo Recorrente de Validação

### 4.1 Framework VARP (Validação Automática de Respostas de IA Preditiva)

Proposta de processo institucional para validar respostas do agente de IA antes de liberar para stakeholders:

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE VARP — CreditLab AI Governance                          │
│                                                                 │
│  FASE 1: PRÉ-GERAÇÃO (Antes do LLM responder)                  │
│  ├── [AUTO] Query BigQuery → Extrai métricas de referência      │
│  ├── [AUTO] Salva métricas em validation_log (timestamp)        │
│  └── [AUTO] Injeta contexto verificado no prompt (RAG)          │
│                                                                 │
│  FASE 2: PÓS-GERAÇÃO (Imediatamente após a resposta)           │
│  ├── [AUTO] Extrai números da resposta (regex/NLP)              │
│  ├── [AUTO] Compara com métricas de referência (tolerância 5%)  │
│  ├── [AUTO] Gera score de confiança (0–100)                     │
│  └── Score ≥ 80 → Libera | Score < 80 → Fila revisão humana    │
│                                                                 │
│  FASE 3: REVISÃO HUMANA (Score < 80 ou perguntas críticas)     │
│  ├── [HUMAN] Analista de Dados valida manualmente               │
│  ├── [HUMAN] Registra no validation_log com justificativa       │
│  └── [HUMAN] Aprova ou rejeita a resposta                       │
│                                                                 │
│  FASE 4: MONITORAMENTO CONTÍNUO (Mensal)                        │
│  ├── [AUTO] Dashboard de taxa de erro do agente por categoria   │
│  ├── [AUTO] Alertas se taxa de erros > 10% no mês              │
│  └── [HUMAN] Revisão do prompt e base de dados a cada 30 dias  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementação do Score de Confiança

```python
# =============================================================================
# VALIDADOR AUTOMÁTICO DE RESPOSTAS LLM — CreditLab AI Governance
# =============================================================================
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class ValidationResult:
    score:          float           # 0.0 a 100.0
    status:         str             # 'APROVADO' | 'REVISAR' | 'REPROVADO'
    divergences:    list[dict]      # Lista de divergências encontradas
    timestamp:      str = field(default_factory=lambda: datetime.now().isoformat())
    approved_by:    Optional[str] = None  # Nome do analista, se revisão humana

def extract_numbers_from_text(text: str) -> dict[str, float]:
    """Extrai números e percentuais de um texto de resposta LLM."""
    patterns = {
        'pct_carteira':  r'(\d+(?:\.\d+)?)\s*%.*(?:carteira|total)',
        'default_rate':  r'inadimplência.*?(\d+(?:[.,]\d+)?)\s*%',
        'default_media': r'média geral.*?(\d+(?:[.,]\d+)?)\s*%',
        'ticket_medio':  r'\$\s*(\d{1,3}(?:[.,]\d{3})*)',
        'renda_media':   r'renda.*?\$\s*(\d{1,3}(?:[.,]\d{3})*)',
        'taxa_juros':    r'juros.*?(\d+(?:[.,]\d+)?)\s*%',
        'pct_grade_bc':  r'(?:grade|grades)\s+[BC].*?(\d+(?:\.\d+)?)\s*%',
    }
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).replace('.','').replace(',','.')
            try:
                extracted[key] = float(val)
            except ValueError:
                pass
    return extracted

def validate_llm_response(
    llm_response: str,
    ground_truth: dict[str, float],
    tolerances: dict[str, float] = None
) -> ValidationResult:
    """
    Compara números extraídos da resposta LLM com os valores reais do BigQuery.
    
    Args:
        llm_response: Texto da resposta gerada pelo LLM
        ground_truth: Métricas reais extraídas do BigQuery
        tolerances:   Tolerância máxima por métrica (default 5%)
    
    Returns:
        ValidationResult com score, status e lista de divergências
    """
    if tolerances is None:
        tolerances = {
            'pct_carteira':  5.0,   # ±5 pp
            'default_rate':  2.0,   # ±2 pp (mais crítico)
            'default_media': 2.0,
            'ticket_medio':  10.0,  # ±10%
            'renda_media':   10.0,
            'taxa_juros':    1.5,
            'pct_grade_bc':  5.0,
        }
    
    # Peso de cada métrica no score final
    weights = {
        'default_rate':  0.30,  # Mais crítico para decisões de crédito
        'default_media': 0.20,
        'pct_carteira':  0.15,
        'pct_grade_bc':  0.15,
        'taxa_juros':    0.10,
        'ticket_medio':  0.05,
        'renda_media':   0.05,
    }
    
    extracted = extract_numbers_from_text(llm_response)
    divergences = []
    weighted_score = 0.0
    total_weight = 0.0
    
    for metric, expected in ground_truth.items():
        if metric not in extracted:
            divergences.append({
                'metric':  metric,
                'issue':   'AUSENTE',
                'expected': expected,
                'found':    None,
            })
            continue
        
        found = extracted[metric]
        delta = abs(found - expected)
        tol   = tolerances.get(metric, 5.0)
        w     = weights.get(metric, 0.10)
        
        if delta <= tol:
            weighted_score += w * 100
        else:
            score_metric = max(0, 100 - (delta - tol) * 10)
            weighted_score += w * score_metric
            divergences.append({
                'metric':    metric,
                'issue':     'DIVERGÊNCIA',
                'expected':  expected,
                'found':     found,
                'delta':     round(delta, 2),
                'tolerance': tol,
            })
        total_weight += w
    
    final_score = weighted_score / total_weight if total_weight > 0 else 0.0
    
    if final_score >= 80:
        status = 'APROVADO'
    elif final_score >= 60:
        status = 'REVISAR'
    else:
        status = 'REPROVADO'
    
    return ValidationResult(
        score=round(final_score, 1),
        status=status,
        divergences=divergences,
    )


# --- Exemplo de uso ---
RESPOSTA_AGENTE = """
Os clientes que tomam empréstimos para consolidação de dívidas representam 48% da
carteira total. A taxa de inadimplência desse segmento é de 12,3%, ligeiramente
abaixo da média geral de 14,1%. O ticket médio é de $15.200 e a renda média anual
é de $72.000. A maioria dos clientes está classificada nas grades B e C (62%),
indicando um perfil de risco moderado. A taxa de juros média é de 13,8%.
"""

# Valores reais extraídos do BigQuery (exemplo)
GROUND_TRUTH_BQ = {
    'pct_carteira':  47.8,
    'default_rate':  13.6,
    'default_media': 14.4,
    'ticket_medio':  14890,
    'renda_media':   71200,
    'taxa_juros':    13.9,
    'pct_grade_bc':  61.0,
}

result = validate_llm_response(RESPOSTA_AGENTE, GROUND_TRUTH_BQ)
print(f"\n🤖 RESULTADO DA VALIDAÇÃO AUTOMÁTICA")
print(f"   Score de Confiança: {result.score}/100")
print(f"   Status: {result.status}")
print(f"   Timestamp: {result.timestamp}")

if result.divergences:
    print(f"\n   ⚠️  Divergências encontradas:")
    for d in result.divergences:
        print(f"   - {d['metric']}: esperado={d.get('expected')}, "
              f"encontrado={d.get('found')}, delta={d.get('delta','N/A')}")
else:
    print("   ✅ Nenhuma divergência material encontrada.")
```

### 4.3 Calendário de Revisão

| Frequência | Ação | Responsável | Ferramenta |
|------------|------|-------------|------------|
| **Em tempo real** | Validação automática a cada resposta | Sistema (VARP) | Cloud Run + BigQuery |
| **Diário** | Revisão manual das respostas com score 60–80 | Analista de Dados | Planilha de tracking |
| **Semanal** | Dashboard de KPIs de qualidade do agente | Tech Lead de Dados | Looker Studio |
| **Mensal** | Calibração de tolerâncias e pesos | Analista Sênior + Head de Risco | Notebook Python |
| **Trimestral** | Retreinamento/ajuste fino do prompt e retrieval | Engenheiro de IA | Vertex AI Studio |
| **Anual** | Auditoria externa de vieses e conformidade regulatória | Jurídico + Risco | Relatório formal |

### 4.4 KPIs de Qualidade do Agente

Estes indicadores devem ser monitorados no dashboard de governança de IA:

```sql
-- View BigQuery: mart.ai_agent_quality_metrics
SELECT
    DATE_TRUNC(timestamp, WEEK)                        AS semana,
    COUNT(*)                                           AS total_respostas,
    COUNTIF(status = 'APROVADO') / COUNT(*)            AS taxa_aprovacao,
    COUNTIF(status = 'REPROVADO') / COUNT(*)           AS taxa_reprovacao,
    AVG(score)                                         AS score_medio,
    COUNTIF(score < 60)                                AS n_criticos,
    -- Alerta automático: se taxa_reprovacao > 10%, gerar alerta no Cloud Monitoring
    IF(COUNTIF(status='REPROVADO')/COUNT(*) > 0.10,
       'ALERTA_CRITICO', 'OK')                         AS status_semana
FROM `creditlab-dataplatform.mart.validation_log`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY 1
ORDER BY 1 DESC
```

---

## 5 · Conclusão e Posicionamento Estratégico

A resposta do agente de IA é um ponto de partida **plausível mas insuficiente** para uso em decisões de negócio. Os principais riscos são a **falta de rastreabilidade** (não cita período nem critério de status) e **precisão limitada** em algumas métricas.

A solução recomendada não é desativar o agente, mas sim implementar o pipeline **VARP + RAG com BigQuery** descrito neste documento. Com grounding em dados verificados, o agente passa de um "gerador de textos plausíveis" para um **assistente analítico auditável** — alinhado com as exigências regulatórias do mercado de crédito (BACEN, LGPD) e com os padrões de IA responsável.

> **Meta:** Score médio de validação ≥ 85/100 dentro de 60 dias após implementação do pipeline VARP.

---

*CreditLab · Documento de Governança de IA Generativa · Confidencial*
