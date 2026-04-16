# credit-risk-analysis
Análise estatística de inadimplência em crédito P2P com Python, BigQuery. Inclui testes de hipótese, regressão logística e framework de validação de LLMs para dados financeiros.
# 📊 Credit Risk Analysis — Lending Club Dataset
Link : https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download&select=rejected_2007_to_2018Q4.csv.gz

Análise estatística de inadimplência em carteiras de crédito P2P, com pipeline de dados no Google Cloud Platform, dashboard self-service e framework de validação de respostas de IA Generativa.

---

## Estrutura do Repositório

```
credit-risk-analysis/
│
├── README.md                          ← este arquivo
├── credit_risk_analysis.ipynb         ← análise completa em Python
└── llm_validation_framework.md        ← framework de validação de LLMs
```

---

## Problema de Negócio

Este projeto endereça três desafios reais de fintechs de crédito:

1. **Autonomia analítica** — como permitir que áreas de negócio respondam suas próprias perguntas sem depender do time de dados para cada consulta
2. **Diagnóstico de inadimplência** — a taxa de default subiu nos últimos trimestres. O que está causando isso, com evidência estatística?
3. **Governança de IA Generativa** — um agente de LLM foi integrado para responder perguntas sobre a carteira. Como garantir que as respostas são confiáveis antes de liberar para stakeholders?

---

## 📁 Dataset

**Lending Club Loan Data** — disponível publicamente no [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Contém dados reais de empréstimos P2P (2007–2018Q4) com informações de:
- Perfil do tomador (renda, score FICO, tempo de emprego, DTI)
- Características do empréstimo (valor, taxa de juros, prazo, finalidade)
- Classificação de risco (grade A–G, sub-grade)
- Desfecho (Fully Paid, Charged Off, Default, Current)

> **Nota:** Este projeto utiliza exclusivamente o arquivo `accepted_2007_to_2018Q4.csv`. O arquivo `rejected` foi descartado por não conter desfecho financeiro observável.

---

## Arquitetura GCP

```
CSV (Kaggle / Cloud Storage)
        │
        ▼
  Google Cloud Storage          ← zona de landing (raw)
        │
        ▼
     BigQuery                   ← data warehouse
     └── credit_risk.lending_club.accepted_loans
        │
   ┌────┴──────────────┐
   ▼                   ▼
Looker Studio      Vertex AI
(Dashboard)        (LLM Agent + Validação)
```

A ingestão é feita via `gcsfs` + `pandas.read_gbq`, usando Application Default Credentials (ADC) — sem necessidade de chave explícita em ambientes GCP.

---

## 📓 Notebook — `credit_risk_analysis.ipynb`

### Estrutura

| Seção | Conteúdo |
|-------|----------|
| 0 · Premissas | Decisões metodológicas e definições adotadas |
| 1 · Ingestão | Leitura via BigQuery (`read_gbq`) |
| 2 · Limpeza e QA | Pipeline de limpeza, feature engineering, definição de inadimplência |
| 3 · EDA | Inadimplência por grade, finalidade, evolução temporal e correlações |
| 4 · Teste de Hipótese | Teste Z, Qui-Quadrado e Regressão Logística |
| 5 · Insight Não-Óbvio | Interação entre Grade e DTI — subgrupos de risco dentro do Grade D |
| 6 · Recomendações | 5 recomendações estratégicas com raciocínio detalhado |

### Principais Decisões Metodológicas

**Definição de inadimplência:** apenas `Charged Off` e `Default` são considerados inadimplentes. `Current` e `Late` foram excluídos por não terem desfecho definitivo — evitando survivorship bias.

**Testes estatísticos utilizados:**
- **Teste Z de proporções** (unilateral à direita) — valida se Grade D-G tem inadimplência maior que A-C
- **Qui-Quadrado de independência** — verifica associação entre grade e inadimplência
- **Regressão Logística multivariada** — isola o efeito de cada variável controlando as demais

### Principais Achados

| Achado | Evidência | Impacto |
|--------|-----------|---------|
| Grades D-G têm inadimplência 2–4× maior | Teste Z e χ² (p < 0.001) | Alto |
| DTI é co-preditor independente do grade | Coeficientes da regressão logística | Alto |
| Grade D com DTI baixo ≈ risco real de Grade B | Heatmap Grade × Quintil de DTI | Médio-Alto |
| Tempo de emprego não neutraliza grade alto | Análise Grade × emp_length | Médio |

### Tecnologias

```python
pandas · numpy · matplotlib · seaborn
scipy · statsmodels · sklearn
google-cloud-bigquery · gcsfs
```

---

## 🤖 LLM Validation Framework — `llm_validation_framework.md`

Documento que propõe e implementa um processo de validação para respostas de agentes de IA sobre dados financeiros.

### O problema

Agentes de LLM integrados a sistemas de dados financeiros podem gerar respostas plausíveis mas numericamente incorretas — o que em crédito pode levar a decisões erradas de política, precificação ou aprovação.

### O framework VARP

**V**alidação **A**utomática de **R**espostas de IA **P**reditiva — pipeline em 4 fases:

```
Fase 1 · Pré-geração    → query BigQuery antes do LLM responder
                           injeta métricas reais como contexto (RAG)

Fase 2 · Pós-geração    → extrai números da resposta via regex
                           compara com BigQuery (tolerância por métrica)
                           gera score de confiança 0–100

Fase 3 · Revisão humana → score < 80 → fila de revisão
                           analista valida e registra decisão

Fase 4 · Monitoramento  → dashboard semanal de qualidade do agente
                           alertas se taxa de erro > 10% no mês
```

### Conteúdo do documento

- Validação quantitativa da resposta do agente (tabela com Δ por métrica)
- Avaliação qualitativa (acertos, erros e omissões)
- Proposta de melhoria via prompt engineering e RAG com BigQuery
- Implementação do score de confiança em Python
- Calendário de revisão recorrente (diário → semanal → mensal → trimestral)
- KPIs de qualidade do agente em SQL (BigQuery)

---

## ⚙️ Como Reproduzir

### 1. Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
pip install gcsfs google-cloud-bigquery google-cloud-storage db-dtypes
```

### 2. Autenticação GCP

```bash
gcloud auth application-default login
```

### 3. Configurar o bucket e tabela

No notebook, ajuste as variáveis:

```python
BUCKET_NAME = 'seu-bucket'
BLOB_PATH   = 'all_lending_club_loan_data/accepted_2007_to_2018Q4.csv'

# ou, se já tiver no BigQuery:
df_raw = pd.read_gbq(
    "SELECT * FROM `seu-projeto.seu-dataset.accepted_loans`",
    project_id="seu-projeto"
)
```

### 4. Executar

Abra o notebook no **Vertex AI Workbench**, **Colab Enterprise** ou localmente e execute célula por célula a partir da Seção 1.

---

## 📌 Limitações e Próximos Passos

- A análise utiliza amostra dos últimos 2 anos disponíveis — recomenda-se validar com base completa
- Os cortes de DTI < 20% e FICO > 680 para sub-segmentação do Grade D foram identificados exploratoriamente e precisam ser validados com análise de ponto de corte ótimo (curva ROC)
- O dataset é americano (2007–2018) — padrões de comportamento de crédito podem diferir em outros mercados e períodos
- Correlação não implica causalidade — os achados são hipóteses a testar, não verdades absolutas

---

## 👩‍💻 Autora

**Bárbara Fernandes Paes**  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
