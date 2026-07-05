"""
app.py — Interface Streamlit para consulta aos pareceres do CNE
==================================================================
Interface web amigável para pesquisadores sem experiência em Python.

Como rodar localmente:
    pip install streamlit anthropic voyageai pandas numpy pyarrow
    streamlit run app.py

Como rodar a partir do Google Colab (com túnel público):
    !pip install streamlit anthropic voyageai pandas numpy pyarrow pyngrok -q
    !streamlit run app.py &>/content/logs.txt &
    from pyngrok import ngrok
    print(ngrok.connect(8501))

Pré-requisito: a base de conhecimento (chunks.parquet + embeddings.npy)
já deve ter sido gerada pelo notebook 01_indexar_cne.ipynb.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pareceres do CNE — Consulta (By Priscila Planelis)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTES — ajuste os caminhos conforme sua estrutura
# ---------------------------------------------------------------------------

PASTA_BASE       = "./base_cne"          # onde estão chunks.parquet e embeddings.npy
PASTA_RESULTADOS = "./resultados_cne"    # onde salvar as análises geradas
MODELO_EMBEDDING = "voyage-4-lite"
MODELO_LLM       = "claude-sonnet-4-6"
TOP_K_PADRAO     = 20

# IDs dos arquivos no Google Drive (extraídos do link de compartilhamento)
# Link: https://drive.google.com/file/d/SEU_ID_AQUI/view?usp=sharing
#                                        ^^^^^^^^^^^ isso aqui é o ID
DRIVE_FILE_ID_CHUNKS     = "COLE_AQUI_O_ID_DO_CHUNKS_PARQUET"
DRIVE_FILE_ID_EMBEDDINGS = "1148nlnMEVdJkCmg4wsSJ1dRSWfNtO3iA"

PROMPT_SISTEMA = """Você é um pesquisador especialista em política educacional brasileira,
com foco em análise de documentos normativos do Conselho Nacional de Educação (CNE).

Sua tarefa é analisar trechos de pareceres do CNE e responder perguntas de pesquisa
com rigor acadêmico, baseando-se EXCLUSIVAMENTE no conteúdo dos documentos fornecidos.

Cada trecho é rotulado com seu tipo:
  [CNE]         voz do Conselho — relato, voto, decisão da câmara
  [TRANSCRICAO] argumentação de terceiros — IES, SERES, AGU, partes interessadas
  [TABELA]      dados estruturados de avaliação

Diretrizes:
- Baseie suas conclusões principalmente em trechos [CNE]
- Trechos [TRANSCRICAO] representam posições de terceiros, não do CNE
- Cite sempre o número do parecer, a câmara e o ano ao referir uma informação
- Se os trechos não contiverem informação suficiente, diga isso explicitamente
- Identifique convergências e contradições entre pareceres de diferentes períodos
- Use linguagem acadêmica clara e objetiva
- Ao final, liste os pareceres consultados para essa resposta"""

VALORES_HOMOLOGADO = [
    "Homologado", "Não Homologado", "Parcialmente", "Arquivado",
    "Recurso", "Reexaminado", "Aguardando Homologação", "Indeterminado",
]
VALORES_CAMARA = ["CEB", "CES", "CP", "CAE"]
VALORES_TIPO_CHUNK = {
    "Apenas voz do CNE (recomendado)": "cne",
    "Apenas argumentação de terceiros": "transcricao",
    "Apenas tabelas de avaliação": "tabela",
    "Todos os tipos": None,
}

# ---------------------------------------------------------------------------
# CARREGAMENTO DA BASE — cacheado para não recarregar a cada interação
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_base():
    """
    Carrega chunks + embeddings uma única vez por sessão do servidor.

    Se os arquivos não existirem localmente, baixa do Google Drive
    automaticamente na primeira execução. Downloads subsequentes usam
    o arquivo já salvo em disco — não baixa de novo a cada reinício
    do app, apenas quando o disco do servidor é reiniciado do zero
    (ex: redeploy no Streamlit Cloud).
    """
    pasta = Path(PASTA_BASE)
    pasta.mkdir(exist_ok=True)
    arq_chunks = pasta / "chunks.parquet"
    arq_emb = pasta / "embeddings.npy"

    if not arq_chunks.exists() or not arq_emb.exists():
        try:
            import gdown
        except ImportError:
            st.error(
                "Biblioteca `gdown` não instalada. Adicione `gdown` ao requirements.txt "
                "ou rode `pip install gdown`."
            )
            return None, None

        with st.spinner("Baixando base de conhecimento do Google Drive (primeira execução)..."):
            if not arq_chunks.exists():
                gdown.download(id=DRIVE_FILE_ID_CHUNKS, output=str(arq_chunks), quiet=False)
            if not arq_emb.exists():
                gdown.download(id=DRIVE_FILE_ID_EMBEDDINGS, output=str(arq_emb), quiet=False)

    if not arq_chunks.exists() or not arq_emb.exists():
        return None, None

    df = pd.read_parquet(arq_chunks)
    emb = np.load(arq_emb)
    return df, emb


@st.cache_resource(show_spinner=False)
def carregar_clientes(voyage_key: str, anthropic_key: str):
    """Instancia os clientes de API uma única vez por sessão."""
    import voyageai
    import anthropic
    return (
        voyageai.Client(api_key=voyage_key),
        anthropic.Anthropic(api_key=anthropic_key),
    )


# ---------------------------------------------------------------------------
# FUNÇÕES DE BUSCA E ANÁLISE
# ---------------------------------------------------------------------------

def filtrar_base(df, emb, camara=None, ano_inicio=None, ano_fim=None,
                  homologado=None, tipo_chunk=None):
    """Aplica filtros ao dataframe e retorna df + embeddings filtrados."""
    mask = pd.Series([True] * len(df))
    if camara:
        mask &= df["camara"].str.upper() == camara.upper()
    if ano_inicio:
        mask &= df["ano"] >= ano_inicio
    if ano_fim:
        mask &= df["ano"] <= ano_fim
    if homologado:
        mask &= df["homologado"] == homologado
    if tipo_chunk:
        mask &= df["tipo_chunk"] == tipo_chunk
    idx = df[mask].index.tolist()
    return df.loc[idx].reset_index(drop=True), emb[idx]


def buscar(pergunta, df, emb, cliente_voyage, top_k=TOP_K_PADRAO):
    """Busca semântica por similaridade cosseno."""
    res = cliente_voyage.embed([pergunta], model=MODELO_EMBEDDING, input_type="query")
    vq = np.array(res.embeddings[0])
    nq = np.linalg.norm(vq)
    nc = np.linalg.norm(emb, axis=1)
    nc = np.where(nc == 0, 1e-10, nc)
    sim = np.dot(emb, vq) / (nc * nq)
    top = np.argsort(sim)[::-1][:top_k]
    result = df.iloc[top].copy()
    result["similaridade"] = sim[top]
    return result


def montar_contexto(chunks):
    """Formata os chunks recuperados, incluindo o tipo de voz de cada trecho."""
    trechos = []
    for _, row in chunks.iterrows():
        tipo = str(row.get("tipo_chunk", "cne")).upper()
        id_p = f"Parecer {row.get('camara', 'CNE')} {row.get('numero_parecer', '?')}/{row.get('ano', '?')}"
        header = f"[{id_p} | {tipo} | relevância: {row['similaridade']:.2f}]"
        trechos.append(f"{header}\n{row['texto']}")
    return "\n\n---\n\n".join(trechos)


def salvar_resultado(resultado: dict):
    """Salva a análise em disco para consulta posterior."""
    pasta = Path(PASTA_RESULTADOS)
    pasta.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in resultado["pergunta"][:50].lower()).strip("_")
    base = pasta / f"{ts}_{slug}"

    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)


def analisar(pergunta, df_base, emb_base, cliente_voyage, cliente_anthropic,
             camara=None, ano_inicio=None, ano_fim=None, homologado=None,
             tipo_chunk=None, top_k=TOP_K_PADRAO):
    """Pipeline completo: filtro → busca semântica → Claude → resultado."""
    df_f, emb_f = df_base, emb_base
    if any([camara, ano_inicio, ano_fim, homologado, tipo_chunk]):
        df_f, emb_f = filtrar_base(df_base, emb_base, camara, ano_inicio, ano_fim, homologado, tipo_chunk)

    if len(df_f) == 0:
        return {"erro": "Nenhum documento corresponde aos filtros selecionados. Tente ampliar os critérios."}

    chunks = buscar(pergunta, df_f, emb_f, cliente_voyage, top_k)
    contexto = montar_contexto(chunks)
    prompt_usuario = f"Pergunta de pesquisa:\n{pergunta}\n\nTrechos dos pareceres do CNE:\n\n{contexto}"

    resposta = cliente_anthropic.messages.create(
        model=MODELO_LLM,
        max_tokens=4096,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": prompt_usuario}],
    )
    texto_resposta = resposta.content[0].text
    custo = (resposta.usage.input_tokens * 3 + resposta.usage.output_tokens * 15) / 1_000_000

    resultado = {
        "pergunta": pergunta,
        "resposta": texto_resposta,
        "timestamp": datetime.now().isoformat(),
        "modelo": MODELO_LLM,
        "filtros": {"camara": camara, "ano_inicio": ano_inicio, "ano_fim": ano_fim,
                    "homologado": homologado, "tipo_chunk": tipo_chunk},
        "chunks_recuperados": len(chunks),
        "pareceres_consultados": (
            chunks[["arquivo", "numero_parecer", "camara", "ano", "homologado", "tipo_chunk", "similaridade"]]
            .drop_duplicates(subset=["arquivo"])
            .to_dict("records")
        ),
        "tokens_entrada": resposta.usage.input_tokens,
        "tokens_saida": resposta.usage.output_tokens,
        "custo_usd": round(custo, 4),
    }
    salvar_resultado(resultado)
    return resultado


# ---------------------------------------------------------------------------
# INTERFACE — BARRA LATERAL (configuração e filtros)
# ---------------------------------------------------------------------------

st.title("📚 Consulta aos Pareceres do CNE")
st.caption("Análise de qualidade na educação com base nos pareceres do Conselho Nacional de Educação")

df_base, emb_base = carregar_base()

if df_base is None:
    st.error(
        "Não foi possível carregar a base de conhecimento.\n\n"
        "Verifique se `DRIVE_FILE_ID_CHUNKS` e `DRIVE_FILE_ID_EMBEDDINGS` estão "
        "configurados corretamente no topo do `app.py`, e se os arquivos no Google "
        "Drive estão compartilhados como \"Qualquer pessoa com o link\"."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuração")

    with st.expander("Chaves de API", expanded=True):
        voyage_key = st.text_input("Chave Voyage AI", type="password", help="Obtenha em voyageai.com")
        anthropic_key = st.text_input("Chave Anthropic", type="password", help="Obtenha em console.anthropic.com")

    st.divider()
    st.header("🔍 Filtros de pesquisa")

    camara_sel = st.selectbox(
        "Câmara",
        options=["Todas"] + VALORES_CAMARA,
        help="Filtra por câmara do CNE",
    )

    col1, col2 = st.columns(2)
    with col1:
        ano_min = int(df_base["ano"].min()) if df_base["ano"].notna().any() else 2015
        ano_max = int(df_base["ano"].max()) if df_base["ano"].notna().any() else 2026
        ano_range = st.slider(
            "Período",
            min_value=ano_min, max_value=ano_max,
            value=(ano_min, ano_max),
        )

    homologado_sel = st.multiselect(
        "Status de homologação",
        options=VALORES_HOMOLOGADO,
        default=[],
        help="Deixe vazio para incluir todos os status",
    )

    tipo_chunk_label = st.radio(
        "Fonte do trecho",
        options=list(VALORES_TIPO_CHUNK.keys()),
        index=0,
        help="Recomendado: apenas voz do CNE, para excluir argumentação de terceiros",
    )

    top_k = st.slider(
        "Trechos consultados por pergunta",
        min_value=5, max_value=40, value=TOP_K_PADRAO,
        help="Mais trechos = análise mais ampla, porém mais lenta e cara",
    )

    st.divider()
    st.header("📊 Estatísticas da base")
    st.metric("Total de documentos", df_base["arquivo"].nunique())
    st.metric("Total de chunks", len(df_base))

    with st.expander("Distribuição por câmara"):
        st.dataframe(
            df_base.groupby("camara")["arquivo"].nunique().rename("documentos"),
            use_container_width=True,
        )
    with st.expander("Distribuição por homologação"):
        st.dataframe(
            df_base.groupby("homologado")["arquivo"].nunique().rename("documentos"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# INTERFACE — ÁREA PRINCIPAL (chat de perguntas)
# ---------------------------------------------------------------------------

if not voyage_key or not anthropic_key:
    st.info("👈 Insira suas chaves de API na barra lateral para começar a fazer perguntas.")
    st.stop()

cliente_voyage, cliente_anthropic = carregar_clientes(voyage_key, anthropic_key)

# Inicializa histórico de conversa na sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

# Exibe histórico de mensagens
for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "pareceres" in msg:
            with st.expander(f"📄 {len(msg['pareceres'])} pareceres consultados"):
                st.dataframe(
                    pd.DataFrame(msg["pareceres"])[
                        ["camara", "numero_parecer", "ano", "homologado", "tipo_chunk", "similaridade"]
                    ].sort_values("similaridade", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

# Campo de entrada da pergunta
pergunta_usuario = st.chat_input("Faça uma pergunta sobre os pareceres do CNE...")

if pergunta_usuario:
    # Monta os filtros a partir da barra lateral
    camara_filtro = None if camara_sel == "Todas" else camara_sel
    ano_inicio_filtro = ano_range[0] if ano_range[0] != ano_min else None
    ano_fim_filtro = ano_range[1] if ano_range[1] != ano_max else None
    tipo_chunk_filtro = VALORES_TIPO_CHUNK[tipo_chunk_label]

    # Se múltiplos status de homologação foram selecionados, roda uma análise
    # combinando-os (aplica-se apenas um por vez na filtragem; se vazio, todos)
    homologado_filtro = homologado_sel[0] if len(homologado_sel) == 1 else None

    # Exibe a pergunta do usuário
    st.session_state.historico.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    # Gera e exibe a resposta
    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos relevantes e analisando com Claude..."):
            resultado = analisar(
                pergunta_usuario, df_base, emb_base, cliente_voyage, cliente_anthropic,
                camara=camara_filtro, ano_inicio=ano_inicio_filtro, ano_fim=ano_fim_filtro,
                homologado=homologado_filtro, tipo_chunk=tipo_chunk_filtro, top_k=top_k,
            )

        if "erro" in resultado:
            st.warning(resultado["erro"])
            st.session_state.historico.append({"role": "assistant", "content": resultado["erro"]})
        else:
            st.markdown(resultado["resposta"])

            col1, col2, col3 = st.columns(3)
            col1.caption(f"💬 {resultado['chunks_recuperados']} trechos consultados")
            col2.caption(f"🎯 Tokens: {resultado['tokens_entrada']} + {resultado['tokens_saida']}")
            col3.caption(f"💰 Custo estimado: US$ {resultado['custo_usd']:.4f}")

            with st.expander(f"📄 {len(resultado['pareceres_consultados'])} pareceres consultados"):
                st.dataframe(
                    pd.DataFrame(resultado["pareceres_consultados"])[
                        ["camara", "numero_parecer", "ano", "homologado", "tipo_chunk", "similaridade"]
                    ].sort_values("similaridade", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

            st.session_state.historico.append({
                "role": "assistant",
                "content": resultado["resposta"],
                "pareceres": resultado["pareceres_consultados"],
            })

# ---------------------------------------------------------------------------
# RODAPÉ — botão para limpar conversa
# ---------------------------------------------------------------------------

if st.session_state.historico:
    if st.sidebar.button("🗑️ Limpar conversa"):
        st.session_state.historico = []
        st.rerun()
