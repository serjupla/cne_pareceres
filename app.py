"""
app.py — Interface Streamlit para consulta aos pareceres do CNE
==================================================================
Interface web amigável para pesquisadores sem experiência em Python.

Como rodar localmente:
    pip install streamlit anthropic voyageai pandas numpy pyarrow requests
    streamlit run app.py

Como rodar a partir do Google Colab (com túnel público):
    !pip install streamlit anthropic voyageai pandas numpy pyarrow requests pyngrok -q
    !streamlit run app.py &>/content/logs.txt &
    from pyngrok import ngrok
    print(ngrok.connect(8501))

Pré-requisito: a base de conhecimento (chunks.parquet + embeddings.npy)
já deve ter sido gerada pelo notebook 01_indexar_cne.ipynb, e os IDs dos
arquivos no Google Drive configurados nos secrets — veja README_app.md.
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
    page_title="Pareceres do CNE — Consulta",
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

def _baixar_arquivo_drive(file_id: str, destino: Path, rotulo: str = "arquivo") -> bool:
    """
    Baixa um arquivo público do Google Drive para `destino`, exibindo uma
    barra de progresso com a porcentagem concluída.

    Implementação manual (sem depender do gdown) porque arquivos grandes
    (>100 MB, caso do embeddings.npy) fazem o Drive exibir uma página de
    aviso ("não foi possível verificar vírus") em vez do binário direto.

    O Drive usa dois formatos de confirmação diferentes dependendo do
    tamanho do arquivo e da versão do endpoint:
      1. Cookie "download_warning_..." + parâmetro confirm na mesma URL
      2. Página HTML com um <form> contendo campos ocultos (id, confirm,
         uuid, etc.) que precisam ser reenviados para uma URL diferente
         (drive.usercontent.google.com) — formato mais recente, usado
         para arquivos grandes.

    Retorna True se o download foi bem-sucedido e o conteúdo é binário
    válido (não uma página HTML de aviso).
    """
    import re
    import requests

    def eh_html(resp) -> bool:
        return "text/html" in resp.headers.get("Content-Type", "")

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    resposta = session.get(URL, params={"id": file_id}, stream=True)

    if eh_html(resposta):
        # Formato 1 (mais antigo): token vem num cookie de aviso
        token = None
        for chave, valor in resposta.cookies.items():
            if chave.startswith("download_warning"):
                token = valor
                break

        if token:
            resposta = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    if eh_html(resposta):
        # Formato 2 (atual): a página HTML traz um <form> com campos ocultos
        # que precisam ser reenviados como querystring para a action do form.
        html = resposta.text
        action_match = re.search(r'<form[^>]+action="([^"]+)"', html)

        if action_match:
            action_url = action_match.group(1).replace("&amp;", "&")

            campos = {}
            for tag_match in re.finditer(r"<input\b[^>]*>", html):
                tag = tag_match.group(0)
                nome_match = re.search(r'name="([^"]+)"', tag)
                valor_match = re.search(r'value="([^"]*)"', tag)
                if nome_match:
                    campos[nome_match.group(1)] = valor_match.group(1) if valor_match else ""

            if campos:
                resposta = session.get(action_url, params=campos, stream=True)

    if eh_html(resposta):
        # Nenhuma das duas estratégias funcionou. Verifica se a página HTML
        # contém uma mensagem específica do Drive (arquivo não encontrado,
        # sem permissão, etc.) para facilitar o diagnóstico.
        html_final = resposta.text.lower()
        if "cannot be viewed" in html_final or "does not exist" in html_final or "acesso negado" in html_final:
            raise ValueError(
                f"O Google Drive recusou o acesso ao arquivo (ID: {file_id}). "
                "Verifique se ele está compartilhado como \"Qualquer pessoa com o link\"."
            )
        return False

    # ── Download com barra de progresso ──────────────────────────────────
    total_bytes = int(resposta.headers.get("Content-Length", 0))
    baixado_bytes = 0
    barra = st.progress(0.0, text=f"Baixando {rotulo}... 0%")

    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        for chunk in resposta.iter_content(chunk_size=262144):  # 256 KB por chunk
            if not chunk:
                continue
            f.write(chunk)
            baixado_bytes += len(chunk)

            if total_bytes > 0:
                fracao = min(baixado_bytes / total_bytes, 1.0)
                mb_baixado = baixado_bytes / (1024 * 1024)
                mb_total = total_bytes / (1024 * 1024)
                barra.progress(
                    fracao,
                    text=f"Baixando {rotulo}... {fracao * 100:.0f}% ({mb_baixado:.1f} MB / {mb_total:.1f} MB)",
                )
            else:
                # Alguns downloads não informam o tamanho total no cabeçalho —
                # nesse caso mostra apenas os MB já baixados, sem porcentagem
                mb_baixado = baixado_bytes / (1024 * 1024)
                barra.progress(0.0, text=f"Baixando {rotulo}... {mb_baixado:.1f} MB baixados")

    barra.empty()

    # Confirmação final: os primeiros bytes não podem ser marcação HTML
    with open(destino, "rb") as f:
        inicio = f.read(20)
    if inicio.strip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        destino.unlink()
        return False

    return True


@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def carregar_base():
    """
    Carrega chunks + embeddings uma única vez por sessão do servidor.

    Os IDs dos arquivos no Google Drive vêm de st.secrets (nunca ficam
    hardcoded no código-fonte). Se os arquivos não existirem localmente,
    baixa do Drive automaticamente na primeira execução. Downloads
    subsequentes usam o arquivo já salvo em disco.
    """
    drive_id_chunks = st.secrets.get("DRIVE_FILE_ID_CHUNKS", "")
    drive_id_embeddings = st.secrets.get("DRIVE_FILE_ID_EMBEDDINGS", "")

    if not drive_id_chunks or not drive_id_embeddings:
        st.error(
            "IDs dos arquivos do Google Drive não configurados nos secrets. "
            "Adicione `DRIVE_FILE_ID_CHUNKS` e `DRIVE_FILE_ID_EMBEDDINGS` ao "
            "arquivo de secrets — veja `README_app.md`."
        )
        return None, None

    pasta = Path(PASTA_BASE)
    pasta.mkdir(exist_ok=True)
    arq_chunks = pasta / "chunks.parquet"
    arq_emb = pasta / "embeddings.npy"

    if not arq_chunks.exists():
        try:
            sucesso = _baixar_arquivo_drive(drive_id_chunks, arq_chunks, rotulo="chunks.parquet")
        except ValueError as e:
            st.error(str(e))
            return None, None
        if not sucesso:
            st.error(
                "Falha ao baixar `chunks.parquet` do Google Drive. "
                "Verifique se o arquivo está compartilhado como "
                "\"Qualquer pessoa com o link\" e se o ID em `DRIVE_FILE_ID_CHUNKS` "
                "está correto."
            )
            return None, None

    if not arq_emb.exists():
        try:
            sucesso = _baixar_arquivo_drive(drive_id_embeddings, arq_emb, rotulo="embeddings.npy")
        except ValueError as e:
            st.error(str(e))
            return None, None
        if not sucesso:
            st.error(
                "Falha ao baixar `embeddings.npy` do Google Drive. "
                "Verifique se o arquivo está compartilhado como "
                "\"Qualquer pessoa com o link\" e se o ID em `DRIVE_FILE_ID_EMBEDDINGS` "
                "está correto."
            )
            return None, None

    df = pd.read_parquet(arq_chunks)
    emb = np.load(arq_emb)

    # Validação de integridade — chunks e embeddings devem ter o mesmo tamanho.
    # Uma divergência aqui indica download parcial ou arquivos de execuções
    # diferentes do notebook de indexação.
    if len(df) != emb.shape[0]:
        st.error(
            f"**Base inconsistente.** `chunks.parquet` tem {len(df)} linhas, mas "
            f"`embeddings.npy` tem {emb.shape[0]} vetores — deveriam ser iguais.\n\n"
            "Apague a pasta `base_cne/` no servidor e reinicie o app para forçar "
            "um novo download. Se persistir, os dois arquivos podem ser de execuções "
            "diferentes do notebook de indexação — gere-os novamente juntos."
        )
        return None, None

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
    st.info(
        "Verifique a mensagem de erro acima, ou confirme se os secrets "
        "`DRIVE_FILE_ID_CHUNKS` e `DRIVE_FILE_ID_EMBEDDINGS` estão configurados "
        "corretamente. Veja `README_app.md` para instruções."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuração")

    # As chaves de API vêm de st.secrets (arquivo .streamlit/secrets.toml local,
    # ou painel de Secrets no Streamlit Cloud) — nunca ficam visíveis na tela
    # nem são digitadas pelo usuário. Veja README_app.md para configurar.
    voyage_key = st.secrets.get("VOYAGE_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")

    if voyage_key and anthropic_key:
        st.success("✓ Chaves de API carregadas")
    else:
        st.error(
            "Chaves de API não configuradas. Veja `README_app.md` "
            "para instruções de como configurar o arquivo de secrets."
        )
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
    st.info("Configure as chaves de API nos *secrets* do Streamlit para começar a fazer perguntas. Veja `README_app.md`.")
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
