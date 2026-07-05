# Interface de Consulta — Pareceres do CNE

Interface web (Streamlit) para pesquisadores fazerem perguntas sobre os pareceres do CNE sem precisar programar.

## Pré-requisitos

1. Ter executado o notebook `01_indexar_cne.ipynb` e gerado a base de conhecimento
2. Ter em mãos as chaves de API do [Voyage AI](https://voyageai.com) e da [Anthropic](https://console.anthropic.com)

## Passo 0 — Configurar as chaves de API com segurança

As chaves **nunca** devem ser digitadas na interface nem ficar visíveis no código.
O Streamlit tem um sistema próprio de *secrets* para isso.

### Rodando localmente ou no Colab

1. Crie uma pasta chamada `.streamlit` no mesmo diretório do `app.py`
2. Dentro dela, crie um arquivo chamado `secrets.toml` com este conteúdo:
   ```toml
   VOYAGE_API_KEY = "sua-chave-voyage-aqui"
   ANTHROPIC_API_KEY = "sua-chave-anthropic-aqui"
   ```
   (Use o arquivo `secrets.toml.example` incluído como modelo — só renomear e preencher.)

3. O `.gitignore` incluído já impede que esse arquivo suba para o GitHub por engano.

Estrutura final:
```
projeto_cne/
├── app.py
├── .gitignore
├── .streamlit/
│   └── secrets.toml       ← suas chaves reais, nunca vai para o GitHub
└── secrets.toml.example   ← modelo, esse pode ir para o GitHub
```

### Rodando no Streamlit Cloud (deploy permanente)

Não crie o arquivo `secrets.toml` manualmente — o próprio painel do Streamlit Cloud
tem um campo para colar os *secrets*:

1. No painel do seu app em [share.streamlit.io](https://share.streamlit.io), clique em **Settings → Secrets**
2. Cole o mesmo conteúdo TOML:
   ```toml
   VOYAGE_API_KEY = "sua-chave-voyage-aqui"
   ANTHROPIC_API_KEY = "sua-chave-anthropic-aqui"
   ```
3. Salve — o app reinicia automaticamente com as chaves disponíveis

Com isso, quem acessa o app pelo link **nunca vê as chaves**, apenas faz perguntas.
Você mantém controle total sobre o uso e o custo da API.

## Passo 1 — Compartilhar a base no Google Drive

O arquivo `embeddings.npy` costuma passar de 100 MB, o que o GitHub não aceita em push normal.
A solução é deixá-lo no Google Drive e o app baixa automaticamente na primeira execução.

1. No Google Drive, clique com o botão direito em `chunks.parquet` → **Compartilhar** →
   **Qualquer pessoa com o link** → copie o link
2. Repita para `embeddings.npy`
3. Cada link tem este formato:
   ```
   https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
   ```
   O trecho entre `/d/` e `/view` é o **ID do arquivo** — é isso que você precisa.

4. Abra o `app.py` e cole os dois IDs no topo do arquivo:
   ```python
   DRIVE_FILE_ID_CHUNKS     = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
   DRIVE_FILE_ID_EMBEDDINGS = "1XyZ9876543210AbCdEfGhIjKl"
   ```

> Na primeira vez que alguém abrir o app, ele baixa os arquivos do Drive e salva em
> `./base_cne/` no servidor. Nas execuções seguintes, usa o arquivo já salvo — não
> baixa de novo, a menos que o servidor seja reiniciado do zero (redeploy).

## Passo 2 — Instalar dependências

```bash
pip install streamlit anthropic voyageai pandas numpy pyarrow gdown
```

## Passo 3 — Rodar o app

```bash
streamlit run app.py
```

Uma aba do navegador abre automaticamente em `http://localhost:8501`. Insira suas chaves de API na barra lateral e comece a fazer perguntas.

---

## Rodando a partir do Google Colab (sem instalar nada localmente)

```python
# Célula 1 — instalar dependências
!pip install streamlit anthropic voyageai pandas numpy pyarrow gdown pyngrok -q

# Célula 2 — copiar o app.py e o secrets.toml (a base é baixada automaticamente do Drive)
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/app.py /content/app.py
!mkdir -p /content/.streamlit
!cp /content/drive/MyDrive/secrets.toml /content/.streamlit/secrets.toml

# Célula 3 — rodar o Streamlit em background e abrir túnel público
!cd /content && streamlit run app.py &>/content/logs.txt &

from pyngrok import ngrok
tunnel = ngrok.connect(8501)
print(f"Acesse o app em: {tunnel.public_url}")
```

> A primeira vez que usar o ngrok, crie uma conta gratuita em [ngrok.com](https://ngrok.com) e configure o token com `ngrok.set_auth_token("seu-token")`.

---

## Hospedagem permanente (opcional)

Para deixar o app acessível permanentemente, sem depender do Colab:

1. Crie um repositório no GitHub com **apenas** `app.py` e um `requirements.txt` —
   não é preciso subir a pasta `base_cne/`, pois o app baixa do Drive sozinho:
   ```
   streamlit
   anthropic
   voyageai
   pandas
   numpy
   pyarrow
   gdown
   ```
2. Acesse [share.streamlit.io](https://share.streamlit.io), conecte sua conta GitHub
3. Selecione o repositório e clique em "Deploy"
4. As chaves de API podem ser configuradas como *secrets* no painel do Streamlit Cloud, em vez de digitadas manualmente pelo usuário

### Sobre os limites do Google Drive

O Drive impõe um limite informal de downloads diários para arquivos muito acessados por links
públicos (geralmente após ~100 downloads/dia do mesmo arquivo). Para um app de uso
interno com poucos pesquisadores, isso raramente é um problema — o download só
acontece uma vez por reinício do servidor, não a cada pergunta. Se o projeto crescer
para uso público intenso, considere migrar os arquivos para um bucket S3, Backblaze B2,
ou o Hugging Face Hub (que aceita arquivos grandes gratuitamente e não tem esse limite).

### Download falhando (arquivo corrompido)

Arquivos maiores que 100 MB — o caso típico do `embeddings.npy` — passam por uma
página de aviso do Google Drive ("não foi possível verificar vírus"). O app já lida
com isso automaticamente (usa `fuzzy=True` no gdown e verifica se o conteúdo baixado
é HTML por engano), mas se ainda assim o erro `chunks.parquet tem X linhas, mas
embeddings.npy tem Y vetores` aparecer:

1. Apague a pasta `base_cne/` no ambiente onde o app roda (no Colab: `!rm -rf base_cne`)
2. Recarregue a página do app para forçar um novo download
3. Se persistir, o problema pode ser o link de compartilhamento do Drive — confirme
   que está como "Qualquer pessoa com o link" e que o ID copiado é o do arquivo
   `embeddings.npy` (não de uma pasta)

Se o problema continuar mesmo assim, a alternativa mais robusta é hospedar o
`embeddings.npy` no [Hugging Face Hub](https://huggingface.co/docs/hub/datasets-adding)
como dataset — ele foi feito especificamente para arquivos binários grandes e não tem
o problema da página de confirmação do Drive.

---

## Funcionalidades da interface

- **Chat conversacional** — histórico de perguntas e respostas na mesma tela
- **Filtros na barra lateral** — câmara, período, status de homologação, tipo de trecho (voz do CNE vs. transcrições)
- **Painel de pareceres consultados** — mostra quais documentos embasaram cada resposta, com score de relevância
- **Custo estimado** — exibido após cada pergunta, para acompanhar o gasto com a API
- **Cache automático** — a base de 50 mil chunks é carregada uma única vez, tornando as perguntas seguintes instantâneas
- **Download automático da base** — busca `chunks.parquet` e `embeddings.npy` do Google Drive na primeira execução
