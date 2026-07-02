import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.readers.file import PDFReader

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Assistente Virtual do IFPI-FLO",
    page_icon="🎓",
    layout="wide"
)

st.title("Assistente Virtual do IFPI-FLO 🤖")
st.caption("Resolução Normativa Nº 253/2025 — Assistente Virtual")
st.markdown("---")

# Carrega as variáveis de ambiente (.env)
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Erro: Chave da API da Groq (GROQ_API_KEY) não encontrada. Verifique seu arquivo .env.")
    st.stop()

#  CONSTANTES 
DATA_DIR = "./data"
PERSIST_DIR = "./storage_ifpi_final"
LLM_MODEL = "llama-3.1-8b-instant"

# CONFIGURAÇÕES 
@st.cache_resource(show_spinner=False)
def configure_settings():
    # O System Prompt é a "alma" inquebrável do assistente
    regras_sistema = (
        "Você é o Assistente Virtual Oficial do IFPI - Campus Floriano, especialista na Resolução 253/2025. "
        "Seja educado, mas aja com absoluta autoridade normativa. Você DEVE seguir estas regras estritamente:\n"
        "1. Nunca use termos de incerteza (ex: 'pode ser', 'depende', 'talvez'). Afirme com convicção baseando-se no texto.\n"
        "2. Justifique todas as respostas detalhando o raciocínio matemático ou lógico das regras da instituição.\n"
        "3. É OBRIGATÓRIO citar a base legal (Artigo, Parágrafo ou Inciso) no final de TODAS as respostas.\n"
        "4. Fique atento às divisões exatas de notas e prazos (ex: 72 horas para atestados e segunda chamada, notas de Técnico vs Superior)."
    )

    Settings.llm = Groq(
        model=LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0, 
        max_tokens=1024,
        system_prompt=regras_sistema, # <--- LINHA ADICIONADA AQUI
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    
    Settings.node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

configure_settings()

# PROMPT
custom_qa_prompt = PromptTemplate(
    "INSTRUÇÕES DE LÓGICA E ROTEAMENTO:\n"
    "1. Avalie se a dúvida é de aluno ou professor.\n"
    "2. Para regras institucionais (NAPNE, uniforme, atestados, infrações), use todo o contexto, ignorando o filtro de modalidade.\n"
    "3. No Técnico Integrado, o uso de uniforme é SEMPRE OBRIGATÓRIO no período regular.\n"
    "4. CÁLCULO DE NOTAS:\n"
    "   - Aprovação: Média >= 7,0 E Freq. >= 75%.\n"
    "   - Prova Final (Técnico/Médio): Média entre 2,0 e 6,9 E Freq. >= 75%.\n"
    "   - Exame Final (Superior): Média entre 4,0 e 6,9 E Freq. >= 75%.\n"
    "   - Subsequente/Concomitante (Art. 96): Conhecimento vale até 8,0 pontos e aspectos qualitativos valem até 2,0 pontos.\n"
    "5. Se a resposta não estiver no texto, diga apenas: 'Infelizmente, não encontrei essa informação detalhada na Organização Didática.'\n\n"
    
    "Contexto recuperado:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Pergunta do usuário: {query_str}\n\n"
    "Resposta estruturada, justificada e com citação legal:"
)

#  CARREGAMENTO DO ÍNDICE 
@st.cache_resource(show_spinner="Indexando o documento com a nova estrutura vetorial...")
def get_query_engine():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        if not os.path.exists(DATA_DIR):
            st.error(f"❌ Pasta '{DATA_DIR}' não encontrada. Crie a pasta e coloque o PDF dentro.")
            st.stop()

        reader = SimpleDirectoryReader(
            input_dir=DATA_DIR,
            file_extractor={".pdf": PDFReader()},
        )
        documents = reader.load_data()

        if not documents:
            st.error("❌ Nenhum PDF encontrado na pasta ./data")
            st.stop()

        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    # response_mode="tree_summarize" força o modelo a processar logicamente todos os chunks antes de responder
    return index.as_query_engine(
        response_mode="tree_summarize", 
        similarity_top_k=5, 
        text_qa_template=custom_qa_prompt,
    )

# INTERFACE 
query_engine = get_query_engine()

# SIDEBAR 
with st.sidebar:
    st.header("⚙️ Filtros da Consulta")
    
    modalidade_selecionada = st.selectbox(
        "Selecione o nível de ensino para guiar o assistente:",
        [
            "Geral (Pesquisar em todo o documento)", 
            "Educação Superior (Graduação)", 
            "Técnico Integrado ao Ensino Médio", 
            "Técnico Concomitante/Subsequente"
        ],
        help="O assistente filtrará as regras de calendário, notas e faltas específicas para o nível escolhido."
    )
    
    st.markdown("---")
    
    if st.button("🔄 Limpar conversa", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "✅ Conversa reiniciada. Pode fazer uma nova pergunta!"
        }]
        st.rerun()

    st.markdown("---")
    st.caption("Resolução Normativa Nº 253/2025 — IFPI Campus Floriano")


# LÓGICA DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "✅ Olá! Sou o **Assistente Virtual da Organização Didática IFPI**.\n\n"
                   "Irei te ajudar a entender os termos segundo a Resolução Normativa Nº 253/2025. "
                   "Utilize o menu lateral para selecionar o seu nível de ensino e faça sua pergunta!\n\n"
                   "> ⚠️ *Importante: Sou uma Inteligência Artificial e posso cometer erros."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta sobre a Organização Didática..."):
    # Salva a pergunta original para mostrar na tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a Resolução e cruzando dados..."):
            try:
                # Injeção dinâmica do contexto
                if modalidade_selecionada != "Geral (Pesquisar em todo o documento)":
                    prompt_enviado = f"O usuário faz parte da modalidade '{modalidade_selecionada}'. Priorize as regras dessa modalidade para responder, mas considere as regras institucionais gerais se o assunto for aplicável a todos. Pergunta: {prompt}"
                else:
                    prompt_enviado = prompt

                # Consulta o banco de dados
                response = query_engine.query(prompt_enviado)
                answer = str(response)
                st.markdown(answer)

                # Mostrar fontes de validação
                with st.expander("📚 Fontes consultadas"):
                    for i, node in enumerate(response.source_nodes[:6], 1):
                        page = node.metadata.get("page_label") or node.metadata.get("page", "N/A")
                        filename = node.metadata.get("file_name", "Documento")
                        st.markdown(f"**{i}. {filename} — Página {page}**")
                        preview = node.text[:300] + "..." if len(node.text) > 300 else node.text
                        st.caption(f"_{preview}_")
                        if i < len(response.source_nodes[:6]):
                            st.divider()

                # Adiciona a resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Erro na requisição: {str(e)}")