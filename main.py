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
    Settings.llm = Groq(
        model=LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0, 
        max_tokens=1024,
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    
    # Aumentado para manter Artigos e Parágrafos sempre unidos no contexto
    Settings.node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

configure_settings()

#  PROMPT  
custom_qa_prompt = PromptTemplate(
    "Você é o Assistente Virtual Oficial do IFPI - Campus Floriano, especialista na Resolução 253/2025. "
    "Seja sempre educado, acolhedor e amigável em suas respostas, mas vá direto ao ponto.\n\n"
    
    "INSTRUÇÕES DE ROTEAMENTO E LÓGICA:\n"
    "1. IDENTIFICAÇÃO: Avalie se a dúvida é de aluno ou professor e foque nos direitos/deveres correspondentes.\n"
    "2. REGRAS GERAIS: Se o documento tratar de regras institucionais (como NAPNE, uniformes, vestimenta, infrações), utilize todo o contexto, mesmo que o usuário tenha selecionado uma modalidade específica.\n"
    "3. LÓGICA DE NOTAS (PENSE PASSO A PASSO): Analise a modalidade do aluno antes de calcular os direitos a exame:\n"
    "   - Aprovação Direta (Todos os níveis): Média >= 7,0 E Frequência >= 75%.\n"
    "   - Prova Final (Cursos Técnicos/Médio): Média entre 2,0 e 6,9 E Frequência >= 75%.\n"
    "   - Exame Final (Ensino Superior): Média entre 4,0 e 6,9 E Frequência >= 75%.\n"
    "   - Reprovação Direta: Frequência < 75% OU Média abaixo do mínimo exigido para a modalidade.\n"
    "4. COMPOSIÇÃO DA NOTA: Fique extremamente atento à divisão exata dos pontos descrita nos incisos. Por exemplo, no Técnico Concomitante/Subsequente, o conhecimento vale até 8,0 pontos e os aspectos qualitativos valem obrigatoriamente até 2,0 pontos.\n"
    "5. BASE LEGAL: Toda resposta deve terminar com a citação do Artigo. Ex: (Art. 152).\n"
    "6. SINCERIDADE: Se a resposta realmente não estiver no texto, diga de forma gentil: 'Infelizmente, não encontrei essa informação detalhada no documento da Organização Didática atual.'\n\n"
    
    "Contexto recuperado:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Pergunta do usuário: {query_str}\n\n"
    "Resposta amigável:"
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