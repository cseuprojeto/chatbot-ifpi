# Assistente Virtual da Organização Didática IFPI

## Objetivo do Projeto
[cite_start]Desenvolver um sistema inteligente de RAG (Retrieval-Augmented Generation) para automatizar a consulta de normas acadêmicas, utilizando Deep Learning para processar e responder dúvidas sobre a Resolução Normativa Nº 253/2025 do IFPI[cite: 1132].

## Tecnologias Utilizadas
* **Linguagem**: Python
* **Framework**: Streamlit (Interface) e LlamaIndex (Orquestração de Dados)
* **LLM**: Groq (Llama-3.1-8b-instant)
* **Embeddings**: HuggingFace (BAAI/bge-m3)

## Funcionalidades
* Busca semântica avançada em documentos PDF.
* Roteamento de respostas baseado em perfis (Aluno vs. Professor).
* Citação automática de base legal (Artigos e Incisos).

## Instruções de Instalação
1. Clone este repositório: `git clone [seu-link]`
2. Instale as dependências: `pip install -r requirements.txt`
3. Configure sua chave Groq no arquivo `.env`: `GROQ_API_KEY=sua_chave_aqui`

## Instruções de Execução
1. Execute o comando: `streamlit run app.py`
2. Acesse `http://localhost:8501` no seu navegador.