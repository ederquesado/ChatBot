# 🤖 AI Chatbots com Streamlit + LangChain + Ollama

Este repositório contém **duas aplicações de chatbot** utilizando LLMs locais com Ollama:

1. 👾 **Chatbot Conversacional Simples**
2. 📚 **Chatbot com Documentos (RAG)**

Ambos demonstram integração prática entre:
- Interface web (Streamlit)
- LLM local (Ollama)
- Orquestração com LangChain

---

## 📌 Projetos

---

# 👾 1. Chatbot Conversacional

Chatbot simples com memória de conversa, utilizando modelo local via Ollama.

## 🚀 Funcionalidades

- Conversa em linguagem natural
- Histórico de mensagens (context-aware)
- Respostas em português
- Streaming de resposta em tempo real
- Execução 100% local

---

## 🧠 Arquitetura

Usuário → Streamlit → Prompt + Histórico → Ollama (phi3) → Resposta

# 📚 Chatbot com Documentos (RAG + Streamlit + Ollama)

Chatbot interativo que permite conversar com arquivos PDF usando técnicas de **RAG (Retrieval-Augmented Generation)**.

O sistema combina:
- Busca semântica (FAISS + embeddings)
- Modelos locais (Ollama)
- Interface web (Streamlit)

---

## 🚀 Funcionalidades

- Upload de múltiplos PDFs
- Busca inteligente por similaridade semântica
- Conversa contextual (memória de chat)
- Respostas baseadas no conteúdo dos documentos
- Exibição das fontes utilizadas (com página)

---

## 🧠 Arquitetura

Pipeline do sistema:
PDF → Chunking → Embeddings → FAISS → Retriever → LLM → Resposta

### Componentes principais:

- **Loader**: `PyPDFLoader`
- **Splitter**: `RecursiveCharacterTextSplitter`
- **Embeddings**: `BAAI/bge-m3`
- **Vector Store**: FAISS
- **LLM**: Ollama (phi3)
- **Orquestração**: LangChain
- **UI**: Streamlit

---

## ⚙️ Como funciona

1. Usuário envia PDFs
2. Documentos são divididos em chunks
3. Cada chunk vira embedding vetorial
4. FAISS armazena e indexa os vetores
5. Pergunta do usuário é:
   - contextualizada com histórico
   - usada para recuperar documentos relevantes
6. LLM gera resposta com base no contexto

---

## 🛠️ Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (LLM local)
- PyPDF


## 📦 Instalação

### 1. Clone o projeto
git clone <seu-repo>
cd <seu-repo>

### 2 .Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### 3. Instale dependências
pip install -r requirements.txt

### Configurar Ollama
Instale o Ollama:

👉 https://ollama.com/

Baixe o modelo:
ollama pull phi3

Executar
streamlit run app.py
