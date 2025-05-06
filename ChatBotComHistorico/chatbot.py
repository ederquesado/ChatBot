import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Função para instanciar o modelo
def model_ollama(model="phi3", temperature=0.1):
    return ChatOllama(model=model, temperature=temperature)

# Função que executa a cadeia de resposta do modelo
def model_response(user_query, chat_history):
    llm = model_ollama()
    
    system_prompt = """Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}."""
    language = "português"
    user_prompt = "{input}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })

# Função principal que roda o Streamlit
def main():
    st.set_page_config(page_title="Seu assistente virtual 👾", page_icon="👾")
    st.title("Seu Assistente Virtual")

    # Inicializa o histórico de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Olá, eu sou o seu assistente virtual! Como posso te ajudar?")]

    # Renderiza o histórico
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Entrada do usuário
    user_query = st.chat_input("Digite sua mensagem")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            resp = st.write_stream(model_response(user_query, st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(content=resp))

# Executa o app
if __name__ == "__main__":
    main()