import streamlit as st
import os
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURACIÓN DEL FRONTEND ---
st.set_page_config(page_title="NutriVet Agent v3.0", page_icon="🩺", layout="wide")
st.title("🩺 NutriVet AI - Agente Clínico Autónomo")
st.markdown("¡Hola! Soy el Agente NutriVet. Pregúntame sobre riesgos genéticos, dietas para razas específicas, o pídeme que calcule la porción diaria según el peso de tu mascota.")

# --- 2. CREDENCIALES ---
os.environ["OPENAI_API_KEY"] = "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" # <-- PEGA TU TOKEN AQUÍ
os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"

# --- 3. INICIALIZACIÓN DE LA BASE DE DATOS (RAG) ---
@st.cache_resource(show_spinner="Iniciando conexiones sinápticas y cargando base médica...")
def inicializar_rag():
    datos_crudos = """
    {
      "mascotas": [
        {
          "especie": "Perro", "raza": "Golden Retriever",
          "riesgos_geneticos": ["Displasia de cadera y codo", "Obesidad", "Cáncer"],
          "dieta_organica_recomendada": ["Salmón salvaje (Omega-3)", "Arándanos", "Caldo de huesos"],
          "alimentos_a_evitar": ["Exceso de carbohidratos (arroz blanco, maíz)"]
        },
        {
          "especie": "Perro", "raza": "Bulldog Francés",
          "riesgos_geneticos": ["Problemas respiratorios", "Alergias cutáneas", "Sensibilidad gastrointestinal"],
          "dieta_organica_recomendada": ["Carne de pavo o conejo", "Camote", "Aceite de coco"],
          "alimentos_a_evitar": ["Pollo comercial", "Trigo, soya y lácteos"]
        },
        {
          "especie": "Gato", "raza": "Gato Persa",
          "riesgos_geneticos": ["Enfermedad Renal Poliquística (PKD)", "Bolas de pelo", "Cristales urinarios"],
          "dieta_organica_recomendada": ["Pollo hervido con alto contenido de caldo", "Pasta de malta orgánica", "Aceite de pescado"],
          "alimentos_a_evitar": ["Pienso/pellets secos 100%", "Alimentos con alto fósforo"]
        },
        {
          "especie": "Gato", "raza": "Maine Coon",
          "riesgos_geneticos": ["Cardiomiopatía Hipertrófica", "Desgaste articular", "Displasia de cadera felina"],
          "dieta_organica_recomendada": ["Corazón de res (Taurina)", "Mejillón de labios verdes", "Yema de huevo de codorniz"],
          "alimentos_a_evitar": ["Dietas bajas en proteína de origen animal"]
        }
      ]
    }
    """
    datos_json = json.loads(datos_crudos)
    documentos = []
    for mascota in datos_json["mascotas"]:
        contenido = f"Raza: {mascota['raza']}. Riesgos: {', '.join(mascota['riesgos_geneticos'])}. Dieta recomendada: {', '.join(mascota['dieta_organica_recomendada'])}. Evitar: {', '.join(mascota['alimentos_a_evitar'])}."
        documentos.append(Document(page_content=contenido, metadata={"raza": mascota["raza"]}))

    try:
        documentos.extend(PyPDFLoader('WSAVA-Nutrition-Assessment-Guidelines-2011-JSAP.pdf').load())
    except Exception as e:
        print(f"Nota: Iniciando sin PDF. Motivo: {e}")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documentos)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = inicializar_rag()
llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

# --- 4. CREACIÓN DE LAS HERRAMIENTAS DEL AGENTE (Act 2.2) ---

prompt_rag = ChatPromptTemplate.from_template("Responde basándote SOLO en este contexto médico: {context}\n\nConsulta: {input}")
cadena_rag = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_rag))

# --- 4. CREACIÓN DE LAS HERRAMIENTAS DEL AGENTE (Act 2.2) ---

# Herramienta 1: El Consultor Clínico (RAG) - Simplificada para evitar errores de argumentos
from langchain.tools import tool

@tool
def consultar_base_medica(query: str):
    """Úsalo SIEMPRE para buscar riesgos genéticos, dietas y alimentos prohibidos para razas."""
    return cadena_rag.invoke({"input": query})["answer"]

@tool
def calcular_porcion(peso: float):
    """Úsalo para calcular la porción diaria. El input DEBE ser el número del peso en kg."""
    gramos = peso * 25
    return f"Según el peso de {peso} kg, la porción recomendada es de {gramos} gramos."

tools = [consultar_base_medica, calcular_porcion]

# --- 5. ORQUESTACIÓN DEL AGENTE Y MEMORIA (Act 2.3) ---

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_agente = ChatPromptTemplate.from_messages([
    ("system", "Eres NutriVet AI, un veterinario experto. Tienes dos herramientas. Si preguntan por razas/dietas, usa 'Consultor_Medico_RAG'. Si dan un peso y piden cantidades, usa 'Calculadora_Nutricional'. Eres muy empático, usa emojis y responde en español. Usa el historial de la conversación para recordar de qué mascota o raza están hablando."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt_agente)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# --- 6. INTERFAZ DE CHAT DE STREAMLIT ---
if "mensajes_ui" not in st.session_state:
    st.session_state.mensajes_ui = [{"role": "assistant", "content": "¡Hola! Dime, ¿qué mascota tienes y en qué te puedo ayudar hoy? 🐾"}]

# Mostrar historial de chat
for msg in st.session_state.mensajes_ui:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capturar mensaje del usuario
if prompt_usuario := st.chat_input("Ej: Tengo un Golden Retriever..."):
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    st.session_state.mensajes_ui.append({"role": "user", "content": prompt_usuario})
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando herramientas y consultando historial..."):
            respuesta = agent_executor.invoke({"input": prompt_usuario})
            contenido_respuesta = respuesta["output"]
            st.markdown(contenido_respuesta)
            st.session_state.mensajes_ui.append({"role": "assistant", "content": contenido_respuesta})