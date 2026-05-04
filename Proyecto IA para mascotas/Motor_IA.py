import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tabulate import tabulate 


load_dotenv()
# Configuración de Clientes
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("GITHUB_TOKEN"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("GITHUB_TOKEN"), check_embedding_ctx_length=False)

# 1. Función para preparar el PDF (Solo se corre una vez o cuando cambies el PDF)
def preparar_base_conocimiento(pdf_path):
    print("\n[SISTEMA] Procesando guías técnicas de la WSAVA...")
    loader = PyPDFLoader(pdf_path)
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(paginas)

    # Creamos la base de datos de vectores (ChromaDB)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db" # Aquí se guardará la "memoria" del PDF
    )
    return vectorstore

def cargar_base_datos():
    with open('datos_razas.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def mostrar_menu_veterinario(datos):
    # Creamos una tabla con las razas disponibles en tu JSON
    tabla_razas = []
    for mascota in datos['mascotas']:
        tabla_razas.append([mascota['raza'], mascota['especie'], "✔ Disponible"])
    
    print("\n" + "="*60)
    print(" 🐾 ASISTENTE DE NUTRICIÓN VETERINARIA - DUOC UC 🐾 ")
    print("="*60)
    print("\nCatálogo de Pacientes Configurados:")
    print(tabulate(tabla_razas, headers=["Raza", "Especie", "Estado"], tablefmt="fancy_grid"))
    print("\nAyuda disponible: Guía WSAVA 2011 cargada.")
    print("-" * 60)

def asistente_final():
    datos_mascotas = cargar_base_datos()
    
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = preparar_base_conocimiento("WSAVA-Nutrition-Assessment-Guidelines-2011-JSAP.pdf")
    
    # Variable para guardar la última respuesta y mostrarla después de limpiar
    ultima_respuesta = ""
    
    while True:
        # --- COMANDO PARA LIMPIAR PANTALLA ---
        # 'nt' es para Windows, 'posix' para Linux/Mac
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Mostramos la interfaz (la tabla) siempre arriba
        mostrar_menu_veterinario(datos_mascotas)
        
        # Si hay una respuesta anterior, la mostramos debajo de la tabla
        if ultima_respuesta:
            print(f"\n📋 ÚLTIMO DIAGNÓSTICO SUGERIDO:\n{ultima_respuesta}")
            print("-" * 40)

        user_input = input("\n🐾 Ingrese síntoma o consulta (o 'salir'): ")
        
        if user_input.lower() == 'salir': 
            print("\nCerrando consulta. ¡Que tenga un buen día en la clínica!")
            break

        docs_relevantes = vectorstore.similarity_search(user_input, k=3)
        contexto_pdf = "\n\n".join([doc.page_content for doc in docs_relevantes])
        contexto_json = json.dumps(datos_mascotas, indent=2)

        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": f"""Eres un experto veterinario de Duoc UC. 
                        RESPONDE BASÁNDOTE EN:
                        1. Datos de Razas (JSON): {contexto_json}
                        2. Guía Clínica (PDF): {contexto_pdf}
                        
                        Usa el formato Markdown para tus respuestas.
                        Si usas la guía clínica, indica siempre: '(Fuente: Protocolo WSAVA)'."""
                    },
                    {"role": "user", "content": user_input}
                ],
                model="gpt-4o-mini",
            )
            
            # Guardamos la respuesta para que no se borre al limpiar la pantalla
            ultima_respuesta = response.choices[0].message.content
            
        except Exception as e:
            ultima_respuesta = f"❌ Error en el sistema: {e}"

if __name__ == "__main__":
    asistente_final()