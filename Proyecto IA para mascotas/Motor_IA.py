import os
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Credenciales
os.environ["OPENAI_API_KEY"] = "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" # <-- Pon tu token "github_pat_..." aquí
os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"

print("Iniciando el Asistente Nutricional...")

# 2. Cargar tus datos (El JSON incrustado y el PDF)
print("Cargando base de datos genética...")

# Incrustamos los datos directamente para evitar errores de lectura de archivos
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

# Transformamos este texto en Documentos que LangChain pueda entender
datos_json = json.loads(datos_crudos)
documentos_json = []
for mascota in datos_json["mascotas"]:
    contenido = f"Raza: {mascota['raza']}. Riesgos: {', '.join(mascota['riesgos_geneticos'])}. Dieta recomendada: {', '.join(mascota['dieta_organica_recomendada'])}. Evitar: {', '.join(mascota['alimentos_a_evitar'])}."
    doc = Document(page_content=contenido, metadata={"raza": mascota["raza"]})
    documentos_json.append(doc)

print("Cargando manual veterinario WSAVA (PDF)...")
try:
    pdf_loader = PyPDFLoader('WSAVA-Nutrition-Assessment-Guidelines-2011-JSAP.pdf')
    documentos_pdf = pdf_loader.load()
except Exception as e:
    print(f"Aviso: No se pudo cargar el PDF ({e}). Continuando solo con genética...")
    documentos_pdf = []

# Juntamos todo el conocimiento
documentos_totales = documentos_json + documentos_pdf
print("Documentos procesados exitosamente.")

# 3. Cortar el texto y vectorizar (Tu Base de Conocimiento)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documentos_totales)

print("Generando base de datos vectorial...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. El Cerebro (El Modelo y el Prompt)
# CAMBIO 1: Subimos la temperatura de 0.2 a 0.7. 
# Esto le da al modelo más "creatividad" e imaginación al redactar, sin perder los hechos.
llm = ChatOpenAI(model="gpt-4o", temperature=0.7) 

# CAMBIO 2: Rediseñamos el Prompt para darle una "Personalidad"
template = """Eres un experto nutricionista veterinario, pero por sobre todo, eres muy empático, cercano y te encantan los animales. 
Tu objetivo es ayudar a los dueños a cuidar a sus mascotas explicándoles las cosas de forma fácil de entender.

Utiliza la siguiente información clínica para basar tu diagnóstico médico:
{context}

Instrucciones de comportamiento:
1. Saluda al usuario de forma cálida y felicítalo por preocuparse por la salud de su mascota.
2. Explica los riesgos genéticos de forma suave, sin asustar al dueño.
3. Al recomendar la dieta orgánica, explica *POR QUÉ* esos ingredientes son buenos (desarrolla la idea, no hagas solo una lista).
4. Usa un tono conversacional, como si estuvieran charlando en tu consulta clínica.
5. Puedes usar emojis 🐾 para hacer el texto más amigable.
6. Aportarás datos médicos SOLO del contexto entregado, pero tienes total libertad creativa para estructurar tu charla.

Pregunta del dueño: {question}
Respuesta experta:"""

prompt = ChatPromptTemplate.from_template(template)

# 5. La Cadena de IA
cadena_nutricion = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- ZONA DE PRUEBAS ---
if __name__ == "__main__":
    print("\n¡Sistema listo! Hagamos una prueba.")
    pregunta_usuario = "Hola, tengo un Gato Persa. ¿Qué dieta orgánica me recomiendas y qué problemas de salud debo prevenir?"
    
    print(f"\nProcesando consulta: '{pregunta_usuario}'...\n")
    
    respuesta = cadena_nutricion.invoke(pregunta_usuario)
    print(respuesta)