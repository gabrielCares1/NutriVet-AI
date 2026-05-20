# 🐾 NutriVet AI v3.0

> **Agente Clínico Autónomo para Asesoría Nutricional Veterinaria**

NutriVet AI es un sistema avanzado de inteligencia artificial diseñado para brindar recomendaciones nutricionales precisas, basadas en literatura clínica oficial, minimizando alucinaciones y garantizando la tenencia responsable de mascotas.

---

## 🏗️ Arquitectura del Sistema
El sistema implementa un patrón **ReAct (Reason + Act)**, permitiendo al agente razonar sobre las necesidades del usuario y ejecutar herramientas de forma autónoma.

### 🧠 Componentes Principales
* **Cerebro (LLM):** GPT-4o (vía GitHub Models API).
* **Orquestador:** `AgentExecutor` que coordina el flujo de trabajo.
* **Memoria:** `ConversationBufferMemory` persistente (vía `st.session_state`).
* **Herramientas (StructuredTools):**
    * 🩺 **Consultor_Medico_RAG:** Recuperación semántica de datos clínicos con ChromaDB.
    * 🧮 **Calculadora_Nutricional:** Lógica determinista para porciones diarias.

---

## 🛠️ Justificación de Diseño
La arquitectura sigue los principios de **Clean Architecture**, separando la lógica del agente de las herramientas ejecutables.
* **Robustez:** El uso de `StructuredTool` (decorador `@tool`) mejora la precisión en el *Function Calling*.
* **Escalabilidad:** El diseño modular permite integrar nuevas herramientas (ej. gestión de historiales clínicos) sin modificar el núcleo del agente.

---

## 🚀 Guía de Despliegue

### 1. Requisitos previos
Asegúrate de tener Python 3.10+ instalado.

### 2. Instalación:
Clona el repositorio e instala las dependencias:
```bash
py -m pip install -r Requerimientos.txt
```
3. Configuración
Crea o edita el archivo app.py y configura tu API Key:
```bash
os.environ["OPENAI_API_KEY"] = "TU_TOKEN_DE_GITHUB_AQUI"
```
4. Ejecución
Inicia la interfaz de Streamlit:
```bash
py -m streamlit run app.py
```

Resultados esperados
El sistema es capaz de:

Identificar razas y riesgos genéticos (vía RAG).

Calcular porciones diarias personalizadas.

Mantener el contexto histórico de la consulta (Memoria).

Actuar con un tono empático y profesional.


