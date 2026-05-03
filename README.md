# 🩺 NutriVet AI v2.0: Plataforma de Diagnóstico Nutricional RAG

**NutriVet AI** es una solución de ingeniería de software que aplica modelos de lenguaje de gran tamaño (LLMs) y técnicas de recuperación de información para transformar la medicina preventiva en mascotas. En esta versión 2.0, el sistema evoluciona de un motor de consola a una plataforma interactiva completa.

---

## 🚀 Novedades de la Versión 2.0
* **Interfaz Gráfica (Frontend):** Implementación de una interfaz web reactiva y amigable orientada a entornos clínicos utilizando Streamlit.
* **Optimización de Caché:** Mejoras en el rendimiento del sistema almacenando en caché la carga del modelo y la base de datos vectorial para consultas más rápidas.
* **Gestión de Errores Mejorada:** Manejo robusto de excepciones al leer documentos (PDFs/JSON) y advertencias clínicas automatizadas para los usuarios.

## 🛠️ Stack Tecnológico Actualizado
* **Lenguaje:** Python 3.10+
* **Orquestación de IA:** [LangChain](https://www.langchain.com/) (LCEL & Chain of Thought).
* **Modelos de Lenguaje:** GPT-4o a través de GitHub Models API.
* **Embeddings:** text-embedding-3-small.
* **Base de Datos Vectorial:** [ChromaDB](https://www.trychroma.com/).
* **Frontend y Despliegue:** [Streamlit](https://streamlit.io/).

---

## 📂 Estructura del Proyecto
```text
NutriVet-AI/
├── app.py                # Aplicación web principal (Streamlit + RAG)
├── Motor_IA.py           # Core del motor lógico y pruebas de consola
├── datos_razas.json      # Base de conocimientos genéticos
├── WSAVA-Manual.pdf      # Literatura médica de referencia
├── README.md             # Documentación técnica
└── Requerimientos.txt    # Dependencias del sistema
