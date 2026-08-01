# Adaptación morfológica del guaraní con Few-shot y RAG

Este repositorio contiene un pipeline experimental para generar adaptaciones
morfológicas en guaraní. El sistema combina ejemplos *few-shot*, recuperación de
información gramatical mediante RAG y un modelo de lenguaje configurable.

## Estructura del repositorio

| Ruta | Descripción |
| --- | --- |
| `guaranidb.py` | Divide el corpus gramatical y crea una base vectorial Chroma. |
| `fewshot_rag_with_grammar.py` | Ejecuta la recuperación RAG, genera predicciones y calcula las métricas. |
| `data/guarani.txt` | Corpus gramatical utilizado para construir la base vectorial. |
| `data/guarani-train.tsv` | Conjunto de entrenamiento; debe añadirse antes de ejecutar el pipeline. |
| `data/guarani-dev.tsv` | Conjunto de desarrollo; debe añadirse antes de ejecutar el pipeline. |
| `.env.example` | Plantilla de configuración sin credenciales. |
| `requirements.txt` | Dependencias de Python. |

Las carpetas `guarani_db/` y `resultados/` son generadas localmente y no se
suben a GitHub.

## Requisitos

- Python 3.10 o posterior.
- Una clave de OpenAI para la etapa RAG.
- Una clave para el proveedor que generará las predicciones. Si se selecciona
  OpenAI, la misma clave puede utilizarse en ambas etapas.
- Los conjuntos `guarani-train.tsv` y `guarani-dev.tsv`.

Los archivos TSV deben estar separados por tabulaciones e incluir estas
columnas:

```text
ID	Source	Change	Target
```

## Instalación

1. Clone el repositorio y entre en su carpeta:

   ```bash
   git clone https://github.com/USUARIO/NOMBRE-DEL-REPOSITORIO.git
   cd NOMBRE-DEL-REPOSITORIO
   ```

2. Cree y active un entorno virtual:

   En Linux o macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   En Windows PowerShell:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Instale las dependencias:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copie la plantilla de configuración:

   En Linux o macOS:

   ```bash
   cp .env.example .env
   ```

   En Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Abra `.env`, seleccione un proveedor y complete las claves necesarias. No
   suba este archivo a GitHub.

## Configuración de proveedores

| `PROVIDER` | Variable de la clave |
| --- | --- |
| `OPENAI` | `OPENAI_API_KEY` |
| `GEMINI` | `GOOGLE_API_KEY` |
| `GROQ` | `GROQ_API_KEY` |
| `MISTRAL` | `MISTRAL_API_KEY` |
| `DEEPSEEK` | `DEEPSEEK_API_KEY` |
| `ANTHROPIC` | `ANTHROPIC_API_KEY` |

La recuperación RAG utiliza OpenAI independientemente del proveedor elegido
para las predicciones. Por eso `OPENAI_API_KEY` es obligatoria en todos los
casos. Los nombres de los modelos pueden modificarse en `.env`.

## Uso

1. Coloque los archivos de datos en `data/`:

   ```text
   data/
   ├── guarani.txt
   ├── guarani-dev.tsv
   └── guarani-train.tsv
   ```

2. Cree la base vectorial:

   ```bash
   python guaranidb.py
   ```

   El script genera `guarani_db/` con la colección Chroma `guarani`.

3. Ejecute el experimento:

   ```bash
   python fewshot_rag_with_grammar.py
   ```

El programa muestra BLEU, ChrF++ y exactitud, y guarda un TSV detallado dentro
de `resultados/`.

## Rutas personalizadas

Las rutas pueden cambiarse en `.env`:

```dotenv
DEV_PATH=data/guarani-dev.tsv
TRAIN_PATH=data/guarani-train.tsv
GUARANI_TXT_PATH=data/guarani.txt
PERSIST_DIR=guarani_db
OUTPUT_DIR=resultados
```

`DEV_PATH` y `TRAIN_PATH` también aceptan una URL. Las rutas locales relativas
se interpretan desde la carpeta del script.

## Cómo publicar el repositorio en GitHub

Después de crear un repositorio vacío en GitHub, ejecute desde esta carpeta:

```bash
git init
git add .
git status
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE-DEL-REPOSITORIO.git
git push -u origin main
```

Revise `git status` antes del *commit* y confirme que `.env`, `guarani_db/` y
`resultados/` no aparezcan en la lista.

## Datos, atribución y licencia

El archivo `data/guarani.txt` parece corresponder a *Gramática Guaraní y
orientaciones básicas para la escritura de los nombres toponímicos en guaraní y
otras terminologías*, 4.ª edición (2023). Antes de publicar el repositorio,
verifique la fuente exacta, complete la referencia bibliográfica y confirme que
dispone de permiso para redistribuir el texto. Haga la misma verificación para
los conjuntos de entrenamiento y desarrollo.

Este proyecto no incorpora todavía un archivo `LICENSE`. Antes de hacerlo
público, elija una licencia para el código que sea compatible con los permisos
de los datos y del corpus.
