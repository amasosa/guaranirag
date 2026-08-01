"""Local Few-shot + RAG pipeline for morphological adaptation in Guaraní.

Configuration:
    1. Install the dependencies with pip.
    2. Create a .env file next to this script with the required keys.
    3. Place the Chroma database in ./guarani_db or define PERSIST_DIR.
    4. Run: python fewshot_rag_with_grammar.py

The provider is selected using PROVIDER. Supported values: OPENAI, GEMINI,
MISTRAL, GROQ, DEEPSEEK, and ANTHROPIC.
"""

import os
import time
from pathlib import Path

import pandas as pd
import sacrebleu
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

print("pandas:", pd.__version__)
print("sacrebleu:", sacrebleu.__version__)

# Provider selection
provider = os.environ.get("PROVIDER", "ANTHROPIC").upper()
SUPPORTED_PROVIDERS = {"OPENAI", "GEMINI", "MISTRAL", "GROQ", "DEEPSEEK", "ANTHROPIC"}
if provider not in SUPPORTED_PROVIDERS:
    raise ValueError(
        f"Unsupported provider: {provider}. "
        f"Use one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )

# Display the selected provider
print(f"Provider: {provider}")

# LLM model selection

# Model by provider
openai_model   = "gpt-5.2"          
gemini_model   = "gemini-2.5-pro"   
groq_model     = "llama-3.3-70b-versatile"   
mistral_model  = "mistral-large-latest"
deepseek_model = "deepseek-chat"         
anthropic_model = "claude-sonnet-4-5-20250929"

# Name of the environment variable required by each provider
SECRET_NAMES = {
    "OPENAI":   "OPENAI_API_KEY",
    "GEMINI":   "GOOGLE_API_KEY",
    "GROQ":     "GROQ_API_KEY",
    "MISTRAL":  "MISTRAL_API_KEY",
    "DEEPSEEK": "DEEPSEEK_API_KEY",
    "ANTHROPIC": "ANTHROPIC_API_KEY"
}

# Assign the selected provider to the environment
os.environ["PROVIDER"] = provider
PROVIDER = provider.upper()

secret_name = SECRET_NAMES[provider]

# Read the key from the environment or the local .env file
api_key = os.environ.get(secret_name)
if not api_key:
    raise RuntimeError(
        f"The '{secret_name}' variable was not found. "
        f"Define it in the environment or in the {SCRIPT_DIR / '.env'} file."
        )

# Default models. They can be overridden in .env.
os.environ.setdefault("OPENAI_MODEL", openai_model)
os.environ.setdefault("GEMINI_MODEL", gemini_model)
os.environ.setdefault("GROQ_MODEL", groq_model)
os.environ.setdefault("MISTRAL_MODEL", mistral_model)
os.environ.setdefault("DEEPSEEK_MODEL", deepseek_model)
os.environ.setdefault("ANTHROPIC_MODEL", anthropic_model)


DEFAULT_MODELS = {
    "OPENAI": os.environ.get("OPENAI_MODEL", "GPT-5.2"),
    "GEMINI": os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
    "GROQ":   os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
    "MISTRAL":os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
    "DEEPSEEK":os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    "QWEN":os.environ.get("QWEN_MODEL", "qwen-7b-chat"),
    "ANTHROPIC":os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
}

print(f"| Provider: {provider} | Credential {secret_name} loaded.")

# Client-model initialization

def make_llm(provider: str = PROVIDER, temperature: float = 0, max_tokens: int = 256):
    provider = provider.upper()
    if provider == "OPENAI":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):            
            raise ValueError("OPENAI_API_KEY is missing.")
        llm = ChatOpenAI(model=DEFAULT_MODELS["OPENAI"], temperature=temperature, max_tokens=max_tokens)
        name = DEFAULT_MODELS["OPENAI"]
    elif provider == "GEMINI":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is missing.")
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODELS["GEMINI"],
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        name = DEFAULT_MODELS["GEMINI"]
    elif provider == "GROQ":  # META
        from langchain_groq import ChatGroq

        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is missing.")
        llm = ChatGroq(model=DEFAULT_MODELS["GROQ"], temperature=temperature, max_tokens=max_tokens)
        name = DEFAULT_MODELS["GROQ"]
    elif provider == "MISTRAL":
        from langchain_mistralai import ChatMistralAI

        if not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY is missing.")
        llm = ChatMistralAI(model=DEFAULT_MODELS["MISTRAL"], temperature=temperature, max_tokens=max_tokens)
        name = DEFAULT_MODELS["MISTRAL"]
    elif provider == "DEEPSEEK":
        from langchain_deepseek import ChatDeepSeek

        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is missing.")
        llm = ChatDeepSeek(model=DEFAULT_MODELS["DEEPSEEK"], temperature=temperature, max_tokens=max_tokens)
        name = DEFAULT_MODELS["DEEPSEEK"]
    elif provider == "ANTHROPIC":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is missing.")
        llm = ChatAnthropic(model=DEFAULT_MODELS["ANTHROPIC"], temperature=temperature, max_tokens=max_tokens)
        name = DEFAULT_MODELS["ANTHROPIC"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return llm, name

llm, active_model_name = make_llm()
print(f"Active provider: {PROVIDER} — Model: {active_model_name}")

# Dataset connection

# Paths can be URLs or local files. Relative paths are resolved
# relative to the folder containing this script.
DEFAULT_DEV_PATH = "data/guarani-dev.tsv"
DEFAULT_TRAIN_PATH = "data/guarani-train.tsv"


def resolve_data_source(value: str):
    """Return a URL unchanged or convert a local path to an absolute path."""
    if "://" in value:
        return value

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


DEV_PATH = resolve_data_source(os.environ.get("DEV_PATH", DEFAULT_DEV_PATH))
TRAIN_PATH = resolve_data_source(os.environ.get("TRAIN_PATH", DEFAULT_TRAIN_PATH))

# Data loading
dev_df = pd.read_csv(DEV_PATH, sep="\t", dtype=str).fillna("")
train_df = pd.read_csv(TRAIN_PATH, sep="\t", dtype=str).fillna("")

# Display the first two rows of the datasets
dev_df.head(2)
train_df.head(2)


# Description by change type
# Description associated with a grammatical change type.
mapa = {
    "ASPECT:INM": "Aspecto Intermitente, partícula de aspecto",
    "ASPECT:IPFV": "Aspecto imperfectivo, partícula de aspecto",
    "PERSON:1_PL_EXC": "Primera persona del plural exclusivo",
    "PERSON:1_PL_INC": "Primera persona del plural inclusiva",
    "PERSON:1_SI": "Primera persona del singular",
    "PERSON:2_PL": "Segunda persona del plural",
    "PERSON:2_SI": "Segunda persona del singular",
    "PERSON:3_PL": "Tercera persona del plural",
    "PERSON:3_SI": "Tercera persona del singular (él/ella)",
    "TENSE:FUT_SIM": "Futuro perfecto",
    "TENSE:PAS_IMP": "Pretérito imperfecto",
    "TENSE:PAS_PLU": "Pretérito pluscuamperfecto (partícula -va'ekue)",
    "TENSE:PAS_REC": "Pretérito reciente (partícula -kuri)",
    "TENSE:PRE_SIM": "Tiempo presente",
    "TYPE:AFF": "Forma afirmativa",
    "TYPE:NEG": "Forma negativa",
    "PERSON:1_SI_INC": "Primera persona singular",
    "PERSON:1_SI_EXC": "Primera persona singular",
    }

# RAG generation

# Local path of the persisted database. It can be changed in .env.
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", SCRIPT_DIR / "guarani_db")).expanduser()
if not PERSIST_DIR.is_absolute():
    PERSIST_DIR = SCRIPT_DIR / PERSIST_DIR
PERSIST_DIR = PERSIST_DIR.resolve()

# Verify that the persisted vector database exists
if not PERSIST_DIR.is_dir():
    raise FileNotFoundError(
        f"The vector database does not exist: {PERSIST_DIR}. "
        "Copy the guarani_db folder there or define PERSIST_DIR in the .env file."
        )
# Display contents
print("Files in persist_dir:", [path.name for path in list(PERSIST_DIR.iterdir())[:20]])

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME
)

# Chroma database
COLLECTION_NAME = "guarani"

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# OpenAI is used for RAG retrieval, regardless of the provider
# selected to generate the predictions.
api_key_rag = os.environ.get("OPENAI_API_KEY")
if not api_key_rag:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. This key is required for the RAG stage; "
        f"define it in the environment or in {SCRIPT_DIR / '.env'}."
    )
# Instantiate the client
from langchain_openai import ChatOpenAI

llmRag = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=api_key_rag)

promptRag = ChatPromptTemplate.from_messages([
    ("system",
     "Responde usando SOLO el contexto. No recuperar viñetas numericas."
     "Si no está en el contexto, di: 'No tengo información suficiente en el contexto.'"),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}\n\nRespuesta:")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | promptRag
    | llmRag
    | StrOutputParser()
)

respuesta = rag_chain.invoke("Forma afirmativa")
print(respuesta)

# Retrieve the RAG value for the 18 change types
mapa_rag = {}

for clave, valor in mapa.items():
    resultado = rag_chain.invoke(valor)
    mapa_rag[clave] = {
        "original": valor,
        "rag": resultado
    }

for clave, valor in mapa_rag.items():
    print(f"CLAVE: {clave}")
    print(f"Original: {valor['original']}")
    print(f"RAG: {valor['rag']}")
    print("-" * 100)

# Function to retrieve RAG data
def obtener_original_rag(clave, mapa_rag):
    info = mapa_rag.get(clave)

    if info is None:
      print(f"⚠️ Change not founded in mapa_rag: {clave}")
      return clave, ""

    return info["original"], info["rag"]

# Test the obtener_original_rag function
original, rag = obtener_original_rag("TENSE:PRE_SIM", mapa_rag)

print("Original:", original)
print("RAG:", rag)

# Helper functions

RANDOM_STATE = 1   # Reproducibility when sampling examples
EXAMPLES_K   = 10  # Number of few-shot examples

def get_examples(change_tag, k=EXAMPLES_K):
    """Return a subset of k examples with the same change."""
    subset = train_df[train_df["Change"] == change_tag]
    return subset.sample(min(k, len(subset)), random_state=RANDOM_STATE)

def build_prompt(examples, new_source, change_tag, tipo_cambio, gramatica=""):
    """Build a prompt with the grammar and few-shot examples."""

    # Role and task instructions
    role_instruction = [
        "Eres un experto lingüista computacional especializado en la morfosintaxis del idioma Guaraní.\n",
        "TAREA",
        "Transformar una oración \"Source\" aplicando exactamente un cambio gramatical para generar un \"Target\"\n",
        "OBJETIVO DEL CAMBIO",
        f"Convierte la oración Source a {tipo_cambio}.\n"
        ]

    # Grammar
    guarani_grammar_context = [
        f"GRAMÁTICA DE {tipo_cambio}",
        gramatica
        ]

    # Few-shot examples
    shots = []
    for _, row in examples.iterrows():
        shots.append(
            f"Source: {row['Source']}\n"
            f"Target: {row['Target']}"
        )

    # Constraints
    restriccion = [
        "\nRESTRICCIONES",
        "- No inventes información nueva.",
        "- No cambies palabras que no necesiten modificarse.",
        "- La salida debe ser únicamente la oración Target transformada.",
        "- No agregues etiquetas ni comillas.",
        "- IMPORTANTE: no incluir los pasos de tu razonamiento ni explicaciones adicionales."
        ]

    # Final query that the model must complete
    query = (
        "\nORACIÓN A TRANSFORMAR\n"
        f"Source: {new_source}\n"
        f"Target:"
    )

    # Combine everything in the correct order: role instruction -> Guarani grammar context -> examples -> constraints -> task
    prompt_parts = [
        "\n".join(role_instruction),
        "\n".join(guarani_grammar_context),
        "\nEJEMPLOS\n" + "\n".join(shots) if shots else "",
        "\n".join(restriccion),
        query
    ]

    return "\n".join(prompt_parts)

# Prediction

refs = []
predictions = []
gramatica = []

print(f"Using model: {active_model_name}")

for i, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Processing guarani-dev"):
    change = str(row.get("Change", "")).strip()
    source = str(row.get("Source", "")).strip()
    target = str(row.get("Target", "")).strip()
    row_id = row.get("ID", str(i))

    examples = get_examples(change)

    tipo_cambio, gramatica_significado = obtener_original_rag(change, mapa_rag)

    prompt = build_prompt(
        examples,
        source,
        change,
        tipo_cambio,
        gramatica_significado
    )

    refs.append(target)
    pred = ""   # Default value; prevents an error if llm.invoke fails

    try:
      respuesta = llm.invoke(prompt)
      pred = str(getattr(respuesta, "content", respuesta)).strip()

    except Exception as e:
      # Display a notification on the screen
      print(f"⚠️ Error in row {row_id}: {type(e).__name__} → {e}")

    # Save results
    predictions.append(pred)
    gramatica.append(gramatica_significado)
    if active_model_name == "mistral-large-latest":
      time.sleep(1)  # Adjust if the provider imposes rate limits
    else:
      time.sleep(0.1)

# Add columns to the DataFrame
dev_df["Prediction"] = predictions
dev_df["_provider"] = os.environ.get("PROVIDER", "OPENAI").upper()
dev_df["_model"] = active_model_name
dev_df["prompt"] = "PROMPT5"
dev_df["RecuperacionRAG"] = gramatica

dev_df.head(2)

# Metrics
bleu  = sacrebleu.corpus_bleu(predictions, [refs]).score
chrf  = sacrebleu.corpus_chrf(predictions, [refs]).score
acc   = 100 * sum(p == r for p, r in zip(predictions, refs)) / len(refs)

print(f"\nResultados DEV con {PROVIDER} 🟢")
print("==============================")
print(f"BLEU:     {bleu:.2f}")
print(f"ChrF++:   {chrf:.2f}")
print(f"Accuracy: {acc:.2f}%")
print("==============================")

# Create the results dataset
# Create the DataFrame with the results
results_df = pd.DataFrame({
    'ID': dev_df['ID'],
    'Source': dev_df['Source'],
    'Change': dev_df['Change'],
    'Target': refs,
    'Prediction': predictions,
    'recuperacionRAG': gramatica,
    '_provider': dev_df['_provider'],
    '_model': dev_df['_model']
})

# Boolean column to mark correct predictions
results_df['Correct'] = results_df['Target'].str.strip() == results_df['Prediction'].str.strip()

# Save to a TSV file with a unique name
results_filename = f"fewshot_rag_with_grammar_{PROVIDER}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.tsv"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", SCRIPT_DIR / "results")).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_path = OUTPUT_DIR / results_filename
results_df.to_csv(results_path, index=False, sep='\t')

print(f"\nResultados exportados a: {results_path.resolve()}")
