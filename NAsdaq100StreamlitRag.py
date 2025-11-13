import streamlit as st
import os
import json
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. Konfiguration laden ---
# Lädt die Schlüssel aus deiner .env-Datei (nur für lokale Entwicklung)
load_dotenv()

# Funktion, um Secrets zu laden (funktioniert lokal UND in der Cloud)
def get_secret(key):
    # Prüfen, ob st.secrets vorhanden ist (Streamlit Cloud)
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    # Fallback auf os.getenv (lokale .env-Datei)
    return os.getenv(key)

# Lade die Schlüssel mit der neuen Funktion
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

# Embedding-Modell (muss dasselbe sein wie beim Erstellen der CSV)
EMBEDDING_MODEL = "text-embedding-3-small"
# LLM-Modell für die Antwortgenerierung
LLM_MODEL = "gpt-4o-mini"


# --- 2. Clients initialisieren (mit Streamlit-Caching) ---

@st.cache_resource
def get_openai_client():
    """Initialisiert den OpenAI-Client."""
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY nicht gefunden! Bitte in den Streamlit Secrets eintragen.")
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

@st.cache_resource
def get_supabase_client():
    """Initialisiert den Supabase-Client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("SUPABASE_URL oder SUPABASE_KEY nicht gefunden! Bitte in den Streamlit Secrets eintragen.")
        return None
    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client

client = get_openai_client()
supabase = get_supabase_client()

# Überprüfen, ob Clients erfolgreich geladen wurden
if not client or not supabase:
    st.stop()


# --- 3. RAG-Kernfunktionen ---

def get_query_embedding(query_text: str):
    """Erstellt ein Embedding für die Benutzeranfrage."""
    try:
        response = client.embeddings.create(
            input=[query_text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Query-Embeddings: {e}")
        return None

def find_similar_chunks(embedding: list):
    """
    Ruft die SQL-Funktion 'match_nasdaq_chunks' in Supabase auf,
    um die ähnlichsten Text-Chunks zu finden.
    """
    try:
        # Dies ruft die Funktion auf UND FÜHRT SIE AUS!
        response = supabase.rpc('match_nasdaq_chunks', {
            'query_embedding': embedding,
            'match_threshold': 0.4, # Mindest-Ähnlichkeit (0.0 bis 1.0)
            'match_count': 5        # Anzahl der Chunks, die zurückgegeben werden sollen
        }).execute() # <--- ***** DAS WAR DIE KORREKTUR *****
        
        return response.data
    except Exception as e:
        st.error(f"Fehler bei der Vektorsuche in Supabase: {e}")
        return []

def get_llm_answer(query: str, context_chunks: list) -> str:
    """
    Erstellt ein Prompt mit dem Kontext und der Frage
    und sendet es an das LLM, um eine Antwort zu generieren.
    """
    
    # 1. Kontext formatieren
    context_text = ""
    sources = []
    if context_chunks:
        for chunk in context_chunks:
            # Füge den Text-Inhalt hinzu
            context_text += f"\n---\n{chunk['content']}\n---\n"
            
            # (Optional) Metadaten für die Quellenangabe extrahieren
            try:
                # 'metadata' ist bereits ein dict (oder None), da es von Supabase als JSONB kommt
                metadata = chunk.get('metadata', {})
                if metadata and metadata.get('Ticker'):
                    sources.append(f"{metadata.get('Ticker', 'N/A')} ({metadata.get('Company', 'N/A')})")
            except Exception as e:
                # Fehler bei Metadaten-Verarbeitung protokollieren, aber App nicht stoppen
                print(f"Fehler beim Parsen der Metadaten: {e}")
                
    # 2. System-Prompt erstellen
    system_prompt = f"""
    Du bist ein Assistent für Finanzanalysen, spezialisiert auf den Nasdaq-100.
    Antworte auf die Frage des Benutzers basierend *ausschließlich* auf dem folgenden Kontext.
    Wenn der Kontext die Antwort nicht enthält, sage: "Ich habe dazu keine Informationen in meiner Datenbank."
    Sei präzise und zitiere Fakten, wenn möglich.

    Kontext:
    {context_text if context_text else "Kein Kontext gefunden."}
    """

    # 3. LLM aufrufen
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0 # Wir wollen faktenbasierte, nicht-kreative Antworten
        )
        
        answer = response.choices[0].message.content
        
        # (Optional) Eindeutige Quellen zur Antwort hinzufügen
        if sources:
            unique_sources = sorted(list(set(sources)))
            answer += "\n\n**Quellen:**\n- " + "\n- ".join(unique_sources)
            
        return answer

    except Exception as e:
        st.error(f"Fehler beim Aufruf des LLM: {e}")
        return "Es gab einen Fehler bei der Generierung der Antwort."


# --- 4. Streamlit UI-Setup ---

st.title("📈 Nasdaq-100 RAG Chatbot")
st.caption("Dieser Bot beantwortet Fragen basierend auf den Daten Ihrer Supabase Vektor-Datenbank.")

# Chat-Verlauf im Session State initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Alten Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Chat-Logik (Eingabe -> RAG -> Ausgabe) ---
if query := st.chat_input("Stelle eine Frage (z.B. 'Was ist die Geschäftsstrategie von Apple?')"):
    
    # 1. Benutzereingabe anzeigen und speichern
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. RAG-Prozess starten
    with st.chat_message("assistant"):
        with st.spinner("Analysiere Daten... (Embedding -> Vektorsuche -> LLM)"):
            
            # Schritt A: Query embedden
            query_embedding = get_query_embedding(query)
            
            if query_embedding:
                # Schritt B: Ähnliche Chunks in Supabase finden
                relevant_chunks = find_similar_chunks(query_embedding)
                
                # Schritt C: Antwort mit LLM generieren
                answer = get_llm_answer(query, relevant_chunks)
            else:
                answer = "Fehler: Konnte die Anfrage nicht verarbeiten."

        # 3. Antwort anzeigen und speichern
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

