"""
app.py
------
Interface web Flask pour le chatbot RAG Puls-Events.
Lance le serveur avec : python scripts/app.py
Puis ouvre http://localhost:5000 dans ton navigateur.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TOP_K           = int(os.getenv("TOP_K", 5))
INDEX_DIR       = Path("C:/rag_index")

RAG_PROMPT_TEMPLATE = """Tu es un assistant specialise dans les evenements culturels d'Ile-de-France pour Puls-Events.
Reponds UNIQUEMENT a partir des evenements fournis dans le contexte ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.
Reponds toujours en francais, de maniere conviviale. Mentionne le lieu et la date pour chaque evenement.

CONTEXTE :
{context}

QUESTION : {question}

REPONSE :"""

# ── HTML de l'interface ────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Puls-Events — Assistant RAG</title>
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0D1B3E;
  color: #fff;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ── */
header {
  background: #0A1525;
  border-bottom: 2px solid #E84393;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.logo { font-size: 22px; }
.header-info h1 { font-size: 17px; font-weight: 700; color: #fff; }
.header-info p  { font-size: 11px; color: #8899BB; margin-top: 1px; }
.badge {
  background: #E84393;
  color: #fff;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 20px;
  margin-left: 6px;
}
.status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #8899BB;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #00C9A7;
}
.status-dot.loading { background: #E84393; animation: pulse 1s infinite; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Zone de chat ── */
#chat {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  scroll-behavior: smooth;
}

/* ── Messages ── */
.msg {
  display: flex;
  gap: 10px;
  max-width: 85%;
  animation: fadeUp .25s ease;
}
.msg.user { align-self: flex-end; flex-direction: row-reverse; }

@keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

.avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
}
.avatar.bot  { background: #132045; border: 2px solid #E84393; }
.avatar.user { background: #1A3A6B; border: 2px solid #00C9A7; }

.bubble {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 13.5px;
  line-height: 1.6;
}
.msg.bot  .bubble { background: #132045; border: 1px solid #1E3464; border-top-left-radius: 3px; }
.msg.user .bubble { background: #1A3A6B; border: 1px solid #2E6DB4; border-top-right-radius: 3px; }

/* Sources */
.sources {
  margin-top: 7px;
  padding: 7px 10px;
  background: #0A1525;
  border-radius: 8px;
  font-size: 11.5px;
  color: #8899BB;
  border-left: 3px solid #E84393;
}
.sources strong { color: #E84393; display: block; margin-bottom: 3px; font-size: 11px; }

/* Chips de suggestion */
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 8px;
}
.chip {
  background: #132045;
  border: 1px solid #1E3464;
  color: #8899BB;
  font-size: 11.5px;
  padding: 5px 11px;
  border-radius: 20px;
  cursor: pointer;
  transition: border-color .15s, color .15s;
}
.chip:hover { border-color: #E84393; color: #E84393; }

/* Indicateur de frappe */
.typing-wrap {
  display: flex; gap: 10px; align-items: flex-end;
}
.typing {
  display: flex; gap: 4px;
  padding: 12px 14px;
  background: #132045;
  border-radius: 14px;
  border: 1px solid #1E3464;
  border-top-left-radius: 3px;
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #E84393;
  animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }

/* ── Zone de saisie ── */
#input-area {
  padding: 12px 16px;
  background: #0A1525;
  border-top: 1px solid #1E3464;
  display: flex;
  gap: 10px;
  flex-shrink: 0;
}
#q {
  flex: 1;
  background: #132045;
  border: 1.5px solid #1E3464;
  border-radius: 10px;
  padding: 10px 14px;
  color: #fff;
  font-size: 13.5px;
  outline: none;
  transition: border-color .15s;
  font-family: inherit;
  resize: none;
  min-height: 42px;
  max-height: 110px;
}
#q:focus { border-color: #E84393; }
#q::placeholder { color: #4A5A7A; }

#btn {
  background: #E84393;
  color: #fff;
  border: none;
  border-radius: 10px;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 13.5px;
  font-weight: 700;
  font-family: inherit;
  transition: background .15s;
  white-space: nowrap;
  align-self: flex-end;
}
#btn:hover:not(:disabled) { background: #C0326D; }
#btn:disabled { background: #2A3A5A; cursor: not-allowed; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D1B3E; }
::-webkit-scrollbar-thumb { background: #1E3464; border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div class="logo">🎭</div>
  <div class="header-info">
    <h1>Puls-Events <span class="badge">POC RAG</span></h1>
    <p>Assistant événements culturels — Île-de-France</p>
  </div>
  <div class="status">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-txt">Prêt</span>
  </div>
</header>

<div id="chat"></div>

<div id="input-area">
  <textarea id="q" rows="1"
    placeholder="Posez votre question sur les événements culturels IDF..."
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button id="btn" onclick="send()">Envoyer âœˆï¸</button>
</div>

<script>
// ── Variables globales ────────────────────────────────────────────────────────
var chat    = document.getElementById('chat');
var inp     = document.getElementById('q');
var btn     = document.getElementById('btn');
var sdot    = document.getElementById('status-dot');
var stxt    = document.getElementById('status-txt');
var isReady = true;

// ── Message d'accueil ─────────────────────────────────────────────────────────
(function init() {
  var d = document.createElement('div');
  d.className = 'msg bot';
  d.innerHTML = '<div class="avatar bot">🤖</div>' +
    '<div>' +
      '<div class="bubble">Bonjour ! Je suis votre assistant Puls-Events. ' +
      'Posez-moi vos questions sur les événements culturels en Île-de-France !</div>' +
      '<div class="chips">' +
        '<span class="chip" onclick="send(this.innerText)">🎵 Concerts à Paris</span>' +
        '<span class="chip" onclick="send(this.innerText)">Activites enfants</span>' +
        '<span class="chip" onclick="send(this.innerText)">🆓 Événements gratuits</span>' +
        '<span class="chip" onclick="send(this.innerText)">🎨 Expositions en cours</span>' +
        '<span class="chip" onclick="send(this.innerText)">🌟 Événement insolite</span>' +
      '</div>' +
    '</div>';
  chat.appendChild(d);
})();

// ── Utilitaires ───────────────────────────────────────────────────────────────
function scrollBottom() {
  setTimeout(function(){ chat.scrollTop = chat.scrollHeight; }, 30);
}

function setStatus(loading) {
  if (loading) {
    sdot.className = 'status-dot loading';
    stxt.innerText = 'Recherche…';
  } else {
    sdot.className = 'status-dot';
    stxt.innerText = 'Prêt';
  }
}

function addMsg(role, html, sources) {
  var d = document.createElement('div');
  d.className = 'msg ' + role;

  var av = document.createElement('div');
  av.className = 'avatar ' + (role === 'bot' ? 'bot' : 'user');
  av.textContent = role === 'bot' ? '🤖' : '👤';

  var inner = document.createElement('div');

  var bub = document.createElement('div');
  bub.className = 'bubble';
  bub.innerHTML = html;
  inner.appendChild(bub);

  if (sources && sources.length > 0) {
    var src = document.createElement('div');
    src.className = 'sources';
    src.innerHTML = '<strong>📎 Sources consultées</strong>' +
      sources.map(function(s){ return '• ' + s; }).join('<br>');
    inner.appendChild(src);
  }

  d.appendChild(av);
  d.appendChild(inner);
  chat.appendChild(d);
  scrollBottom();
  return d;
}

function addTyping() {
  var d = document.createElement('div');
  d.id = 'typing-msg';
  d.className = 'msg bot';
  d.innerHTML = '<div class="avatar bot">🤖</div>' +
    '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  chat.appendChild(d);
  scrollBottom();
}

function removeTyping() {
  var t = document.getElementById('typing-msg');
  if (t) t.remove();
}

// ── Envoi de message ──────────────────────────────────────────────────────────
function send(text) {
  var question = (text || inp.value).trim();
  if (!question || !isReady) return;

  inp.value = '';
  inp.style.height = 'auto';
  isReady = false;
  btn.disabled = true;
  setStatus(true);

  addMsg('user', question.replace(/</g,'&lt;').replace(/>/g,'&gt;'), null);
  addTyping();

  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/ask', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.timeout = 30000;

  xhr.onload = function() {
    removeTyping();
    if (xhr.status === 200) {
      try {
        var data = JSON.parse(xhr.responseText);
        var answer = (data.answer || 'Pas de réponse.').replace(/\\n/g, '<br>');
        addMsg('bot', answer, data.sources || []);
      } catch(e) {
        addMsg('bot', 'âŒ Erreur lors du décodage de la réponse.', null);
      }
    } else {
      addMsg('bot', 'âŒ Erreur serveur (' + xhr.status + '). Vérifiez que app.py tourne.', null);
    }
    isReady = true;
    btn.disabled = false;
    setStatus(false);
  };

  xhr.onerror = function() {
    removeTyping();
    addMsg('bot', 'âŒ Impossible de contacter le serveur. Vérifiez que <code>python scripts/app.py</code> est lancé.', null);
    isReady = true;
    btn.disabled = false;
    setStatus(false);
  };

  xhr.ontimeout = function() {
    removeTyping();
    addMsg('bot', 'â±ï¸ Délai dépassé (30s). Le LLM Mistral met trop de temps à répondre.', null);
    isReady = true;
    btn.disabled = false;
    setStatus(false);
  };

  xhr.send(JSON.stringify({ question: question }));
}

// ── Auto-resize textarea ───────────────────────────────────────────────────────
inp.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 110) + 'px';
});
</script>

</body>
</html>"""

# ── Application Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)

# Initialisation de la chaîne RAG au démarrage
_chain     = None
_retriever = None

def init_rag():
    """Initialise la chaîne RAG LangChain au démarrage du serveur."""
    global _chain, _retriever

    if not MISTRAL_API_KEY:
        print("âŒ MISTRAL_API_KEY non définie dans .env")
        return False

    if not (INDEX_DIR / "index.faiss").exists():
        print(f"âŒ Index FAISS introuvable : {INDEX_DIR}")
        print("   Lancez d'abord : python scripts/vectorize.py")
        return False

    print("â³ Chargement de l'index FAISS…")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
    )
    vs = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    def fmt(docs):
        return "\n\n".join(
            f"- {d.metadata.get('title','?')} "
            f"({d.metadata.get('city','')}, "
            f"{(d.metadata.get('date_start') or '')[:10]})\n"
            f"  {d.page_content}"
            for d in docs
        )

    _chain = (
        {"context": _retriever | fmt, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Système RAG prêt !")
    return True


@app.route("/")
def index():
    """Page principale — interface web du chatbot."""
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Endpoint POST /ask
    Body JSON : { "question": "..." }
    Retourne   : { "answer": "...", "sources": [...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "question" not in data:
        return jsonify({"answer": "Question manquante.", "sources": []}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"answer": "Question vide.", "sources": []}), 400

    if _chain is None:
        return jsonify({"answer": "Le système RAG n'est pas initialisé. Vérifiez les logs du serveur.", "sources": []}), 503

    try:
        answer = _chain.invoke(question)
        docs   = _retriever.invoke(question)
        sources = list({
            f"{d.metadata.get('title','?')} — {d.metadata.get('city','')} "
            f"({(d.metadata.get('date_start') or '')[:10]})"
            for d in docs
        })
        return jsonify({"answer": answer, "sources": sources[:3]})
    except Exception as e:
        return jsonify({"answer": f"Erreur lors de la génération : {str(e)}", "sources": []}), 500


# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = init_rag()
    if ok:
        print("🌐 Interface disponible sur http://localhost:5000")
        print("   Appuyez sur Ctrl+C pour arrêter.\n")
    app.run(debug=False, port=5000, host="127.0.0.1")


