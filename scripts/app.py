from flask import Flask, request, jsonify, render_template_string
import os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INDEX_DIR = Path("C:/rag_index")

RAG_PROMPT_TEMPLATE = """Tu es un assistant specialise dans les evenements culturels d Ile-de-France pour Puls-Events.
Reponds UNIQUEMENT a partir des evenements fournis dans le contexte ci-dessous.
Si l information n est pas dans le contexte, dis-le clairement.
Reponds toujours en francais, de maniere conviviale. Mentionne le lieu et la date pour chaque evenement.

CONTEXTE :
{context}

QUESTION : {question}

REPONSE :"""

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Puls-Events RAG</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#0D1B3E;color:#fff;height:100vh;display:flex;flex-direction:column}
header{background:#0A1525;border-bottom:2px solid #E84393;padding:14px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:20px}
.badge{background:#E84393;font-size:11px;font-weight:bold;padding:2px 10px;border-radius:20px;margin-left:8px}
.sub{font-size:12px;color:#8899BB;margin-top:2px}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;max-width:85%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.av{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.av.b{background:#132045;border:2px solid #E84393}
.av.u{background:#1A3A6B;border:2px solid #00C9A7}
.bub{padding:10px 14px;border-radius:14px;font-size:14px;line-height:1.6}
.msg.bot .bub{background:#132045;border:1px solid #1E3464;border-top-left-radius:3px}
.msg.user .bub{background:#1A3A6B;border:1px solid #2E6DB4;border-top-right-radius:3px}
.src{margin-top:6px;padding:7px 10px;background:#0A1525;border-radius:7px;font-size:12px;color:#8899BB;border-left:3px solid #E84393}
.src b{color:#E84393;display:block;margin-bottom:3px}
#bar{padding:14px 20px;background:#0A1525;border-top:1px solid #1E3464;display:flex;gap:10px}
#q{flex:1;background:#132045;border:1px solid #1E3464;border-radius:10px;padding:10px 14px;color:#fff;font-size:14px;outline:none}
#q:focus{border-color:#E84393}
#q::placeholder{color:#4A5A7A}
#btn{background:#E84393;color:#fff;border:none;border-radius:10px;padding:10px 18px;cursor:pointer;font-size:14px;font-weight:bold}
#btn:hover{background:#C0326D}
#btn:disabled{background:#4A5A7A;cursor:not-allowed}
.dots{display:flex;gap:4px;padding:10px 14px;background:#132045;border-radius:14px;border:1px solid #1E3464}
.dot{width:7px;height:7px;background:#E84393;border-radius:50%;animation:b 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes b{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-7px)}}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:4px}
.chip{background:#132045;border:1px solid #1E3464;color:#8899BB;font-size:12px;padding:5px 11px;border-radius:18px;cursor:pointer}
.chip:hover{border-color:#E84393;color:#E84393}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#0D1B3E}::-webkit-scrollbar-thumb{background:#1E3464;border-radius:3px}
</style>
</head>
<body>
<header>
  <span style="font-size:24px">🎭</span>
  <div>
    <h1>Puls-Events <span class="badge">POC RAG</span></h1>
    <div class="sub">Assistant evenements culturels — Ile-de-France</div>
  </div>
</header>
<div id="chat">
  <div class="msg bot">
    <div class="av b">🤖</div>
    <div>
      <div class="bub">Bonjour ! Je suis l assistant Puls-Events. Que souhaitez-vous decouvrir ?</div>
      <div class="chips">
        <span class="chip" id="c1">🎵 Concerts a Paris</span>
        <span class="chip" id="c2">👨‍👩‍👧 Activites enfants</span>
        <span class="chip" id="c3">🆓 Evenements gratuits</span>
        <span class="chip" id="c4">🎨 Expositions en cours</span>
        <span class="chip" id="c5">🌟 Evenement insolite</span>
      </div>
    </div>
  </div>
</div>
<div id="bar">
  <input type="text" id="q" placeholder="Posez votre question sur les evenements culturels IDF...">
  <button id="btn">Envoyer</button>
</div>
<script>
var chat = document.getElementById('chat');
var inp  = document.getElementById('q');
var btn  = document.getElementById('btn');

function addMsg(role, html, sources){
  var d = document.createElement('div');
  d.className = 'msg ' + role;
  var av = document.createElement('div');
  av.className = 'av ' + (role==='bot'?'b':'u');
  av.textContent = role==='bot'?'🤖':'👤';
  var inner = document.createElement('div');
  var bub = document.createElement('div');
  bub.className = 'bub';
  bub.innerHTML = html;
  inner.appendChild(bub);
  if(sources && sources.length){
    var src = document.createElement('div');
    src.className = 'src';
    src.innerHTML = '<b>📎 Sources</b>' + sources.map(function(s){return '• '+s;}).join('<br>');
    inner.appendChild(src);
  }
  d.appendChild(av);
  d.appendChild(inner);
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function addTyping(){
  var d = document.createElement('div');
  d.className = 'msg bot';
  d.id = 'typing';
  d.innerHTML = '<div class="av b">🤖</div><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function send(question){
  if(!question) return;
  inp.value = '';
  btn.disabled = true;
  addMsg('user', question, null);
  addTyping();
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/ask', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = function(){
    if(xhr.readyState === 4){
      var t = document.getElementById('typing');
      if(t) t.remove();
      if(xhr.status === 200){
        var d = JSON.parse(xhr.responseText);
        addMsg('bot', d.answer.replace(/\n/g,'<br>'), d.sources);
      } else {
        addMsg('bot', 'Erreur serveur : ' + xhr.status, null);
      }
      btn.disabled = false;
    }
  };
  xhr.send(JSON.stringify({question: question}));
}

btn.addEventListener('click', function(){ send(inp.value.trim()); });
inp.addEventListener('keydown', function(e){ if(e.key==='Enter'){ send(inp.value.trim()); } });
document.getElementById('c1').addEventListener('click', function(){ send('Quels concerts ont lieu a Paris prochainement ?'); });
document.getElementById('c2').addEventListener('click', function(){ send('Y a-t-il des activites pour enfants en Ile-de-France ?'); });
document.getElementById('c3').addEventListener('click', function(){ send('Quels evenements gratuits peut-on trouver a Paris ?'); });
document.getElementById('c4').addEventListener('click', function(){ send('Quelles expositions sont en cours en Ile-de-France ?'); });
document.getElementById('c5').addEventListener('click', function(){ send('Recommande-moi un evenement insolite ou original a Paris.'); });
</script>
</body>
</html>"""

app = Flask(__name__)
vs_store = {}

def init_chain():
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k":5})
    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY, temperature=0.3, max_tokens=1024)
    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context","question"])
    def fmt(docs):
        return "\n\n".join("- "+d.metadata.get("title","?")+" ("+d.metadata.get("city","")+", "+(d.metadata.get("date_start") or "")[:10]+")\n  "+d.page_content for d in docs)
    chain = ({"context": retriever | fmt, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    vs_store["retriever"] = retriever
    vs_store["chain"] = chain

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("question","")
    try:
        answer = vs_store["chain"].invoke(q)
        docs = vs_store["retriever"].invoke(q)
        sources = list({d.metadata.get("title","?")+" — "+d.metadata.get("city","")+" ("+( d.metadata.get("date_start") or "")[:10]+")" for d in docs})
        return jsonify({"answer": answer, "sources": sources[:3]})
    except Exception as e:
        return jsonify({"answer": "Erreur : "+str(e), "sources": []})

if __name__ == "__main__":
    print("Chargement du systeme RAG...")
    init_chain()
    print("Systeme pret ! Ouvrez http://localhost:5000")
    app.run(debug=False, port=5000)
