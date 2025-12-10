# 📚 RAG Chatbot pour Roman "Wa7ch" - Système d'Extraction et de Questions-Réponses en Arabe

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-compatible-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)

*Un système RAG complet pour extraire et interroger le contenu du roman "Wa7ch" en arabe*

[Fonctionnalités](#-fonctionnalités) • [Installation](#-installation) • [Démarrage Rapide](#-démarrage-rapide) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Contact](#-contact)

</div>

---

## 📋 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Structure du Projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Démarrage Rapide](#-démarrage-rapide)
- [Architecture du Système](#-architecture-du-système)
- [Explication Détaillée du Code](#-explication-détaillée-du-code)
- [Configuration](#-configuration)
- [Dépannage](#-dépannage)
- [Contact et Support](#-contact-et-support)

---

## 🌟 Aperçu du Projet

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** complet pour le roman arabe "Wa7ch". Il comprend deux composants principaux :

1. **📊 Extraction et Indexation** : Conversion d'un PDF en arabe vers une base de données vectorielle
2. **💬 Chatbot Intelligent** : Interface de conversation pour poser des questions sur le roman

### Objectifs du Projet

- 🎯 **Extraire efficacement** le texte arabe d'un PDF avec préservation de la structure
- 🏗️ **Créer une base vectorielle** pour la recherche sémantique
- 🤖 **Implémenter un chatbot** avec interface utilisateur intuitive
- 🔒 **Maintenir la confidentialité** avec des modèles locaux (Ollama)
- 🌍 **Support optimal de l'arabe** pour le traitement du langage naturel

---

## ✨ Fonctionnalités

### 📥 Extraction de Documents

- ✅ **Extraction de PDF en arabe** avec LlamaParse
- ✅ **Préservation des tableaux et structures** complexes
- ✅ **Conversion en Markdown** structuré
- ✅ **Support asynchrone** pour le traitement de fichiers

### 🗃️ Base de Données Vectorielle

- ✅ **Découpage intelligent** en paragraphes
- ✅ **Embeddings locaux** avec Ollama (mxbai-embed-large)
- ✅ **Stockage persistant** avec ChromaDB
- ✅ **Recherche sémantique** optimisée pour l'arabe

### 💬 Chatbot Intelligent

- ✅ **Interface Web moderne** avec Streamlit
- ✅ **Design responsive** et adaptatif
- ✅ **Questions suggérées** pour une expérience utilisateur améliorée
- ✅ **Historique de conversation** persistant
- ✅ **Recherche RAG** en temps réel

### 🔧 Fonctionnalités Techniques

- ✅ **Sécurisation des clés API** (avec avertissement)
- ✅ **Gestion des erreurs** robuste
- ✅ **Logs détaillés** pour le débogage
- ✅ **Configuration flexible** via variables d'environnement

---

## 📁 Structure du Projet

```
rag-roman-wa7ch/
│
├── 📓 dataembeddings.ipynb           # Notebook Jupyter pour l'extraction et indexation
├── 🤖 chatbotmain.py                 # Application chatbot Streamlit
│
├── 📄 Wa7ch.pdf                      # Roman original (PDF)
├── 📝 Wa7ch.md                       # Texte extrait en Markdown
│
├── 📁 philo_db/                      # Base de données vectorielle ChromaDB
│   ├── chroma.sqlite3
│   ├── chroma-collections.parquet
│   └── ...
│
├── 📁 rag/                           # Alternative pour la base vectorielle
│   └── philo_db/
│
├── 📋 requirements.txt               # Dépendances Python
└── 📖 README.md                      # Documentation complète
```

---

## 🔧 Prérequis

### Système d'Exploitation
- Windows 10/11, macOS 10.14+, ou Linux Ubuntu 18.04+
- 8GB RAM minimum (16GB recommandé)
- 2GB d'espace disque libre

### Logiciels
- Python 3.11 ou supérieur
- pip (gestionnaire de packages Python)
- Git (pour cloner le repository)
- Ollama (pour les modèles locaux)

---

## 🚀 Installation

### Étape 1 : Cloner le Repository

```bash
git clone https://github.com/Oussama-fahim/chatbot-to-answer-questions-about-my-story-wa7ch.git
cd rag-roman-wa7ch
```

### Étape 2 : Créer un Environnement Virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 : Installer les Dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 : Installer et Configurer Ollama

```bash
# Télécharger Ollama depuis https://ollama.ai/
# Installer le modèle d'embedding
ollama pull mxbai-embed-large

# Installer le modèle de langage
ollama pull llama3.1
```

### Étape 5 : Configurer les Clés API

```bash
# Créer un fichier .env
echo "LLAMA_CLOUD_API_KEY=votre_clé_api_ici" > .env
```

**⚠️ Important** : Remplacez la clé API dans le notebook par une variable d'environnement pour la sécurité.

---

## ⚡ Démarrage Rapide

### Phase 1 : Extraction et Indexation

```bash
# Lancer Jupyter Notebook
jupyter notebook dataembeddings.ipynb

# Exécuter toutes les cellules dans l'ordre :
# 1. Importation des bibliothèques
# 2. Configuration du parser Llama
# 3. Extraction PDF -> Markdown
# 4. Création de la base vectorielle
```

### Phase 2 : Lancer le Chatbot

```bash
# Démarrer le serveur Streamlit
streamlit run chatbotmain.py

# Ouvrir votre navigateur à l'adresse :
# http://localhost:8501
```

---

## 🏗️ Architecture du Système

### Diagramme d'Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Wa7ch.pdf     │───▶│  LlamaParse     │───▶│  Wa7ch.md       │
│   (PDF Arabe)   │    │  (Extraction)   │    │  (Markdown)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Interface      │◀───│   Système RAG   │◀───│  ChromaDB       │
│  Streamlit      │    │   (Recherche)   │    │  (Vecteurs)     │
│  (Chatbot)      │───▶│                 │───▶│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Ollama LLM     │◀───│   Embeddings    │◀───│  Découpage      │
│  (llama3.1)     │    │   (mxbai)       │    │  (Paragraphes)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Flux de Données

1. **Extraction** : PDF → Texte structuré (Markdown)
2. **Préparation** : Découpage en paragraphes → Documents
3. **Embedding** : Transformation texte → Vecteurs
4. **Indexation** : Stockage dans ChromaDB
5. **Recherche** : Question → Récupération de contexte
6. **Génération** : Contexte + Question → Réponse

---

## 💻 Explication Détaillée du Code

### Partie 1 : Notebook `dataembeddings.ipynb`

#### Étape 1 : Importation des Bibliothèques

```python
import os
from llama_parse import LlamaParse
from llama_parse.base import ResultType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from llama_cloud_services.parse.utils import Language
from langchain_community.embeddings.ollama import OllamaEmbeddings
```

**Explication** :
- `LlamaParse` : Outil spécialisé pour l'extraction de PDF complexes
- `Chroma` : Base de données vectorielle légère et efficace
- `OllamaEmbeddings` : Modèle local pour générer des embeddings
- `Language.ARABIC` : Support spécifique pour la langue arabe

#### Étape 2 : Configuration du Parser

```python
LLAMA_API_KEY = "votre_clé_api"
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY

parser_ar = LlamaParse(
    result_type=ResultType.MD,
    language=Language.ARABIC,
    verbose=True
)
```

**Points Importants** :
- ⚠️ **Sécurité** : La clé API doit être stockée dans des variables d'environnement
- `ResultType.MD` : Format Markdown pour préserver la structure
- `Language.ARABIC` : Optimisation pour le traitement de l'arabe

#### Étape 3 : Extraction PDF → Markdown

```python
import nest_asyncio
nest_asyncio.apply()

pdf_files = [("Wa7ch.pdf", parser_ar)]

with open("Wa7ch.md", 'w', encoding='utf-8') as f:
    for file_name, parser in pdf_files:
        documents = parser.load_data(file_name)
        for doc in documents:
            f.write(doc.text + "\n\n")
```

**Fonctionnement** :
- `nest_asyncio` : Permet l'exécution asynchrone dans Jupyter
- `load_data()` : Envoie le PDF au cloud pour traitement
- Encodage UTF-8 : Essentiel pour les caractères arabes

#### Étape 4 : Création de la Base Vectorielle

```python
# 1. Lecture du fichier Markdown
with open("Wa7ch.md", encoding='utf-8') as f:
    markdown_content = f.read()

# 2. Découpage en paragraphes
paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]

# 3. Création des documents
documents = [Document(page_content=paragraph) for paragraph in paragraphs]

# 4. Initialisation des embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# 5. Création de la base vectorielle
persist_directory = "philo_db"
vecdb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="rag-chroma"
)

# 6. Persistance des données
vecdb.persist()
```

**Détails Techniques** :
- Découpage par `\n\n` : Simple mais efficace pour le Markdown
- `OllamaEmbeddings` : Modèle local, pas besoin d'internet après téléchargement
- `persist()` : Sauvegarde sur disque pour réutilisation

### Partie 2 : Script `chatbotmain.py`

#### Configuration Initiale

```python
import ollama
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from streamlit_float import *

st.set_page_config(
    page_title="روبوت رواية وحش",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

**Composants** :
- `streamlit` : Framework pour applications web interactives
- `Chroma` : Client pour interroger la base vectorielle
- `streamlit_float` : Extension pour interface avancée

#### Interface Utilisateur

```python
# CSS personnalisé
st.markdown("""
<style>
    .custom-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    /* ... autres styles ... */
</style>
""", unsafe_allow_html=True)

# Header professionnel
st.markdown("""
<div class="custom-header">
    <h1 style="margin:0; font-size: 2.5rem; font-weight: bold;">📚 روبوت رواية وحش</h1>
    <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">مساعد ذكي للإجابة على جميع أسئلتك حول رواية "وحش"</p>
</div>
""", unsafe_allow_html=True)
```

**Design** :
- Interface bilingue (français/arabe)
- Gradients et animations modernes
- Design responsive pour mobile

#### Logique RAG

```python
def retrieve_from_db(question):
    retriever = vecdb.as_retriever()
    retrieved_docs = retriever.invoke(question)
    retrieved_docs_txt = retrieved_docs[1].page_content if len(retrieved_docs) > 1 else ""
    return retrieved_docs_txt

def generate_response(user_message: str, chat_history: list=[], doc=""):
    system_msg = """أنت روبوت محادثة متخصص في رواية "وحش". 
    يجب أن تجيب على جميع الأسئلة المتعلقة برواية "وحش" باللغة العربية فقط.
    كن دقيقاً في المعلومات وواضحاً في الشرح.
    إذا لم يكن هناك إجابة كافية في السياق، رجاءً أجب بـ "آسف، لا أملك معلومات كافية للإجابة على هذا السؤال".
    """
    
    my_message = [{"role": "system", "content": system_msg.format(document=doc, question=user_message)}]
    
    for chat in chat_history:                      
        my_message.append({"role": chat["name"], "content": chat["msg"]})
    
    my_message.append({"role": "user", "content": user_message})

    response = ollama.chat(                      
        model="llama3.1",
        messages=my_message
    ) 
    return response["message"]["content"]
```

**Fonctionnement RAG** :
1. **Retrieval** : Trouve les paragraphes pertinents dans ChromaDB
2. **Contextualisation** : Combine question + contexte extrait
3. **Génération** : Utilise Ollama pour produire une réponse naturelle

#### Gestion de l'État

```python
def main():
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    
    # Interface utilisateur
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chat_container = st.container()
        # Affichage de l'historique
    
    with col2:
        # Questions suggérées
        suggested_questions = [
            "ما هي قصة رواية وحش؟",
            "من هو بطل الرواية؟",
            "ما هي المواضيع الرئيسية في الرواية؟"
        ]
    
    # Zone de saisie
    user_message = st.chat_input("اكتب سؤالك عن رواية وحش هنا...")
    
    # Traitement
    if user_message:
        doc = retrieve_from_db(user_message)
        response = generate_response(user_message, chat_history=st.session_state.chat_log, doc=doc)
        
        # Mise à jour de l'historique
        st.session_state.chat_log.append({"name": "user", "msg": user_message})
        st.session_state.chat_log.append({"name": "assistant", "msg": response})
```

**Session State** :
- Persistance de l'historique pendant la session
- Gestion asynchrone des interactions
- Mise à jour en temps réel

---

## ⚙️ Configuration

### Variables d'Environnement

```bash
# .env file
LLAMA_CLOUD_API_KEY=votre_clé_api_llama
OLLAMA_HOST=http://localhost:11434
```

### Modèles Ollama Requis

```bash
# Embeddings
ollama pull mxbai-embed-large:latest

# Modèle de langage
ollama pull llama3.1:latest

# Vérification
ollama list
```

### Configuration ChromaDB

```python
# Dans le notebook
persist_directory = "philo_db"  # Chemin local
collection_name = "rag-chroma"  # Nom de la collection

# Dans le chatbot
persist_directory = "rag/philo_db"  # Chemin alternatif
```

---

## 🔧 Dépannage

### Problèmes Courants

#### 1. Erreur : "LlamaParse API key not found"
```bash
Solution:
export LLAMA_CLOUD_API_KEY="votre_clé"
# Ou ajouter à .env
```

#### 2. Ollama ne répond pas
```bash
# Vérifier le service
ollama serve

# Vérifier les modèles
ollama list

# Redémarrer
pkill ollama
ollama serve
```

#### 3. Encodage arabe incorrect
```python
# Ajouter encoding='utf-8' à tous les open()
with open("fichier.md", 'r', encoding='utf-8') as f:
    content = f.read()
```

#### 4. Erreur de dépendances
```bash
# Mettre à jour pip
pip install --upgrade pip

# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

#### 5. Streamlit ne se lance pas
```bash
# Vérifier le port
streamlit run chatbotmain.py --server.port 8501

# Désactiver le cache
streamlit run chatbotmain.py --server.fileWatcherType none
```

### Logs de Débogage

```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier la connexion Ollama
import requests
response = requests.get("http://localhost:11434/api/tags")
print(response.json())
```

---

## 📞 Contact et Support

### Développeur Principal

**Nom complet** : Oussama fahim  
**Email** : Oussamafahim2017@gmail.com  
**Téléphone** : +212 645468306 

---

## 🤝 Contribution

### Comment Contribuer

1. **Fork** le repository
2. **Créer une branche** (`git checkout -b feature/nouvelle-fonctionnalite`)
3. **Commit** les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. **Push** sur la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. **Créer une Pull Request**

### Zones d'Amélioration

- 🌐 **Support multilingue** (français, anglais)
- 📊 **Analytiques d'utilisation** du chatbot
- 🎨 **Thèmes personnalisables** pour l'interface
- 🔍 **Amélioration de la recherche** sémantique
- 📱 **Application mobile** native

---

## 📚 Ressources Supplémentaires

### Documentation Officielle

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Tutoriels Recommandés

- [Introduction au RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [Traitement du Langage Naturel en Arabe](https://github.com/aub-mind/arabert)
- [Déploiement d'Applications Streamlit](https://streamlit.io/cloud)

---
### Technologies Utilisées

- **LlamaParse** pour l'extraction robuste de PDF
- **Ollama** pour l'exécution locale de modèles
- **ChromaDB** pour le stockage vectoriel efficace
- **Streamlit** pour l'interface utilisateur intuitive

### Inspiration

- Communauté open-source des modèles de langage
- Projets éducatifs en traitement automatique de l'arabe
- Innovations récentes en systèmes RAG

---

<div align="center">

## ⭐ Supportez le Projet

Si ce projet vous a été utile, pensez à lui donner une étoile sur GitHub !

**Développé avec ❤️ par oussama fahim**

</div>
