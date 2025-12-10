import ollama
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from streamlit_float import *

# Configuration de la page
st.set_page_config(
    page_title="روبوت رواية وحش",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design professionnel
st.markdown("""
<style>
    /* Styles généraux */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header personnalisé */
    .custom-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    
    /* Messages de chat */
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Message de l'utilisateur */
    [data-testid="stChatMessage"]:has(div:first-child [data-testid="stChatMessageAvatar"]:contains("user")) {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        margin-left: 20%;
    }
    
    /* Message de l'assistant */
    [data-testid="stChatMessage"]:has(div:first-child [data-testid="stChatMessageAvatar"]:contains("assistant")) {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-right: 5px solid #9c27b0;
        margin-right: 20%;
    }
    
    /* Zone de saisie */
    .stChatInput {
        background: white;
        border-radius: 25px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
    }
    
    /* Boutons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Cartes d'information */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    
    /* Animation pour les nouveaux messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .custom-header h1 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Chroma database
persist_directory = "rag/philo_db"
vecdb = Chroma(
    persist_directory=persist_directory,
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    collection_name="rag-chroma"
)

# RAG retrieval logic
def retrieve_from_db(question):
    retriever = vecdb.as_retriever()
    retrieved_docs = retriever.invoke(question)
    retrieved_docs_txt = retrieved_docs[1].page_content if len(retrieved_docs) > 1 else ""
    return retrieved_docs_txt

# Header professionnel
st.markdown("""
<div class="custom-header">
    <h1 style="margin:0; font-size: 2.5rem; font-weight: bold;">📚 روبوت رواية وحش</h1>
    <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">مساعد ذكي للإجابة على جميع أسئلتك حول رواية "وحش"</p>
</div>
""", unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; color: white;">
        <h2>🎯 حول الروبوت</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>📖 رواية وحش</h4>
        <p>روبوت متخصص في الإجابة على جميع الأسئلة المتعلقة برواية "وحش" باللغة العربية</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🔍 كيفية الاستخدام</h4>
        <p>• اكتب أي سؤال عن الرواية</p>
        <p>• احصل على إجابات دقيقة</p>
        <p>• استفسر عن الشخصيات والأحداث</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>⚡ المميزات</h4>
        <p>• إجابات فورية</p>
        <p>• معلومات دقيقة</p>
        <p>• واجهة سهلة الاستخدام</p>
    </div>
    """, unsafe_allow_html=True)

def generate_response(user_message: str, chat_history: list=[], doc=""):
    system_msg = """أنت روبوت محادثة متخصص في رواية "وحش". 
    يجب أن تجيب على جميع الأسئلة المتعلقة برواية "وحش" باللغة العربية فقط.
    كن دقيقاً في المعلومات وواضحاً في الشرح.
    إذا لم يكن هناك إجابة كافية في السياق، رجاءً أجب بـ "آسف، لا أملك معلومات كافية للإجابة على هذا السؤال".
    لا تستخدم عبارات مثل "بناءً على السياق المقدم" أو "وفقًا للوثيقة المقدمة".
    ركز على أحداث وشخصيات الرواية فقط.
    استخدم المعلومات من الوثائق ذات الصلة: {document}
    المدخلات المستخدم: {question}
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

def main():
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    
    # Section de chat principale
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Conteneur pour l'historique du chat
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_log:
                st.markdown("""
                <div style='text-align: center; padding: 3rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <h3 style='color: #2c3e50;'>مرحباً! 👋</h3>
                    <p style='color: #7f8c8d; font-size: 1.1rem;'>أنا مساعدك الخاص للرد على استفساراتك حول رواية "وحش"</p>
                    <p style='color: #95a5a6;'>اطرح عليّ أي سؤال عن الرواية وسأجيبك فوراً</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for chat in st.session_state.chat_log:
                    with st.chat_message(chat["name"]):
                        st.write(chat["msg"])

    with col2:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h4 style='color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem;'>💡 أسئلة مقترحة</h4>
        </div>
        """, unsafe_allow_html=True)
        
        suggested_questions = [
            "ما هي قصة رواية وحش؟",
            "من هو بطل الرواية؟",
            "ما هي المواضيع الرئيسية في الرواية؟",
            "حدثني عن الشخصيات الرئيسية",
            "ما هو الصراع الأساسي في القصة؟"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=question):
                st.session_state.user_input = question
                st.rerun()

    # Zone de saisie flottante
    footer_container = st.container()
    with footer_container:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        user_message = st.chat_input("اكتب سؤالك عن رواية وحش هنا...", key="user_input")
    
    footer_container.float("bottom: 0rem; background: transparent;")

    # Traitement du message
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.spinner("🔍 جاري البحث في الرواية..."):
            doc = retrieve_from_db(user_message)
            response = generate_response(user_message, chat_history=st.session_state.chat_log, doc=doc)

        if response:
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_log.append({"name": "user", "msg": user_message})
            st.session_state.chat_log.append({"name": "assistant", "msg": response})
            
            # Auto-scroll vers le bas
            st.markdown("""
            <script>
                window.scrollTo(0, document.body.scrollHeight);
            </script>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()