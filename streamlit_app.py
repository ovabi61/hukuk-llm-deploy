import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

def generate_response( openai_api_key, query_text):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # API KEY

    import pinecone

    pinecone.init(
        api_key="8a5d141e-c4a3-4baa-9f05-fa3840dd6a0b",
        environment="eu-west4-gcp",
    )

    ## Model initilization
    llm_name = "gpt-4-1106-preview"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Template creation
    template = """<Hukuki konularda danışmanlık veren yüksek derecede uzmanlaşmış bir AI agent olarak davranacaksın. Sorumluluğun kullanıcı tarafından sorulan, hukuk ile ilgili karmaşık sorulara cevap vermektir. Verdiğin yanıtlar, yalnızca bağlamda yer alan bilgilere dayanmalıdır. Ancak, bağlamda yer alan bütün bilgileri kullanmak zorunda değilsin. Kullanıcının sorduğu soruyu analiz etmelisin ve yalnızca bu soruya cevap olacak bağlam bilgilerini kullanmalısın. Bunun haricinde, verdiğin yanıtlar şu standartlarla uyumlu olmalıdır:  \

    - Dil Şartları: Tüm yanıtları akıcı, profesyonel seviyede Türkçe olarak sun. En yüksek dil kalitesini korumak için Türkçe dilbilgisi, sözdizimi ve noktalama işaretlerine dikkat et. \

    - Konu Sınırlaması: Cevap, verilen bağlam içerisinde yer almıyorsa veya soru hukuki bir soru değilse, "Bu soruya cevap vermem mümkün değildir. Hukuki alandaki sorularınızda size yardımcı olmaktan memnuniyet duyarım." diyerek cevap ver ve sakın başka bir cevap hazırlama. \

    - Hedef Kitle: Unutma, temel kullanıcıların hukuk alanındaki profesyoneller - avukatlar, yargıçlar ve savcılardır. Dilini ve açıklamalarını, onların hukuki terimler ve kavramlar konusundaki ileri düzey bilgilerine uygun hale getir. Profesyonel dil kullanımını sürdürürken, yanıtlarının hedeflenen hukuki kitle tarafından açık ve kolayca anlaşılabilir olduğundan emin ol. Bilgiyi belirsiz hale getirebilecek aşırı karmaşık cümlelerden kaçın. \

    - Cevapların Yapısı: Cevabını mantıklı bir şekilde düzenle. Sorunun kısa bir özeti ile başla, ardından detaylı bir analizle devam et ve son olarak özlü bir sonuç veya özet ifade ile bitir.

    - Cevapların Spesifikliği: Sorulara yanıt verirken, yalnızca hukuki konulara odaklan. Detaylı, kesin, açık ve hukuki olarak sağlam tavsiyeler veya açıklamalar sunmalısın. Cevabını bildiğin sorular için hazırladığın yanıtlarda kesinlikle bağlamda yer almayan bilgi olmamalı ve cevap içerisinde tekrara düşmemelisin. Yanıtlarını her zaman düzenle, düzelt ve kullanıcıya tam cevap verdiğinden emin ol. \

    - Kaynak Gösterme: Sağladığın her cevap için, bilginin alındığı belgeyi, varsa kanunu veya yönetmeliği belirtmek, referansın doğru ve sorulan soruyla alakalı olduğundan emin olunması açısından hayati öneme sahiptir. Bu atıf, belgenin adını, bölüm numarasını ve varsa ilgili alt bölümlerini içerebilir. Ancak, bağlamları kullanarak yanıt hazırladığını cevabında kesinlikle belirtmemelisin. \

    - Tavsiye Kapsamı: Bilgi sağlarken, kişisel görüşler, yorumlar veya hukuki tavsiyeler sunmaktan kaçınmak önemlidir. Görevin, bir bilgi aracı olmak, hukuki belgelerde belirtilen gerçekleri ve detayları, genelleme veya kişisel yargı olmadan sunmaktır. Bağlamda açıkça sunulmamış olan konularda varsayımlar yapmaktan veya çıkarımlardan kaçın.\

    - Sorgu Yönetimi: Her soruya odaklanarak yanıt ver, yanıt oluşturmadan önce soruyu tamamen anladığından emin ol. Çok yönlü veya belirsiz hukuki soruların olduğu durumlarda, daha fazla bilgi alabilmek için ekstra sorular sorabilirsin. Gerekirse, bağlamda bulunan incelikleri veya farklı yorumları belirt, ele alınan hukuki konunun kapsamlı bir görünümünü sağla. \

    - Etik Hususlar: Yanıtlarında her zaman en yüksek hukuki etik standartlarına bağlı kal. Tüm cevaplarında gizliliğe, tarafsızlığa ve hukuk sisteminin bütünlüğüne saygı göster. \

    Bu yönergeleri takip ederek, görevin, hukuki karar alma sürecini iyileştirmek, hukuki eğitime yardımcı olmak ve doğru, iyi kaynaklanmış ve profesyonelce ifade edilmiş hukuki bilgilerle adli sistemini desteklemektir.> \

    <kullanıcı sorusu>{question}</kullanıcı sorusu>

    <bağlam>{context}</bağlam>
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Vector Database mounting
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    persist_directory = 'docs/chroma/'
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    index_name = "llm-hukuk-butun"
    vectordb = Pinecone.from_existing_index(index_name, embeddings)

    # Chain initilization
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),  # search_type='mmr'
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": query_text})
    return result["result"]

# Page title
st.set_page_config(page_title='📖 Plus Lawyer Chatbot')
st.title('📖 Plus Lawyer Chatbot')

# File upload
#uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Lütfen sorunuzu giriniz: ', placeholder = 'Karakter limiti yoktur, sorularınızı açık bir şekilde sorabilirsiniz.')#, disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (query_text))
    submitted = st.form_submit_button('Gönder', disabled=not(query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Analiz Ediliyor...'):
            response = generate_response(openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
