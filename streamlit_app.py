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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def generate_response( openai_api_key, query_text):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # API KEY

    from pinecone import Pinecone
    pc = Pinecone(api_key="063b7989-52e9-44ff-9c57-2fb86b57cc66")
    os.environ["PINECONE_API_KEY"] = "063b7989-52e9-44ff-9c57-2fb86b57cc66"

    ## Model initilization
    llm_name = "gpt-4-1106-preview"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Template creation
    template = """<Hukuki konularda danışmanlık veren yüksek derecede uzmanlaşmış bir AI agent olarak davranacaksın. Sorumluluğun kullanıcı tarafından sorulan, hukuk ile ilgili karmaşık sorulara cevap vermektir. Verdiğin yanıtlar, yalnızca bağlamda yer alan bilgilere dayanmalıdır. Ancak, bağlamda yer alan bütün bilgileri kullanmak zorunda değilsin. Kullanıcının sorduğu soruyu analiz etmelisin ve yalnızca bu soruya cevap olacak bağlam bilgilerini kullanmalısın. Bunun haricinde, verdiğin yanıtlar şu standartlarla uyumlu olmalıdır:  \

    - Dil Şartları: Tüm yanıtları akıcı, profesyonel seviyede Türkçe olarak sun. En yüksek dil kalitesini korumak için Türkçe dilbilgisi, sözdizimi ve noktalama işaretlerine dikkat et. \

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
    from langchain.vectorstores import Chroma,Pinecone
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    index_name = "llm-hukuk-butun-1250"
    vectordb = Pinecone.from_existing_index(index_name, embeddings)

    question = query_text


    prompt = ChatPromptTemplate.from_template("Bu soruya cevap verir misin: {foo}")
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_llm = chain.invoke({"foo": question})
    print('Soruya cevap verildi. Augmentation DONE !')

    fin_question = question + ' ' + result_llm


    ff = vectordb.similarity_search(
        fin_question,  # our search query
        k=9  # return 3 most relevant docs
    )
    print('Augmente edilmis soruyla vectordb source tarama DONE')
    context = ""
    for ind,i in enumerate(ff):
        ind = "[{0}] ".format(str(ind+1))
        context = context + ind + '\"' + i.page_content + '\"' + '\n\n'

    prompt = """Bu sorguyla “{question}” ilgili pasajlar aşağıdadır. Bu pasajları, sorguyla alakalarına 
            göre sırala ve cevap olarak sadece pasajların alakalıdan alakasıza sıralı olduğu örnek çıktı formatındaki gibi 
            çıktı oluştur ve sıralamada bütün pasaj numaraları bulunsun. Yanıtta herhangi bir açıklama yapmak zorunda değilsin, mantık ve muhakemeyi arka planda kendin
            yapabilirsin. Çıktı içerisinde sıra bilgisini yazmana gerek yok ilk sıradaki en alakalı son sıradaki en az 
            alakalı olarak kabul edilecektir, sadece pasaj numaraları çıktı içerisinde yer alsın.
            Senden beklediğimiz örnek format: "[3, 2, ...]" 
            {context}
            """

    prompt = ChatPromptTemplate.from_template(prompt)
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_llm = chain.invoke({"question": question, "context":context})
    print('Siralama tamamlandi!')

    fin_passages = []
    for i in result_llm:
        if i.isnumeric():
            fin_passages.append(int(i))

    newcontext = ""
    try:
        new_array = []
        for ind in fin_passages[:5] :
            new_array.append(ff[ind-1].page_content)
            newcontext = newcontext  + '\"' + ff[ind-1].page_content + '\"' + '\n\n'
            print('Yeni context siralama ile olusturuldu!')

    except:
        newcontext = newcontext + context
        print('Siralama sicti, eldekilerle devam!')


    prompt = QA_CHAIN_PROMPT
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_llm = chain.invoke({"question": question, "context":newcontext})

    return result_llm

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
