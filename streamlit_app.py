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
from langchain_pinecone.vectorstores import PineconeVectorStore

def generate_response( openai_api_key, query_text):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # API KEY

    ## Model initilization
    llm_name = "gpt-4-1106-preview"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # vectordb = Pinecone.from_documents(splits,embeddings,index_name="llm-hukuk")
    index_names = ["llm-hukuk-kanun", "llm-hukuk-butun-1250", "llm-hukuk-talimat"]
    pinecone_api_keys = [
        "8a5d141e-c4a3-4baa-9f05-fa3840dd6a0b",
        "063b7989-52e9-44ff-9c57-2fb86b57cc66",
        "ddf2ca90-ae60-4769-a7ec-23a97bf44845",
    ]
    vector_databases = [
        PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_keys[i],
        )
        for i, index_name in enumerate(index_names)
    ]
    print("Vectordbs olusturuldu!")
    
    # Template creation
    template = """<Hukuki konularda danışmanlık veren yüksek derecede uzmanlaşmış bir AI agent olarak davranacaksın. Sorumluluğun kullanıcı tarafından sorulan, hukuk ile ilgili karmaşık sorulara cevap vermektir. Verdiğin yanıtlar, bağlamda yer alan bilgilere dayanmalıdır. Ancak, bağlamda yer alan bütün bilgileri kullanmak zorunda değilsin. Kullanıcının sorduğu soruyu analiz etmelisin ve yalnızca bu soruya cevap olacak bağlam bilgilerini kullanmalısın. Bunun haricinde, verdiğin yanıtlar şu standartlarla uyumlu olmalıdır:  \

    - Dil Şartları: Tüm yanıtları akıcı, profesyonel seviyede Türkçe olarak sun. En yüksek dil kalitesini korumak için Türkçe dilbilgisi, sözdizimi ve noktalama işaretlerine dikkat et. \

    - Konu Sınırlaması: Bağlam içerisinde soruya cevap olabilecek bilgiler yer almıyorsa, "Bu soruya cevap vermem mümkün değildir. Hukuki alandaki sorularınızda size yardımcı olmaktan memnuniyet duyarım." diyerek cevap ver ve sakın başka bir cevap hazırlama. \

    - Hedef Kitle: Unutma, temel kullanıcıların hukuk alanındaki profesyoneller - avukatlar, yargıçlar ve savcılardır. Dilini ve açıklamalarını, onların hukuki terimler ve kavramlar konusundaki ileri düzey bilgilerine uygun hale getir. Profesyonel dil kullanımını sürdürürken, yanıtlarının hedeflenen hukuki kitle tarafından açık ve kolayca anlaşılabilir olduğundan emin ol. Bilgiyi belirsiz hale getirebilecek aşırı karmaşık cümlelerden kaçın. \

    - Cevapların Yapısı: Cevabını mantıklı bir şekilde düzenle. Sorunun kısa bir özeti ile başla, ardından detaylı bir analizle devam et ve son olarak özlü bir sonuç veya özet ifade ile bitir.

    - Cevapların Spesifikliği: Sorulara yanıt verirken, yalnızca hukuki konulara odaklan. Detaylı, kesin, açık ve hukuki olarak sağlam tavsiyeler veya açıklamalar sunmalısın. Hazırladığın yanıtları bağlamda yer alan bilgilere dayandırmaya özen göster ve cevap içerisinde tekrara düşme. Yanıtlarını her zaman düzenle, düzelt ve kullanıcıya tam cevap verdiğinden emin ol. \

    - Kaynak Gösterme: Sağladığın her cevap için, bilginin alındığı belgeyi, varsa kanunun veya yönetmeliğin ismini belirtmek, referansın doğru ve sorulan soruyla alakalı olduğundan emin olunması açısından kesinlikle hayati öneme sahiptir. Bu atıf, belgenin adını, bölüm numarasını ve kanun veya yönetmelik madde numaralarını içerebilir. Ancak, bağlamları kullanarak yanıt hazırladığını cevabında kesinlikle belirtmemelisin. \

    - Tavsiye Kapsamı: Bilgi sağlarken, kişisel görüşler, yorumlar veya hukuki tavsiyeler sunmaktan kaçınmak önemlidir. Görevin, bir bilgi aracı olmak, hukuki belgelerde belirtilen gerçekleri ve detayları, genelleme veya kişisel yargı olmadan sunmaktır. Bağlamda açıkça sunulmamış olan konularda varsayımlar yapmaktan veya çıkarımlardan kaçın.\

    - Sorgu Yönetimi: Her soruya odaklanarak yanıt ver, yanıt oluşturmadan önce soruyu tamamen anladığından emin ol. Çok yönlü veya belirsiz hukuki soruların olduğu durumlarda, daha fazla bilgi alabilmek için ekstra sorular sorabilirsin. Gerekirse, bağlamda bulunan incelikleri veya farklı yorumları belirt, ele alınan hukuki konunun kapsamlı bir görünümünü sağla. \

    - Etik Hususlar: Yanıtlarında her zaman en yüksek hukuki etik standartlarına bağlı kal. Tüm cevaplarında gizliliğe, tarafsızlığa ve hukuk sisteminin bütünlüğüne saygı göster. \

    Bu yönergeleri takip ederek, görevin, hukuki karar alma sürecini iyileştirmek, hukuki eğitime yardımcı olmak ve doğru, iyi kaynaklanmış ve profesyonelce ifade edilmiş hukuki bilgilerle adli sistemini desteklemektir.> \

    <kullanıcı sorusu>{question}</kullanıcı sorusu>

    <bağlam>{context}</bağlam>
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


    question = query_text


    prompt = ChatPromptTemplate.from_template(
        "Türk hukuku hakkında danışmanlık veren yüksek derecede uzmanlaşmış bir AI agent olarak davranacaksın. Sorumluluğun kullanıcı tarafından sorulan, hukuk ile ilgili karmaşık sorulara cevap vermektir. Temel kullanıcıların hukuk alanındaki profesyoneller - avukatlar, yargıçlar ve savcılardır. Sorulara yanıt verirken, yalnızca hukuki konulara odaklan. Detaylı, kesin, açık ve hukuki olarak sağlam tavsiyeler veya açıklamalar sunmalısın. Yanıtlarını her zaman düzenle, düzelt ve kullanıcıya tam cevap verdiğinden emin ol. <kullanıcı sorusu>{foo}</kullanıcı sorusu>"
    )
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_llm_aug = chain.invoke({"foo": question})
    print("Soruya cevap verildi. Augmentation DONE !")
    
    fin_question = question + " " + result_llm_aug
    
    ff = [
        vector_database.similarity_search(
            fin_question, k=7  # our search query  # return 7 most relevant docs
        )
        for vector_database in vector_databases
    ]
    ff = [ff[i][j] for i in range(len(ff)) for j in range(len(ff[i]))]
    
    print("Augmente edilmis soruyla vectordb source tarama DONE")
    context = ""
    for ind, i in enumerate(ff):
        ind = "'[{0}]' ".format(str(ind + 1))
        context = context + ind + '"{' + i.page_content + '}"' + "\n\n"
    
    prompt = """Türk hukuku hakkında bilgi sahibi yüksek derecede uzmanlaşmış bir AI agent olarak davranacaksın. Görevin, sana verilen pasajların, kullanıcı sorgusuyla olan alakalarını değerlendirmek ve bir sıralama yapmaktır.
    
    Kullanıcı sorgusu “{question}” ile ilgili pasajlar aşağıdadır. Pasaj numaraları, pasajların başında '[pasaj numarası]' şeklinde yer almaktadır.
    
    Bu pasajları, sorguyla alakalarına göre sıralamalısın. Cevap olarak sadece pasajların numaralarının alakalıdan alakasıza sıralı olduğu örnek çıktı formatındaki gibi bir çıktı oluşturmalısın. Oluşturduğun cevapta bütün pasajların numaraları bulunmalıdır. Yanıtta herhangi bir açıklama yapmak zorunda değilsin, mantık ve muhakemeyi arka planda kendin yapmalısın. Çıktı içerisinde sıra bilgisini yazmana gerek yoktur. İlk sıradaki en alakalı, son sıradaki en az alakalı olarak kabul edilecektir. Sadece pasaj numaralarına çıktı içerisinde yer ver.
    
    Kullanılması gereken cevap formatı: "[3, 2, …]” 
    
    <pasajlar>{context}</pasajlar>
    """
    
    prompt = ChatPromptTemplate.from_template(prompt)
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_ordering = chain.invoke({"question": question, "context": context})
    print("Siralama tamamlandi!")
    
    newcontext = ""
    try:
        # check if the response is valid or not
        for idx, i in enumerate(result_ordering):
            if idx == 0:
                assert (
                    result_ordering[idx] == "["
                ), "The first response character should be '['"
                continue
    
            if idx == len(result_ordering) - 1:
                assert (
                    result_ordering[idx] == "]"
                ), "The last response character should be ']'"
                continue
    
            assert (
                i.isnumeric() or i == "," or i == " "
            ), f"Invalid character in idx {idx}: {i}"
    
        # extract the passage numbers
        fin_passages = [int(s) for s in result_ordering[1:-1].split(",")]
    
        new_array = []
        for ind in fin_passages[:7]:
            new_array.append(ff[ind - 1].page_content)
            newcontext = newcontext + '"' + ff[ind - 1].page_content + '"' + "\n\n"
            print("Yeni context siralama ile olusturuldu!")
    
    except Exception as exception:
        print(exception)
        newcontext = newcontext + context
        print("Siralama sicti, eldekilerle devam!")
    
    
    prompt = QA_CHAIN_PROMPT
    model = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    result_llm_cevap = chain.invoke({"question": question, "context": newcontext})

    return result_llm_cevap

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
