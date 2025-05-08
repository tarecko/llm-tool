import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from openai import OpenAI
import os

# 1. إعداد الصفحة
st.set_page_config(page_title="Review Analyzer", layout="wide")

# 2. اختيار اللغة
lang = st.sidebar.selectbox("🌐 Language / اللغة", ["English", "العربية"])

if lang == "العربية":
    st.title("📝 محلل التقييم")
    st.markdown(" ناهد الصالح ")
else:
    st.title("📝 Review Analyzer")
    st.markdown(" Nahed Al-Saleh ")
# 3. تحميل نموذج التصنيف (BERT)
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# 4. تصنيف التقييم
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item() + 1
    return "Positive" if predicted_class >= 4 else "Negative"

# 5. إعداد GPT Client
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# 6. تحليل المراجعة السلبية
def analyze_with_gpt(review, lang):
    if lang == "العربية":
        prompt = f"""مراجعة المستخدم:\n{review}\n\n
هذه مراجعة سلبية. استخرج أسباب السلبية في شكل نقاط، ثم قدم توصيات في شكل نقاط.\n
الإخراج المطلوب:\n
🟥 الأسباب:\n- ...\n- ...\n\n
🟩 التوصيات:\n- ...\n- ...
"""
        sys_msg = "أنت مساعد ذكي تحلل المراجعات السلبية وتقترح توصيات لتحسين تجربة المستخدم."
    else:
        prompt = f"""User review:\n{review}\n\n
This is a negative review. Extract the reasons for dissatisfaction as bullet points, and then provide recommendations as bullet points.\n
Output format:\n
🟥 Reasons:\n- ...\n- ...\n\n
🟩 Recommendations:\n- ...\n- ...
"""
        sys_msg = "You are a smart assistant that analyzes negative user reviews and suggests improvements."

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )
    return chat_response.choices[0].message.content

# 7. إدخال المستخدم
default_text = "Example: التطبيق لا يعمل بشكل جيد ويستغرق وقتًا طويلاً للتحميل." if lang == "العربية" else "Example: The app crashes frequently and takes too long to load."
review = st.text_area("✍️ Enter your review here" if lang == "English" else "✍️ أدخل تقييمك هنا", height=250, value=default_text)

# 8. زر التحليل
if st.button("🔍 Analyze Review" if lang == "English" else "🔍 تحليل المراجعة"):
    if not openai_api_key:
        st.error("❌ Please enter your OpenAI API key.")
    else:
        with st.spinner("Analyzing..."):
            sentiment = classify_sentiment(review)
            if lang == "العربية":
                st.subheader("🔍 التصنيف:")
                st.success("✅ إيجابي" if sentiment == "Positive" else "❌ سلبي")
            else:
                st.subheader("🔍 Sentiment:")
                st.success("✅ Positive" if sentiment == "Positive" else "❌ Negative")

            if sentiment == "Negative":
                result = analyze_with_gpt(review, lang)
                st.markdown("🤖 **Analysis Result:**" if lang == "English" else "🤖 **نتيجة التحليل:**")
                st.markdown(result)
            else:
                st.info("No further analysis needed." if lang == "English" else "لا حاجة لمزيد من التحليل.")


st.markdown("[Visit my blog](https://tarecko.blogspot.com)")
