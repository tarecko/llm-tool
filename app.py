import streamlit as st
import openai

# إعداد واجهة Streamlit
st.set_page_config(page_title="تحليل التقييمات", layout="centered")
st.title("🧠 نظام تحليل التقييمات بالذكاء الصناعي")

# إعداد مفتاح OpenAI
openai.api_key = "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"

# تعريف الدالة الرئيسية
def analyze_review_with_gpt(text):
    prompt = f"""
أنت مساعد ذكي لتحليل التقييمات. بناءً على التقييم التالي، حدد أولاً هل هو إيجابي أم سلبي.
إذا كان سلبيًا، فقم بتحليل أسباب السلبية في شكل نقاط ثم اقترح حلولًا عملية أيضًا في شكل نقاط.

التقييم:
{text}

النتيجة:
1. التصنيف (إيجابي/سلبي): 
2. الأسباب:
- 
3. التوصيات:
-
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# واجهة المستخدم
user_input = st.text_area("✍️ أدخل التقييم هنا:", height=200)

if st.button("🔍 تحليل التقييم"):
    if not user_input.strip():
        st.warning("يرجى إدخال التقييم أولاً.")
    else:
        with st.spinner("جاري التحليل باستخدام GPT..."):
            result = analyze_review_with_gpt(user_input)
            st.markdown(result)
