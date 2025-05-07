import streamlit as st
import openai
from openai import OpenAI
import os

# واجهة Streamlit
st.set_page_config(page_title="تحليل التقييمات", layout="centered")
st.title("🤖 ناهد الصالح تحليل التقييمات ")

# إعداد مفتاح OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# دالة تحليل التقييم
def analyze_review_with_gpt(review_text):
    prompt = f"""
نص التقييم:
{review_text}

حدد هل التقييم سلبي أو إيجابي، وإذا كان سلبيًا، قدم تحليلًا للأسباب المحتملة على شكل نقاط، ثم قدم توصيات عملية أيضًا على شكل نقاط.

صيغة النتيجة:
- النوع: إيجابي / سلبي
- الأسباب:
1.
2.
- التوصيات:
1.
2.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # يمكن تغييره إلى gpt-4o لاحقًا
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# واجهة المستخدم
user_input = st.text_area("📝 أدخل التقييم هنا:", height=200)

if st.button("🔍 تحليل"):
    if not user_input.strip():
        st.warning("يرجى إدخال نص التقييم.")
    else:
        with st.spinner("جاري التحليل باستخدام GPT..."):
            try:
                result = analyze_review_with_gpt(user_input)
                st.markdown(result)
            except Exception as e:
                st.error(f"حدث خطأ أثناء الاتصال بـ OpenAI: {e}")