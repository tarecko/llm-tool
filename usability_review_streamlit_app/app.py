import streamlit as st
import torch
import openai
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# إعداد واجهة Streamlit بالعربية
st.set_page_config(page_title="تحليل التقييمات", layout="centered")
st.title("🔍 نظام تحليل التقييمات باستخدام الذكاء الاصطناعي")

# إعداد مفتاح OpenAI الخاص بك (يرجى عدم مشاركته مع الآخرين)
openai.api_key = "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"

# تحميل النموذج والمدقق
@st.cache_resource
def load_model():
    class ReviewClassifier(nn.Module):
        def __init__(self, hidden_size=768, num_classes=6):
            super(ReviewClassifier, self).__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
            self.conv1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            lstm_output, _ = self.lstm(sequence_output)
            lstm_output = lstm_output.permute(0, 2, 1)
            conv_output = self.relu(self.conv1(lstm_output))
            pooled_output = torch.mean(conv_output, dim=2)
            out = self.dropout(pooled_output)
            return self.fc(out)

    model = ReviewClassifier()
    model.load_state_dict(torch.load("phase1_model.pth", map_location=torch.device("cpu")))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

label_map = {
    0: "Satisfaction - إيجابي",
    1: "Satisfaction - سلبي",
    2: "Completeness - إيجابي",
    3: "Completeness - سلبي",
    4: "Correctness - إيجابي",
    5: "Correctness - سلبي"
}

def classify_review(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(tokens["input_ids"], tokens["attention_mask"])
        prediction = torch.argmax(outputs, dim=1).item()
    return label_map[prediction], prediction

def analyze_negativity_with_gpt(text):
    prompt = f"""
أعطني تحليلًا نقديًا لهذا التقييم من حيث الأسباب المحتملة للانطباع السلبي، ثم اقتراحات عملية لتحسين التجربة. اعرض النتائج في شكل نقاط.

النص:
{text}

النتيجة:
- الأسباب:
1.
- التوصيات:
1.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# واجهة المستخدم
review_input = st.text_area("📝 أدخل التقييم هنا:", height=200)

if st.button("🔍 تحليل التقييم"):
    if not review_input.strip():
        st.warning("يرجى إدخال نص التقييم أولاً.")
    else:
        label_text, label_id = classify_review(review_input)
        st.success(f"📌 التصنيف: {label_text}")

        if "سلبي" in label_text:
            with st.spinner("يتم تحليل السلبية باستخدام GPT..."):
                result = analyze_negativity_with_gpt(review_input)
                st.markdown(result)
        else:
            st.info("✅ التقييم إيجابي، لا حاجة لتحليل إضافي.")