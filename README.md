Explanation:

1. The code uses two main libraries:

transformers: To load a multilingual BERT model for sentiment classification.

openai: To analyze negative reviews using the GPT-4o-mini model.



2. First, it loads the BERT sentiment model (from Hugging Face) to determine if the review is positive or negative (1–5 stars scale).


3. If the review is negative, it sends it to GPT-4o-mini to extract the reasons and recommendations.


4. Streamlit builds the web interface, which includes:

A main title

A text box to input the review

A button to analyze the review

The output (sentiment and analysis)

Your personal blog link at the bottom



شرح بالعربية:

1.   الكود يستخدم مكتبتين رئيسيتين:

transformers: لتحميل نموذج BERT لتصنيف المشاعر (إيجابي / سلبي).

openai: لاستخدام نموذج GPT-4o-mini لتحليل المراجعة السلبية واقتراح توصيات.



2.   يتم أولًا تحميل نموذج BERT متعدد اللغات لتصنيف المشاعر.


3.   يتم استخدام النموذج لتحديد ما إذا كانت المراجعة إيجابية أم سلبية (من 1 إلى 5 نجوم).


4.   إذا كانت سلبية، يتم إرسالها إلى GPT-4o-mini لتحليل أسباب السلبية ثم تقديم توصيات.


5.   Streamlit يُستخدم لبناء واجهة رسومية بسيطة تحتوي على:

عنوان رئيسي

مربع نص لإدخال المراجعة

زر لتحليل المراجعة

عرض التصنيف والتحليل إن وجد

رابط الموقع الشخصي في الأسفل.
https://tarecko.blogspot.com
