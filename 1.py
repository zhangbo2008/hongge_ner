from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# # Sentiment analysis pipeline
# analyzer = pipeline("sentiment-analysis")

# # Question answering pipeline, specifying the checkpoint identifier
# oracle = pipeline(
#     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
# )

# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")
tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
recognizer = pipeline("ner", model=model, tokenizer=tokenizer)


print(recognizer('我叫克拉拉，我住在加州伯克利。'))