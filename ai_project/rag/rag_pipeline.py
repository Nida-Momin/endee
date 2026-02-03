from transformers import pipeline

class RAGPipeline:
    def __init__(self):
        self.qa = pipeline(
            "question-answering",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base"
        )

    def answer(self, question, context):
        result = self.qa(question=question, context=context)
        return result["answer"]
