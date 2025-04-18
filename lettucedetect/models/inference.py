import json
import os
import re
from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationDataset,
)

PROMPT_QA = """
Briefly answer the following question:
{question}
Bear in mind that your response should be strictly based on the following {num_passages} passages:
{context}
In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
output:
"""

PROMPT_SUMMARY = """
Summarize the following text:
{text}
output:
"""

PROMPT_LLM = """
<task>
You will act as an expert annotator to evaluate an answer against a provided source text.
The source text will be given within <source>... </source> XML tags.
The answer  will be given within <answer>... </answer> XML tags.

For each answer, follow these steps:
Step 1: Read and fully understand the answer in german. The answer is a text containing information related to the source text.
Step 2: Thoroughly analyze how the answer relates to the information in the source text. Determine whether the answer contains hallucinations. Hallucinations are sentences that contain one of the following information:
    a. conflict: instances where the answer presents direct contraction or opposition to the original source.
    b. baseless info: instances where the generated answer includes information which is not inferred from the original source.
Step 3: Determine whether the answer contains any hallucinations. If no hallucinations are found, return an empty list.
Step 4: Compile the labeled hallucinated spans found into a JSON dict, with a key "hallucination list" and its value is a list of
hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
list": [hallucination span1, hallucination span2, ...]}}. In case of no hallucinations, please output an empty list : {{"hallucination
list": []}}.
Output only the JSON dict.

</task>

Given below are three examples for you to comprehend the task.
<example1>


Source: Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs? Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen.
Answer: Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 69 Millionen.

1.The answer states that Paris is capital of France. This matches the source and is correct.
2.The answer states that the population of France is 69 million. This condradicts the source that the population is actually 67 million. 
Hallucination -> "Die Bevölkerung von Frankreich beträgt 69 Millionen."
Therefore, output only {{"hallucination list": ["Die Bevölkerung Frankreichs beträgt 69 Millionen." ]}}
</example1>

<example2>
Source: Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs?  Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung von Frankreich beträgt 67 Millionen.
Answer: Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung von Frankreich beträgt 67 Millionen, und die Amtssprache ist Spanisch.

1.The answer states that Paris is capital of France. This matches the source and is correct.
2.The answer states that the population of France is 69 million. This matches the source and is correct.
3. The answer states that the language spoken in France is Spanish. This is incorrect and not supported by the source.
Hallucination -> "die Amtssprache ist Spanisch"
Therefore, output only {{"hallucination list": ["die Amtssprache ist Spanisch" ]}}

</example2>

<example3>
Source: Was ist die Hauptstadt von Österreich? Wie hoch ist die Bevölkerung Österreich? Österreich ist ein Land in Europa. Die Hauptstadt von Österreich ist Wien. Die Bevölkerung Österreichs beträgt 9.1 Millionen.
Answer: Die Hauptstadt von Österreich ist Wien. Die Bevölkerung Österreichs beträgt 9.1 Millionen.
1.The answer states that Vienna is capital of Austria. This matches the source and is correct.
2.The answer states that the population of Austria is 9.1 million. This matches the source and is correct.
Hallucination -> No hallucinations found
Therefore, output only {{"hallucination list": []}}
</example3>

\n 
<source>
{context}
</source>
\n 
<answer>
{answer}
</answer>
)"""


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Given a context and an answer, returns predictions.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        pass


class TransformerDetector(BaseDetector):
    def __init__(self, model_path: str, max_length: int = 4096, device=None, **kwargs):
        """Initialize the TransformerDetector.

        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.

        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return PROMPT_SUMMARY.format(text=context_str)
        else:
            return PROMPT_QA.format(
                question=question, num_passages=len(context), context=context_str
            )

    def _predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        # Use the shared tokenization logic from RagTruthDataset
        encoding, labels, offsets, answer_start_token = (
            HallucinationDataset.prepare_tokenized_input(
                self.tokenizer, context, answer, self.max_length
            )
        )

        # Create a label tensor: mark tokens before answer as -100 (ignored) and answer tokens as 0.
        labels = torch.full_like(encoding.input_ids[0], -100, device=self.device)
        labels[answer_start_token:] = 0
        # Move encoding to the device
        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }
        labels = torch.tensor(labels, device=self.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits
        token_preds = torch.argmax(logits, dim=-1)[0]
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Mask out predictions for context tokens.
        token_preds = torch.where(labels == -100, labels, token_preds)

        if output_format == "tokens":
            # return token probabilities for each token (with the tokens as well, if not -100)
            token_probs = []
            input_ids = encoding["input_ids"][0]  # Get the input_ids tensor from the encoding dict
            for i, (token, pred, prob) in enumerate(zip(input_ids, token_preds, probabilities)):
                if not labels[i].item() == -100:
                    token_probs.append(
                        {
                            "token": self.tokenizer.decode([token]),
                            "pred": pred.item(),
                            "prob": prob[1].item(),  # Get probability for class 1 (hallucination)
                        }
                    )
            return token_probs
        elif output_format == "spans":
            # Compute the answer's character offset (the first token of the answer).
            if answer_start_token < offsets.size(0):
                answer_char_offset = offsets[answer_start_token][0].item()
            else:
                answer_char_offset = 0

            spans: list[dict] = []
            current_span: dict | None = None

            # Iterate over tokens in the answer region.
            for i in range(answer_start_token, token_preds.size(0)):
                # Skip tokens marked as ignored.
                if labels[i].item() == -100:
                    continue

                token_start, token_end = offsets[i].tolist()
                # Skip special tokens with zero length.
                if token_start == token_end:
                    continue

                # Adjust offsets relative to the answer text.
                rel_start = token_start - answer_char_offset
                rel_end = token_end - answer_char_offset

                is_hallucination = (
                    token_preds[i].item() == 1
                )  # assuming class 1 indicates hallucination.
                confidence = probabilities[i, 1].item() if is_hallucination else 0.0

                if is_hallucination:
                    if current_span is None:
                        current_span = {
                            "start": rel_start,
                            "end": rel_end,
                            "confidence": confidence,
                        }
                    else:
                        # Extend the current span.
                        current_span["end"] = rel_end
                        current_span["confidence"] = max(current_span["confidence"], confidence)
                else:
                    # If we were building a hallucination span, finalize it.
                    if current_span is not None:
                        # Extract the hallucinated text from the answer.
                        span_text = answer[current_span["start"] : current_span["end"]]
                        current_span["text"] = span_text
                        spans.append(current_span)
                        current_span = None

            # Append any span still in progress.
            if current_span is not None:
                span_text = answer[current_span["start"] : current_span["end"]]
                current_span["text"] = span_text
                spans.append(current_span)

            return spans
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        prompt = self._form_prompt(context, question)
        return self._predict(prompt, answer, output_format)


class LLMDetector(BaseDetector):
    def __init__(self, model: str = "gpt-4o", temperature: int = 0):
        """Initialize the LLMDetector.

        :param model: OpenAI model.
        :param temperature: model temperature.
        """
        self.model = model
        self.temperature = temperature

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.
        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return PROMPT_SUMMARY.format(text=context_str)
        else:
            return PROMPT_QA.format(
                question=question, num_passages=len(context), context=context_str
            )

    def _create_labels(self, llm_content: str, answer: str) -> list:
        """Create hallucination labels for each answer."""
        labels = []
        match_dict = re.search(r"\{.*?\}", llm_content, re.DOTALL)
        try:
            hal_dict = match_dict.group(0)
            hal_dict = json.loads(hal_dict)
        except json.JSONDecodeError:
            return labels

        for hal in hal_dict["hallucination list"]:
            match = re.search(re.escape(hal), answer)
            if match:
                labels.append({"start": match.start(), "end": match.end(), "text": hal})

        return labels

    def _get_openai_client(self) -> OpenAI:
        """Get OpenAI client configured from environment variables.

        :return: Configured OpenAI client
        :raises ValueError: If API key is not set
        """
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"

        return OpenAI(
            api_key=api_key,
        )

    def _predict(self, context: str, answer: str, output_format: str = "spans") -> list:
        """Prompts the ChatGPT model to predict hallucination spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: works only for "spans" and returns grouped spans.
        """
        client = self._get_openai_client()

        if output_format == "spans":
            llm_prompt = PROMPT_LLM.format(context=context, answer=answer)
            llm_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": llm_prompt},
                ],
                temperature=self.temperature,
            )
            llm_content = llm_response.choices[0].message.content
            predictions = self._create_labels(llm_content, answer)
            return predictions
        else:
            raise ValueError(
                "Invalid output_format. This model can only predict hallucination spans. Use spans."
            )

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        """Predict hallucination spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "spans" to return grouped spans.
        """
        prompt = self._form_prompt(context, question)
        return self._predict(prompt, answer, output_format=output_format)


class HallucinationDetector:
    def __init__(self, method: str = "transformer", **kwargs):
        """Facade for the hallucination detector.

        :param method: "transformer" for the model-based approach.
        :param kwargs: Additional keyword arguments passed to the underlying detector.
        """
        if method == "transformer":
            self.detector = TransformerDetector(**kwargs)
        elif method == "llm":
            self.detector = LLMDetector(**kwargs)
        else:
            raise ValueError("Unsupported method. Choose 'transformer' or 'llm'.")

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        """
        return self.detector.predict(context, answer, question, output_format)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        """
        return self.detector.predict_prompt(prompt, answer, output_format)
