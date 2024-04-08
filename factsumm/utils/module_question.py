from typing import List, Dict, Optional
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from requests.exceptions import HTTPError
import torch
from factsumm.utils.utils import unflatten


def load_qg(model: str, device: str, batch_size: int = None):
    """
    Load Question Generation model from HuggingFace hub and return either a batch processing or a single-instance processing function based on batch_size.

    Args:
        model (str): Model name to be loaded.
        device (str): Device info ('cpu' or 'cuda').
        batch_size (int, optional): Number of sentence-entity pairs to process in a single batch. If None, processes one sentence at a time.

    Returns:
        function: A question generation function that either processes inputs in batches or individually.
    """
    logging.debug("Loading Question Generation Pipeline...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    except (HTTPError, OSError) as e:
        logging.warning(f"Input model is not supported by HuggingFace Hub or failed to load. Error: {e}")
        return None

    def generate_question(sentences: List[str], total_entities: List):
        """
        Generation question using context and entity information

        Args:
            sentences (List[str]): list of sentences
            total_entities (List): list of entities

        Returns:
            List[Dict] list of question and answer (entity) pairs

        """
        qa_pairs = []

        for sentence, line_entities in zip(sentences, total_entities):
            dedup = {}
            for entity in line_entities:
                entity = entity["word"]
                if entity in dedup:
                    continue

                template = f"answer: {entity}  context: {sentence} </s>"

                tokens = tokenizer(
                    template,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                outputs = model.generate(**tokens, max_length=64)

                question = tokenizer.decode(outputs[0])
                question = question.replace("</s>", "")
                question = question.replace("<pad> question: ", "")

                qa_pairs.append({
                    "question": question,
                    "answer": entity,
                })

                dedup[entity] = True

        return qa_pairs

    def batch_generate_questions(batch_sentences: List[List[str]], batch_total_entities: List[List[Dict]]):
        """
        Generate questions using context and entity information, filling in as many templates to the model as possible to maximize parallelization.

        Args:
            batch_sentences (List[List[str]]): List of lists of sentences, each list corresponding to a batch.
            batch_total_entities (List[List[Dict]]): List of lists of entities for each sentence, corresponding to each batch.

        Returns:
            List[List[Dict]]: List of lists of question and answer (entity) pairs for each batch, maximizing parallel processing.
        """
        all_templates = []  # Accumulate all templates across all batches
        all_entities = []
        all_questions = []
        structure = []
        # Prepare all templates and their corresponding entities
        for sentences, entities_list in zip(batch_sentences, batch_total_entities):
            sub_structure = []
            for sentence, entities in zip(sentences, entities_list):
                dedup = set()  # Use a set for efficient deduplication
                for entity_dict in entities:
                    entity = entity_dict["word"]
                    if entity not in dedup:
                        sub_structure.append(0)
                        dedup.add(entity)
                        template = f"answer: {entity} context: {sentence} </s>"
                        all_templates.append(template)
                        all_entities.append(entity)
            structure.append(sub_structure)


        # Process all templates in large batches
        for batch_start in range(0, len(all_templates), batch_size):
            batch_end = batch_start + batch_size
            batch_templates = all_templates[batch_start:batch_end]

            # Tokenize batch of templates
            tokens = tokenizer(batch_templates, padding=True, max_length=512, truncation=True, return_tensors="pt").to(
                device)

            # Generate questions for the batch
            outputs = model.generate(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, max_length=64)
            questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            questions = [question.replace("question: ", "") for question in questions]
            all_questions.extend(questions)
        all_qa_pairs = [{"question": question, "answer": entity} for question, entity in zip(all_questions, all_entities)]
        batch_qa_pairs = unflatten(all_qa_pairs, structure)
        return batch_qa_pairs

    if batch_size is not None:
        return lambda sentences, entities: batch_generate_questions(sentences, entities)
    else:
        return lambda sentence, entities: generate_question(sentence, entities)


def load_qa(model: str, device: str, batch_size: Optional[int] = None ):
    """
    Load Question Answering model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info

    Returns:
        function: question answering function

    """
    logging.debug("Loading Question Answering Pipeline...")

    try:
        qa = pipeline(
            "question-answering",
            model=model,
            tokenizer=model,
            framework="pt",
            device=-1 if device == "cpu" else 0,
            batch_size=batch_size,
            torch_dtype=torch.bfloat16
        )
    except (HTTPError, OSError):
        logging.warning("Input model is not supported by HuggingFace Hub")

    def answer_question(context: str, qa_pairs: List):
        """
        Answer question via Span Prediction

        Args:
            context (str): context to be encoded
            qa_pairs (List): Question & Answer pairs generated from Question Generation pipe

        """
        answers = []
        qa.handle_impossible_answers = True
        for qa_pair in qa_pairs:
            pred = qa(
                question=qa_pair["question"],
                context=context,
            )["answer"]
            answers.append({
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "prediction": pred if pred != "" else "<unanswerable>"
            })
        return answers

    def batch_answer_question(contexts: List[str], qa_pairs_list: List[List[Dict]]):
        """
        Batch answer questions via Span Prediction for a list of contexts and corresponding question-answer pairs.

        Args:
            contexts (List[str]): List of contexts to be encoded.
            qa_pairs_list (List[List[Dict]]): List of lists of question-answer pairs generated from the Question Generation pipe.

        Returns:
            List[List[Dict]]: List of lists containing answers for each question in qa_pairs, for each context.
        """
        all_ref_answers = []
        all_questions = []
        # Assuming each context in `contexts` corresponds to a list of QA pairs in `qa_pairs_list`
        for context, qa_pairs in zip(contexts, qa_pairs_list):
            for qa_pair in qa_pairs:
                all_questions.append({"question": qa_pair["question"], "context": context})
                all_ref_answers.append(qa_pair["answer"])

        # Process the batch of questions for the current context
        qa.handle_impossible_answer = True
        predictions = qa(all_questions)
        predictions = [prediction["answer"] for prediction in predictions]
        # Organize predictions with their corresponding questions and answers
        all_answers = []
        for qa_pair, answer, pred in zip(all_questions, all_ref_answers, predictions):
            all_answers.append({"question": qa_pair["question"], "answer": answer, "prediction": pred if pred != "" else "<unanswerable>"})
        batch_answers = unflatten(all_answers, qa_pairs_list)
        return batch_answers

    if batch_size is None:
        return lambda context, qa_pair: answer_question(context, qa_pair)
    else:
        return lambda contexts, qa_pairs_list: batch_answer_question(contexts, qa_pairs_list)
