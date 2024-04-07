import logging
from typing import List, Tuple, Dict, Optional
import torch
from requests import HTTPError
from transformers import LukeForEntityPairClassification, LukeTokenizer, pipeline


def load_ner(model: str, device: str, batch_size: Optional[int] = None) -> object:
    """
    Load Named Entity Recognition model from HuggingFace hub

    Args:
        model (str): model name to be loaded
        device (str): device info
        batch_size (int): batch size

    Returns:
        object: Pipeline-based Named Entity Recognition model

    """
    logging.debug("Loading Named Entity Recognition Pipeline...")

    try:
        ner = pipeline(
            task="ner",
            model=model,
            tokenizer=model,
            ignore_labels=[],
            framework="pt",
            device=-1 if device == "cpu" else 0,
            aggregation_strategy="simple",
            batch_size=batch_size,
            torch_dtype=torch.bfloat16
        )
    except (HTTPError, OSError):
        logging.warning("Input model is not supported by HuggingFace Hub")
        raise

    def extract_entities(sentences: List[str]):
        total_entities = ner(sentences)

        result = []
        for line_entities in total_entities:
            result.append([entity for entity in line_entities if entity["entity_group"] != "O"])

        return result
    return extract_entities


def load_rel(model: str, device: str, batch_size: Optional[int] = None):
    """
    Load LUKE for Relation Extraction model and return its applicable function based on batch processing preference.

    Args:
        model (str): Model name to be loaded.
        device (str): Device info ('cpu' or 'cuda').
        use_batch (bool): Whether to use batch processing.
        batch_size (int): The number of sentences to process in a single batch, applicable if use_batch is True.

    Returns:
        function: A function for relation extraction that either processes inputs in batches or individually.
    """
    logging.debug("Loading Relation Extraction Pipeline...")

    try:
        tokenizer = LukeTokenizer.from_pretrained(model)
        model = LukeForEntityPairClassification.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    except (HTTPError, OSError) as e:
        logging.warning("Input model is not supported by HuggingFace Hub or failed to load. Error: {}".format(e))
        return None

    def extract_relation_single(sentences: List) -> List[Tuple]:
        """
        Extraction Relation based on Entity Information

        Args:
            sentence (str): original sentence containing context
            head_entity (Dict): head entity containing position information
            tail_entity (Dict): tail entity containing position information

        Returns:
            List[Tuple]: list of (head_entity, relation, tail_entity) formatted triples

        """
        triples = []

        # TODO: batchify
        for sentence in sentences:
            tokens = tokenizer(
                sentence["text"],
                entity_spans=[
                    (sentence["spans"][0][0], sentence["spans"][0][-1]),
                    (sentence["spans"][-1][0], sentence["spans"][-1][-1]),
                ],
                return_tensors="pt",
            ).to(device)
            outputs = model(**tokens)
            predicted_id = int(outputs.logits[0].argmax())
            relation = model.config.id2label[predicted_id]

            if relation != "no_relation":
                triples.append((
                    sentence["text"]
                    [sentence["spans"][0][0]:sentence["spans"][0][-1]],
                    relation,
                    sentence["text"]
                    [sentence["spans"][-1][0]:sentence["spans"][-1][-1]],
                ))

        return triples

    def extract_relation_batch(sentences: List[Dict], batch_size: int) -> List[Tuple]:
        triples = []
        for batch_start in range(0, len(sentences), batch_size):
            batch_sentences = sentences[batch_start:batch_start + batch_size]
            batch_texts = [sentence["text"] for sentence in batch_sentences]
            batch_spans = [
                [(sentence["spans"][0][0], sentence["spans"][0][-1]),
                 (sentence["spans"][-1][0], sentence["spans"][-1][-1])]
                for sentence in batch_sentences
            ]
            batch_tokens = tokenizer(
                batch_texts,
                entity_spans=batch_spans,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**batch_tokens)
            for idx, sentence in enumerate(batch_sentences):
                predicted_id = int(outputs.logits[idx].argmax())
                relation = model.config.id2label[predicted_id]
                if relation != "no_relation":
                    head_entity = sentence["text"][sentence["spans"][0][0]:sentence["spans"][0][-1]]
                    tail_entity = sentence["text"][sentence["spans"][-1][0]:sentence["spans"][-1][-1]]
                    triples.append((head_entity, relation, tail_entity))
        return triples

    if batch_size is not None:
        return lambda sentences: extract_relation_batch(sentences, batch_size=batch_size)
    else:
        return lambda sentence: extract_relation_single(sentence)
