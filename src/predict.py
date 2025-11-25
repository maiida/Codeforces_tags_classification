import os
from pathlib import Path
import torch
import numpy as np
import weaviate
import ast
from groq import Groq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# Import the shared configuration
from src.config import FOCUS_TAGS, CODEBERT_MODEL_NAME
from src.utils import parse_tags


class CodeBertPredictor:
    """
    A predictor class for the fine-tuned CodeBERT model.
    """
    def __init__(self, model_name=CODEBERT_MODEL_NAME, device=None):
        """
        Initializes the predictor by loading the model and tokenizer.
        """
        # Use model from config if not specified
        model_name = CODEBERT_MODEL_NAME

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading CodeBERT model from: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tags_eval = FOCUS_TAGS
        
            

    def _make_input(self, description, code):
        """Prepares the input string for the model."""
        return f"[DESC] {description} [CODE] {code}"

    def _probs_to_tag_lists(self, probs, threshold=0.5, force_one=True):
        """
        Convert probabilities to a list of tag lists.
        """
        tag_lists = []
        for p in probs:
            indices = np.where(p >= threshold)[0]
            if len(indices) == 0 and force_one:
                indices = [int(np.argmax(p))]
            tag_lists.append([self.tags_eval[i] for i in indices])
        return tag_lists

    def predict(self, data, batch_size=32, threshold=0.5, max_length=512):
        """
        Makes predictions on a given dataset.
        """
        import time
        start_time = time.time()

        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Predicting with CodeBERT"):
                batch_data = data[i:i + batch_size]
                batch_texts = [self._make_input(row['description_clean'], row['code_clean']) for row in batch_data]

                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                outputs = self.model(**enc)
                logits = outputs.logits
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(batch_probs)

        probs = np.concatenate(all_probs, axis=0)
        predicted_tags = self._probs_to_tag_lists(probs, threshold=threshold)

        inference_time = time.time() - start_time
        return predicted_tags, inference_time


class RetrievalPredictor:
    """
    A predictor class using a retrieval-based approach from a Weaviate vector store.
    """
    def __init__(self, embed_model_name="BAAI/bge-m3", index_name="Test"):
        """
        Initializes the predictor by connecting to Weaviate and setting up the index.
        """
        cluster_url = os.getenv("WEAVIATE_CLUSTER_URL", "1scymfxircmnej7ffsnba.c0.europe-west3.gcp.weaviate.cloud")
        api_key = os.getenv("WEAVIATE_API_KEY")

        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
            print("Successfully connected to Weaviate.")
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            client = None

        self.client = client
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        
        if self.client:
            vector_store = WeaviateVectorStore(
                weaviate_client=self.client,
                index_name=index_name,
            )
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
            )
        else:
            self.index = None

    def _build_query(self, description, code):
        """Prepares the query string from a data row."""
        return f"{description}\n\n{code}"

    def predict_single(self, description, code, k=3, min_votes=1):
        """
        Predicts tags for a single example.
        """
        if not self.index:
            print("Predictor not initialized due to Weaviate connection failure.")
            return []
            
        query_text = self._build_query(description, code)
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query_text)

        tag_counts = Counter()
        for node in nodes:
            tags = parse_tags(node.metadata.get("tags_filtered"))
            tag_counts.update(tags)

        pred_tags = [tag for tag, cnt in tag_counts.items() if cnt >= min_votes]
        return pred_tags

    def predict(self, data, k=3, min_votes=1):
        """
        Makes predictions on a given dataset.
        """
        import time
        start_time = time.time()

        all_preds = []
        for row in tqdm(data, desc="Predicting with Retrieval"):
            preds = self.predict_single(row['description_clean'], row['code_clean'], k=k, min_votes=min_votes)
            all_preds.append(preds)

        inference_time = time.time() - start_time
        return all_preds, inference_time

    def __del__(self):
        """Ensure the Weaviate client is closed on object deletion."""
        if self.client:
            self.client.close()
            print("Weaviate client closed.")


class LLMPredictor:
    """
    A predictor class using an LLM (Groq API with Llama 3.1 8B).
    """

    SYSTEM_PROMPT = """
You are an AI assistant tasked with classifying competitive programming problems into the correct algorithmic tags.

Each problem can belong to one or more tags from a predefined set.
Read the problem statement carefully and determine which tags best fit the content.

Use only the tags from the list below. Do NOT invent new tags.

Available Tags (multi-label: zero, one or several per problem):
    • math
    • graphs
    • strings
    • number theory
    • trees
    • geometry
    • games
    • probabilities

Few-Shot Examples

Example 1:
Problem:
numbers 1, 2, 3, ... n each integer from 1 to n once are written on a board. in one operation you can erase any two numbers a and b from the board and write one integer a + b divided by 2 rounded up instead.you should perform the given operation n - 1 times and make the resulting number that will be left on the board as small as possible. for example, if n = 4 , the following course of action is optimal choose a = 4 and b = 2 , so the new number is 3 , and the whiteboard contains 1, 3, 3 choose a = 3 and b = 3 , so the new number is 3 , and the whiteboard contains 1, 3 choose a = 1 and b = 3 , so the new number is 2 , and the whiteboard contains 2 . it s easy to see that after n - 1 operations, there will be left only one number. your goal is to minimize it.

Tags:
["math"]

Example 2:
Problem:
you are given an undirected graph consisting of n vertices. a number is written on each vertex the number on vertex i is a i . initially there are no edges in the graph.you may add some edges to this graph, but you have to pay for them. the cost of adding an edge between vertices x and y is a x + a y coins. there are also m special offers, each of them is denoted by three numbers x , y and w , and means that you can add an edge connecting vertices x and y and pay w coins for it. you don t have to use special offers if there is a pair of vertices x and y that has a special offer associated with it, you still may connect these two vertices paying a x + a y coins for it.what is the minimum number of coins you have to spend to make the graph connected recall that a graph is connected if it s possible to get from any vertex to any other vertex using only the edges belonging to this graph.

Tags:
["graphs"]


Example 3:
Problem:
now that kuroni has reached 10 years old, he is a big boy and doesn t like arrays of integers as presents anymore. this year he wants a bracket sequence as a birthday present. more specifically, he wants a bracket sequence so complex that no matter how hard he tries, he will not be able to remove a simple subsequence!we say that a string formed by n characters or is simple if its length n is even and positive, its first n divided by 2 characters are , and its last n divided by 2 characters are . for example, the strings and are simple, while the strings and are not simple.kuroni will be given a string formed by characters and the given string is not necessarily simple . an operation consists of choosing a subsequence of the characters of the string that forms a simple string and removing all the characters of this subsequence from the string. note that this subsequence doesn t have to be continuous. for example, he can apply the operation to the string , to choose a subsequence of bold characters, as it forms a simple string , delete these bold characters from the string and to get . kuroni has to perform the minimum possible number of operations on the string, in such a way that no more operations can be performed on the remaining string. the resulting string does not have to be empty.since the given string is too large, kuroni is unable to figure out how to minimize the number of operations. can you help him do it instead a sequence of characters a is a subsequence of a string b if a can be obtained from b by deletion of several possibly, zero or all characters.

Tags:
["strings"]



Example 4:
Problem:
there are n positive integers a 1, a 2, ..., a n . for the one move you can choose any even value c and divide by two all elements that equal c .for example, if a= 6,8,12,6,3,12 and you choose c=6 , and a is transformed into a= 3,8,12,3,3,12 after the move.you need to find the minimal number of moves for transforming a to an array of only odd integers each element shouldn t be divisible by 2 .

Tags:
["number theory"]


Example 5:
Problem:
you are given a tree with n vertices. you are allowed to modify the structure of the tree through the following multi-step operation choose three vertices a , b , and c such that b is adjacent to both a and c . for every vertex d other than b that is adjacent to a , remove the edge connecting d and a and add the edge connecting d and c . delete the edge connecting a and b and add the edge connecting a and c . as an example, consider the following tree the following diagram illustrates the sequence of steps that happen when we apply an operation to vertices 2 , 4 , and 5 it can be proven that after each operation, the resulting graph is still a tree.find the minimum number of operations that must be performed to transform the tree into a star. a star is a tree with one vertex of degree n - 1 , called its center, and n - 1 vertices of degree 1 .

Tags:
["graphs", "trees"]


Example 6:
Problem:
vasya has got three integers n , m and k . he d like to find three integer points x 1, y 1 , x 2, y 2 , x 3, y 3 , such that 0 <= x 1, x 2, x 3 <= n , 0 <= y 1, y 2, y 3 <= m and the area of the triangle formed by these points is equal to nm divided by k .help vasya! find such points if it s possible . if there are multiple solutions, print any of them.

Tags:
["number theory", "geometry"]


Example 7:
Problem:
the little girl loves problems on games very much. here s one of them.two players have got a string s, consisting of lowercase english letters. they play a game that is described by the following rules the players move in turns in one move the player can remove an arbitrary letter from string s. if the player before his turn can reorder the letters in string s so as to get a palindrome, this player wins. a palindrome is a string that reads the same both ways from left to right, and vice versa . for example, string abba is a palindrome and string abc isn t. determine which player will win, provided that both sides play optimally well the one who moves first or the one who moves second.

Tags:
["games"]


Example 8:
Problem:
petya loves lucky numbers. we all know that lucky numbers are the positive integers whose decimal representations contain only the lucky digits 4 and 7. for example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.petya and his friend vasya play an interesting game. petya randomly chooses an integer p from the interval pl, pr and vasya chooses an integer v from the interval vl, vr also randomly . both players choose their integers equiprobably. find the probability that the interval min v, p , max v, p contains exactly k lucky numbers.

Tags:
["probabilities"]


When I give you a new problem statement, you must:

1. Read and understand the problem.
2. Decide which tags from the list above apply.
3. Respond only with a valid JSON array of strings, containing the applicable tags.
4. If no tag clearly applies, respond with an empty list: [].

Do NOT add explanations, text, or formatting. Output must be JSON only.

Output format (examples):
["math", "number theory"]
["graphs"]
["games", "probabilities"]
[]
"""

    def __init__(self, api_key=None, model_name="llama-3.1-8b-instant"):
        """
        Initializes the LLM predictor.
        """

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.focus_tags = FOCUS_TAGS

    def _call_groq(self, description, code=None):
        """
        Calls the Groq API for a single prediction.
        """
        if code:
            user_content = f"Problem statement:\n{description}\n\nSolution code:\n{code}\n"
        else:
            user_content = f"Problem statement:\n{description}\n"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            top_p=1.0,
            stream=False,
        )

        return completion.choices[0].message.content

    def predict_single(self, description, code=None, max_retries=3):
        """
        Predicts tags for a single example with retry logic.
        """
        import time
        import json

        for attempt in range(max_retries):
            try:
                response = self._call_groq(description, code)
                tags = json.loads(response)
                if isinstance(tags, list):
                    return [t for t in tags if t in self.focus_tags]
                return []
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 5 * (attempt + 1)
                    print(f"Rate limit hit, sleeping {wait}s then retry...")
                    time.sleep(wait)
                else:
                    print(f"Error calling Groq API: {e}")
                    return []
        return []

    def predict(self, data, include_code=True):
        """
        Makes predictions on a given dataset.
        """
        import time
        start_time = time.time()

        all_preds = []
        for row in tqdm(data, desc="Predicting with LLM"):
            code = row.get('code_clean') if include_code else None
            preds = self.predict_single(row['description_clean'], code)
            all_preds.append(preds)

        inference_time = time.time() - start_time
        return all_preds, inference_time