from enum import Enum
from sacremoses import MosesPunctNormalizer,MosesTokenizer
from abc import ABC, abstractmethod
import string
import os
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import numpy
import stanza
import json 
import spacy_udpipe
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
import simplemma

class Query:
    def __init__(self,num,x,operation,y,qrels,result_name):
        self.documents = 0
        self.relevant = 0
        self.precision = 0
        self.recall = 0

        self.__eval(x,y,operation)
        self.__get_relevant(num,qrels,result_name)
        self.__get_precision_recall(num,qrels)
    
    def __eval(self,documents_x,documents_y,operation):
        operator:Operator = None

        if operation == "AND":
            operator = And(documents_x)
        
        elif operation == "OR":
            operator = Or(documents_x)

        elif operation == "AND NOT":
            operator = AndNot(documents_x)
        
        operator.function(documents_y)
        
        self.documents = operator.result_documents
    
    def __get_relevant(self,num,qrels:dict[str,set[str]],result_name):
        used = set()
        with open(result_name,"w") as result_file:
            print(17)
            for document in sorted(self.documents):
                if document in qrels[num] and document not in used:
                    self.relevant += 1
                    result_file.write(f"{num} {document}\n")
                    used.add(document)
    
    def __get_precision_recall(self,num,qrels):
        self.precision = self.relevant / len(self.documents) if self.documents else 0
        self.recall = self.relevant / len(qrels[num]) if qrels.get(num) else 0

class Language(ABC):
    def __init__(self,tag,documents,topics_train,result_filename,qrels_filename,NLP):
        self.tag = tag
        self.documents = documents
        self.topics_train = topics_train
        self.indexInvertor = IndexInvertor(NLP)
        self.result_filename = result_filename
        self.qrels_filename = qrels_filename
        self.queries:list[Query] = []
        self.avarage_queries = None
        self.tokens = 0

        if not os.path.exists("inverted_index____" + self.tag + ".json"):
            self.__parse_documents()
        
        with open("inverted_index____" + self.tag + ".json", "r", encoding="utf-8") as f:
            inverted_index = json.load(f)

        qrels = self.__parse_qrels()
        self.__parse_queries(qrels,inverted_index,NLP)
        self.__get_average_queries()

    def get_tokens(self,text):
        tokenizer = MosesTokenizer(lang = self.tag)

        return tokenizer.tokenize(text)
    
    def __get_average_queries(self):
        metrics = numpy.array([(len(queries.documents), queries.relevant, queries.precision, queries.recall) for queries in self.queries])
        self.avarage_queries = numpy.mean(metrics, axis=0)

    def __parse_qrels(self):
        qrels:dict[str,set[str]] = {}
        with open(self.qrels_filename,"r") as result_file:
            for line in result_file:
                num, _, docid, relevant_num = line.strip().split(" ")

                if num not in qrels and relevant_num == "1":
                    qrels[num] = {docid}
                
                elif num in qrels and relevant_num == "1":
                    qrels[num].add(docid)
                    
        return qrels

    def __parse_documents(self):
        for xml_name in os.listdir(self.documents):
            xml_path = os.path.join(self.documents, xml_name)

            try:
                self.indexInvertor.read_xml(xml_path, self)
            
            except Exception:
                pass


            self.indexInvertor.sort_values()

        with open("inverted_index____" + self.tag + ".json", "w", encoding="utf-8") as f:
            json.dump(self.indexInvertor.terms, f, ensure_ascii=False, indent=2)

    def __parse_queries(self,qrels,indexInvertor,NLP):
        tree = ET.parse(self.topics_train)
        root = tree.getroot()

        for top in root.findall('top'):
            documents_x = []
            documents_y = []

            num = top.findtext('num').strip()
            query_text = top.findtext('query').strip()
            parts = query_text.split(" ")

            if len(parts) == 3:
                token_x, operator_tag, token_y = parts

            elif len(parts) == 4 and parts[1]== "AND" and parts[2] == "NOT":
                token_x = parts[0]
                operator_tag = "AND NOT"
                token_y = parts[3]

            normalize_token_x = IndexInvertor(NLP).normalize(token_x)
            normalize_token_y = IndexInvertor(NLP).normalize(token_y)

            try:
                documents_x = indexInvertor[normalize_token_x]
                documents_y = indexInvertor[normalize_token_y]

            except Exception:
                pass

            query = Query(num,documents_x,operator_tag,documents_y,qrels,self.result_filename)
            self.queries.append(query)
    
    def add_token(self):
        self.tokens += 1

class Czech(Language):
    def __init__(self, tag = "cs",DOCUMENTS = "documents_cs",train ="topics-train_cs.xml",RESULT_FILENAME = "results-cs.dat",QRELS_TRAIN ="qrels-train_cs.txt",NLP = "cs"):
        super().__init__(tag,DOCUMENTS,train,RESULT_FILENAME,QRELS_TRAIN,NLP)

class English(Language):
    def __init__(self, tag = "en",DOCUMENTS = "documents_en",train="topics-train_en.xml",RESULT_FILENAME = "results-en.dat",QRELS_TRAIN = "qrels-train_en.txt",NLP = "en"):
        super().__init__(tag,DOCUMENTS,train,RESULT_FILENAME,QRELS_TRAIN,NLP)

class IndexInvertor:
    def __init__(self,nlp):
        self.nlp = nlp
        self.terms: dict[str, list[str]] = {}

    def __add(self, token: str, id_document: str):
        if token not in self.terms:
            self.terms[token] = [id_document]
        elif id_document not in self.terms[token]:
            self.terms[token].append(id_document)

    def normalize(self, word: str) -> str:
        word_clean = ''.join(c.lower() for c in word if c not in string.punctuation)
        if  len(word_clean) > 0:
            return simplemma.lemmatize(word_clean, self.nlp)
        
        return ""
    
    def read_xml(self, xml_path: str, language: Language):
        print(f"Reading: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for doc in root.findall('DOC'):
            id_document = doc.findtext('DOCID')

            texts = []

            for descendant in doc.findall("*"):
                if descendant.tag == "DOCID":
                    continue

                if descendant.text:
                    texts.append(descendant.text.strip())

            completeText = " ".join(texts)
            tokens = language.get_tokens(completeText)

            for token in tokens:
                normalized_token = self.normalize(token)

                if normalized_token.isalpha():
                    language.add_token()
                    self.__add(normalized_token, id_document)

    def sort_values(self):
        for key in self.terms:
            self.terms[key] = sorted(self.terms[key])

class Operator(ABC):
    def __init__(self,x):
        self.result_documents = []
        self.used = set(x)

    @abstractmethod
    def function(self,y):
        pass

class And(Operator):
    def __init__(self, x):
        super().__init__(x)

    def function(self,y):
        for document_y in y:
            if document_y in self.used and document_y not in self.result_documents:
                self.result_documents.append(document_y)
    
class Or(Operator):
    def __init__(self, x):
        super().__init__(x)

    def function(self,y):
        for document_x in self.used:
            self.result_documents.append(document_x)
        
        for document_y in y:
            if document_y not in self.used and document_y not in self.result_documents:
                self.result_documents.append(document_y)

class AndNot(Operator):
    def function(self,y):
        for document_y in y:
            if document_y not in y and document_y not in self.result_documents:
                self.result_documents.append(document_y)


def google_task_questions(inverted_index):
    number_of_unique_terms = len(inverted_index)
    number_of_all_postings = 0
    term_with_highest_document_frequency = ""
    highest_document_frequency = 0
    sum_posting_len = 0

    for term, postings in inverted_index.items():
        len_postings = len(postings)
        number_of_all_postings += 1    

        if len_postings > highest_document_frequency:
            highest_document_frequency = len_postings
            term_with_highest_document_frequency = term
        
        sum_posting_len += len_postings

    average_document_frequency = round(sum_posting_len / number_of_unique_terms,2)

    print(number_of_unique_terms)
    print(number_of_all_postings)
    print(term_with_highest_document_frequency, highest_document_frequency)
    print(highest_document_frequency)
    print(average_document_frequency)

def information_retrivial():
    languages:list[Language] = [Czech(),English()]

    for language in languages:
        print(language.avarage_queries)
        print(language.tokens)
    
    with open("inverted_index____" + "cs" + ".json", "r", encoding="utf-8") as f:
            inverted_index_cs = json.load(f)
    
    with open("inverted_index____" + "en" + ".json", "r", encoding="utf-8") as f:
            inverted_index_en = json.load(f)
    
    google_task_questions(inverted_index_cs, "cs")
    google_task_questions(inverted_index_en, "en")

if __name__ == "__main__":
    information_retrivial()
