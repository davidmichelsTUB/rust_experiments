import unittest
import pytest
import numpy as np
import regex
from sklearn.feature_extraction.text import CountVectorizer
from adapter import RustyCountVectorizer

from sklearn.datasets import fetch_20newsgroups
NEWS_PATH = "data/news.csv"


def load_news():
    data = fetch_20newsgroups(data_home="data").get("data")
    return data

WORD = r"[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]"
_tokenizer = regex.compile(rf"(?<!{WORD}){WORD}{{2,}}(?!{WORD})")
class TestAdapterCountVectorizer(unittest.TestCase):
    
    
    def test_vocab(self):
        x = load_news()
        

        sk = CountVectorizer(tokenizer=_tokenizer.findall, token_pattern=None)


        sk_out = sk.fit(x)
        rusty = RustyCountVectorizer()
        rusty_out = rusty.fit(x)
    

        sk_terms = set(sk.vocabulary_)
        ru_terms = set(rusty.vocabulary_)

        print("in sk, not rusty:", sk_terms - ru_terms)
        print(len(sk.vocabulary_),len(rusty.vocabulary_))
        print("in rusty, not sk:", ru_terms - sk_terms)
        
        assert (sk_out.vocabulary_ == rusty_out.vocabulary_)
    def test_rusty_scaler(self):
        x = load_news()
        sk = CountVectorizer(tokenizer=_tokenizer.findall, token_pattern=None)

        sk_out = sk.fit_transform(x)
        rusty = RustyCountVectorizer()
        rusty_out = rusty.fit_transform(x)
        assert (sk_out == rusty_out).all()
            