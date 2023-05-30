import re
import string

import pandas as pd
import pathlib
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from textblob import Word, WordList, Sentence
from textblob import TextBlob
from redlines import Redlines

from heapq import nlargest
from collections import defaultdict
from itertools import compress
from dataclasses import dataclass


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


@dataclass
class Mood:
    emoji: str
    sentiment: float


class SentimentAnalysis:
    def __init__(self, text: str):
        self.text: str = text
        self.blob = TextBlob(text)

    def get_mood(self, threshold: float = 0.3) -> Mood:
        sentiment: float = self.blob.sentiment.polarity
        friendly_threshold: float = threshold
        hostile_threshold: float = -threshold

        if sentiment >= friendly_threshold:
            return Mood(emoji='😍😃😊🥰', sentiment=sentiment)

        elif sentiment <= hostile_threshold:
            return Mood(emoji='😢😡😭😞', sentiment=sentiment)
        else:
            return Mood(emoji='😐😑😒😕', sentiment=sentiment)

    def polarity_and_subjectivity(self,) -> str:
        result: str = ''
        for sentence in self.blob.sentences:
            result += str(sentence)
            sentiment = round(sentence.sentiment.polarity, 2)
            subjectivity = round(sentence.sentiment.subjectivity, 2)
            color = 'green' if sentiment > 0 else 'red'
            result += f'\n🔎 :blue[polarity]: :{color}[{sentiment}] 🚥 :orange[subjectivity]: :violet[{subjectivity}] \n\n'
        return result

    def get_word_count_df(self,) -> pd.DataFrame:
        word_count: defaultdict = self.blob.word_counts
        return pd.DataFrame(data={'words': word_count.keys(), 'count': word_count.values()})


class SpellingCorrection:
    def __init__(self, text: str):
        self.text: str = text.lower()
        self.blob: TextBlob = TextBlob(text.lower())
        self.correct: str = str(TextBlob(text.lower()).correct())

    def spelling_correction(self,) -> str:
        response = f'✅ {self.correct} 🖊'
        return response

    def redlines(self) -> str:
        response = Redlines(source=self.text, test=self.correct).compare()
        return '🔎' + str(response)

    def get_explanation(self) -> tuple[dict, list]:
        input_text: list[str] = self.text.split()
        correct_text: list[str] = self.correct.split()
        mistake_mask: list[bool] = [
            True if a != b else False for a, b in zip(input_text, correct_text)]
        response: dict = {}
        for i, mistake in enumerate(mistake_mask):
            if mistake:
                word_mistake = input_text[i]
                word_correction = correct_text[i]

                m = Word(word_mistake)
                w = Word(word_correction)

                response[word_mistake] = {
                    'correction': m.spellcheck(), 'definition': w.definitions}
        return response, list(compress(data=input_text, selectors=mistake_mask))


class PartsOfSpeechTagging:
    def __init__(self, text: str):
        self.text: str = text
        self.blob: TextBlob = TextBlob(text)
        self.pos: dict = dict(list(TextBlob(text=text).tags))

    def parts_of_speech_tagging(self,) -> pd.DataFrame:
        return pd.DataFrame(data=self.pos, index=['Part of Speech Tags'])

    def parts_of_speech_map(self,) -> dict:
        tags = defaultdict(lambda: [])
        for word, tag in self.blob.tags:
            tags[tag].append(word)
        return tags

    def part_of_speech_help(self) -> dict[str:str]:
        pos_txt: list = pathlib.Path(
            './parts-of-speech-tags.txt').read_text().splitlines()
        pos_tag_map: dict[str:str] = dict(
            [tuple(txt.split(sep='\t')) for txt in pos_txt[2:]])
        return pos_tag_map


class TextSummarize:
    def __init__(self, text: str):
        self.text: str = text
        self.sentences: list[Sentence] = TextBlob(text).sentences
        self.english_stopwords: list = stopwords.words('english')

    def preprocessing(self,) -> str:
        res: str = re.sub(pattern='\s', repl=' ', string=self.text)
        res = re.sub(pattern=f'[{string.punctuation}]', repl='', string=res)
        res = res.lower()
        return res

    def unique_token_frequency_scaled_map(self,) -> dict[str:float]:
        processed_text: str = self.preprocessing()
        words: WordList = TextBlob(processed_text).words
        word_count: defaultdict = TextBlob(processed_text).word_counts
        tokens: set = set(words) - set(self.english_stopwords)
        token_frequency: dict[str:int] = {k: word_count[k] for k in tokens}
        W: int = max(token_frequency.values())
        token_frequency_scaled: dict[str:float] = {
            k: v/W for k, v in token_frequency.items()}
        return token_frequency_scaled

    def sentence_scores(self,) -> defaultdict[str:float]:
        weights: dict[str:float] = self.unique_token_frequency_scaled_map()
        sentence_scores = defaultdict(lambda: 0)
        for sentence in self.sentences:
            for word in sentence.words:
                try:
                    score = weights[word.lower()]
                except:
                    score = 0
                sentence_scores[str(sentence)] += score
        return sentence_scores

    def get_summary(self, n=3) -> tuple[str, str]:
        score_dict: defaultdict[str:float] = self.sentence_scores()
        summary: list = nlargest(n=n, iterable=score_dict, key=score_dict.get)
        markup: str = ''
        for sentence in self.sentences:
            if str(sentence).strip() in summary:
                markup += f':blue[{sentence}]'
            else:
                markup += str(sentence)
            markup += '\n\n'

        return markup, '\n\n'.join(summary)

    def get_token_frequency_df(self,) -> pd.DataFrame:
        weights: dict[str:float] = self.unique_token_frequency_scaled_map()
        df = pd.DataFrame(
            data={'Tokens': weights.keys(), 'Weights': weights.values()})
        return df
