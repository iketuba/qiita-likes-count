import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import MeCab
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class Analyzer:
    def __init__(self):
        dir_ref = "model"
        self.model = pickle.load(open(f"{dir_ref}/model.pkl", "rb"))
        self.model_tfidf_title = pickle.load(open(f"{dir_ref}/tfidf_title.pkl", "rb"))
        self.model_tfidf_body = pickle.load(open(f"{dir_ref}/tfidf_body.pkl", "rb"))
        self.model_tfidf_tags = pickle.load(open(f"{dir_ref}/tfidf_tags.pkl", "rb"))
        self.model_svd_title = pickle.load(open(f"{dir_ref}/svd_title.pkl", "rb"))
        self.model_svd_body = pickle.load(open(f"{dir_ref}/svd_body.pkl", "rb"))
        self.model_svd_tags = pickle.load(open(f"{dir_ref}/svd_tags.pkl", "rb"))
        self.wakati = MeCab.Tagger("-Owakati")

    def __create_dataframe(self, title, body, tags, followers_count, organization, items_count):
        df = pd.DataFrame()
        df["title"] = [title]
        df["body"] = [body]
        df["tags"] = [tags]
        df["followers_count"] = [followers_count]
        df["organization"] = [organization]
        df["items_count"] = [items_count]
        self.df = df

    def __wakati_process(self, x):
        # 入力の文字列を句点で区切る
        sentences = x.split("。")
        # 分かち書きできる文章だけを集めるself.
        wakati_sentences = []
        for sentence in sentences:
            try:
                wakati_sentence = " ".join(self.wakati.parse(sentence).split())
                if wakati_sentence:
                    wakati_sentences.append(wakati_sentence)
            except:
                pass
        return " 。 ".join(wakati_sentences)

    def __tags_process(self, x):
        x_split = re.split(",|、|，", x)
        return " ".join(x_split)

    def __preprocess(self):
        # organizationの値が"はい"ならTrue, "いいえ"ならFalseに変換
        self.df["organization"] = self.df["organization"].map({"はい": True, "いいえ": False})

        self.df["body"] = self.df["body"].apply(self.__wakati_process)
        self.df["title"] = self.df["title"].apply(self.__wakati_process)
        self.df["tags"] = self.df["tags"].apply(self.__tags_process)

        # tf-idf化
        tfidf_title = self.model_tfidf_title.transform(self.df["title"])
        tfidf_body = self.model_tfidf_body.transform(self.df["body"])
        tfidf_tags = self.model_tfidf_tags.transform(self.df["tags"])
        # 次元削減
        svd_title = self.model_svd_title.transform(tfidf_title)
        svd_body = self.model_svd_body.transform(tfidf_body)
        svd_tags = self.model_svd_tags.transform(tfidf_tags)
        # データフレームに変換
        data_title_vec = pd.DataFrame(svd_title, columns=[f"title_{i}" for i in range(20)])
        data_body_vec = pd.DataFrame(svd_body, columns=[f"body_{i}" for i in range(20)])
        data_tags_vec = pd.DataFrame(svd_tags, columns=[f"tags_{i}" for i in range(20)])

        # 元のデータフレームと結合
        self.df = pd.concat([self.df, data_title_vec], axis=1)
        self.df = self.df.drop(["title"], axis=1)
        self.df = pd.concat([self.df, data_body_vec], axis=1)
        self.df = self.df.drop(["body"], axis=1)
        self.df = pd.concat([self.df, data_tags_vec], axis=1)
        self.df = self.df.drop(["tags"], axis=1)

    def predict(self, title, body, tags, followers_count, organization, items_count):
        self.__create_dataframe(title, body, tags, followers_count, organization, items_count)
        self.__preprocess()
        like_pred = self.model.predict(self.df)[0]
        # 予測値がマイナスになる場合は0にする
        if like_pred < 0:
            like_pred = 0
        return like_pred