from SequenceEncoder import SequenceEncoder
from TextTransformers import LDATransformer, PVTransformer, BoNGTransformer, NBLogCountRatioTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import time
import numpy as np


class PredictiveModel():
    hardcoded_prediction = None

    def __init__(self, nr_events, case_id_col, label_col, encoder_kwargs, transformer_kwargs, cls_kwargs, text_col=None,
                 text_transformer_type=None, cls_method="rf"):
        
        self.text_col = text_col
        self.case_id_col = case_id_col
        self.label_col = label_col

        self.encoder = SequenceEncoder(nr_events=nr_events, case_id_col=case_id_col, label_col=label_col, **encoder_kwargs)
        self.transformer = self._get_text_transformer(text_transformer_type, **transformer_kwargs)
        self.cls = self._get_cls_method(cls_method, **cls_kwargs)

    def _get_text_transformer(self, text_transformer_type, **transformer_kwargs):
        if text_transformer_type == "LDATransformer":
            return LDATransformer(**transformer_kwargs)
        elif text_transformer_type == "BoNGTransformer":
            return BoNGTransformer(**transformer_kwargs)
        elif text_transformer_type == "NBLogCountRatioTransformer":
            return NBLogCountRatioTransformer(**transformer_kwargs)
        elif text_transformer_type == "PVTransformer":
            return PVTransformer(**transformer_kwargs)
        print("Transformer type not known")
        return None

    def  _get_cls_method(self, cls_method, **cls_kwargs):
        if cls_method == "logit":
            return LogisticRegression(**cls_kwargs)
        elif cls_method == "rf":
            return RandomForestClassifier(**cls_kwargs)
        print("Classifier method not known")
        return None

    def fit(self, dt_train):
        preproc_start_time = time.time()

        train_encoded = self.encoder.fit_transform(dt_train)
        train_x = train_encoded.drop([self.case_id_col, self.label_col], axis=1)
        train_y = train_encoded[self.label_col]

        if self.transformer:
            train_x = self._fit_transform_x(train_x, train_y)

        preproc_end_time = time.time() - preproc_start_time

        cls_start_time = time.time()
        self._train_cls(train_x, train_y)
        cls_time = time.time() - cls_start_time
        return None

    def _fit_transform_x(self, train_x, train_y):
        text_cols = [col for col in train_x.columns.values if col.startswith(self.text_col)]
        # Alternative syntax: train_x = list(map(lambda col: train_x[col].astype('str'), text_cols))
        for col in text_cols:
            train_x[col] = train_x[col].astype('str')
        train_text = self.transformer.fit_transform(train_x[text_cols], train_y)
        return pd.concat([train_x.drop(text_cols, axis=1), train_text], axis=1)

    def _train_cls(self, train_x, train_y):
        if len(train_y.unique()) < 2:  # less than 2 classes are present
            self.hardcoded_prediction = train_y.iloc[0]
            self.cls.classes_ = train_y.unique()
        else:
            self.cls.fit(train_x, train_y)
        return None

    def predict_proba(self, dt_test):
        encode_start_time = time.time()
        test_encoded = self.encoder.transform(dt_test)
        test_encode_time = time.time() - encode_start_time

        # Transformation
        test_preproc_start_time = time.time()
        test_x = test_encoded.drop([self.case_id_col, self.label_col], axis=1)
        if self.transformer:
            test_x, test_encoded = self._transform(test_x, test_encoded)

        test_preproc_time = time.time() - test_preproc_start_time

        # Prediction
        test_start_time = time.time()
        predictions_proba = self._cls_predict_proba(test_x)
        test_time = time.time() - test_start_time

        return predictions_proba

    def _transform(self, test_x, test_encoded):
        text_cols = [col for col in test_x.columns.values if col.startswith(self.text_col)]
        for col in text_cols:
            test_encoded[col] = test_encoded[col].astype('str')
        test_text = self.transformer.transform(test_encoded[text_cols])
        test_x = pd.concat([test_x.drop(text_cols, axis=1), test_text], axis=1)
        return test_x, test_encoded

    def _cls_predict_proba(self, test_x):
        if self.hardcoded_prediction:  # e.g. model was trained with one class only
            return np.array([1.0, 0.0] * test_x.shape[0]).reshape(test_x.shape[0], 2)
        return self.cls.predict_proba(test_x)