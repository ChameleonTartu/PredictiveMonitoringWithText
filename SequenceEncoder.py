import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

class SequenceEncoder():
    
    def __init__(self, nr_events, event_nr_col, case_id_col, label_col, static_cols=None, dynamic_cols=None, 
                 last_state_cols=None, cat_cols=None, oversample_fit=True, minority_label="positive", fillna=True, 
                 random_state=None):
        self.nr_events = nr_events
        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols
        self.last_state_cols = last_state_cols
        self.cat_cols = cat_cols
        self.event_nr_col = event_nr_col
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.oversample_fit = oversample_fit
        self.minority_label = minority_label
        self.random_state = random_state
        self.fillna = fillna
        
        self.fitted_columns = None
        
        
    def fit(self, X):
        return self
        
    def fit_transform(self, X):
        data = self._encode(X)
        if self.oversample_fit:
            data = self._oversample(data)
        return data
        
    def transform(self, X):
        data = self._encode(X)
        return data
        
    def _encode(self, X):
        # endoce static cols
        if self.label_col not in self.static_cols:
            self.static_cols.append(self.label_col)
        if self.case_id_col not in self.static_cols:
            self.static_cols.append(self.case_id_col)
        data_final = X[X[self.event_nr_col] == 1][self.static_cols]
        
        # encode dynamic cols
        for i in range(1, self.nr_events+1):
            data_selected = X[X[self.event_nr_col] == i][[self.case_id_col] + self.dynamic_cols]
            data_selected.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.dynamic_cols]
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="right")
            
        # encode last state cols
        for col in self.last_state_cols:
            data_final = pd.merge(data_final, X[X[self.event_nr_col] == self.nr_events][[self.case_id_col, col]], on=self.case_id_col, how="right")
            for idx, row in data_final.iterrows():
                current_nr_events = self.nr_events - 1
                while pd.isnull(data_final.loc[idx, col]) and current_nr_events > 0:
                    data_final.loc[idx, col] = X[(X[self.case_id_col] == row[self.case_id_col]) & (X[self.event_nr_col] == current_nr_events)].iloc[0][col]
                    current_nr_events -= 1
                    
        
        # make categorical
        dynamic_cat_cols = [col for col in self.cat_cols if col in self.dynamic_cols]
        static_cat_cols = [col for col in self.cat_cols if col in self.static_cols]
        catecorical_cols = ["%s_%s"%(col, i) for i in range(1, self.nr_events+1) for col in dynamic_cat_cols] + static_cat_cols
        cat_df = data_final[catecorical_cols]
        cat_dict = cat_df.T.to_dict().values()
        vectorizer = DV( sparse = False )
        vec_cat_dict = vectorizer.fit_transform(cat_dict)
        cat_data = pd.DataFrame(vec_cat_dict, columns=vectorizer.feature_names_)
        data_final = pd.concat([data_final.drop(catecorical_cols, axis=1), cat_data], axis=1)
        
        if self.fitted_columns is not None:
            missing_cols = self.fitted_columns[~self.fitted_columns.isin(data_final.columns)]
            for col in missing_cols:
                data_final[col] = 0
            data_final = data_final[self.fitted_columns]
        else:
            self.fitted_columns = data_final.columns
        
        # fill NA
        if self.fillna:
            data_final = data_final.fillna(0)
            
        return data_final
    
    def _oversample(self, X):
        oversample_count = sum(X[self.label_col] != self.minority_label) - sum(X[self.label_col] == self.minority_label)

        if oversample_count > 0 and sum(X[self.label_col] == self.minority_label) > 0:
            oversampled_data = X[X[self.label_col]==self.minority_label].sample(oversample_count, replace=True, random_state=self.random_state)
            X = pd.concat([X, oversampled_data])

        return X
        