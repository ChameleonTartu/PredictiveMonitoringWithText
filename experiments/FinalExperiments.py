import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import sys
sys.path.append('..')
from PredictiveMonitor import PredictiveMonitor

data_filepath = "data_preprocessed.csv"
data = pd.read_csv(data_filepath, sep=";", encoding="utf-8")

static_cols = ['case_name', 'label', 'omakapital', 'kaive', 'state_tax_lq', 'lab_tax_lq', 'bilansimaht', 'puhaskasum', 'kasumlikkus', 'status_code', 'pr_1', 'pr_2', 'pr_3', 'pr_4', 'pr_5', 'pr_6', 'deg_1', 'deg_2', 'deg_3', 'deg_4', 'deg_5', 'deg_6', 'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'td_6', 'td_5', 'td_4', 'td_3', 'td_2', 'td_1', 'tdp_6', 'tdp_5', 'tdp_4', 'tdp_3', 'tdp_2', 'tdp_1', 'tdd_6', 'tdd_5', 'tdd_4', 'tdd_3', 'tdd_2', 'tdd_1', 'tdi_6', 'tdi_5', 'tdi_4', 'tdi_3', 'tdi_2', 'tdi_1', 'tdip_6', 'tdip_5', 'tdip_4', 'tdip_3', 'tdip_2', 'tdip_1', 'md_6', 'md_5', 'md_4', 'md_3', 'md_2', 'md_1', 'decl_6', 'decl_5', 'decl_4', 'decl_3', 'decl_2', 'decl_1', 'age', 'exp_payment_isna', 'state_tax_lq_isna', 'lab_tax_lq_isna', 'tax_debt_isna', 'tax_declar_isna', 'debt_balances_isna', 'debtors_name_isna', 'aasta_isna', 'kaive_isna', 'bilansimaht_isna', 'omakapital_isna', 'puhaskasum_isna', 'kasumlikkus_isna', 'bilanss_client_isna', 'status_code_isna', 'pr_1_isna', 'deg_1_isna', 'score_1_isna', 'td_1_isna', 'tdp_1_isna', 'tdd_1_isna', 'tdi_1_isna', 'tdip_1_isna', 'md_1_isna', 'decl_1_isna', 'age_isna']

cat_cols = ['tax_declar', 'month', 'status_code', 'exp_payment_isna', 'state_tax_lq_isna', 'lab_tax_lq_isna', 'tax_debt_isna', 'tax_declar_isna', 'debt_balances_isna', 'debtors_name_isna', 'aasta_isna', 'kaive_isna', 'bilansimaht_isna', 'omakapital_isna', 'puhaskasum_isna', 'kasumlikkus_isna', 'bilanss_client_isna', 'status_code_isna', 'pr_1_isna', 'deg_1_isna', 'score_1_isna', 'td_1_isna', 'tdp_1_isna', 'tdd_1_isna', 'tdi_1_isna', 'tdip_1_isna', 'md_1_isna', 'decl_1_isna', 'age_isna']

case_id_col = "case_name"
label_col = "label"
event_nr_col = "event_nr"
text_col = "event_description_lemmas"
pos_label = "unsuccessful"

# divide into train and test data
train_names, test_names = train_test_split( data[case_id_col].unique(), train_size = 4.0/5, random_state = 22 )
train = data[data[case_id_col].isin(train_names)]
test = data[data[case_id_col].isin(test_names)]

confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
text_transformer_types = [None, "LDATransformer", "PVTransformer", "BoNGTransformer", "NBLogCountRatioTransformer"]
out_filename_bases = {None:"base", "LDATransformer":"lda", "PVTransformer":"pv", "BoNGTransformer":"bong", "NBLogCountRatioTransformer":"nb"}
cls_methods = ["rf", "logit"]

for cls_method in cls_methods:
    if cls_method == "rf":
            cls_kwargs = {"n_estimators":500, "random_state":22}
    else:
        cls_kwargs = {"random_state":22}
    
    optimal_params = pd.read_csv("cv_results/optimal_params_%s"%cls_method, sep=";")
        
    for text_transformer_type in text_transformer_types:
        
        for conf in confidences:
        
            if text_transformer_type is None:
                transformer_kwargs = None
                dynamic_cols = ["debt_sum", "max_days_due", "exp_payment", "tax_declar", "month", "tax_debt", "debt_balances",
                                "bilanss_client"]
                last_state_cols = []
            elif text_transformer_type == "LDATransformer":
                k = int(optimal_params[optimal_params.confidence==conf].topic_k)
                tfidf = (optimal_params[optimal_params.confidence==conf].topic_tfidf == "tfidf").iloc[0]
                transformer_kwargs = {"num_topics":k, "tfidf":tfidf, random_seed:22}
                dynamic_cols = ["debt_sum", "max_days_due", "exp_payment", "tax_declar", "month", "tax_debt", "debt_balances",
                                "bilanss_client", "event_description_lemmas"]
                last_state_cols = []
            elif text_transformer_type == "PVTransformer":
                size = int(optimal_params[optimal_params.confidence==conf].doc2vec_size)
                window = int(optimal_params[optimal_params.confidence==conf].doc2vec_window)
                transformer_kwargs = {"size":size, "window":window, random_seed:22, epochs:10}
                dynamic_cols = ["debt_sum", "max_days_due", "exp_payment", "tax_declar", "month", "tax_debt", "debt_balances",
                                "bilanss_client", "event_description_lemmas"]
                last_state_cols = []
            elif text_transformer_type == "BoNGTransformer":
                ngram_max = int(optimal_params[optimal_params.confidence==conf].bow_ngram)
                nr_selected = int(optimal_params[optimal_params.confidence==conf].bow_selected)
                tfidf = (optimal_params[optimal_params.confidence==conf].bow_tfidf == "tfidf").iloc[0]
                transformer_kwargs = {"ngram_max":ngram_max, "tfidf":tfidf, "nr_selected":nr_selected}
                dynamic_cols = ["debt_sum", "max_days_due", "exp_payment", "tax_declar", "month", "tax_debt", "debt_balances",
                                "bilanss_client"]
                last_state_cols = ["event_description_lemmas"]
            elif text_transformer_type == "NBLogCountRatioTransformer":
                ngram_max = int(optimal_params[optimal_params.confidence==conf].nb_ngram)
                nr_selected = int(optimal_params[optimal_params.confidence==conf].nb_selected)
                alpha = float(optimal_params[optimal_params.confidence==conf].nb_alpha)
                transformer_kwargs = {"ngram_max":ngram_max, "alpha":alpha, "nr_selected":nr_selected}
                dynamic_cols = ["debt_sum", "max_days_due", "exp_payment", "tax_declar", "month", "tax_debt", "debt_balances",
                                "bilanss_client"]
                last_state_cols = ["event_description_lemmas"]
                
            encoder_kwargs = {"event_nr_col":event_nr_col, "static_cols":static_cols, "dynamic_cols":dynamic_cols,
                              "last_state_cols":last_state_cols, "cat_cols":cat_cols, "oversample_fit":True,
                              "minority_label":pos_label, "fillna":True, "random_state":22}

            # train
            predictive_monitor = PredictiveMonitor(event_nr_col=event_nr_col, case_id_col=case_id_col, label_col=label_col, pos_label=pos_label, encoder_kwargs=encoder_kwargs, cls_kwargs=cls_kwargs, transformer_kwargs=transformer_kwargs, text_col=text_col, text_transformer_type=text_transformer_type, cls_method=cls_method)
            predictive_monitor.train(train)

            # test
            predictive_monitor.test(test, confidences=[conf], evaluate=True, output_filename="final_results/%s_%s"%(out_filename_bases[text_transformer_type], cls_method), outfile_mode='a')