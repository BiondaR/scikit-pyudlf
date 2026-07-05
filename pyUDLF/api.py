from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType as it
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import os
import tempfile
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class UDLF:
    def __init__(self, task, rks_path=None, list_path=None, classes_path=None, images_path=None, persist=False, bin_path=None, log_path='./outputs/log.txt', output_path='./outputs/output'):
        """
        Unified Class: Configures, Executes and Analyzes UDLF.
        
        :param kwargs: Pass the hyperparameters directly here (e.g. L=100, K=20, list_file="...").
        """
        self.bin_path = bin_path
        self.task = task.upper()
        self.rks_path = rks_path
        self.persist = persist
        self.list_path = list_path
        self.classes_path = classes_path
        self.images_path = images_path
        self.log_path = log_path
        self.output_path = output_path
        
        self.input_data = self._build_config()
        self.output_data = None # Will store the log after fit
        self.is_fitted = False
        self._temp_dir = None
        self._cached_matrix = None

    def __repr__(self):
        method = self.input_data.get_method_name()
        method_str = method[0].strip() if method else "not configured"
        status = "fitted" if self.is_fitted else "not executed"
        return (
            f"UDLF(task={self.task!r}, method={method_str!r}, "
            f"persist={self.persist}, status={status!r})"
        )

    def _build_config(self):
        """Internal method that builds the configuration dynamically."""
        if self.bin_path is not None:
            udlf.setBinaryPath(self.bin_path)
        input_data = it.InputType()
        
        # 1. Basic Configurations
        input_data.set_task(self.task)
        
        # 2. File Configurations 
        if self.images_path is not None and None in [self.rks_path, self.list_path, self.classes_path]:
            raise ValueError("The paths for rankings, classes and lists must be passed along with the images path")

        return input_data
        
    def _set_inputs(self, rks_path, list_path, images_path, classes_path, input_format):
        if self.task == "UDL":
            self.input_data.set_input_file(rks_path)
            with open(rks_path) as f:
                size = sum(1 for _ in f)
        elif self.task == "FUSION":
            self.input_data.set_input_files(rks_path)
            with open(rks_path[0]) as f:
                size = sum(1 for _ in f)
            
        self.input_data.set_dataset_size(size)

        self.input_data.set_lists_file(list_path)
        if classes_path:
            self.input_data.set_classes_file(classes_path)
        if images_path:
            self.input_data.set_input_images_path(images_path)

        self.input_data.set_input_file_format(input_format)
        if input_format == "RK":
            self.input_data.set_rk_format('NUM')
        elif input_format == "MATRIX":
            self.input_data.set_input_matrix_type('DIST')

    def _set_effectiveness(self, vals=None, effec_eval=True):
        if vals is None:
            vals = [1, 2, 5, 10, 15]
        self.input_data.set_effectiveness_eval(effec_eval)
        if effec_eval == True:
            self.input_data.set_effectiveness_recalls_to_compute(vals)
            self.input_data.set_effectiveness_precisions_to_compute(vals)
            self.input_data.set_effectiveness_compute_map(True)
            self.input_data.set_effectiveness_compute_recall(True)
            self.input_data.set_effectiveness_compute_precisions(True)

    def _set_output(self, log_path, output_path):
        self.input_data.set_output_file(True)
        self.input_data.set_output_file_format('MATRIX')
        self.input_data.set_output_matrix_type('DIST')
        self.input_data.set_output_log_file_path(log_path)
        self.input_data.set_output_file_path(output_path)
        
    def _check_fitted(self):
        if self.is_fitted:
            logger.warning("Model has already been fitted. Call fit() again to apply changes.")

    # --- Method-specific parameters ---
    def base(self, L=1000):  # NONE has no explicit default in METHOD_PARAMS
        self._check_fitted()
        if self.task == 'FUSION':
            raise ValueError ("[ERROR] fusion method not implemented")
        self.input_data.set_method_name("NONE")
        self.input_data.set_method_parameters(method="NONE", l=L)

    def contextrr(self, L=25, K=7, T=5, NBYK=1, OPTIMIZATIONS=True):
        self._check_fitted()
        self.input_data.set_method_name("CONTEXTRR")
        self.input_data.set_method_parameters(method="CONTEXTRR", l=L, k=K, t=T, nbyk=NBYK, optimizations=OPTIMIZATIONS)

    def corgraph(self, L=200, K=25, TH_START=0.35, TH_END=1.0, TH_INC=0.005, CORRELATION='PEARSON'):
        self._check_fitted()
        self.input_data.set_method_name("CORGRAPH")
        self.input_data.set_method_parameters(method="CORGRAPH", l=L, k=K, threshold_start=TH_START, threshold_end=TH_END, threshold_inc=TH_INC, correlation=CORRELATION)

    def cprr(self, L=400, K=20, T=2):
        self._check_fitted()
        self.input_data.set_method_name("CPRR")
        self.input_data.set_method_parameters(method="CPRR", l=L, k=K, t=T)

    def rkgraph(self, L=700, K=20, T=1, P=0.95):
        self._check_fitted()
        self.input_data.set_method_name("RKGRAPH")
        self.input_data.set_method_parameters(method="RKGRAPH", l=L, k=K, t=T, p=P)

    def recknngraph(self, L=200, K=15, EPS=0.0125):
        self._check_fitted()
        self.input_data.set_method_name("RECKNNGRAPH")
        self.input_data.set_method_parameters(method="RECKNNGRAPH", l=L, k=K, epsilon=EPS)

    def rlrecom(self, L=400, K=8, EPS=0.0125, LAMBDA=2.0):
        self._check_fitted()
        if self.task == 'FUSION':
            raise ValueError ("[ERROR] fusion method not implemented")
        self.input_data.set_method_name("RLRECOM")
        self.input_data.set_method_parameters(method="RLRECOM", l=L, k=K, lambda_rlrecom=LAMBDA, epsilon=EPS) 

    def rlsim(self, TOPK=15, CK=700, T=3, METRIC='INTERSECTION'):
        self._check_fitted()
        self.input_data.set_method_name("RLSIM")
        self.input_data.set_method_parameters(method="RLSIM", topk=TOPK, ck=CK, t=T, metric=METRIC)

    def lhrr(self, L=1400, K=18, T=2):
        self._check_fitted()
        self.input_data.set_method_name("LHRR")
        self.input_data.set_method_parameters(method="LHRR", l=L, k=K, t=T)

    def bfstree(self, L=1400, K=20, CORRELATION='RBO'):
        self._check_fitted()
        if self.task == 'FUSION':
            raise ValueError ("[ERROR] fusion method not implemented")
        self.input_data.set_method_name("BFSTREE")
        self.input_data.set_method_parameters(method="BFSTREE", l=L, correlation_metric=CORRELATION, k=K)

    def rdpac(self, L=400, L_MULT=2, P=0.60, PL=0.99, K_START=1, K_INC=1, K=15):
        self._check_fitted()
        self.input_data.set_method_name("RDPAC")
        self.input_data.set_method_parameters(method="RDPAC", l=L, l_mult=L_MULT, p=P, pl=PL, k_start=K_START, k_inc=K_INC, k_end=K)

    def rfe(self, L=400, K=20, T=2, PA=0.1, TH_CC=0, RK_BY_EMB=False, EXPORT_EMBS=False, PERF_CCS=True, EMB_PATH='embeddings.txt', CCS_PATH='ccs.txt'):
        self._check_fitted()
        self.input_data.set_method_name("RFE")
        self.input_data.set_method_parameters(method="RFE", l=L, k=K, t=T, pa=PA, th_cc=TH_CC, rerank_by_emb=RK_BY_EMB, export_embeddings=EXPORT_EMBS, perform_ccs=PERF_CCS, embeddings_path=EMB_PATH, ccs_path=CCS_PATH)

    # fit
    def fit(self, X=None, y=None, distance='euclidean'):
        """Executes C++ and retrieves the output object (logs and metrics)."""
        print(f"Executing {self.input_data.get_method_name()[0]} via UDLF...")

        if self.persist == True:
            abs_log = os.path.abspath(self.log_path)
            abs_out = os.path.abspath(self.output_path)
            os.makedirs(os.path.dirname(abs_log), exist_ok=True)
            os.makedirs(os.path.dirname(abs_out), exist_ok=True)
            self._set_output(abs_log, abs_out)
        else:
            self._temp_dir = tempfile.mkdtemp()
            self._set_output(self._temp_dir +'/log.txt', self._temp_dir +'/output')

        if self.rks_path != None:
            missing = [name for name, val in [
                ("list_path", self.list_path),
                ("classes_path", self.classes_path),
            ] if val is None]
            if missing:
                raise ValueError(
                    f"The following parameters are required when rks_path is provided: {missing}"
                )
            self._set_inputs(self.rks_path, self.list_path, self.images_path, self.classes_path, "RK")
            self._set_effectiveness()
            if X is not None:
                print(f"Ignoring X and executing for file {self.rks_path}")
        else:
            if X is None:
                raise RuntimeError('Provide an input file')
            else:
                if self._temp_dir is None: 
                    self._temp_dir = tempfile.mkdtemp()
                
                # A. Process Matrix X
                if distance == 'precomputed':
                    if X.shape[0] != X.shape[1]:
                        raise ValueError('The distance matrix must be square!')
                    else:
                        dist_matrix = X   
                else: 
                    dist_matrix = euclidean_distances(X, X)
                temp_rks = os.path.join(self._temp_dir, "input_dist.txt")
                np.savetxt(temp_rks, dist_matrix, fmt='%.1f')

                temp_list = os.path.join(self._temp_dir, "list.txt")
                np.savetxt(temp_list, np.arange(X.shape[0]), fmt='%d')
                if y is None:
                    self._set_effectiveness(effec_eval=False)
                    temp_classes = ""
                else:
                    self._set_effectiveness()
                    temp_classes = os.path.join(self._temp_dir, "classes.txt")
                    with open(temp_classes, 'w') as f:
                        for i, label in enumerate(y):
                            f.write(f"{i}:{label}\n")

                self._set_inputs(rks_path=temp_rks, list_path=temp_list, classes_path=temp_classes, images_path="", input_format="MATRIX")
        
        # Now get_output=True to retrieve the logs from code
        self.output_data = udlf.run(
            self.input_data, 
            get_output=True, 
            compute_individual_gain=False, 
            visualization=False
        )
        self._cached_matrix = None
        self.is_fitted = True
        return self

    def transform(self):
        """Reads the matrix generated by C++ and returns it."""
        if not self.is_fitted:
            raise RuntimeError("Run fit() first!")
            
        if getattr(self, '_cached_matrix', None) is None:
            txt_path = self.input_data.get_output_file_path()[0] + '.txt'
            self._cached_matrix = np.loadtxt(txt_path)
            
        return self._cached_matrix

    def fit_transform(self, X=None, y=None):
        return self.fit(X, y).transform()

    def get_ranked_list(self):
        return np.argsort(self.transform(), axis=1).astype(int)

    def _get_retrieval_results_df(self, mode):
        if self.task == "FUSION":
            return pd.DataFrame.from_dict(self.output_data.get_log(), orient='index', columns=['Valor'])

        elif self.task == "UDL":
            data = self.output_data.get_log().copy()
            data.pop('Time', None)
            df_all = pd.DataFrame.from_dict(data, orient='index')
            
            if mode == 'all':
                return df_all
            elif mode in ['Before', 'After', 'Gain']:
                return df_all[[mode]]
            else:
                raise ValueError("Mode must be 'all', 'Before', 'After' or 'Gain'")

    def _get_retrieval_results_dict(self, mode):
        data = self.output_data.get_log().copy()
        time_val = data.get('Time')
        metrics_only = {k: v for k, v in data.items() if k != 'Time'}

        if self.task == "FUSION":
            return data

        if mode == 'all':
            all_dict = {'Time': time_val}
            for sub_key in ['Before', 'After', 'Gain']:
                all_dict[sub_key] = {k: v[sub_key] for k, v in metrics_only.items()}
            return all_dict

        elif mode in ['Before', 'After', 'Gain']:
            result = {'Time': time_val}
            for k, v in metrics_only.items():
                result[k] = v.get(mode)
            return result
        else:
            raise ValueError("Mode must be 'all', 'Before', 'After' or 'Gain'")

    def get_metrics(self, mode='all', output_type="dict"):
        """Returns the UDLF evaluation results."""
        if not self.is_fitted:
            raise RuntimeError("No metrics found. Run fit() first!")
            
        if output_type == "df":
            return self._get_retrieval_results_df(mode=mode)
        elif output_type == "dict":
            return self._get_retrieval_results_dict(mode=mode)
        else:
            raise ValueError('[ERROR] what kind of type is this?!??!?!')
            return None
        
    def get_labels_from_classes_file(self):
        """
        Reads the UDLF classes file (ID:Class format) and returns vector y (labels).
        """
        labels = []
        classes_file = self.input_data.get_classes_file()
        # Unwrap nested lists until we get a string
        while isinstance(classes_file, list):
            classes_file = classes_file[0]
        if not classes_file or classes_file.strip() == "":
            raise RuntimeError('Class file does not exist')
        with open(classes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split at the last ':' in case the filename has ':'
                # Ex: '001.jpg:carro' -> ['001.jpg', 'carro']
                parts = line.rsplit(':', 1)
                if len(parts) == 2:
                    labels.append(parts[1]) # Get only the class
                else:
                    logger.warning("Line without ':' separator ignored: %r", line)
                    labels.append(line) # Fallback if there is no ':'
                    
        # Return as numpy array. Can be an array of strings or ints.
        return np.array(labels)


    def _aggregate_scores(self, scores, y, mode):
        """
        Internal function that packages results based on the 'mode'.
        :param scores: List of metrics with the same size as y (one per query).
        """
        if mode == 'global':
            return np.mean(scores)
            
        elif mode == 'index':
            # Simple dictionary with query position and its score
            return {i: score for i, score in enumerate(scores)}
            
        elif mode == 'class':
            # Groups scores by class and calculates the average for each
            class_scores = defaultdict(list)
            for score, cls in zip(scores, y):
                class_scores[cls].append(score)
                
            return {cls: np.mean(vals) for cls, vals in class_scores.items()}
            
        else:
            raise ValueError("The 'mode' parameter must be 'global', 'class' or 'index'.")

    def precision_at_k(self, y=None, ignore_self=True, k=10, mode='global'):
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            
            if ignore_self:
                ranking = ranking[ranking != i]
                
            top_k = ranking[:k]
            relevance = (y[top_k] == query_class).astype(int)
            scores.append(np.sum(relevance) / k)
            
        return self._aggregate_scores(scores, y, mode)


    def recall_at_k(self, y=None, k=10, ignore_self=True, mode='global'):
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            total_relevant = np.sum(y == query_class)
            
            if ignore_self:
                ranking = ranking[ranking != i]
                total_relevant -= 1
                
            if total_relevant <= 0:
                scores.append(0.0)
                continue
                
            top_k = ranking[:k]
            relevance = (y[top_k] == query_class).astype(int)
            scores.append(np.sum(relevance) / total_relevant)
            
        return self._aggregate_scores(scores, y, mode)


    def f1_score_at_k(self, y=None, k=10, ignore_self=True, mode='global'):
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            total_relevant = np.sum(y == query_class)
            
            if ignore_self:
                ranking = ranking[ranking != i]
                total_relevant -= 1
                
            top_k = ranking[:k]
            relevance = (y[top_k] == query_class).astype(int)
            
            p = np.sum(relevance) / k
            r = np.sum(relevance) / total_relevant if total_relevant > 0 else 0.0
            
            if p + r > 0:
                scores.append(2 * (p * r) / (p + r))
            else:
                scores.append(0.0)
                
        return self._aggregate_scores(scores, y, mode)


    def average_precision_at_k(self, y=None, k=10, ignore_self=True, mode='global'):
        """
        If mode='global', the returned result is mAP (Mean Average Precision).
        """
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            
            if ignore_self:
                ranking = ranking[ranking != i]
                
            top_k = ranking[:k]
            relevance = (y[top_k] == query_class).astype(int)
            
            if np.sum(relevance) == 0:
                scores.append(0.0)
                continue
                
            precisions = [np.sum(relevance[:j+1]) / (j+1) for j, is_rel in enumerate(relevance) if is_rel == 1]
            scores.append(np.mean(precisions))
            
        return self._aggregate_scores(scores, y, mode)


    def reciprocal_rank(self, y=None, ignore_self=True, mode='global'):
        """
        If mode='global', the returned result is MRR (Mean Reciprocal Rank).
        """
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            
            if ignore_self:
                ranking = ranking[ranking != i]
                
            relevance = (y[ranking] == query_class).astype(int)
            first_correct_idx = np.where(relevance == 1)[0]
            
            if len(first_correct_idx) > 0:
                scores.append(1.0 / (first_correct_idx[0] + 1))
            else:
                scores.append(0.0)
                
        return self._aggregate_scores(scores, y, mode)

    def ndcg_at_k(self, y=None, k=10, ignore_self=True, mode='global'):
        """
        Calculates NDCG @ K. 
        Measures the quality of the ranking penalizing hits that appear further from the top.
        """
        if y is None:
            y = self.get_labels_from_classes_file()

        scores = []
        
        # Precalculates logarithmic discount (log base 2 of position i+1)
        # Since positions start at 1, the divisors are log2(2), log2(3), ..., log2(k+1)
        discounts = np.log2(np.arange(2, k + 2))
        
        for i, ranking in enumerate(self.get_ranked_list()):
            query_class = y[i]
            
            if ignore_self:
                ranking = ranking[ranking != i]
                
            top_k = ranking[:k]
            relevance = (y[top_k] == query_class).astype(int)
            
            # 1. Calculates the real DCG of the query
            dcg = np.sum(relevance / discounts)
            
            # 2. Calculates the IDCG (Ideal DCG - The maximum this query could reach)
            total_relevant = np.sum(y == query_class)
            if ignore_self:
                total_relevant -= 1
                
            # The ideal scenario is to have '1s' filling the first N possible positions
            ideal_relevance = np.zeros(k)
            ideal_relevance[:min(k, total_relevant)] = 1
            idcg = np.sum(ideal_relevance / discounts)
            
            # 3. Normalize (NDCG)
            if idcg > 0:
                scores.append(dcg / idcg)
            else:
                scores.append(0.0)
                
        return self._aggregate_scores(scores, y, mode)
    
    def jaccard_overlap_at_k(self, y=None, k=10, ignore_self=True, mode='global'):
        """
        UNSUPERVISED Metric. 
        Measures the similarity (Jaccard/Intersection) of the Top-K before and after re-ranking.
        
        :param ranked_lists_after: Index matrix after UDLF.
        :param ranked_lists_before: Raw index matrix (e.g., FAISS or B-CNN).
        :param y: Optional. Used ONLY if you want to aggregate by mode='class'.
        """
        if y is None and mode == 'class':
            y = self.get_labels_from_classes_file()

        ranked_lists_after = self.get_ranked_list()
        if (self.rks_path is None or self.rks_path == "") and self._temp_dir is not None:
            temp_rks = os.path.join(self._temp_dir, "input_dist.txt")
            ranked_lists_before = np.argsort(np.loadtxt(temp_rks), axis=1)
        elif self.rks_path != None and self.rks_path != "":
            ranked_lists_before = np.loadtxt(self.rks_path)
        else:
            raise RuntimeError('ranked list before processing not found')

        scores = []
        for i in range(ranked_lists_after.shape[0]):
            ranking_after = ranked_lists_after[i]
            ranking_before = ranked_lists_before[i]
            
            if ignore_self:
                ranking_after = ranking_after[ranking_after != i]
                ranking_before = ranking_before[ranking_before != i]
                
            # Get the K neighbors from each list
            top_k_after = set(ranking_after[:k])
            top_k_before = set(ranking_before[:k])
            
            # Formula: (Intersection) / K 
            # (How many original items remained at the top?)
            intersection = len(top_k_after.intersection(top_k_before))
            scores.append(intersection / k)
            
        # If the user asks for aggregation by class, they MUST have passed 'y'
        if mode == 'class' and y is None:
            raise ValueError("To use mode='class', you must provide the vector 'y' to guide the grouping.")
            
        # If global or index, we don't need real y to aggregate
        dummy_y = y if y is not None else np.zeros(len(scores))
        
        return self._aggregate_scores(scores, dummy_y, mode)

    @staticmethod
    def compare(models, X=None, y=None, k=10):
        """
        Compares multiple models on the same dataset.
        All models are re-fitted here to ensure they run on the same data.

        :param models: dict {name: model} or list of UDLF instances
        :param X: features or distance matrix (optional if models have rks_path)
        :param y: vector of labels
        :param k: K for the @K metrics
        """
        if isinstance(models, dict):
            named_models = models
        else:
            named_models = {repr(m): m for m in models}

        if y is not None:
            y = np.array(y)

        rows = []
        for name, model in named_models.items():
            print(f"[{name}] executing fit()...")
            model.fit(X=X, y=y)

            if y is None:
                y_eval = model.get_labels_from_classes_file()
            else:
                y_eval = y
            y_eval = np.array(y_eval)

            udlf_metrics = model.get_metrics(mode='After')

            rows.append({
                'model':      name,
                f'P@{k}':    model.precision_at_k(y=y_eval, k=k),
                f'R@{k}':    model.recall_at_k(y=y_eval, k=k),
                f'NDCG@{k}': model.ndcg_at_k(y=y_eval, k=k),
                'mAP':       model.average_precision_at_k(y=y_eval, k=k),
                'MRR':       model.reciprocal_rank(y=y_eval),
                'MAP_udlf':  udlf_metrics.get('MAP'),
            })

        return pd.DataFrame(rows).set_index('model')
