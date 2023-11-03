import itertools
import json
import string
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

pd.options.mode.chained_assignment = None  # default='warn'


def get_column_type(column, column_threshold=0.5, entity_threshold=0.5):
    """
    Determines whether a column is text or numeric based on the majority of its cells.
    Returns 1 if text column, 0 if numeric column.
    """
    str_cells = []
    for cell in column:
        if isinstance(cell, str):
            str_cells.append(cell)

    text_cells = []
    for cell in str_cells:
        for char in cell:
            if char in string.ascii_letters:
                text_cells.append(cell)
                break
    num_text_cells = len(text_cells) / len(str_cells)

    if num_text_cells > entity_threshold:
        return 1
    elif len(str_cells) / len(column) > column_threshold:
        return 1
    else:
        return 0


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def get_col_link(cols_in_table, num_columns):
    col_link_set = set()
    for table_name in cols_in_table:
        indices = cols_in_table[table_name]
        for c in itertools.combinations(indices, 2):
            col_link_set.add(c)

    col_link_arr = np.zeros((num_columns, num_columns))
    for i, j in combinations(range(num_columns), 2):
        if (i, j) not in col_link_set and (j, i) not in col_link_set:
            col_link_arr[i, j] = 1
            col_link_arr[j, i] = 1
    col_link_csr = csr_matrix(col_link_arr)

    return col_link_csr


def cluster(
    n_clusters,
    column_embeddings,
    col_link_csr,
    all_distance,
):
    clusters = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="l2",
        compute_distances=True,
        linkage="complete",
        connectivity=col_link_csr,
    )
    clusters.fit_predict(column_embeddings)
    labels = clusters.labels_  # .tolist()
    all_distance[n_clusters] = metrics.silhouette_score(column_embeddings, labels)

    result_dict = defaultdict(set)
    for col_index, label in enumerate(labels.tolist()):
        result_dict[label].add(col_index)

    pred_edges = set()
    for col_index_set in result_dict:
        set1 = result_dict[col_index_set]
        set2 = result_dict[col_index_set]
        cur_pred_edges = set()
        for s1 in set1:
            for s2 in set2:
                cur_pred_edges.add(tuple(sorted((s1, s2))))
        pred_edges = pred_edges.union(cur_pred_edges)

    return pred_edges


def plot_distance(distance_dict, title, num_columns, min_k):
    distance_list = distance_dict.items()
    distance_list = sorted(distance_list)
    x, y = zip(*distance_list)
    algorithm_k = max(distance_dict, key=distance_dict.get)

    overlapping = 1
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("number of clusters")
    plt.ylabel("silhouette score")
    plt.axvline(
        x=num_columns,
        color="red",
        linestyle="dashed",
        label="groundtruth k",
        alpha=overlapping,
        lw=3,
    )
    plt.axvline(
        x=algorithm_k,
        linestyle="dotted",
        color="green",
        label="algorithm k",
        alpha=overlapping,
        lw=3,
    )
    plt.axvline(x=min_k, color="black", label="min k")
    plt.show()


class AlignIntegrationIds(BaseModel):
    method: str = "bert"  # fasttext or turl or bert
    benchmark: str = "Align Benchmark"  # Align Benchmark or Real Benchmark

    table_index: int = 3
    vec_length: int = 300
    get_cluster_name: object = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.method == "fasttext":
            self.table_index = 3
            self.vec_length = 300
        elif self.method == "bert":
            self.table_index = 3
            self.vec_length = 768
        elif self.method == "turl":
            self.table_index = 2
            self.vec_length = 312

        self.get_cluster_name = lambda x: x.stem.split("_", self.table_index)[-1]

    def run(self):
        embedding_root = Path(self.method) / self.benchmark

        final_precision = {}
        final_recall = {}
        final_f_measure = {}

        start_time = time.time_ns()
        for embedding_path in embedding_root.iterdir():
            try:
                precision, recall, f_measure = self.run_one(embedding_path)

                final_precision[embedding_path.stem] = precision
                final_recall[embedding_path.stem] = recall
                final_f_measure[embedding_path.stem] = f_measure

            except:
                continue
        end_time = time.time_ns()
        total_time = int(end_time - start_time) / 10**9

        self.measure_summary(final_precision, final_recall, final_f_measure, total_time)

    def measure_summary(
        self,
        precision,
        recall,
        f_measure,
        total_time,
    ):
        total_precision = 0
        total_recall = 0
        total_f_measure = 0
        average_precision = 0
        average_recall = 0
        average_f_measure = 0

        for item in precision:
            total_precision += precision[item]
        for item in recall:
            total_recall += recall[item]
        for item in f_measure:
            total_f_measure += f_measure[item]

        average_precision = total_precision / len(precision)
        average_recall = total_recall / len(recall)
        average_f_measure = total_f_measure / len(f_measure)
        print("-------------------------------------")
        print("Result by:", self.method, " in ", self.benchmark)
        print("Average precision:", average_precision)
        print("Average recall", average_recall)
        print("Average f measure", average_f_measure)
        print("Total time", total_time)
        print("-------------------------------------")

    def run_one(self, embedding_path):
        (
            column_embeddings,
            col_link_csr,
            cols_in_table,
            gt_edges,
            col_name_set,
        ) = self.load(embedding_path)

        min_k = 1
        max_k = 0
        for item in cols_in_table:
            if len(cols_in_table[item]) > min_k:
                min_k = len(cols_in_table[item])
            max_k += len(cols_in_table[item])

        all_distance = {}
        record_current_precision = {}
        record_current_recall = {}
        record_current_f_measure = {}
        record_result_edges = {}
        for col_id in range(min_k, max_k):
            pred_edges = cluster(
                col_id,
                column_embeddings,
                col_link_csr,
                all_distance,
            )

            current_true_positive = len(gt_edges.intersection(pred_edges))
            current_precision = current_true_positive / len(pred_edges)
            current_recall = current_true_positive / len(gt_edges)

            record_current_precision[col_id] = current_precision
            record_current_recall[col_id] = current_recall
            record_current_f_measure[col_id] = 0
            if (current_precision + current_recall) > 0:
                record_current_f_measure[col_id] = (
                    2 * current_precision * current_recall
                ) / (current_precision + current_recall)
            record_result_edges[col_id] = pred_edges

        algorithm_k = max(all_distance, key=all_distance.get)
        precision = record_current_precision[algorithm_k]
        recall = record_current_recall[algorithm_k]
        f_measure = record_current_f_measure[algorithm_k]

        plot_distance(all_distance, embedding_path.stem, len(col_name_set), min_k)

        return precision, recall, f_measure

    def load(self, embedding_path):
        with open(embedding_path) as f:
            embedding = json.load(f)

        column_embeddings = []
        col_id_dict = {}
        cols_in_table = defaultdict(set)
        cols_in_same_cluster = defaultdict(set)
        col_id = 0
        col_name_set = set()
        cluster_name = self.get_cluster_name(embedding_path)
        for table_name in embedding:
            table_path = Path(self.benchmark) / cluster_name / table_name
            if not table_path.exists():
                continue

            db_table = pd.read_csv(
                table_path,
                encoding="latin1",
                warn_bad_lines=True,
                error_bad_lines=False,
            )

            # fill empty columns with empty list
            col_embedding_dict = embedding[table_name]
            if len(col_embedding_dict) == 0:
                for column in list(db_table.columns):
                    col_embedding_dict[column] = []

            for column in col_embedding_dict:
                try:
                    if get_column_type(db_table[column].tolist()) == 1:
                        col_name_set.add(column)
                        all_embeddings = col_embedding_dict[column]

                        if self.method == "turl":
                            entities_only = all_embeddings["entities_only"]
                        elif self.method in ["fasttext", "bert"]:
                            entities_only = all_embeddings
                        else:
                            raise Exception("Invalid method")

                        if (
                            isinstance(entities_only, float)
                            or not entities_only
                            or not all_embeddings
                        ):
                            entities_only = np.random.uniform(-1, 1, self.vec_length)
                        column_embeddings.append(entities_only)

                        col_id_dict[(table_name, column)] = col_id
                        cols_in_table[table_name].add(col_id)
                        cols_in_same_cluster[column].add(col_id)

                        col_id += 1
                except:
                    continue
        num_columns = col_id

        gt_edges = set()
        for column in cols_in_same_cluster:
            col_ids = cols_in_same_cluster[column]
            cur_gt_edges = set(map(tuple, map(sorted, combinations(col_ids, 2))))
            gt_edges = gt_edges.union(cur_gt_edges)

        column_embeddings = np.array(column_embeddings)

        col_link_csr = get_col_link(cols_in_table, num_columns)

        return (
            column_embeddings,
            col_link_csr,
            cols_in_table,
            gt_edges,
            col_name_set,
        )


if __name__ == "__main__":
    AlignIntegrationIds().run()
