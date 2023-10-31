from pathlib import Path
import os
import math
import glob
import json
import string
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
import pandas as pd
import itertools
from sklearn import metrics
import time

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


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def main(
    method="bert",  # fasttext or turl or bert
    input_folder_name="Align Benchmark",  # Align Benchmark or Real Benchmark
):
    if method == "fasttext":
        table_index = 3
        vec_length = 300
    elif method == "bert":
        table_index = 3
        vec_length = 768
    elif method == "turl":
        table_index = 2
        vec_length = 312
    else:
        raise Exception("Invalid method")

    embedding_root = Path(method) / input_folder_name
    embedding_paths = embedding_root.glob("**/*")

    final_precision = {}
    final_recall = {}
    final_f_measure = {}
    start_time = time.time_ns()
    for embedding_path in embedding_paths:
        try:
            tablename = embedding_path.name
            with open(embedding_path) as f:
                embedding = json.load(f)

            column_embeddings = []
            track_columns = {}  # for debugging only
            track_tables = {}
            record_same_cluster = {}
            i = 0
            count_tables = 0
            all_columns = set()
            total_columns = 0
            # change below to 3 for bert and fast text
            cluster_name = tablename.split("_", table_index)[-1]
            cluster_name = cluster_name.split(".", 1)[0]
            real_table_path = input_folder_name + "/" + cluster_name + "/"
            for table in embedding:
                try:
                    real_table = pd.read_csv(
                        real_table_path + table,
                        encoding="latin1",
                        warn_bad_lines=True,
                        error_bad_lines=False,
                    )
                except:
                    # print("table not found")
                    break
                table_columns = embedding[table]
                if len(table_columns) == 0:
                    for column in list(real_table.columns):
                        table_columns[column] = []
                for column in table_columns:
                    try:
                        if get_column_type(real_table[column].tolist()) == 1:
                            # print("String column ", column)
                            all_columns.add(column)
                            total_columns += 1
                            all_embeddings = table_columns[column]
                            if table_index == 2:
                                entities_only = all_embeddings["entities_only"]
                            else:
                                # fast text and bert
                                entities_only = all_embeddings
                            if (
                                isinstance(entities_only, float) == True
                                or len(entities_only) == 0
                                or len(all_embeddings) == 0
                            ):
                                entities_only = np.random.uniform(-1, 1, vec_length)
                            column_embeddings.append(entities_only)
                            track_columns[(table, column)] = i
                            if table in track_tables:
                                track_tables[table].add(i)
                            else:
                                track_tables[table] = {i}

                            if column not in record_same_cluster:
                                record_same_cluster[column] = {i}
                            else:
                                record_same_cluster[column].add(i)
                            i += 1

                    except:
                        continue
                count_tables += 1
            # =============================================================================
            #             if count_tables > 10:
            #                 break
            # =============================================================================

            all_true_edges = set()
            for col_index_set in record_same_cluster:
                set1 = record_same_cluster[col_index_set]
                set2 = record_same_cluster[col_index_set]
                current_true_edges = set()
                for s1 in set1:
                    for s2 in set2:
                        current_true_edges.add(tuple(sorted((s1, s2))))
                all_true_edges = all_true_edges.union(current_true_edges)

            # =============================================================================
            #         relationship = open("track_col.csv", 'w', encoding='utf-8')
            #         for i in track_columns.items():
            #             relationship.write(str(i[0][0]) + "," + str(i[0][1]) + ','+ str(i[1]) + '\n')
            #         relationship.close()
            # =============================================================================
            x = np.array(column_embeddings)
            zero_positions = set()
            for table in track_tables:
                indices = track_tables[table]
                all_combinations = findsubsets(indices, 2)
                for each in all_combinations:
                    zero_positions.add(each)

            arr = np.zeros((len(track_columns), len(track_columns)))
            for i in range(0, len(track_columns) - 1):
                for j in range(i + 1, len(track_columns)):
                    # print(i, j)
                    if (
                        (i, j) not in zero_positions
                        and (j, i) not in zero_positions
                        and i != j
                    ):
                        arr[i][j] = 1
                        arr[j][i] = 1
            # convert to sparse matrix representation
            s = csr_matrix(arr)

            all_distance = {}
            all_labels = {}
            record_current_precision = {}
            record_current_recall = {}
            record_current_f_measure = {}
            min_k = 1
            max_k = 0
            record_result_edges = {}

            for item in track_tables:
                # print(item, len(track_tables[item]))
                if len(track_tables[item]) > min_k:
                    min_k = len(track_tables[item])
                max_k += len(track_tables[item])

            for i in range(min_k, min(max_k, max_k)):
                # clusters = KMeans(n_clusters=14).fit(x)
                clusters = AgglomerativeClustering(
                    n_clusters=i,
                    metric="l2",
                    compute_distances=True,
                    linkage="complete",
                    connectivity=s,
                )
                clusters.fit_predict(x)
                labels = clusters.labels_  # .tolist()
                all_labels[i] = labels.tolist()
                all_distance[i] = metrics.silhouette_score(x, labels)
                result_dict = {}
                wrong_results = set()
                for col_index, label in enumerate(all_labels[i]):
                    if label in result_dict:
                        result_dict[label].add(col_index)
                    else:
                        result_dict[label] = {col_index}

                all_result_edges = set()
                for col_index_set in result_dict:
                    set1 = result_dict[col_index_set]
                    set2 = result_dict[col_index_set]
                    current_result_edges = set()
                    for s1 in set1:
                        for s2 in set2:
                            current_result_edges.add(tuple(sorted((s1, s2))))
                    all_result_edges = all_result_edges.union(current_result_edges)

                current_true_positive = len(
                    all_true_edges.intersection(all_result_edges)
                )
                current_precision = current_true_positive / len(all_result_edges)
                current_recall = current_true_positive / len(all_true_edges)

                record_current_precision[i] = current_precision
                record_current_recall[i] = current_recall
                record_current_f_measure[i] = 0
                if (current_precision + current_recall) > 0:
                    record_current_f_measure[i] = (
                        2 * current_precision * current_recall
                    ) / (current_precision + current_recall)
                record_result_edges[i] = all_result_edges
            distance_list = all_distance.items()
            distance_list = sorted(distance_list)
            x, y = zip(*distance_list)
            algorithm_k = max(all_distance, key=all_distance.get)
            final_precision[tablename] = record_current_precision[algorithm_k]
            final_recall[tablename] = record_current_recall[algorithm_k]
            final_f_measure[tablename] = record_current_f_measure[algorithm_k]
            overlapping = 1
            plt.plot(x, y)
            plt.title(tablename)
            plt.xlabel("number of clusters")
            plt.ylabel("silhouette score")
            plt.axvline(
                x=len(all_columns),
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
            # plt.axvline(x = max_k, color = 'black', label = 'max k')
            # plt.legend(bbox_to_anchor = (1.0, 1), loc = 'lower right', borderaxespad=3)
            plt.show()

            # =============================================================================
        except:
            continue

    end_time = time.time_ns()
    total_time = int(end_time - start_time) / 10**9
    total_precision = 0
    total_recall = 0
    total_f_measure = 0
    average_precision = 0
    average_recall = 0
    average_f_measure = 0

    for item in final_precision:
        total_precision += final_precision[item]
    for item in final_recall:
        total_recall += final_recall[item]
    for item in final_f_measure:
        total_f_measure += final_f_measure[item]

    average_precision = total_precision / len(final_precision)
    average_recall = total_recall / len(final_recall)
    average_f_measure = total_f_measure / len(final_f_measure)
    print("-------------------------------------")
    print("Result by:", method, " in ", input_folder_name)
    print("Average precision:", average_precision)
    print("Average recall", average_recall)
    print("Average f measure", average_f_measure)
    print("Total time:", total_time)


if __name__ == "__main__":
    main()
