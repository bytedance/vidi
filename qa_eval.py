"""
A standalone evaluation pipeline.
Copyright 2025 Bytedance.
"""
import os
import os.path as osp
import argparse
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# base colors for visualization
BASE_COLORS = ['blue', 'red', 'green', 'orange', 'cyan', 'grey', 'brown', 'purple', 'pink', 'olive', 'black',\
               'indianred', 'chocolate', 'darkolivegreen', 'gold', 'darkcyan', 'slategrey', 'darkblue', 'indigo'\
               'deeppink', 'sienna', 'crimson', 'darkseagreen', 'dodgerblue', 'navy', 'violet', 'tan', 'teal']

# draw plots w.r.t different attributes
def draw_plot(result_rates, attribute, plot_name, output_dir=''):
    if attribute in ["ultra-short", "short", "medium", "long", "ultra-long"]:
        sub_folder = "duration_category"
    elif attribute in ["keyword", "phrase", "sentence"]:
        sub_folder = "query_format"
    elif attribute in ["audio", "vision", "vision+audio"]:
        sub_folder = "query_modality"
    else:
        sub_folder = ""
    output_path = osp.join(output_dir, sub_folder)
    if not osp.isdir(output_path): 
        os.mkdir(output_path)

    thres = np.linspace(0, 1, 101)
    # calculate AUC for each model
    auc_scores = {algo: np.trapz(result_rate, thres)*100 for algo, result_rate in result_rates.items()}
    colors = {algo: BASE_COLORS[idx] for idx, (algo, _) in enumerate(result_rates.items())}

    # sort models by AUC scores
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=False)

    # plot settings
    plt.figure(figsize=(10, 8))
    for algo, _ in sorted_auc_scores:
        plt.plot(thres, result_rates[algo], label=f'{algo} [{auc_scores[algo]:.2f}%]', linewidth=3, color=colors[algo])

    # title and labels
    plt.title(f'Accuracy-{plot_name} Plot for {attribute}', fontsize=30)
    plt.xlabel(f'{plot_name} Threshold', fontsize=24)
    plt.ylabel(f'Accuracy', fontsize=24)

    # set x and y axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # define custom tick intervals
    x_ticks = np.arange(0, 1.1, 0.1)  # Ticks from 0 to 1 with a step of 0.1
    y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # grid and legend
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='best', fontsize=24)    

    # save the plot
    plt.savefig(osp.join(output_path, f'{attribute}_{plot_name}_plot.png'), dpi=300, bbox_inches='tight')
    if attribute == 'overall':
        plt.savefig(osp.join(output_path, f'{attribute}_{plot_name}_plot.pdf'), dpi=300, bbox_inches='tight')    
    plt.close()

# draw radar plot w.r.t attributes
def radar_plot(attributes, all_results, scores, mode, output_dir=''):
    colors = {algo: BASE_COLORS[idx] for idx, (algo, _) in enumerate(all_results.items())}
    num_vars = len(attributes)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop
    _, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for name, values in scores.items():
        values = values.tolist()
        values += values[:1]  # complete the loop
        ax.plot(angles, values, label=name, linewidth=2, color=colors[name])
        ax.fill(angles, values, alpha=0.2, color=colors[name])

    # set the axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes, fontsize=15)

    # set radial range and grid
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.tick_params(axis='y', labelsize=12)
    
    # title and legend
    plt.title(mode+' Scores of Attributes', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.1), fontsize=15)
    # save the plot
    plt.savefig(osp.join(output_dir, mode + '_radar_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# compute IOU scores between predicted and ground-truth timeranges
def overlap_ratio(pred, gt):
    if len(gt) == 0 or gt.shape[0] == 0:
        if len(pred) == 0 or pred.shape[0] == 0:
            return 1.0
        else:
            return 0.0
    # IOU = 0 if no predictions
    if len(pred) == 0 or pred.shape[0] == 0:
        return 0.0
    
    # merge adjacent time slots
    pred = merge_time_spans(pred)
    len_gt = np.sum(gt[:,1] - gt[:,0])

    # filter wrong predictions
    pred = pred[pred[:, 0] <= pred[:, 1]]
    if pred.shape[1] == 0:
        return 0.0
    
    # calculate IoU scores
    intersect = 0    
    len_pred = np.sum(pred[:,1] - pred[:,0])
    union = len_pred+len_gt
    for i in range(len(pred)):
        for j in range(len(gt)):
            sta_time = np.maximum(pred[i][0], gt[j][0])
            end_time = np.minimum(pred[i][1], gt[j][1])
            intersect += np.maximum(0.0, end_time - sta_time)
    union -= intersect

    iou = intersect / (union+1e-16)
    iou = np.maximum(np.minimum(1.0, iou), 0.0)
    return iou

# compute iou scores over thresholds
def success_overlap(results):
    n_query = len(results)
    thres = np.linspace(0, 1, 101)
    success = np.zeros(len(thres))
    iou = np.zeros(n_query)
    for i in range(n_query):
        iou[i] = overlap_ratio(results[i]['answer'], results[i]['gt'])
    for i in range(len(thres)):
        success[i] = np.sum(iou > thres[i]) / float(n_query+1e-16)
    auc = np.trapz(success, thres) # success score is the normalized area under the curve
    return success, auc

# compute precision/recall scores over thresholds
def precision_recall_thres(results):
    precision, recall = compute_precision_recall(results, avg=False)
    thres = np.linspace(0, 1, 101)
    precision_thres = np.zeros(len(thres))
    recall_thres = np.zeros(len(thres))
    for i, t in enumerate(thres):
        precision_thres[i] = np.mean(precision >= t)
        recall_thres[i] = np.mean(recall >= t)

    return precision_thres, recall_thres

# break down results w.r.t attributes
def breakdown_results(attributes, all_results, output_dir):
    thres = np.arange(0, 1.05, 0.05)
    iou_rate, precision_rate, recall_rate = [], [], []
    for _ in range(len(attributes)):
        iou = {algo_id: np.zeros((len(thres))) for algo_id in all_results}
        prec = {algo_id: np.zeros((len(thres))) for algo_id in all_results}
        rec = {algo_id: np.zeros((len(thres))) for algo_id in all_results}
        iou_rate.append(iou)
        precision_rate.append(prec)
        recall_rate.append(rec)
    result_rates = {'IoU': iou_rate, 'Precision': precision_rate, 'Recall': recall_rate}
    # break down results w.r.t attributes
    pre_scores, rec_scores, iou_scores = {}, {}, {}
    for algo_id in all_results:
        results = all_results[algo_id]
        pre_scores[algo_id], rec_scores[algo_id], iou_scores[algo_id] = np.zeros(len(attributes)), np.zeros(len(attributes)), np.zeros(len(attributes))
        n_query = len(results)
        for j in range(len(attributes)):
            if attributes[j] in ["ultra-short", "short", "medium", "long", "ultra-long"]:
                results_ = [results[i] for i in range(n_query) if results[i]['duration_category'] == attributes[j]]
            elif attributes[j] in ["keyword", "phrase", "sentence"]:
                results_ = [results[i] for i in range(n_query) if results[i]['query_format'] == attributes[j]]
            elif attributes[j] in ["audio", "vision", "vision+audio"]:
                results_ = [results[i] for i in range(n_query) if results[i]['query_modality'] == attributes[j]]
            else:
                results_ = results
            iou_rate[j][algo_id], iou_scores[algo_id][j] = success_overlap(results_)
            precision_rate[j][algo_id], recall_rate[j][algo_id] = precision_recall_thres(results_)
            pre_scores[algo_id][j], rec_scores[algo_id][j] = compute_precision_recall(results_)
    # draw plots for each metric
    for i in range(len(attributes)):
        for plot_name in ['IoU', 'Precision', 'Recall']:
            draw_plot(result_rates[plot_name][i], attributes[i], plot_name, output_dir=output_dir)
    return pre_scores, rec_scores, iou_scores

# merge overlapping or adjacnet time ranges
def merge_time_spans(intervals):
    if len(intervals) == 0:
        return np.array([])
    # sort intervals by start time
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0]]
    for current in intervals[1:]:
        _, prev_end = merged[-1]
        curr_start, curr_end = current
        # check for overlap or adjacency
        if curr_start <= prev_end:  # overlap or adjacent
            merged[-1][1] = max(prev_end, curr_end)  # merge intervals
        else:
            merged.append(current)
    return np.array(merged)

# calculate intersection of two timestamp lists
def interval_intersection(intervals1, intervals2):
    try:
        i, j = 0, 0
        result = []
        while i < len(intervals1) and j < len(intervals2):
            a_start, a_end = intervals1[i]
            b_start, b_end = intervals2[j]
            # check uf the two intervals overlap
            if a_start <= b_end and b_start <= a_end:
                result.append((max(a_start, b_start), min(a_end, b_end)))
            # move the pointer of the interval that ends first
            if a_end < b_end:
                i += 1
            else:
                j += 1
    except Exception as e:
        print('interval_intersection error:', e)
        print(intervals1, intervals2)
        return []
    return result

# calculate union of two timestamp lists
def interval_union(intervals1, intervals2):
    # combine both interval lists and sort them by start time
    intervals = sorted(intervals1 + intervals2)
    result = []
    if len(intervals):
        # start with the first interval as the current one to merge
        current_interval = intervals[0]
        # iterate over the rest of the intervals
        for interval in intervals[1:]:
            if interval[0] <= current_interval[1]:
                current_interval[1] = max(current_interval[1], interval[1])
            else:
                result.append(current_interval)
                current_interval = interval
        result.append(current_interval)

    return result

# calculate precision and recall
def compute_precision_recall(results, avg=True):
    gt_all, pred_all, inter_all, union_all = [], [], [], []
    for item in results:
        gt = [[min(interval), max(interval)] for interval in item['gt'] if len(interval) == 2]
        pred = [[min(interval), max(interval)] for interval in item['answer'] if len(interval) == 2]
        inter = interval_intersection(copy.deepcopy(gt), copy.deepcopy(pred))
        union = interval_union(copy.deepcopy(gt), copy.deepcopy(pred))
        gt_all.append(sum([interval[1] - interval[0] for interval in gt]))
        pred_all.append(sum([interval[1] - interval[0] for interval in pred]))
        inter_all.append(sum([interval[1] - interval[0] for interval in inter]))
        union_all.append(sum([interval[1] - interval[0] for interval in union]))

    gt_all, pred_all, inter_all = np.array(gt_all), np.array(pred_all), np.array(inter_all)
    recall = []
    for i, g in zip(inter_all, gt_all):
        if g != 0:
            recall.append(i / g)
    recall = np.array(recall)
    
    precision = []
    for i, g, p in zip(inter_all, gt_all, pred_all):
        if g == 0 and p == 0:
            precision.append(1.)
        elif p != 0:
            precision.append(i / p)
    precision = np.array(precision)
    # calculate average scores
    if avg:
        thres = np.linspace(0, 1, 101)
        precision_thres = np.zeros(len(thres))
        recall_thres = np.zeros(len(thres))
        for i, t in enumerate(thres):
            precision_thres[i] = np.mean(precision >= t)
            recall_thres[i] = np.mean(recall >= t)
        
        precision = np.trapz(precision_thres, thres)
        recall = np.trapz(recall_thres, thres)

    return precision, recall

# load prediction results and merge with ground-truth
def load_result(gt_path, res_path):
    # load ground-truth    
    with open(gt_path) as fq:
        gts = json.load(fq)    
    gt_list = {}
    for gt in gts:
        gt_list[gt['query_id']] = gt

    # load current results
    if not osp.exists(res_path):
        print(res_path, 'not exist!!!!')
        raise Exception
    if res_path.endswith('json'):
        with open(res_path) as fq:
            pred_list = json.load(fq)
    elif res_path.endswith('jsonl'):
        with open(res_path) as fq:
            pred_list = [json.loads(x) for x in fq.readlines()]
    else:
        print(res_path, 'not supported format!!!!')
        raise Exception

    # update results
    for i in range(len(pred_list)):
        query_id = pred_list[i]['query_id']
        if len(pred_list[i]['answer']) == 0 or (len(pred_list[i]['answer'])==1 and len(pred_list[i]['answer'][0])==0):
            pred_list[i]['answer'] = np.array([])
        else:
            pred_list[i]['answer'] = np.array(pred_list[i]['answer'])
            pred_list[i]['answer'][:, 0] = np.floor(pred_list[i]['answer'][:, 0])
            pred_list[i]['answer'][:, 1] = np.ceil(pred_list[i]['answer'][:, 1])
        pred_list[i].update(gt_list[query_id])
        pred_list[i]['gt'] = np.array(pred_list[i]['gt'])

    return pred_list

# print overall results
def print_result(model_name, nums, precision, recall, auc):
    print("-----------------------------------------------------")
    print(f"{model_name} # query={nums}")
    print(f"Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%, IoU: {auc*100:.1f}%")
    print("-----------------------------------------------------")

# print results for each attribute
def print_attribute_result(precision_scores, recall_scores, iou_scores, attributes):
    data = []
    for i, attr in enumerate(attributes):
        for algo_name in iou_scores:
            values = [precision_scores[algo_name][i], recall_scores[algo_name][i], iou_scores[algo_name][i]]
            data.append([attr, algo_name] + values)
    
    df = pd.DataFrame(data, columns=['attribute', 'method', 'precision', 'recall', 'iou'])
    df.set_index(['attribute', 'method'], inplace=True)
    df_formatted = df.applymap(lambda x: f"{x*100:.1f}%")
    print(df_formatted)
    df_formatted.to_csv("results_table.csv")

# evaluate results
def evaluate_results(output_dir, res_path, gt_path):
    all_results = {}
    attributes = np.array([
        "overall", "ultra-short", "short", "medium", "long", "ultra-long", "keyword", "phrase", "sentence", "audio", "vision", "vision+audio"
    ])
    root = "" # path of results of compared methods
    compared_methods = ['results_Gemini-2.0-Flash.json', 'results_Gemini-2.5-Pro.json', 'results_GPT-4o.json']
    compared_paths = [res_path] + [osp.join(root, res_name) for res_name in compared_methods]

    # evaluate all compared models
    for compare_res_path in compared_paths:
        split_tup = osp.splitext(compare_res_path)
        res_name = split_tup[0].replace('results_', '')
        results = load_result(gt_path, compare_res_path)
        all_results[res_name] = results
        _, iou_auc = success_overlap(results)
        pre_auc, rec_auc = compute_precision_recall(results)
        print_result(res_name, len(results), pre_auc, rec_auc, iou_auc)

    # attribute evaluation
    precision_scores, recall_scores, iou_scores = breakdown_results(attributes, all_results, output_dir)
    radar_plot(attributes, all_results, iou_scores, 'IoU', output_dir)
    print_attribute_result(precision_scores, recall_scores, iou_scores, attributes)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="VUE-TR evaluation")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--gt_path", default='VUE-TR_ground_truth.json', type=str, help="The path to file containing ground-truth.")
    parser.add_argument("--output_dir", default='results', help="The path to save results.")
    args = parser.parse_args()
    # Evaluate the results
    evaluate_results(args.output_dir, args.pred_path, args.gt_path)    