#!/usr/bin/python2

DIR = "plots_all/"



import mcrep
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pickle
import time


def filter_top_scores(d, fraction):
    sorted_d = sorted(d.items(), key=lambda x: -x[1])

    return sorted_d[:int(fraction*len(sorted_d))]

def determine_representants(uploads, downloads):
    uploads_sorted = sorted(uploads.items(), key=lambda x: x[1])

    result = []
    for i in list(np.arange(0.1, 1.01, 0.1)):
        offset = 0
        while True:
            try:
                candidate = uploads_sorted[offset + int(i*len(uploads_sorted))]
            except IndexError:
                offset -= 1
                continue
            if downloads[candidate[0]] > 0:
                ratio = uploads[candidate[0]]/downloads[candidate[0]]
                if ratio > 0.2 and ratio < 2:
                    result.append(candidate[0])
                    break
            offset += 1

    return result 

# scores is a dict from nodes to tuples, each tuple being a dict scoring:
# (pimrank, spread pimrank, pagerank)
with open('pr_scores', 'r') as f:
    scores = pickle.load(f)

with open('/home/pim/bigresults/exp/results.pickle', 'r') as f:
    scores1 = pickle.load(f)

with open('/home/pim/bigresults/exp/results_2.pickle', 'r') as f:
    scores2 = pickle.load(f)

with open('/home/pim/bigresults/exp/results_4.pickle', 'r') as f:
    scores4 = pickle.load(f)

flow_scores = (scores1, scores2, scores4)

adaptor = mcrep.adaptor.Adaptor('multichain_exp.csv')
#adaptor = mcrep.adaptor.Adaptor('test.csv')

graph = adaptor.create_interaction_graph()
ordered_graph = adaptor.create_ordered_interaction_graph()

nodes = graph.nodes()

flow_scores_index = [[{},{},{}],[{},{},{}]]
print(flow_scores_index)

for i in xrange(2):
    for j in xrange(3):
        for calc_node in nodes:
            calc_scores = {}
            for node in nodes:
                if node != calc_node:
                    calc_scores[node] = flow_scores[j][calc_node][node][i]
            flow_scores_index[i][j][calc_node] = calc_scores




num_agents = len(nodes)

info = {}
average_tpr_scores = {}
average_tpr_spread_scores = {}
average_tpr_spread_weighted_scores = {}
average_pr_scores = {}
average_flow_scores = {}
average_flow_normal_scores = {}
average_bc_scores = {}
informativeness = {}
cut_off_scores = {}
cut_off_scores_flow_tpr = {}
cut_off_scores_flow_bc = {}
ratio_scores = {}
diff_scores = {}
uploads = {}
downloads = {}
times = {}
last_times = {}


for line in adaptor.iterate_csv():
    time_str = line[-1].strip()
    pubkey1 = line[0]
    pubkey2 = line[1]
    insert_time = time.mktime(time.strptime(time_str, '"%Y-%m-%d %H:%M:%S"'))

    times[pubkey1] = min(times.get(pubkey1, time.time()), insert_time)
    times[pubkey2] = min(times.get(pubkey2, time.time()), insert_time)

    last_times[pubkey1] = max(last_times.get(pubkey1, 0), insert_time)
    last_times[pubkey2] = max(last_times.get(pubkey2, 0), insert_time)
    

    


cut_offs = np.arange(0.01, 1.0, 0.03)

for node in nodes:
    average_pr_scores[node] = 0
    average_tpr_scores[node] = 0
    average_tpr_spread_scores[node] = 0

    upload = 0

    for edge in graph.out_edges([node], data='capacity'):
        upload += edge[2]

    download = 0

    for edge in graph.in_edges([node], data=True):
        download += edge[2]['capacity']

    if download > 1e-9:
        ratio_scores[node] = float(upload)/download
    else:
        ratio_scores[node] = None

    diff_scores[node] = float(upload-download)
    uploads[node] = upload
    downloads[node] = download

reps = determine_representants(uploads, downloads)



for node in nodes:
    average_tpr_scores[node] = float(sum(scores.get(calc_node, ({},{},{}))[0].get(node, 0) 
        for calc_node in nodes))/(num_agents - 1.0)

    average_tpr_spread_scores[node] = float(sum(scores.get(calc_node, ({},{},{}))[1].get(node, 0) 
        for calc_node in nodes))/(num_agents - 1.0)

    average_pr_scores[node] = float(sum(scores.get(calc_node, ({},{},{}))[2].get(node, 0) 
        for calc_node in nodes))/(num_agents - 1.0)

    scores_uploads = zip((scores.get(calc_node, ({},{},{}))[1].get(node, 0) for calc_node in nodes),
                         (uploads[calc_node] for calc_node in nodes))

    average_tpr_spread_weighted_scores[node] = float(sum(score*upload for score, upload in scores_uploads))/(num_agents - 1.0)



#    for calc_node in nodes:
#        print((node, calc_node))
#        print(scores1.get(calc_node, ({}, {})))
#        print(scores1.get(calc_node, ({}, {})).get(node, 0))
#        print(scores1.get(calc_node, ({}, {})).get(node, 0)[0])
#

    average_flow_scores[node] = float(sum(scores1.get(calc_node, {}).get(node, (0,0))[0]
        for calc_node in nodes))/(num_agents - 1.0)

    scores_uploads = zip((scores.get(calc_node, ({},{},{}))[1].get(node, 0) for calc_node in nodes),
                         (uploads[calc_node] for calc_node in nodes))

    average_flow_normal_scores[node] = float(sum(float(score)/(upload+1) for score, upload in scores_uploads))/(num_agents - 1.0)


    average_bc_scores[node] = float(sum(scores1.get(calc_node, {}).get(node, (0,0))[1] 
        for calc_node in nodes))/(num_agents - 1.0)

    infos = [0, 0, 0]
    for i in xrange(3):
        for other_node in nodes:
            if node != other_node:
                if flow_scores_index[0][i][node][other_node] > 0:
                    infos[i] += 1.0/(num_agents-1)

    informativeness[node] = tuple(infos)
        


    cut_off_numbers = []
    cut_off_numbers2 = []
    cut_off_numbers3 = []
    for cut_off in cut_offs:
        tpr_top = filter_top_scores(scores.get(node)[1], cut_off)
        pr_top = filter_top_scores(scores.get(node)[2], cut_off)
        flow_top = filter_top_scores(flow_scores_index[0][0].get(node), cut_off)
        bc_top = filter_top_scores(flow_scores_index[1][0].get(node), cut_off)

        common_count = len(set(pair[0] for pair in tpr_top).intersection(set(pair[0] for pair in pr_top)))
        common_count2 = len(set(pair[0] for pair in tpr_top).intersection(set(pair[0] for pair in flow_top)))
        common_count3 = len(set(pair[0] for pair in bc_top).intersection(set(pair[0] for pair in flow_top)))


        cut_off_numbers.append(float(common_count)/len(tpr_top))
        cut_off_numbers2.append(float(common_count2)/len(tpr_top))
        cut_off_numbers3.append(float(common_count3)/len(flow_top))


    cut_off_scores[node] = tuple(cut_off_numbers)
    cut_off_scores_flow_tpr[node] = tuple(cut_off_numbers2)
    cut_off_scores_flow_bc[node] = tuple(cut_off_numbers3)



ratios = np.fromiter((ratio_scores.get(node) for node in nodes if ratio_scores.get(node) is not None), np.float)
diffs = np.fromiter((diff_scores.get(node) for node in nodes), np.float)
pimranks = np.fromiter((average_tpr_scores.get(node) for node in nodes), np.float)
pimranks_spread = np.fromiter((average_tpr_spread_scores.get(node) for node in nodes), np.float)
pimranks_spread_weighted = np.fromiter((average_tpr_spread_weighted_scores.get(node) for node in nodes), np.float)
pageranks = np.fromiter((average_pr_scores.get(node) for node in nodes), np.float)
netflows = np.fromiter((average_flow_scores.get(node) for node in nodes), np.float)
ratio_netflows = np.fromiter((average_flow_scores.get(node) for node in nodes if ratio_scores.get(node) is not None), np.float)
netflows_normal = np.fromiter((average_flow_normal_scores.get(node) for node in nodes), np.float)
bartercasts = np.fromiter((average_bc_scores.get(node) for node in nodes), np.float)
ratio_pimranks = np.fromiter((average_tpr_scores.get(node) for node in nodes if ratio_scores.get(node) is not None), np.float)
first_seen_times = [times.get(node) for node in nodes]
active_times = np.fromiter(((last_times.get(node) - times.get(node))/3600.0 for node in nodes),np.float)

informativeness1 = np.fromiter(sorted(informativeness.get(node)[0] for node in nodes), np.float)
informativeness2 = np.fromiter(sorted(informativeness.get(node)[1] for node in nodes), np.float)
informativeness4 = np.fromiter(sorted(informativeness.get(node)[2] for node in nodes), np.float)

informativeness1_filter = np.fromiter(sorted(informativeness.get(node)[0] for node in nodes if downloads.get(node) > 0), np.float)
informativeness2_filter = np.fromiter(sorted(informativeness.get(node)[1] for node in nodes if downloads.get(node) > 0), np.float)
informativeness4_filter = np.fromiter(sorted(informativeness.get(node)[2] for node in nodes if downloads.get(node) > 0), np.float)


cut_off_commons = [[]]*len(cut_offs)
for i in range(0, len(cut_offs)):
    cut_off_commons[i] = np.fromiter((cut_off_scores.get(node)[i] for node in nodes), np.float)

cut_off_commons2 = [[]]*len(cut_offs)
for i in range(0, len(cut_offs)):
    cut_off_commons2[i] = np.fromiter((cut_off_scores_flow_tpr.get(node)[i] for node in nodes), np.float)

cut_off_commons3 = [[]]*len(cut_offs)
for i in range(0, len(cut_offs)):
    cut_off_commons3[i] = np.fromiter((cut_off_scores_flow_bc.get(node)[i] for node in nodes), np.float)




npuploads = np.fromiter((uploads.get(node) for node in nodes), np.float)
npdownloads = np.fromiter((downloads.get(node) for node in nodes), np.float)

max_pimrank = max(pimranks)
size_pimranks = np.fromiter(((average_tpr_scores.get(node))/max_pimrank*200.0 for node in nodes), np.float)

max_pimrank_spread = max(pimranks_spread)
size_pimranks_spread = np.fromiter(((average_tpr_spread_scores.get(node))/max_pimrank_spread*200.0 for node in nodes), np.float)

max_pimrank_spread_weighted = max(pimranks_spread_weighted)
size_pimranks_spread_weighted = np.fromiter(((average_tpr_spread_weighted_scores.get(node))/max_pimrank_spread_weighted*200.0 for node in nodes), np.float)


max_netflow = max(netflows)
size_netflows = np.fromiter(((average_flow_scores.get(node))/max_netflow*200.0 for node in nodes), np.float)

max_netflow_normal = max(netflows_normal)
size_netflows_normal = np.fromiter(((average_flow_normal_scores.get(node))/max_netflow_normal*200.0 for node in nodes), np.float)




print(len(ratios))
print(len(diffs))


plt.scatter(ratios, ratio_pimranks)
plt.title('Correlation between ratio and Temporal PageRank')
plt.xlabel('Contribution ratio (upload/download)')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_ratio_tpr.svg')
plt.clf()

plt.scatter(diffs, pimranks)
plt.title('Correlation between difference and Temporal PageRank')
plt.xlabel('Contribution diff (upload-download)')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_diff_pimrank.svg')
plt.clf()

plt.scatter(diffs, netflows)
plt.title('Correlation between difference and Netflow')
plt.xlabel('Contribution diff (upload-download)')
plt.ylabel('Average Netflow score')
plt.savefig(DIR + 'plot_diff_pimrank.svg')
plt.clf()

plt.scatter(ratios, ratio_netflows)
plt.title('Correlation between ratio and average Netflow')
plt.xlabel('Ratio')
plt.ylabel('Netflow')
plt.savefig(DIR + 'plot_ratio_netflow.svg')
plt.clf()


plt.scatter(netflows, bartercasts)
plt.title('Correlation between Netflow and BarterCast')
plt.xlabel('Average Netflow score')
plt.ylabel('Average BarterCast score')
plt.savefig(DIR + 'plot_netflow_bc.svg')
plt.clf()

plt.scatter(pageranks, pimranks_spread)
plt.title('Correlation between pagerank and Temporal PageRank, spread')
plt.xlabel('Average PageRank score')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_pr_tpr.svg')
plt.clf()

plt.scatter(netflows, pimranks_spread)
plt.title('Correlation between (average) Netflow and Temporal PageRank (spread)')
plt.xlabel('Netflow score')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_netflow_pimrank.svg')
plt.clf()

plt.scatter(first_seen_times, pimranks_spread)
plt.title('Correlation between First Seen time and Temporal PageRank (spread)')
plt.xlabel('Moment of first observation')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_first_seen_pimrank.svg')
plt.clf()

plt.scatter(active_times, pimranks_spread)
plt.title('Correlation between active time and Temporal PageRank (spread)')
plt.xlabel('Difference between last and first seen')
plt.ylabel('Average Temporal PageRank score')
plt.savefig(DIR + 'plot_time_active_pimrank.svg')
plt.clf()

plt.boxplot(cut_off_commons, labels=cut_offs)
plt.title('Fraction in common in top fraction')
plt.xticks(rotation='vertical')
plt.ylabel('Fraction')
plt.savefig(DIR + 'plot_boxplot_common.svg')
plt.clf()


plt.boxplot(cut_off_commons2, labels=cut_offs)
plt.title('Fraction in common in top fraction (TPR/Netflow)')
plt.xticks(rotation='vertical')
plt.ylabel('Fraction')
plt.savefig(DIR + 'plot_boxplot_tpr_netflow.svg')
plt.clf()


plt.boxplot(cut_off_commons3, labels=cut_offs)
plt.title('Fraction in common in top fraction (BarterCast/Netflow)')
plt.xticks(rotation='vertical')
plt.ylabel('Fraction')
plt.savefig(DIR + 'plot_boxplot_bc_netflow.svg')
plt.clf()

plt.plot(informativeness1, 'b-')
plt.plot(informativeness2, 'r-')
plt.plot(informativeness4, 'g-')

plt.title('Informativeness curves for different alpha')
plt.xlabel('Identities')
plt.ylabel('Informativeness (fraction)')
plt.savefig(DIR + 'plot_info.svg')
plt.clf()

plt.plot(informativeness1_filter, 'b-')
plt.plot(informativeness2_filter, 'r-')
plt.plot(informativeness4_filter, 'g-')

plt.title('Informativeness curves for different alpha, filtered')
plt.xlabel('Identities')
plt.ylabel('Informativeness (fraction)')
plt.savefig(DIR + 'plot_info_filter.svg')
plt.clf()



rep_num = 0
for rep in reps:
    rep_score = scores.get(rep)[1]
    rep_num += 1
    

    sizes = np.fromiter(((rep_score.get(node))/max_pimrank_spread*200.0 for node in nodes if node != rep), np.float)

    npuploads_but_one = np.fromiter((uploads.get(node) for node in nodes if node != rep), np.float)
    npdownloads_but_one = np.fromiter((downloads.get(node) for node in nodes if node != rep), np.float)


    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(npdownloads_but_one+1, npuploads_but_one+1, s=sizes)
    plt.plot([downloads.get(rep)+1], [uploads.get(rep)+1], 'or')
    plt.title('Comparison of size (Temporal PageRank)')
    plt.xlabel('Downloads (MB)')
    plt.ylabel('Uploads (MB)')
    plt.grid(True)
    plt.savefig(DIR + 'plot_up_down_size_tpr_' + str(rep_num) +  '.svg')
    plt.clf()

for i in xrange(0, 3):
    rep_num = 0
    for rep in reps:
        rep_score = flow_scores[i].get(rep)
        rep_num += 1
        

        unscaled_sizes = [rep_score.get(node)[0] for node in nodes if node != rep]
        max_size = max(unscaled_sizes)
        print(max_size)
        if max_size < 1e-14:
            continue
        sizes = np.fromiter((float(size)/max_size*200 for size in unscaled_sizes), np.float)

        npuploads_but_one = np.fromiter((uploads.get(node) for node in nodes if node != rep), np.float)
        npdownloads_but_one = np.fromiter((downloads.get(node) for node in nodes if node != rep), np.float)


        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(npdownloads_but_one+1, npuploads_but_one+1, s=sizes)
        plt.plot([downloads.get(rep)+1], [uploads.get(rep)+1], 'or')
        plt.title('Comparison of size (Netflow, alpha=' + str(2**i) + ')')
        plt.xlabel('Downloads (MB)')
        plt.ylabel('Uploads (MB)')
        plt.grid(True)
        plt.savefig(DIR + 'plot_up_down_size_flow' + str(i) + '_' + str(rep_num) +  '.svg')
        plt.clf()





plt.xscale('log')
plt.yscale('log')
plt.scatter(npdownloads+1, npuploads+1, s=size_pimranks)
plt.title('Comparison of size (Temporal PageRank)')
plt.xlabel('Downloads (MB)')
plt.ylabel('Uploads (MB)')
plt.grid(True)
plt.savefig(DIR + 'plot_up_down_size_tpr.svg')
plt.clf()



plt.xscale('log')
plt.yscale('log')
plt.scatter(npdownloads+1, npuploads+1, s=size_pimranks_spread)
plt.title('Comparison of size (Temporal PageRank Spread)')
plt.xlabel('Downloads (MB)')
plt.ylabel('Uploads (MB)')
plt.grid(True)
plt.savefig(DIR + 'plot_up_down_size_tpr_spread.svg')
plt.clf()


plt.xscale('log')
plt.yscale('log')
plt.scatter(npdownloads+1, npuploads+1, s=size_pimranks_spread_weighted)
plt.title('Comparison of size (Temporal PageRank Spread, Weighted)')
plt.xlabel('Downloads (MB)')
plt.ylabel('Uploads (MB)')
plt.grid(True)
plt.savefig(DIR + 'plot_up_down_size_tpr_spread_weighted.svg')
plt.clf()




plt.xscale('log')
plt.yscale('log')
plt.scatter(npdownloads+1, npuploads+1, s=size_netflows)
plt.title('Comparison of size (Netflow)')
plt.xlabel('Downloads (MB)')
plt.ylabel('Uploads (MB)')
plt.grid(True)
plt.savefig(DIR + 'plot_up_down_size_flow.svg')
plt.clf()


plt.xscale('log')
plt.yscale('log')
plt.scatter(npdownloads+1, npuploads+1, s=size_netflows_normal)
plt.title('Comparison of size (Netflow, normalized)')
plt.xlabel('Downloads (MB)')
plt.ylabel('Uploads (MB)')
plt.grid(True)
plt.savefig(DIR + 'plot_up_down_size_flow_normal.svg')
plt.clf()


