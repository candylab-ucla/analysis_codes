import os
import glob
import pandas as pd
import pingouin as pg
import numpy as np
import scipy as sp
import statistics

import seaborn as sea
import matplotlib.pyplot as plt
import netplotbrain


home = os.path.expanduser('~')
data_dir = os.path.join(home,'Desktop','tiger')
rs_list = sorted(glob.glob(data_dir + '/matrices/shen/*.csv'))

behav_data = pd.read_csv(data_dir + '/T1_Golden_TIGER_03.03.22.csv')


rs_sub_list = [int(i.split('/')[7].split('-')[1].split('.')[0]) for i in rs_list]
shared_behav_data = behav_data.loc[behav_data['sub'].isin(rs_sub_list)]
shared_behav_data.set_index('sub', inplace=True)


ctl_data = shared_behav_data[shared_behav_data['Group'] == 'CTL']
mdd_data = shared_behav_data[shared_behav_data['Group'] == 'MDD']

shared_subs = shared_behav_data.index.values
ctl_subs = ctl_data.index.values
mdd_subs = mdd_data.index.values


c_node_list = [101, 103, 104, 105, 106, 107, 108, 110, 113, 114, 117, 118, 119, 236, 237, 238, 243, 244, 245, 248,
249, 250, 251, 252, 253, 254, 255, 256]
c_node_list = [i-1 for i in c_node_list]


def read_in_matrices(ids, file_suffix=None, data_dir=data_dir, zscore=False):
    
    all_fc_data = {}
    all_sq_data = {}
            
    for subj in ids:
        
        # read it in and make sure it's symmetric and has reasonable dimensions
        tmp = np.loadtxt(data_dir+'/matrices/shen/sub-{}.csv'.format(subj), delimiter=',')
        assert tmp.shape[0]==tmp.shape[1]>1, "Matrix seems to have incorrect dimensions: {}".format(tmp.shape)
        
        for node in c_node_list:
            tmp[node, :] = np.nan
            tmp[:,node] = np.nan
        # take just the upper triangle and store it in a dictionary
        if ~zscore:
            all_fc_data[subj] = tmp[np.triu_indices_from(tmp, k=1)]
            all_sq_data[subj] = tmp.ravel()
            
        if zscore:
            all_fc_data[subj] = sp.stats.zscore(tmp[np.triu_indices_from(tmp, k=1)])
        
    # Convert dictionary into dataframe
    all_fc_data = pd.DataFrame.from_dict(all_fc_data, orient='index')
    all_fc_data.dropna(axis=1, inplace=True)
    
    all_sq_data = pd.DataFrame.from_dict(all_sq_data, orient='index')
    
    return all_fc_data, all_sq_data


all_fc_data, all_sq_data = read_in_matrices(shared_subs)
ctl_fc_data, ctl_sq_data = read_in_matrices(ctl_subs)
mdd_fc_data, mdd_sq_data = read_in_matrices(mdd_subs)


nets = pd.read_csv('/Users/anurimamummaneni/Desktop/tiger/code/shen_network_labels.csv')
nets['Node'] = nets['Node']-1
net_names = pd.read_csv('/Users/anurimamummaneni/Desktop/tiger/code/shen_network_names.csv')
nets = nets.loc[~nets.index.isin(c_node_list)]



subject = []
network_label = []
network_strength = []



i = 0

for idx, row in all_sq_data.iterrows():
    fc_mat = pd.DataFrame(np.array(row).reshape(268,268))
    print(fc_mat.shape)
        
    for net in np.unique(nets['Network']):
        print(net)
        nodes = nets[nets['Network'] == net]['Node'].values
        net_mat = fc_mat.loc[nodes,nodes]
        
        
        net_strength = net_mat.values[np.triu_indices_from(net_mat.values,1)].mean()
        
        subject.append(shared_subs[i])
        network_label.append(net_names[net_names['Network'] == net]['Label'].values[0])
        network_strength.append(net_strength)
        
        print(net_names[net_names['Network'] == net]['Label'].values[0] + ' Strength = {}'.format(net_strength))
        
    i += 1


net_strength_data = {'sub' : subject, 'net' : network_label, 'strength' : network_strength}
net_strength_df = pd.DataFrame(net_strength_data)
net_strength_df.set_index('sub', inplace=True) #make data frame for all network strength values

ctl_net = net_strength_df.loc[ctl_fc_data.index]
mean_net_ctl = ctl_net.groupby(['net']).mean() #get mean CTL and MDD network strength

mdd_net = net_strength_df.loc[mdd_fc_data.index]
mean_net_mdd = mdd_net.groupby(['net']).mean()


ls = [mean_net_ctl, mean_net_mdd]
net_concat = pd.concat(ls, keys=['CTL', 'MDD'])

#plot differences between network strength in CTL and MDD participants
plt.ion()
fig, axs = plt.subplots(1,2)
for i in range(len(np.unique(net_concat.index.get_level_values(0)))):
    ax = axs.flat[i]
    condition = np.unique(net_concat.index.get_level_values(0))[i]
    ax.bar(net_concat.loc[condition].index, net_concat.loc[condition]['strength'].values)
    ax.set_ylim(0,1.1)
    ax.set_title('{}'.format(condition))

sp.stats.ttest_ind(ctl_net[ctl_net['net'] == 'VA']['strength'], mdd_net[mdd_net['net'] == 'VA']['strength'])



#plot NBS adjacency matrices for MDD and CTL participants
shen_nodes = pd.read_csv('/Users/anurimamummaneni/Desktop/tiger/code/shen_nodes.csv')
import bct #for NBS


p_list_mdd = []
adj_list_mdd = []
null_list_mdd = []

#calculate NBS separately for each network
for net in np.unique(nets['Network']):


    nodes = nets[nets['Network'] == net]['Node'].values.tolist() #ensure only one network at a time is being comapred
    zero_nodes = [x for x in np.arange(268).tolist() if x not in nodes]

        
    mdd_sq = []
    for idx, row in mdd_sq_data.iterrows():
        mdd_sq.append(np.array(row).reshape(268, 268))

    mdd_arr =  np.array(mdd_sq).reshape(268, 268, 57)


    ctl_sq = []
    for idx, row in ctl_sq_data.iterrows():
        ctl_sq.append(np.array(row).reshape(268, 268))
        
    ctl_arr =  np.array(ctl_sq).reshape(268, 268, 21)


    mdd_arr_net = np.zeros(shape=(len(nodes), len(nodes), 57))
    ctl_arr_net = np.zeros(shape=(len(nodes), len(nodes), 21))
    
    #arrange MDD and CTL functional connectivity data into correct input shape for NBS
    for k in range(mdd_arr.shape[2]):
      
        for l in range(len(nodes)):
            for n in range(len(nodes)):
            
                    mdd_arr_net[l, n, k] = mdd_arr[nodes[l], nodes[n], k]
                    
                    
                    
    for k in range(ctl_arr.shape[2]):
      
        for l in range(len(nodes)):
            for n in range(len(nodes)):
            
                    ctl_arr_net[l, n, k] = ctl_arr[nodes[l], nodes[n], k]


        

    try: #apply NBS
        p, adj, null = bct.nbs_bct(mdd_arr_net, ctl_arr_net, tail='right', thresh=np.sqrt(78)*0.2)
        p_list_mdd.append(p)
        adj_list_mdd.append(adj)
        null_list_mdd.append(null)

    except:
        print(net)
    
            
        inds = list(zip(*np.where(adj == 1))) #calculate edges from adjacency matrices per netplotbrain requirements
            
        i = []
        j = []

        for k in inds:
            i.append(nodes[k[0]]+1)
            j.append(nodes[k[1]]+1)
            
        edges = pd.DataFrame({'i' : i, 'j' : j})
        all_nodes = np.unique(i + j)

        node_size = []

        for k in all_nodes:
            all_edges = i + j
            node_size.append((all_edges.count(k) / len(all_edges)) * (15 * len(nodes)))

        nodes_df = shen_nodes.loc[all_nodes] #calculate nodes from shen networks
        nodes_df['degree_centrality'] = node_size

        plt.ion() #plot adjacency matrix separately for each
        col = sea.color_palette("husl", 8)[net-1]
        netplotbrain.plot(template='MNI152NLin2009cAsym', template_style='glass', nodes=nodes_df, node_size='degree_centrality', node_sizevminvmax='absolute', node_color=col, view='LSR', node_alpha=0.6, edges=edges, edge_color=col, edge_alpha=0.5, edge_widthscale=0.8)
        
        label = net_names[net_names['Network'] == net]['Label'].values[0]
        
        plt.savefig('{}_stronger_in_mdd.png'.format(label), dpi=800) #this code calculates edges stronger in MDDs, NBS calculation would need to be reversed to calculate edges stronger in CTLs
        plt.close()

    except:
        print(net)
    



