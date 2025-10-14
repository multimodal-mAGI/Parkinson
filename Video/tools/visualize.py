import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def plot_topology_matrix(topology_matrix, save_name):
    plt.imshow(topology_matrix, interpolation='nearest',cmap=plt.cm.GnBu)
    plt.colorbar() 
    tick_marks = np.arange(25)
    plt.xticks(tick_marks, tick_marks, fontsize=8)
    plt.yticks(tick_marks, tick_marks, fontsize=8)
    plt.savefig(save_name)
    plt.clf()


def main():

    get_graph = np.load('/NAS/dclab/psw/agi/ProtoGCN/graph.npy', allow_pickle=True)
    vis_graph = get_graph[0]
    
    # ❗️ 이 부분을 추가하여 GPU 텐서 오류를 해결합니다.
    if isinstance(vis_graph, torch.Tensor):
        vis_graph = vis_graph.cpu().numpy()
        
    save_name = 'vis_graph.jpg'
    plot_topology_matrix(vis_graph, save_name)


if __name__ == '__main__':
    main()