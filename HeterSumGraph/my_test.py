import os
from dgl.data.utils import load_graphs

for root, _, files in os.walk(
        "C:\\Users\\amirreza\\Desktop\\dars\\arshad\\tez\\project\\HSG\\HeterSumGraph\\cache\\CNNDM\\graphs\\train"):
    indexes=[]
    for file in files:
        from_index = int(file[:-4])
        indexes.append(from_index)
        path = os.path.join(root, file)
        # g, label_dict = load_graphs(path)
        # print(file,"  ", len(g))
    max_indexes=max(indexes)
    print(max_indexes," ",max_indexes+256)
    for i in range(0,max_indexes,256):
        if i not in indexes:
            print("ERROR: ",i)


