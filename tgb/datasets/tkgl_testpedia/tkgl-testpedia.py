import csv
import datetime
import glob, os

def remove_isolated_edges(fname):
    """
    remove isolated edges from the edgelist
    ts,head,tail,relation_type
    """
    edge_dict = {}
    node_dict = {}
    num_lines = 0
    num_single_edges = 0
    num_single_nodes = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if (num_lines == 0):
                num_lines += 1
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            rel_type = row[3]
            num_lines += 1

            # check singleton edges

            if ((head,tail,rel_type) not in edge_dict):
                edge_dict[(head,tail,rel_type)] = 1
            else:
                edge_dict[(head,tail,rel_type)] += 1

            # check singleton nodes
            if (head not in node_dict):
                node_dict[head] = 1
            else:
                node_dict[head] += 1
            
            if (tail not in node_dict):
                node_dict[tail] = 1
            else:
                node_dict[tail] += 1

            
    
    for key,value in edge_dict.items():
        if (value == 1):
            # print (f"key: {key}, value: {value}")
            num_single_edges += 1
    print (f"num_single_edges: {num_single_edges}")

    for key,value in node_dict.items():
        if (value == 1):
            # print (f"key: {key}, value: {value}")
            num_single_nodes += 1
    print (f"num_single_nodes: {num_single_nodes}")



def main():
    fname = "tkgl-smallpedia_edgelist.csv"
    remove_isolated_edges(fname)
   




if __name__ == "__main__":
    main()