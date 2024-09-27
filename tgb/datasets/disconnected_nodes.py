import csv
import datetime
import glob, os
import networkx as nx
from matplotlib import pyplot as plt

def find_isolated_edges(fname):
    """
    find isolated edges from the edgelist
    ts,head,tail,relation_type
    """
    edge_dict = {}
    node_dict = {}
    headtail_dict = {}
    num_lines = 0
    num_single_edges = 0
    num_single_nodes = 0
    num_single_headtails = 0
    disconnected_triples = []
    G = nx.Graph()  
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
                # add edge to the nx graph
                # only add the triple once to the graph, when it first occurs:
                G.add_edge(head, tail, label=rel_type)  # Add edge between head and tail
            else:
                edge_dict[(head,tail,rel_type)] += 1


            if ((head, tail) not in headtail_dict):
                headtail_dict[(head,tail)] = 1
            else:
                headtail_dict[(head,tail)] +=1                

            # check singleton nodes
            if (head not in node_dict):
                node_dict[head] = 1
            else:
                node_dict[head] += 1
            
            if (tail not in node_dict):
                node_dict[tail] = 1
            else:
                node_dict[tail] += 1


    num_triples_static_graph = 0
    # Check for disconnected triples in the static graph
    for edge in G.edges():
        head, tail = edge        
        num_triples_static_graph+=1
        # Check if both the head and tail have no other neighbors (degree 1)
        if G.degree(head) == 1 and G.degree(tail) == 1:
            relation = G.get_edge_data(head, tail)['label']
            disconnected_triples.append((head, relation, tail))

    print("Number of Disconnected triples in static graph:", len(disconnected_triples))
    print("Number of total triples in static graph:", num_triples_static_graph)

    # for each of these static disconnected triples: check if this subject-object combi occurs only once -> then it is removable
    removable_triples = []
    for triple in disconnected_triples:
        if headtail_dict[(triple[0], triple[2])] == 1:
            removable_triples.append(triple)
        
    print("Number of removable triples:", len(removable_triples)) # this is the number of disconnected triples that occur only once in this subject-object combi, and that can thus be removed
    removable_triples = set(removable_triples)

    # Add the connected triples (triples not in disconnected_triples)
    G_connected = nx.Graph()
    num_lines = 0
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

            if (head, rel_type, tail) not in removable_triples:
                G_connected.add_edge(head, tail, label=relation)

    # Draw the graph with connected triples
    # plt.figure(figsize=(8, 6))
    # # pos = nx.spring_layout(G_connected)  # Layout for the nodes
    # nx.draw(G_connected) #, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')
    # plt.savefig("connected_graph.png")

    # plt.figure(figsize=(8, 6))
    # # pos = nx.spring_layout(G)  # Layout for the nodes
    # nx.draw(G) #, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')
    # plt.savefig("original_graph.png")

    for key,value in headtail_dict.items():
        if (value == 1):
            # print (f"key: {key}, value: {value}")
            num_single_headtails += 1
    print (f"num_single_headtailcombis: {num_single_headtails}")
                
    for key,value in edge_dict.items():
        if (value == 1):
            # print (f"key: {key}, value: {value}")
            num_single_edges += 1
    print (f"num_single_edges: {num_single_edges}")

    single_nodes = []
    for key,value in node_dict.items():
        if (value == 1):
            # print (f"key: {key}, value: {value}")
            num_single_nodes += 1
            single_nodes.append(key)
    print (f"num_single_nodes: {num_single_nodes}")

    # single_nodes = set(single_nodes)
    # with open(fname, 'r') as f:
    #     reader = csv.reader(f, delimiter =',')
    #     num_lines_new = 0
    #     for row in reader: 
    #         ts = int(row[0])
    #         head = row[1]
    #         tail = row[2]
    #         rel_type = row[3]

    #         if (head in single_nodes or tail in single_nodes):
    #             continue

    #         else:
                


    #         # check singleton edges

    #         if ((head,tail,rel_type) not in edge_dict):
    #             edge_dict[(head,tail,rel_type)] = 1
    #         else:
    #             edge_dict[(head,tail,rel_type)] += 1

    #         # check singleton nodes
    #         if (head not in node_dict):
    #             node_dict[head] = 1
    #         else:
    #             node_dict[head] += 1
            
    #         if (tail not in node_dict):
    #             node_dict[tail] = 1
    #         else:
    #             node_dict[tail] += 1   


def main():
    names = ['./tkgl_wikidata/tkgl-wikidata_edgelist.csv','./tkgl_smallpedia/tkgl-smallpedia_edgelist.csv', './tkgl_icews/tkgl-icews_edgelist.csv', './tkgl_polecat/tkgl-polecat_edgelist.csv', './tkgl_yago/tkgl-yago_edgelist.csv'] # , './tkgl_wikidata/tkgl-wikidata_edgelist.csv',
    for fname in names:
        print(fname)
        find_isolated_edges(fname)

   




if __name__ == "__main__":
    main()