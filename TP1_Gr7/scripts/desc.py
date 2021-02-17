import os
import heapq
from collections import Counter

#Region[Blue] Database computation analysis

def file_name_to_class(filename):
    file_class = filename.split('_')[0]
    return file_class

def compute_class(topk):
    class_dict = list(k[1] for k in topk)
    return Counter(class_dict)

def return_class(topk):
    class_dict = compute_class(topk)
    max_val = max(class_dict.values())
    p = max_val/len(topk)
    max_keys = [k for k,v in class_dict.items() if v == max_val]
    return p, max_keys

def parse_result(topk, pfx=""):
    toppicture = topk[0]
    p, keys = return_class(topk)
    print(pfx+"BEST PICTURE APPROACH : class of the most similar picture")
    print(pfx+"Predicted class : "+toppicture[1])
    print(pfx)
    print(pfx+"TOPK APPROACH : most represented class in the top k pictures")
    if len(keys) > 1:
        print(pfx+"Several dominant results with a "+str(p*100)[:4]+" % rate")
        classes = ""
        for key in keys:
            classes = classes + key + ", "
        print(pfx+"Possibles classes : "+classes[:-2])
    else:
        print(pfx+"Predicted class "+str(keys[0])+" : "+str(p*100)[:4]+" %")

#EndRegion

#Region[Red] Research for k most similar pictures

def similarities_research(path_to_query, path_to_database, desc, k=3, verbose=False, show_res = False):
    
    if verbose: print(path_to_query)

    topk = []

    for database_file in os.listdir(path_to_database):
        if database_file.endswith(".jpg"):

            sim = desc(path_to_query, os.path.join(path_to_database,database_file), show_res)
            heapq.heappush(topk, (sim, file_name_to_class(database_file), database_file))
            topk = heapq.nlargest(k, topk)

            if verbose: print("| "+database_file+" : "+str(sim))

    return topk

#EndRegion

#Region[Green]

def compute_description(path_to_data, path_to_database, desc):
    print("[COMPUTING] descriptor : "+desc.__name__)
    for query_file in filter(lambda x : x.endswith('.jpg'), os.listdir(path_to_data)):
        topk = similarities_research(os.path.join(path_to_data, query_file),path_to_database, desc)
        print("| Query "+query_file)
        parse_result(topk, pfx="| | ")
        print("| ")

#EndRegion


'''compute_description('data/part2', 'data/part2/database', orb_similarities)
print('= = =')
compute_description('data/part2', 'data/part2/database', rgb_similarities)'''