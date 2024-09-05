import csv

def load_csv_raw(fname):
    """
    load the raw csv file
    """
    out_dict = {}
    num_lines = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if (num_lines == 0):
                num_lines += 1
                continue
            head = int(row[1])
            tail = int(row[2])
            ts = int(row[3])
            out_dict[(ts,head,tail)] = 1
            num_lines += 1
    return out_dict, num_lines


def write2csv(outname, out_dict):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['source','destination','timestamp'])

        for u,v,t in out_dict.keys():
            row = [u, v, t]
            writer.writerow(row)



def main():
    fname = "ml_uci.csv" 
    out_dict, num_lines = load_csv_raw(fname)
    print (f"num_lines: {num_lines}")
    outname = "tgbl-uci_edgelist.csv"
    write2csv(outname, out_dict)

    


if __name__ == "__main__":
    main()