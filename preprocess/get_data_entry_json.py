import csv
import json

def get_data_entry(data_entry_path):
    data_entry_dict = {}
    with open(data_entry_path, 'r') as file:
        items = csv.reader(file)
        items = list(items)

        for item in items[1:]:
            image_index, finding_label = item[0], item[1]
            if finding_label.__contains__('|'):
                labels = finding_label.split('|')
            else:
                labels = [finding_label]
            data_entry_dict[image_index] = labels

    with open('../data/data_entry.json', 'w') as file:
        json.dump(data_entry_dict, file)
    print('get \'data/data_entry.json\' success')

def main():
    data_entry_path = '../data/Data_Entry_2017.csv'
    get_data_entry(data_entry_path)
main()