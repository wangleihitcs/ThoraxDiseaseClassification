import json

def statistics(data_entry_path):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    # 1. get all finding labels
    finding_labels = []
    for image_index in data_dict.keys():
        labels = data_dict[image_index]
        for label in labels:
            if not finding_labels.__contains__(label):
                finding_labels.append(label)
    print('Finding Labels:\n%s' % finding_labels)

    # 2. get label num
    label_num_dict, one_label_num_dict = {}, {}
    for label in finding_labels:
        label_num_dict[label] = 0
        one_label_num_dict[label] = 0
    for image_index in data_dict.keys():
        labels = data_dict[image_index]
        for label in labels:
            label_num_dict[label] += 1
        if labels.__len__() == 1:
            one_label_num_dict[labels[0]] += 1
    Q = len(data_dict)
    for label in finding_labels:
        Qm = label_num_dict[label]
        lambda_label = (Q - Qm + 0.0) / Q
        print('%s: %d %d %f' % (label, label_num_dict[label], one_label_num_dict[label], lambda_label))

    # with open('../data/data_label.json', 'w') as file:
    #     json.dump(label_num_dict, file)

    print('images num = %d' % len(data_dict))

def statistics_train_test(data_entry_path, train_val_list_path):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    # 1. get all finding labels
    finding_labels = []
    for image_index in data_dict.keys():
        labels = data_dict[image_index]
        for label in labels:
            if not finding_labels.__contains__(label):
                finding_labels.append(label)
    print('Finding Labels:\n%s' % finding_labels)

    with open(train_val_list_path, 'r') as file:
        image_indexs = file.readlines()
    # 2. get label num
    label_num_dict, one_label_num_dict = {}, {}
    for label in finding_labels:
        label_num_dict[label] = 0
        one_label_num_dict[label] = 0

    for image_index in image_indexs:
        image_index = image_index.strip()
        labels = data_dict[image_index]
        for label in labels:
            label_num_dict[label] += 1
        if labels.__len__() == 1:
            one_label_num_dict[labels[0]] += 1
    for label in finding_labels:
        print('%s: %d %d' % (label, label_num_dict[label], one_label_num_dict[label]))

def main():
    data_entry_path = '../data/data_entry.json'
    # statistics(data_entry_path)
    train_val_list_path = '../data/train_val_list.txt'
    # statistics_train_test(data_entry_path, train_val_list_path)
    test_list_path = '../data/test_list.txt'
    statistics_train_test(data_entry_path, test_list_path)
main()