import blobfile as bf
import os

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
        
def label_to_class(classes_file, save_path, labels):
    sorted_classes = {}
    with open(classes_file,'r') as fp:
        data = fp.readlines()
        index = 0
        for line in data:
            classname = line.strip()
            sorted_classes[index] = classname
            index += 1
            
    classes = [sorted_classes[i] for i in labels]

    with open(save_path, "w") as f:
        for i in range(len(labels)):
            f.write(f"{labels[i]} {classes[i]}")
            f.write('\n')
        f.close()
        
        
def write_claseses_file(dataset_path, save_path):
    all_files = _list_image_files_recursively(dataset_path)
    class_names = [path.split("/")[-2] for path in all_files]
    sorted_classes = {i: x for i, x in enumerate(sorted(set(class_names)))}

    with open(save_path, "w") as f:
        for index, class_name in sorted_classes.items():
            f.write(class_name)
            f.write('\n')
        f.close()


def label2index(file_path, label):
    class_dict = dict()

    if os.path.exists(file_path):
        with open(file_path,'r') as fp:
            data = fp.readlines()
            index = 0
            for line in data:
                classname = line.strip()
                class_dict[classname] = index
                index += 1
    return class_dict[label]

def index2label(index, file_path):
    class_dict = dict()

    if os.path.exists(file_path):
        with open(file_path,'r') as fp:
            data = fp.readlines()
            index = 0
            for line in data:
                classname = line.strip()
                class_dict[index] = classname
                index += 1
    return class_dict[index]


def load_diversity_quality(file_path, label):
    class_dict = dict()
    if os.path.exists(file_path):
        with open(file_path,'r') as fp:
            data = fp.readlines()
            index = 0
            for line in data:
                line = line.strip().split(",")
                classname = line[0]
                lpips = line[1]
                class_prob = line[2]
                start_prob = line[3]
                class_dict[classname] = {"lpips":lpips, "class_prob":class_prob, "start_prob":start_prob}
                index += 1
    return float(class_dict[label]["lpips"]), float(class_dict[label]["class_prob"]),float(class_dict[label]["start_prob"])


