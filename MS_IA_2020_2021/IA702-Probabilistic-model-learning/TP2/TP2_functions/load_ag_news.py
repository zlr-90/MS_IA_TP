

def load_ag_news(path = "./"):
    train_set = []
    with open(path + "train_ag_news_file.txt", "r") as f:
        for line in f.readlines():
            instance = eval(line)
            train_set.append(instance)
            
    test_set = []
    with open(path + "test_ag_news_file.txt", "r") as f:
        for line in f.readlines():
            instance = eval(line)
            test_set.append(instance)
            
    return train_set, test_set