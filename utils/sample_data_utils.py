import json
from torch.utils.data import Dataset
import random

def explore():
    with open("data/en.zh.test1.jl", "r") as fin:
        for line in fin.readlines():
            print(line)
            input()
            
def collate_fn(batch):
    return tuple(zip(*batch))
            
class CLIRDataset(Dataset):
    def __init__(self, data_path, type) -> None:
        super().__init__()
        self.src_ids = []
        self.queries = []
        self.targets = []
        with open(data_path, "r") as fin:
            for line in fin.readlines():
                """
                {
                    "src_id": "39", 
                    "src_query": "Albedo", 
                    "tgt_results": [["553658", 6], ["1712206", 5], ["1849020", 5], ["1841381", 5], ["1541246", 3], ["5248845", 2], ["1498501", 2], ["5748160", 2], ["718267", 2], ["5392042", 2], ["2764586", 2], ["202402", 2], ["3316208", 2], ["5375638", 2], ["1161946", 2], ["3542927", 2], ["801173", 2], ["5378920", 2], ["3543134", 2], ["1782326", 1], ["939382", 1], ["3779245", 1], ["2938855", 1], ["3316164", 1], ["5702473", 1], ["939409", 1], ["2938822", 1], ["3315916", 1], ["3542015", 1], ["1156740", 1], ["1042704", 1], ["1690586", 1], ["701461", 1], ["3544317", 1], ["5255753", 1], ["3547732", 1], ["5892814", 1], ["3545801", 1], ["3315788", 1], ["1661774", 1], ["3316223", 1], ["6231204", 1], ["3313232", 1], ["3314496", 1], ["5377208", 1], ["3543616", 1], ["1664298", 1], ["5525991", 1], ["1669824", 1], ["1454516", 1]]
                }
                """
                l_data = json.loads(line)
                self.src_ids.append(l_data["src_id"])
                self.queries.append(l_data["src_query"])
                self.targets.append(l_data["tgt_results"])

        if type == 0:
            # Ensure that the sample size is not greater than the dataset
            sample_size = 1600
            # Randomly sample indices
            sampled_indices = random.sample(range(len(self.src_ids)), sample_size)
            # Use the sampled indices to filter the data
            self.src_ids = [self.src_ids[i] for i in sampled_indices]
            self.queries = [self.queries[i] for i in sampled_indices]
            self.targets = [self.targets[i] for i in sampled_indices]

    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, index):
        return self.src_ids[index], self.queries[index], self.targets[index]
    

class CLIRKLDataset(Dataset):
    def __init__(self, data_path, target_data_path, type) -> None:
        super().__init__()
        self.src_ids = []
        self.queries = []
        self.targets = []
        self.target_queries = []
        with open(target_data_path, "r") as fin:
            for line in fin.readlines():
                l_data = json.loads(line)
                self.target_queries.append(l_data["src_query"])
        with open(data_path, "r") as fin:
            for line in fin.readlines():
                """
                {
                    "src_id": "39", 
                    "src_query": "Albedo", 
                    "tgt_results": [["553658", 6], ["1712206", 5], ["1849020", 5], ["1841381", 5], ["1541246", 3], ["5248845", 2], ["1498501", 2], ["5748160", 2], ["718267", 2], ["5392042", 2], ["2764586", 2], ["202402", 2], ["3316208", 2], ["5375638", 2], ["1161946", 2], ["3542927", 2], ["801173", 2], ["5378920", 2], ["3543134", 2], ["1782326", 1], ["939382", 1], ["3779245", 1], ["2938855", 1], ["3316164", 1], ["5702473", 1], ["939409", 1], ["2938822", 1], ["3315916", 1], ["3542015", 1], ["1156740", 1], ["1042704", 1], ["1690586", 1], ["701461", 1], ["3544317", 1], ["5255753", 1], ["3547732", 1], ["5892814", 1], ["3545801", 1], ["3315788", 1], ["1661774", 1], ["3316223", 1], ["6231204", 1], ["3313232", 1], ["3314496", 1], ["5377208", 1], ["3543616", 1], ["1664298", 1], ["5525991", 1], ["1669824", 1], ["1454516", 1]]
                }
                """
                l_data = json.loads(line)
                self.src_ids.append(l_data["src_id"])
                self.queries.append(l_data["src_query"])
                self.targets.append(l_data["tgt_results"])
        
        if type == 0:
            # Ensure that the sample size is not greater than the dataset
            sample_size = 1600
            # Randomly sample indices
            sampled_indices = random.sample(range(len(self.src_ids)), sample_size)
            # Use the sampled indices to filter the data
            self.src_ids = [self.src_ids[i] for i in sampled_indices]
            self.queries = [self.queries[i] for i in sampled_indices]
            self.targets = [self.targets[i] for i in sampled_indices]
            self.target_queries = [self.target_queries[i] for i in sampled_indices]

    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, index):
        return self.src_ids[index], self.queries[index], self.targets[index], self.target_queries[index]
    
    
if __name__ == "__main__":
    explore()