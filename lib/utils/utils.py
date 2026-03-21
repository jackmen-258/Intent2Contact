import os
import logging

def cycle(dl):
    while True:
        for data in dl:
            yield data

def load_latest_checkpoint(folder):
    weights = os.listdir(folder)
    weights_paths = [os.path.join(folder, weight) for weight in weights if weight.endswith('.pt')]
    return max(weights_paths, key=os.path.getctime)

class IntentFilteredDataset:
    def __init__(self, original_dataset, target_intent):
        self.original_dataset = original_dataset
        self.target_intent = target_intent
        
        # 筛选目标意图的样本
        self.filtered_indices = []
        print(f"Filtering {target_intent} samples...")
        
        for i in range(len(original_dataset)):
            sample = original_dataset[i]
            if sample["intent_name"] == target_intent:
                self.filtered_indices.append(i)
        
        print(f"Found {len(self.filtered_indices)} {target_intent} samples out of {len(original_dataset)} total")
        
        if len(self.filtered_indices) == 0:
            raise ValueError(f"No {target_intent} samples found in dataset!")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]
    
def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def makelogger(log_dir,mode='a'):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s'%log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

