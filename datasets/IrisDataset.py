import torch
from torch.utils.data import Dataset  

class IrisDataset(Dataset):
    
    targets = 0
    
    def __init__(self, xy, transforms=None):  

        self.len = xy.shape[0]  
        self.x_data = torch.Tensor(xy.iloc[:, 0:xy.shape[1]-1].values)  # xy[:, :-1] Read all data except the last column.
        self.y_data = torch.LongTensor(xy.iloc[:,xy.shape[1]-1].values.astype(int))  # [-1] Convert the last dimension to array
        self.transforms = transforms
        
        self.targets = self.y_data
    
    def __getitem__(self, index):  
        data_x, data_y = self.x_data[index], self.y_data[index]
        if self.transforms is not None:
            data_x = self.transforms(torch.unsqueeze(data_x,1).numpy())
        return (data_x, data_y)
    
    def __len__(self):  # Data set length
        return self.len