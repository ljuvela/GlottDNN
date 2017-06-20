import cPickle as pickle 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import minimum

class DataContainer():
    
    def __init__(self, tag=None, inputs=None, outputs=None):
        self.tag=tag
        self.inputs=inputs
        self.outputs=outputs
        

class DataObj():

    def read_file_list(self):
        file_list = []
        f = open(self.scp, 'r')
        for line in f:
            file_list.append(line.rstrip())
        n_files = len(file_list)
        return [file_list, n_files]
    
    def evalFileTags(self):
        fileTags = []
        for f in self.file_list:
            bname_with_ext = os.path.basename(f)
            bname = os.path.splitext(bname_with_ext)[0]
            fileTags.append(bname)    
        return fileTags
            
    def setSubsetFlags(self, shuffle=False):
        # initialize subset membership flags
        self.subset_membership = {}
        self.subset_membership['val'] = [False] * self.n_files  
        self.subset_membership['test'] = [False] * self.n_files  
        self.subset_membership['train'] = [False] * self.n_files  
        val_set = 0
        test_set = 0   
        
        # todo: accept shuffling index
        
        # start filling 
        for i in range(self.n_files): 
            if val_set < self.n_validation:
                self.subset_membership['val'][i] = True
                val_set += 1
            elif test_set < self.n_test:
                self.subset_membership['test'][i] = True
                test_set += 1
            else:
                self.subset_membership['train'][i] = True
  

    def __init__(self, key, dim, scp, ext=None, dtype=np.float32, n_validation=1, n_test=1):
        self.key = key
        self.dim = dim
        self.scp = scp
        if ext == None:
            self.ext = "." + key
        else:
            self.ext = ext
        self.dtype = dtype
        self.n_files = 0
        self.file_list, self.n_files = self.read_file_list()
        self.file_tags = self.evalFileTags()
        
        # make train, validation and test set flags
        self.n_validation = n_validation
        self.n_test = n_test
        self.setSubsetFlags()
        
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0.1, 0.9)) # range for compatibility with GlottDNN internal implementation
        
    def compute_stats(self):
        
        if self.dim == 0:
            Warning("Dimension for " + self.key + " was set to 0, skipping normalization")
            return

        self.file_lengths = []
        for fname in self.file_list:
            data = np.fromfile(fname, dtype=self.dtype)
            if len(data) % self.dim == 0:
                data = data.reshape(-1, self.dim)
            else:
                raise ValueError("Data dimension does not match in '" + fname + "' " +
                                 "(expected " + str(self.dim) + ")")
        
            self.std_scaler.partial_fit(data)
            self.minmax_scaler.partial_fit(data)
            self.file_lengths.append(data.shape[0])
            
    def getDataFile(self, file_ind, scaler='minmax'):
        
        data = np.fromfile(self.file_list[file_ind], dtype=self.dtype)
        data = data.reshape(-1, self.dim)
        
        if scaler == 'std':
            data = self.std_scaler.transform(data)
        elif scaler == 'minmax':   
            data = self.minmax_scaler.transform(data)
        elif scaler == None:
            data = data
        else:
            raise ValueError("Unsupported scaler type '" + self.scaler +"', " +
                             "supported types are 'std', 'minmax' and None" )
            
        return data
    
    def deNormData(self, data, scaler):
        if scaler == 'std':
            data = self.std_scaler.inverse_transform(data)
        elif scaler == 'minmax':   
            data = self.minmax_scaler.inverse_transform(data)
        elif scaler == None:
            data = data
        else:
            raise ValueError("Unsupported scaler type '" + self.scaler +"', " +
                             "supported types are 'std', 'minmax' and None" )
        return data


class Dataset:
    
    def __init__(self):
        self.DataObjs = {}
        self.scalers = {}
        self.n_files = 0
        
    def addDataObj(self, obj, scaler='minmax'):
        if self.n_files == 0:
            self.n_files = obj.n_files
            self.DataObjs[obj.key] = obj
            self.scalers[obj.key] = scaler
            
        elif self.n_files == obj.n_files:
            self.DataObjs[obj.key] = obj
            self.scalers[obj.key] = scaler
        else:
            raise ValueError("Length mismatch in scp files: " +
                             "expected " + str(self.n_files) + ", found " + str(obj.n_files) +
                             " (type " + obj.key + ")")
        
    def compute_stats(self):
        for d in self.DataObjs.values():
            d.compute_stats()
            
    def getData(self, ind):
        data = {}
        for key, obj in self.DataObjs.iteritems():
            data[key] = obj.getDataFile(ind, self.scalers[key])
        return data
    
    def getDim(self, keys):
        dim = 0
        for key in keys:
            dim += self.DataObjs[key].dim    
        return dim
    
    def getFileTag(self, ind=None):
        if ind == None:
            return self.DataObjs.values()[0].file_tags
        else:
            return self.DataObjs.values()[0].file_tags[ind]   
        
    def getSubsetFlags(self, subset_key):
        
        return self.DataObjs.values()[0].subset_membership[subset_key]
    
    
    def getDataInputOutput(self, seq_ind, inputs, outputs):
        in_dims = []
        n_frames = 0
        for in_key in inputs:
            in_dims.append(self.DataObjs[in_key].dim)
            n_frames = self.DataObjs[in_key].file_lengths[seq_ind]
        out_dims = []
        for out_key in outputs:
            out_dims.append(self.DataObjs[out_key].dim)
            
        x = np.zeros((n_frames, sum(in_dims)))
        start = 0
        for d, key in zip(in_dims, inputs):
            x[:, start:start+d] = self.DataObjs[key].getDataFile(seq_ind,
                                                                 scaler=self.scalers[key])
            start += d
         
        y = np.zeros((n_frames, sum(out_dims))) 
        start = 0    
        for d, key in zip(out_dims, outputs):
            y[:, start:start+d] = self.DataObjs[key].getDataFile(seq_ind,
                                                                 scaler=self.scalers[key])
            start += d
        
        return [x, y]
    
    def splitData(self, keys, data):
        dim_start = 0
        split_data = {}
        for key in keys:
            data_stream = data[:,dim_start:dim_start+self.DataObjs[key].dim]
            split_data[key] = self.DataObjs[key].deNormData(data_stream,
                                                             scaler=self.scalers[key])
        return split_data
    
        
    def saveDataProvider(self, filepath='./data_provider.dat'):
        pickle.dump(self, open(filepath, "w"))
        
    def loadDataProvider(self, filepath='./data_provider.dat'):
        self = pickle.load(open(filepath, "r"))
        
        
class Dataset_iter:
    
    def __init__(self, dataset, shuffle=False, io_keys=None,
                 subset='test'):
        self.iter_ind = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.io_keys = io_keys
        self.subset = subset
        self.inds = range(dataset.n_files)
        if self.shuffle:
            self.inds = np.random.permutation(self.inds)
        
    def __iter__(self):
        return self
    
    def next(self):
        
        if self.iter_ind < self.dataset.n_files: 
        
            while not self.dataset.getSubsetFlags(self.subset)[self.inds[self.iter_ind]]:
                self.iter_ind += 1
                if not self.iter_ind < self.dataset.n_files:
                    raise StopIteration()    
        
            if self.io_keys == None:
                data = self.dataset.getData(self.inds[self.iter_ind])
            else:
                data = self.dataset.getDataInputOutput(self.inds[self.iter_ind],
                                                       self.io_keys[0],
                                                       self.io_keys[1])
            tag = self.dataset.getFileTag(self.inds[self.iter_ind]) 
            self.iter_ind += 1
            return DataContainer(tag=tag, inputs=data[0], outputs=data[1])
        else:
            raise StopIteration()
        
        
class DatasetBatchBuffer_iter:
    
    def __init__(self, dataset, batch_size=128, shuffle=False, io_keys=None,
                 subset='test'):

        self.iter_ind = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.io_keys = io_keys
        self.subset = subset
        self.inds = range(dataset.n_files)
        if self.shuffle:
            self.inds = np.random.permutation(self.inds)
    
        self.batch_size = batch_size
        self.input_batch = np.zeros((self.batch_size, self.dataset.getDim(io_keys[0])),
                                    dtype=np.float32) # todo: dynamic data type
        self.output_batch = np.zeros((self.batch_size, self.dataset.getDim(io_keys[1])),
                                     dtype=np.float32) # todo: dynamic data type
        self.batch_buffer_ptr = 0

        self.data_len = 0
        self.data_read_ptr = 0
        
        self.stop_iter = False

        
    def __iter__(self):
        return self
    
    def next(self):
        
        # loop until something is returned by the buffer
        while True:
        
            if self.stop_iter:
                raise StopIteration()
        
            # step by minimum of read or write buffer remaining
            step = np.minimum(self.data_len-self.data_read_ptr, self.batch_size-self.batch_buffer_ptr)
            if step > 0:
                self.input_batch[self.batch_buffer_ptr:self.batch_buffer_ptr+step,:] = self.data[0][self.data_read_ptr:self.data_read_ptr+step,:]
                self.output_batch[self.batch_buffer_ptr:self.batch_buffer_ptr+step,:] = self.data[1][self.data_read_ptr:self.data_read_ptr+step,:]
                self.data_read_ptr += step
                self.batch_buffer_ptr += step

            # batch was filled, step data iterator
            if self.batch_buffer_ptr >= self.batch_size:
                self.batch_buffer_ptr = 0
                #return [self.input_batch, self.output_batch]
                return DataContainer(tag=None, inputs=self.input_batch, outputs=self.output_batch)

            # file end was reached, open new file
            if self.data_read_ptr >= self.data_len:
                if self.iter_ind < self.dataset.n_files: 
                    
                    # only read from specified subset 
                    while not self.dataset.getSubsetFlags(self.subset)[self.inds[self.iter_ind]]:
                        self.iter_ind += 1
                        if not self.iter_ind < self.dataset.n_files:
                            raise StopIteration()    
                    
                    self.data = self.dataset.getDataInputOutput(self.inds[self.iter_ind],
                                                            self.io_keys[0],
                                                            self.io_keys[1])
                    self.iter_ind += 1
                    self.data_read_ptr = 0
                    self.data_len = self.data[0].shape[0]
                else:
                    self.stop_iter = True
                    # flush buffer with zeros
                    self.input_batch[self.batch_buffer_ptr:-1,:] = 0.0
                    self.output_batch[self.batch_buffer_ptr:-1,:] = 0.0
                    #return [self.input_batch, self.output_batch]
                    return DataContainer(tag=None, inputs=self.input_batch, outputs=self.output_batch)
                 
            

            
    
    
        