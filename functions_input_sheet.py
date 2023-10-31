from torch.utils.data import Dataset
import random
from scipy.linalg import block_diag
import numpy as np
# import matplotlib.pyplot as plt
import math

# %% input sheets
class input_sheets():
    """
    Generates input sheets for the model.
    
    Returns:
        input sheet class with all input sheet coordinates, their sizes and densities.
    """
    
    def __init__(self, sheet_sizes = np.array(([10,5])), sheet_densities = [1]):

        self.sheet_sizes_x = sheet_sizes[:,0] if sheet_sizes.ndim > 1 else np.array([sheet_sizes[0]])
        self.sheet_sizes_y = sheet_sizes[:,1] if sheet_sizes.ndim > 1 else np.array([sheet_sizes[1]])
        self.sheet_densities = sheet_densities
        self.num_regions = len(sheet_densities)
        
        self.calc_num_receptors()
        
    def calc_num_receptors(self):
        """
        Calculates the total number of receptors for each input region
        """
        self.receptors_each_region = []
        
        for i in range(len(self.sheet_sizes_x)):
            self.receptors_each_region.append(self.sheet_sizes_x[i]*self.sheet_densities[i]*self.sheet_sizes_y[i]*self.sheet_densities[i])
            
        self.total_receptors = np.sum(self.receptors_each_region)
        
        
    def create_input_sheets(self):
        """
        Runs loop over all input sheet parameters to create sheets.
        Adds sheet coordinates to class.
        """
        self.input_sheets = []
        
        for sht_idx in range(len(self.sheet_sizes_y)):
            self.input_sheets.append(self.create_single_input_sheet(sht_idx))
            
        
    def create_single_input_sheet(self, idx):
        """
        Creates coordinates for single input sheet.

        Parameters
        ----------
        idx : int
            Index of input sheet to be created
        """
        return np.meshgrid(np.linspace(1,self.sheet_sizes_x[idx],self.sheet_sizes_x[idx]*self.sheet_densities[idx]), 
                           np.linspace(1,self.sheet_sizes_y[idx],self.sheet_sizes_y[idx]*self.sheet_densities[idx]))
    


# %% Input class
class generate_inputs(Dataset):
    """
    Generates inputs for model.
    
    If more than one input sheet, distributes between these with probability p.
    
    """
    def __init__(self, input_sheets, num_input = 10000, p=[1], input_sizes=[5,10], seed = random.randint(0,10000)):
        
        self.num_input_total = num_input
        self.p = np.asarray((p))
        self.seed = seed
        self.input_sizes = input_sizes
        self.calc_num_patterns_each(input_sheets.num_regions)

        #random.seed(self.seed) # Removed this to set seed globally for whole file
        #np.random.seed(self.seed) # Removed this to set seed globally for whole file
        
        self.gaussian_blobs(input_sheets)


    def __len__(self):
        """
        Number of samples
        """
        return len(self.all_samples)


    def __getitem__(self, idx):
        """
        Returns a sample at index idx
        """
        return self.all_samples[idx]


    def calc_num_patterns_each(self, num_regions):
        # checks num patt each sum to one
        assert len(self.p) == num_regions, 'probabilities do not match number of regions'
        assert math.isclose(np.sum(self.p),1,rel_tol=1e-6), "Probabilities do not sum to 1."
            
        self.num_patterns_each = np.round(self.num_input_total*self.p).astype('int')
        
        self.final_p = (self.num_patterns_each/sum(self.num_patterns_each))*100 
        
        if sum(self.num_patterns_each) != self.num_input_total:
            print(f'total inputs: {sum(self.num_patterns_each)}')
            self.num_input_total = sum(self.num_patterns_each)


    def randomize_inputs(self):
        """
        Randomises inputs within the sample set. Reorders sample position.
        """
        order_index = np.random.choice(self.all_samples.shape[0], 
                                    np.size(self.all_samples,0), replace=False)
        
        self.all_samples =self.all_samples[order_index,:]
        
        self.all_inputs_pos_toget = self.all_inputs_pos_toget[order_index,:]
        
        
    def gaussian_blobs(self, all_input_sheets):
        """
        Create 2D gaussian inputs for each input sheet.
        Cycles through each input sheet

        Parameters
        ----------
        all_input_sheets : class
            Input sheet data
        """
        assert hasattr(all_input_sheets, 'input_sheets'), 'no input sheets, build input sheets'
        
        self.all_inputs = [] # input responses of the afferents
        self.all_inputs_pos = [] # where stimuli were centered, coords
        self.all_inputs_sizes = [] # sizes of each stimuli
        
        # Create for each input sheet
        for i in range(len(self.num_patterns_each)):
        
            input_pat, pos, size = self.create_gauss_blob_input(self.num_patterns_each[i], 
                        all_input_sheets.sheet_sizes_y[i], all_input_sheets.sheet_sizes_x[i], 
                        all_input_sheets.input_sheets[i], all_input_sheets.sheet_densities[i])

            self.all_inputs.append(input_pat)
            self.all_inputs_pos.append(pos)
            self.all_inputs_sizes.append(size)
        
        # append all inputs from each sheet into one dataset
        self.all_samples, self.all_inputs_pos_toget = self.build_dataset()
        
        # calculate the indexes for each input sheet
        self.calc_sample_indexes(all_input_sheets)
        
        # randomise samples within the set- MIGHT NOT BE NECESSARY FOR THIS PROJECT
        self.randomize_inputs()
    

        
    def gauss_resp(self,x,y,center_x,center_y,size):
        """
        Create single 2D gaussian.

        Parameters
        ----------
        center_x : float
            x coordinate position of the stimuli
        center_y : float
            y coordinate position of the stimuli
        size : float
            size of the stimuli
        """
        return np.exp(-((x-center_x)**2 + (y-center_y)**2)/size**2)        
        
    def create_gauss_blob_input(self, num_inputs, sheet_size_y, sheet_size_x, sheet_coords, sheet_density):
        """
        Create all the samples for one input sheet.

        Parameters
        ----------
        num_inputs : int
            number of inputs for this sheet
        sheet_size_y : 
            length of sheet, y direction
        sheet_size_x : int
            length of sheet, x direction
        sheet_coords : array
            coordinates of receptors for each sheet
        sheet_density : int
            density of receptors on sheet

        Returns
        -------
        all_inputs_R : np array
            response of each afferent to a stimuli
        all_inputs_pos : np array
            coordinate location of placed stimuli on the sheet
        all_inputs_size : np array
            size of each input

        """
        R_x = sheet_coords[0]
        R_y = sheet_coords[1]
        
        num_receptors = np.size(R_x,0)*np.size(R_x,1)
        
        all_inputs_R = np.zeros((num_inputs, num_receptors)) 
        all_inputs_pos = np.zeros((num_inputs, 2)) 
        all_inputs_size = np.zeros((num_inputs)) 
        
        for i in range(num_inputs):
            # rand between min and max size
            size = random.uniform(self.input_sizes[0],self.input_sizes[1])
            
            R_center_x = random.uniform(1,sheet_size_x)
            R_center_y = random.uniform(1,sheet_size_y)
                
            R_resp = self.gauss_resp(R_x,R_y,R_center_x,R_center_y,size)
        
            all_inputs_R[i,:] = R_resp.flatten()
            all_inputs_pos[i,:] = [R_center_x, R_center_y]
            all_inputs_size[i] = size
            
        return all_inputs_R, all_inputs_pos, all_inputs_size

    
    def build_dataset(self):
        """
        Appends input samples from each sheet together.

        """
        return block_diag(*self.all_inputs), np.vstack(self.all_inputs_pos) # check for multiple regions


    def calc_sample_indexes(self, all_input_sheets):
        """
        Calculate corresponding sheet indexes for the samples.

        Parameters
        ----------
        all_input_sheets : list
            list of input sheet arrays
        """
        sample_indexes = np.zeros((np.size(self.all_samples,1)))
        
        curr_i = 0
        for i, sheets in enumerate(all_input_sheets.input_sheets):
            total_aff = np.size(sheets[0],0)*np.size(sheets[0],1)
            sample_indexes[curr_i:curr_i+total_aff] = i
            curr_i += total_aff
        
        self.sample_indexes = sample_indexes
