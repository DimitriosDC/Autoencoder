import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstructed_input(model, data_samples, num, receptors_each_region, sheet_size):
    #encoded = []
    reconstructed = []

    data1 = data_samples[num] #.detach().numpy()
    data1 = torch.from_numpy(data1)


    #enc = model.encoder(data1)
    recon = model(data1.float())
    #encoded.append(enc.detach().numpy())
    reconstructed.append(recon.detach().numpy())

    #encoded = np.concatenate(encoded)
    reconstructed = np.concatenate(reconstructed)

    input_sample = data_samples[num] #.numpy()

    r1_receptors = receptors_each_region[0]

    input_sample_r1 = input_sample[:r1_receptors].reshape(sheet_size,sheet_size)
    input_sample_r2 = input_sample[r1_receptors:].reshape(sheet_size,sheet_size)

    max_in = np.max(input_sample)
    min_in = np.min(input_sample)

    re_input_sample_r1 = reconstructed[:r1_receptors].reshape(sheet_size,sheet_size)
    re_input_sample_r2 = reconstructed[r1_receptors:].reshape(sheet_size,sheet_size)

    max_re = np.max(reconstructed)
    min_re = np.min(reconstructed)

    # Create a new figure with a 2x2 grid of subplots
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    # Plot the input r1 array
    ax[0, 0].imshow(input_sample_r1, cmap='Blues_r', vmin=min_in, vmax=max_in)
    ax[0, 0].set_title('Input r1')

    # Plot the input r2 array
    ax[0, 1].imshow(input_sample_r2, cmap='Blues_r', vmin=min_in, vmax=max_in)
    ax[0, 1].set_title('Input r2')

    # Plot the reconstructed r1 array
    ax[1, 0].imshow(re_input_sample_r1, cmap='Blues_r', vmin=min_re, vmax=max_re)
    ax[1, 0].set_title('Reconstructed r1')

    # Plot the reconstructed r2 array
    ax[1, 1].imshow(re_input_sample_r2, cmap='Blues_r', vmin=min_re, vmax=max_re)
    ax[1, 1].set_title('Reconstructed r2')

    # Add a title for the entire plot
    fig.suptitle('Model reconstruction')

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
    print(input_sample_r2.shape)
    print(reconstructed.shape)


def plot_grids(data, x_figures=5, y_figures=10, receptor_split=900, region_shape=30, title=''):

    # flip data if short axis not first
    #if np.size(data,0) > np.size(data,1):
    #    data = data.T

    fig = plt.figure(figsize=(y_figures,x_figures)) # create outer figure
    subfigs2 = fig.subfigures(2,1, height_ratios=[1,13])
    subfigs = subfigs2[1].subfigures(x_figures, y_figures) # create figure subpanels

    subfigs2[0].suptitle(title, fontsize=15)

    for outer_idx, subfig in enumerate(subfigs.flat): # for each suplot create inner figures

        if outer_idx < np.size(data,0): # check that there is enough data for the plot
            subfig.suptitle(f'{outer_idx+1}') # add index of plot

            axs = subfig.subplots(1, 2) # create inner subplot with two panels

            ax_data = data[outer_idx,:] # get data
            ax_data_r1 = ax_data[:receptor_split].reshape(region_shape,region_shape) # reshape region 1 data
            ax_data_r2 = ax_data[receptor_split:].reshape(region_shape,region_shape) # reshape region 2 data

            both_reshape = [ax_data_r1,ax_data_r2] # create list of data

            min_data = np.min(ax_data) # calculate min of data
            max_data = np.max(ax_data) # calculate max of data

            for inner_idx, ax in enumerate(axs.flat): # plot for each region
                ax.imshow(both_reshape[inner_idx], vmin=min_data, vmax=max_data) # create plot
                ax.set_xticks([]) # remove ticks
                ax.set_yticks([]) # remove ticks
