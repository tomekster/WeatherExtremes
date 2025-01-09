import itertools
import numpy as np
import matplotlib.pyplot as plt

LAT_LON = {
  'zurich': (549, 754), # (47.25, 8.5)
  'sf': (511, 230), # (37.75, -122.5)
  'cape': (224, 794), # (-34.0, 18.5)
}

def grid_index_to_lat_long(pair):
    x, y = pair
    return (x*0.25 - 90, y*0.25-180)

def lat_lon_to_grid_index(pair):
    x,y = pair
    return ((x+90) * 4, (y+180) * 4)

def load_data(vals):
    file_path = f"/weather/WeatherExtremes/daily_mean_2m_temperature_1960_1989_AGG.MEAN_aggrwindow_{vals['agg_window']}_percboost_{vals['perc_boost']}/percentiles_0_{vals['perc']}.npy"
    # data = np.load(file_path)
    lat_idx, lon_idx = vals['lat_lon']
    # Open the file using open_memmap
    data = np.lib.format.open_memmap(file_path, mode='r')
    # Only load the required data slice
    result = data[:, lat_idx, lon_idx]
    return result

def generic_plot(subplot_args: dict[str, list],
                 graph_args: dict[str, list],
                 out_fname):
    """
    Generate a figure for every combination of the argument-values in `figure_args`.
    Within each figure, generate a subplot for every combination of the remaining arguments.
    
    Parameters
    ----------
    args_dict : dict[str, list]
        Dictionary of argument_name -> list_of_possible_values. {agg_window, perc_boost, lat_long_pairs}
    figure_args : list[str]
        Which arguments define the figure-level combinations (Cartesian product).
        All other arguments define subplot-level combinations.
    """

    # Precompute the Cartesian products for figure-level arguments and subplot-level arguments
    subplot_combos = list(
        itertools.product(*(subplot_args[arg] for arg in subplot_args))
    )
    graph_combos = list(
        itertools.product(*(graph_args[arg] for arg in graph_args))
    )
    
    # Create a new figure; optional: size, layout, etc.
    # Let's assume one subplot per combination in subplot_combos
    n_subplots = len(subplot_combos)
    fig, axes = plt.subplots(
        n_subplots, 1, 
        figsize=(8, 4 * n_subplots),
        sharex=True
    )
    
    # # Create a dict that tells us the selected argument value
    # # for each subplot-level argument in this combination
    # fig_arg_val_dict = dict(zip(subplot_args, sub_vals))
        
    # # Give the figure a descriptive title
    # fig_title_items = [f"{k}={v}" for k, v in fig_arg_val_dict.items()]
    # fig.suptitle("Params: " + ", ".join(fig_title_items))
    
    # ---------------------------------------------------------------------
    # 3. Iterate over each combination of the figure-level arguments
    #    -> each combination => one Figure
    # ---------------------------------------------------------------------
    for ax_idx, sub_vals in enumerate(subplot_combos):        
        # If there's only one subplot, wrap it in a list for uniform handling
        if n_subplots == 1:
            axes = [axes]

        ax = axes[ax_idx]
        
        # -----------------------------------------------------------------
        # 4. For each combination of the subplot-level arguments
        #    -> each combination => one Subplot
        # -----------------------------------------------------------------
        for graph_idx, graph_vals in enumerate(graph_combos):
            vals_plot = {argname: val for argname, val in zip(subplot_args.keys(), sub_vals)}
            vals_graph = {argname: val for argname, val in zip(graph_args.keys(), graph_vals)}
            vals = vals_plot | vals_graph
            data = load_data(vals)

            # -------------------------------------------------------------
            # 6. Make the actual plot
            # -------------------------------------------------------------
            
            if 'lat_lon' in vals_graph:
                vals_graph['lat_lon'] = grid_index_to_lat_long(vals_graph['lat_lon'])
            
            ax.plot(data, label=", ".join(f"{k}={v}" for k, v in vals_graph.items()), lw=1)
            ax.set_ylabel("Kelvin degrees")
            
            # Add a secondary Y-axis for Celsius
            def kelvin_to_celsius(k):
                return k-273.15 

            def celsius_to_kelvin(c):
                return c+273.15 

            secax = ax.secondary_yaxis('right', functions=(kelvin_to_celsius, celsius_to_kelvin))
            secax.set_ylabel("Celsius degrees")
            
            # Let’s put a legend on each subplot for clarity
            ax.legend()
            
            if 'lat_lon' in vals_plot:
                vals_plot['lat_lon'] = grid_index_to_lat_long(vals_plot['lat_lon'])
            subplot_title_items = [f"{k}={v}" for k, v in vals_plot.items()]
            ax.set_title("Plot params: " + ", ".join(subplot_title_items))
        
        # -----------------------------------------------------------------
        # 7. Done populating subplots. Adjust layout and show (or save).
        # -----------------------------------------------------------------
        plt.tight_layout()
        plt.savefig(out_fname)

if __name__ == '__main__':    
    generic_plot(
        {
            'agg_window': [1],
            'perc_boost': [1],
            'lat_lon': [LAT_LON['zurich'], LAT_LON['sf'], LAT_LON['cape']],
         },
        {
            'perc': [9, 95, 97, 99],
        },
        'vary_percentile.png'
        )
    
    generic_plot(
        {
            'perc_boost': [1],
            'lat_lon': [LAT_LON['zurich'], LAT_LON['sf'], LAT_LON['cape']],
            'perc': [9],
         },
        {
            'agg_window': [1,3,5],            
        },
        'vary_agg_wind.png'
        )
    
    generic_plot(
        {
            'agg_window': [1,3,5],   
            'lat_lon': [LAT_LON['zurich']],
            'perc': [9],
         },
        {
            'perc_boost': [1,3,5],
        },
        'vary_perc_boost.png'
        )