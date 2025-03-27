import itertools
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import numpy as np
from python.hpc import HybridArray
from python.array_math_utils.array_math_utils import max_column_along_rows, argmax_column_along_rows, min_column_along_rows, argmin_column_along_rows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class R_Beta_Space:
    def __init__(self,\
                 N: int,\
                 num_monte: int,\
                 r_range: np.ndarray|list,\
                 beta_range: np.ndarray|list,\
                 **kwargs) -> None:
        self.r_range = np.maximum(np.sort(r_range).reshape(-1), 0.0)
        self.beta_range = np.clip(np.sort(beta_range),0.0,1.0).reshape(-1)
        mus = np.sqrt(2*self.r_range*np.log(N))
        epsilons = np.power(N,-self.beta_range)
        n1s = np.clip((epsilons*N+0.5).astype(np.uint32),np.uint32(1),N)
        assert n1s.dtype == np.uint32
        self.mu_n1_tuples = list(itertools.product(mus,n1s))
        self.r_beta_shape = (self.r_range.size, self.beta_range.size)
        self.r_beta_size = self.r_range.size * self.beta_range.size
        self.r_beta_full_results_shape = (self.r_beta_size, N)
        self.single_rare_weak_shape = (num_monte, N)
        self.r_beta_full_results_reshaped = (self.r_range.size, self.beta_range.size, N)


    def alloc_r_beta_full_results(self, dtype: type = np.float64, use_gpu: bool|None = None) -> HybridArray:
        return HybridArray().realloc(shape=self.r_beta_full_results_shape, dtype=dtype, use_gpu=use_gpu)


    def get_full_result(self, r_beta_full_results: HybridArray) -> np.ndarray:
        r_beta_full_results.uncrop()
        assert r_beta_full_results.shape() == self.r_beta_full_results_shape
        return r_beta_full_results.numpy().reshape(self.r_beta_full_results_reshaped)


    def reshape_selected_column(self,\
                                data: np.ndarray|HybridArray,\
                                select_col: int|np.uint32|float|np.floating|None = None,\
                                is_power: bool|None=None) -> np.ndarray:
        if isinstance(data, HybridArray):
            if select_col is None:
                column = data.numpy()
            else:
                N = data.ncols()
                if isinstance(select_col, (float, np.floating)):
                    assert is_power is not None
                    if is_power:
                        select_col=np.uint32(N**select_col + 0.5)-np.uint32(1)
                    else:
                        select_col = min(np.uint32(N*max(select_col,0.0) + 0.5),np.uint32(N-1))
                elif isinstance(select_col, (int, np.integer)):
                    if select_col < 0:
                        select_col = np.uint32(select_col + N)
                column = data.select_col(select_col).numpy()
                data.uncrop()
        else:
            column = data
        return column.reshape(self.r_beta_shape).astype(np.float32)
    

    def select_by_alpha(self, r_beta_full_results: HybridArray, alpha_selection_methods: list|str|float|tuple) -> dict:
        r_beta_full_results.uncrop()
        assert r_beta_full_results.shape() == self.r_beta_full_results_shape
        if not isinstance(alpha_selection_methods,list):
            alpha_selection_methods = [alpha_selection_methods]
        selection_results = {}
        N = r_beta_full_results.ncols()
        with (HybridArray() as arg_minmax, HybridArray() as val_minmax):
            for alpha_method in alpha_selection_methods:
                if isinstance(alpha_method,tuple):
                    alpha_power = alpha_method[1]
                    selection_results[('alpha_power',alpha_power)] =\
                        self.reshape_selected_column(r_beta_full_results, select_col=alpha_power, is_power=True)
                    continue
                if isinstance(alpha_method, (float, np.floating)):
                    selection_results[('alpha',alpha_method)] =\
                        self.reshape_selected_column(r_beta_full_results, select_col=alpha_method, is_power=False)
                    continue
                if isinstance(alpha_method,str):
                    if alpha_method == 'max_metric': 
                        max_column_along_rows(r_beta_full_results, maxval=val_minmax)
                        selection_results[alpha_method] = self.reshape_selected_column(val_minmax)
                        argmax_column_along_rows(r_beta_full_results, argmax=arg_minmax)
                        argmax_numpy = (self.reshape_selected_column(arg_minmax)+1)/N
                        argmax_numpy[argmax_numpy < 1.5/N] = 0
                        selection_results['argmax_metric'] = argmax_numpy
                        continue
                    if alpha_method == 'min_metric': 
                        min_column_along_rows(r_beta_full_results, minval=val_minmax)
                        selection_results[alpha_method] = self.reshape_selected_column(val_minmax)
                        argmin_column_along_rows(r_beta_full_results, argmin=arg_minmax)
                        argmin_numpy = (self.reshape_selected_column(arg_minmax)+1)/N
                        argmin_numpy[argmin_numpy < 1.5/N] = 0
                        selection_results['argmin_metric'] = argmin_numpy
                        continue
                    if alpha_method == 'first':
                        selection_results['according to lowest p-value'] = self.reshape_selected_column(r_beta_full_results, select_col = 0)
                        continue
                    if alpha_method == 'last':
                        selection_results['according to highest p-value'] = self.reshape_selected_column(r_beta_full_results, select_col = -1)
                        continue
                assert False, f'{alpha_selection_methods=}'
        return selection_results


    def collect_values(self, select_alpha: dict) -> tuple[list[str],list[np.ndarray]]:
        params = []
        data = []
        for key, value in select_alpha.items():
            if isinstance(value, dict):
                sub_list, sub_values = self.collect_values(select_alpha=value)
                params += [f'{key}={sub}' for sub in sub_list]
                data += [sub.reshape(self.r_beta_shape) for sub in sub_values]
                continue
            assert isinstance(value, np.ndarray)
            if isinstance(key, tuple):
                str_key = f'{key[0]}'
                for a in key[1:]:
                    if isinstance(a,str):
                        str_key += ' ' + a
                    else:
                        str_key += f'={a}'
                params.append(str_key)
            else:
                params.append(f'{key}')
            data.append(value.reshape(self.r_beta_shape))
        return params, data
    

    def filterout_key_values(self, subkeys: list[str], params: list[str], collected_data: list[np.ndarray]) -> tuple[list[str],list[np.ndarray]]:
        res_params = []
        res_data = []
        for p,d in zip(params,collected_data):
            if any([k in p for k in subkeys]):
                continue
            res_params.append(p)
            res_data.append(d)
        return res_params, res_data


    def select_best_param(self, collected_data: list[np.ndarray], argmax: bool) -> tuple[np.ndarray, np.ndarray]:
        data = np.array(collected_data)
        assert data.ndim == 3
        assert data.shape[1:] == self.r_beta_shape
        if argmax:
            partitioned = np.partition(data, -2, axis=0)  # partition to get 2 largest
            max_values = partitioned[-1, :, :]  # largest values
            second_best = partitioned[-2, :, :]  # second largest values
            differences = max_values - second_best
            data = data.argmax(axis=0)
        else:
            partitioned = np.partition(data, 2, axis=0)  # partition to get 2 smallest
            min_values = partitioned[0, :, :]  # smallest values
            second_best = partitioned[1, :, :]  # second largest values
            differences = second_best - min_values
            data = data.argmin(axis=0)
        return data.astype(np.uint32), differences


    def heatmap(self,
                data: np.ndarray,\
                title: str|None = None,\
                value_name: str|None = None,\
                data_min: float|None = None,\
                data_max: float|None = None,\
                x_ticks_rotation: float|int|str|None = None,\
                data_font_size: int|float|None=None) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        img = self.__imshow(ax=ax, data=data, cmap='rainbow', data_min=data_min, data_max=data_max, title=title, x_ticks_rotation=x_ticks_rotation)
        if value_name is not None:
            fig.colorbar(img, label=value_name)
        # Add values on top of heatmap
        num_x = self.beta_range.size
        num_y = self.r_range.size
        if data_font_size is None:
            data_font_size = 8 if num_x < 15 else 7
        for i in range(num_y):
            for j in range(num_x):
                ax.text(x=self.beta_range[j], y=self.r_range[i], s=f"{data[i, j]:.2f}",\
                        color="white", ha="center", va="center",\
                            fontsize=data_font_size)
        plt.show()


    def imagemap(self,\
                 data: np.ndarray,\
                 labels: list[str],\
                 title: str|None = None,\
                 x_ticks_rotation: float|int|str|None = None,\
                 **kwargs) -> None:
        num_classes = len(labels)
        assert data.dtype == np.uint32
        assert data.min() >= 0
        assert data.max() < num_classes
        # Create two subplots:
        # Top:    for the image (large)
        # Bottom: for the legend (small)
        fig, (ax_img, ax_legend) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(6, 7),
            gridspec_kw={'height_ratios': [8, 1]}  # Make top subplot larger
        )
        # Choose a colormap with distinct colors (e.g., tab10)
        cmap = plt.get_cmap("tab10")
        # Plot the image in the top 
        img = self.__imshow(ax=ax_img, data=data, cmap=cmap, data_min=0, data_max=num_classes-1,\
                            title=title, x_ticks_rotation=x_ticks_rotation)

        # Create legend patches with consistent normalization
        if num_classes > 1:
            cmap_values = [i / (num_classes-1) for i in range(num_classes)]
        else:
            cmap_values = [0.5]
        patches = [mpatches.Patch(color=cmap(cmap_val), label=labels[i]) for i,cmap_val in enumerate(cmap_values)]
        # In the bottom subplot, just show the legend (turn off any axis lines)
        ax_legend.axis('off')
        ax_legend.legend(
            handles=patches, 
            loc='center',      # Center it horizontally
            ncol=1,
            title="Statistical Methods"
        )
        plt.tight_layout()
        plt.show()


    def __imshow(self, ax: Axes, data: np.ndarray, cmap,\
                 data_min: int|float|None = None,\
                 data_max: int|float|None = None,\
                 title: str|None = None,\
                 x_ticks_rotation: float|int|str|None = None) -> AxesImage:
        d_beta = (self.beta_range[-1] - self.beta_range[0])/(self.beta_range.size-1)
        d_r = (self.r_range[-1] - self.r_range[0])/(self.r_range.size-1)
        extent = (self.beta_range[0] - d_beta*0.5,\
                self.beta_range[-1] + d_beta*0.5,\
                self.r_range[0] - d_r*0.5,\
                self.r_range[-1] + d_r*0.5)
        if data_min is None:
            data_min = data.min()
        if data_max is None:
            data_max = data.max()
        img = ax.imshow(\
            X=data, cmap=cmap, vmin=data_min, vmax=data_max, extent=extent,\
            origin='lower',\
            aspect='auto'  # Force stretching to fill the subplot
        )
        ax.set_xlabel('Beta')
        ax.set_ylabel('r')
        xticks_kwargs = {}
        if x_ticks_rotation is not None:
            xticks_kwargs['rotation'] = x_ticks_rotation
        elif self.beta_range.size > 15:
            xticks_kwargs['rotation'] = 90
        ax.set_xticks(self.beta_range)
        ax.set_xticklabels([f'{val:.2f}' for val in self.beta_range], **xticks_kwargs)
        ax.set_yticks(self.r_range)
        ax.set_yticklabels([f'{val:.2f}' for val in self.r_range])
        # Optional figure title
        if title is not None:
            ax.set_title(title)
        return img
