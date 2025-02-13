import numpy as np
import matplotlib.pyplot as plt

def heatmap_r_beta_range(r_range: list|np.ndarray,\
                         beta_range: list|np.ndarray,\
                         data: np.ndarray,\
                         title: str|None = None,\
                         value_name: str|None = None,\
                         data_min: float|None = None,\
                         data_max: float|None = None,\
                         x_ticks_rotation: float|int|str|None = None,\
                         data_font_size: int|float|None=None,\
                         **kwargs) -> None:
    beta_range = np.sort(np.asarray(beta_range).reshape(-1))
    r_range = np.sort(np.asarray(r_range).reshape(-1))
    num_x = beta_range.size
    num_y = r_range.size
    plt.figure(figsize=(8, 6))
    if data_min is None:
        data_min = data.min()
    if data_max is None:
        data_max = data.max()
    extent = (beta_range[0] - (beta_range[1]-beta_range[0])*0.5,\
              beta_range[-1] + (beta_range[-1]-beta_range[-2])*0.5,\
              r_range[0] - (r_range[1]-r_range[0])*0.5,\
              r_range[-1] + (r_range[-1]-r_range[-2])*0.5)
    plt.imshow(data, aspect='auto', extent=extent, origin='lower',\
               cmap='rainbow', vmin=data_min, vmax=data_max)
    if value_name is not None:
        plt.colorbar(label=value_name)
    plt.xlabel('Beta')
    plt.ylabel('r')
    xticks_kwargs = {}
    if x_ticks_rotation is None:
        if num_x > 15:
            xticks_kwargs['rotation'] = 90
    else:
        xticks_kwargs['rotation'] = x_ticks_rotation
    plt.xticks(ticks=beta_range, labels=[f'{val:.2f}' for val in beta_range], **xticks_kwargs)
    plt.yticks(r_range, [f'{val:.2f}' for val in r_range])
    # Add values on top of heatmap
    if data_font_size is None:
        data_font_size = 8 if num_x < 15 else 7
    for i in range(num_y):
        for j in range(num_x):
            plt.text(x=beta_range[j], y=r_range[i], s=f"{data[i, j]:.2f}",\
                     color="white", ha="center", va="center",\
                        fontsize=data_font_size)

    if title is not None:
        plt.title(title)
    plt.show()

