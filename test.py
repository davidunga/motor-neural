from impl.hatsopoulos import HatsoData
from visualize import plot_event_triggered_response, plot_center_out_trajectories
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_dir = "~/data/hatsopoulos"
    data = HatsoData.Make(data_dir, "CO_01", lag=.1, bin_sz=.01)
    print("Loaded " + str(data))
    plot_event_triggered_response(data, event="max_spd", tradius=[.4, .4], per_ang=False, tlims_mode="align")
    if data.task == "CO":
        plot_center_out_trajectories(data)
    plt.show()
