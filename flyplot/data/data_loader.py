import os
import glob
def data_loader():
    data_path = os.path.join(os.path.dirname(__file__),'datasets')
    fig_paths =os.path.join(data_path,'figs/*')
    img_paths = os.path.join(data_path,'imgs/*.png')
    raw_data_paths = os.path.join(data_path,'raw_data/*.txt')
    # dictionary
    sample_data = {"figs":glob.glob(fig_paths),"imgs":glob.glob(img_paths),"raw_data":glob.glob(raw_data_paths)}
    return sample_data
if __name__ == "__main__":
    print(data_loader())