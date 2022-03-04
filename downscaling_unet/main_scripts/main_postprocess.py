import os, sys
import argparse
from timeit import default_timer as timer
import tensorflow.keras as keras
import xarray as xr
from handle_data_unet import HandleUnetData

class TimeHistory(keras.callbacks.Callback):
    def on_predict_begin(self,logs={}):
        self.batch_times = []

    def on_predict_batch_begin(self,batch,logs={}):
        self.batch_time_start = timer()

    def on_predict_batch_end(self,batch,logs={}):
        self.batch_times.append(timer() - self.batch_time_start)
        print("the time for one batch",self.batch_times)

def main(parser_args):

    # start timing
    t0 = timer()

    method = main.__name__

    #parse arguments
    job_id = parser_args.job_id
    model_path = parser_args.model_path
    input_dir = parser_args.input_dir
    out_dir = parser_args.output_dir
    batch_size = args.batch_size
    hour = args.hour 
    
    #reconstruct the model
    model_recon = keras.models.load_model(model_path)

    #obtain the test dataset
    data_obj = HandleUnetData(input_dir, "test", purpose="test")
    int_data, tart_data, opt_norm = data_obj.normalize("test", daytime=hour)
    time_tracker = TimeHistory()
    callback_list = [time_tracker]
    #inferences
    preds = model_recon.predict(int_data.values,batch_size=batch_size,verbose=2,callbacks=callback_list)
    print("Time", time_tracker.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-in", dest="input_dir", type=str, required=True,
                        help="Directory where input netCDF-files are stored.")
    parser.add_argument("--model_path", "-model",dest="model_path",type=str,required=True,
                        help="The saved model path")
    parser.add_argument("--output_dir", "-out", dest="output_dir",
                        type=str, required=True, help="Output directory where the predictoins are saved.")
    parser.add_argument("--batch_size", "-bs", dest="batch_size", type=int, default=32,
                        help = "Batch size during model training.")
    parser.add_argument("--job_id", "-id", dest="job_id", type=int, help="Job-id from Slurm.")
    parser.add_argument("--hour", "-hour", dest="hour", type=int, default=12,
                        help="Daytime hour for which model will be trained.")

    args = parser.parse_args()
    main(args)


