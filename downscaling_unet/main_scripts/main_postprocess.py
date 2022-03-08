import os, sys
import argparse
from timeit import default_timer as timer
import tensorflow.keras as keras
import xarray as xr
from handle_data_unet import HandleUnetData
import json
class TimeHistory(keras.callbacks.Callback):
    def on_predict_begin(self,logs={}):
        self.batch_times = []

    def on_predict_batch_begin(self,batch,logs={}):
        self.batch_time_start = timer()

    def on_predict_batch_end(self,batch,logs={}):
        self.batch_times.append(timer() - self.batch_time_start)

    def on_predict_end(self,logs={}):
        print("The time for inference per batch",self.batch_times)
        print("The average of inference time per batch is",sum(self.batch_times)/len(self.batch_times))


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
    opt_norm = args.opt_norm
    
    #load the json file for normalizing the test dataset
    with open(opt_norm) as jfl:
        opt_norm_js = json.load(jfl)
    
    #reconstruct the model
    model_recon = keras.models.load_model(model_path)

    #obtain the test dataset
    data_obj = HandleUnetData(input_dir, "test", purpose="test_aug")
    int_data, tart_data = data_obj.normalize("test_aug",  daytime=None, opt_norm=opt_norm_js)

    
    time_tracker = TimeHistory()
    callback_list = [time_tracker]
    
    #inferences
    preds = model_recon.predict(int_data.values,batch_size=batch_size,verbose=2,callbacks=callback_list)
   
    inf_total_time  = timer() - t0
    print("The total inference time is",inf_total_time)

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
    parser.add_argument("--opt_norm_path", "-opt_norm", dest="opt_norm", type=str, default=None,
                        help="opt_norm_path json file where you obtained from normalizing training dataset, will used for noramlizing test dataset")
    args = parser.parse_args()
    main(args)


