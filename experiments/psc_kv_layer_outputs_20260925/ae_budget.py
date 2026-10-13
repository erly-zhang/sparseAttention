import os
"""Measure original AE on an actual calibration map, discard temporary weights.
This is a cost gate, not a trained checkpoint or group generator.
"""
import json,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
RUN=Path(os.environ.get('PSC_RUN_DIR', str(HERE/'run')))
from experiments.run_shareprefill_autoencoder_clustering import SharePrefillAutoencoder,MeanAttentionMapDataset

def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    data=MeanAttentionMapDataset(HERE/'ae/mean_attention_maps',num_layers=32,num_heads=32)
    model=SharePrefillAutoencoder().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    steps=[];valid=[]
    for i in range(8):
        torch.cuda.synchronize();start=time.perf_counter()
        x=data[i].unsqueeze(0).cuda();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): rec,_=model(x);loss=(rec.float()-x).square().mean()
        loss.backward();opt.step();torch.cuda.synchronize()
        if i>=3:steps.append(time.perf_counter()-start)
    with torch.inference_mode():
        for i in range(3):
            torch.cuda.synchronize();start=time.perf_counter();x=data[i].unsqueeze(0).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):rec,_=model(x);loss=(rec.float()-x).square().mean()
            torch.cuda.synchronize();valid.append(time.perf_counter()-start)
    train=sum(steps)/len(steps);evaluation=sum(valid)/len(valid)
    epoch=921*train+103*evaluation
    record={'measured_train_step_seconds':train,'measured_validation_step_seconds':evaluation,
        'estimated_epoch_seconds':epoch,'earliest_patience_stop_epochs':31,
        'estimated_earliest_patience_seconds':epoch*31,'estimated_max1000_seconds':epoch*1000,
        'excludes_checkpoint_io_and_epoch_shuffle_variability':True,'temporary_weights_discarded':True,
        'input_source':'calibration indices0,1,2 mean maps only','not_an_AE_checkpoint':True}
    (RUN/'ae_budget.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
