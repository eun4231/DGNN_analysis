#includes predict_and_save_results function from model/ASTGNN.py with additional codes for profiling
import os
#import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
from lib.utils import re_max_min_normalization
#from time import time
from scipy.sparse.linalg import eigs
def predict_and_save_results2(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
    import torch.profiler
    import torch.utils.data
    from time import time
    import numpy as np
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''

    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # 存储所有batch的input

        start_time = time()
	
	    # nsys profiler
        #torch.cuda.cudart().cudaProfilerStart()

        # pytorch profiler
        with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=4, repeat=1), on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ASTGNN'), with_stack=True, profile_memory=False) as profiler:
        
        # for nsys
        #for i in range(1):
            #count = 0
            
            for batch_index, batch_data in enumerate(data_loader):
                
                # for nsys
                #if args.gpu != -1 and 
                #if count==4: torch.cuda.cudart().cudaProfilerStart()
                #if args.gpu != -1: 
                #torch.cuda.nvtx.range_push("preprocess input - " + str(batch_index))

                encoder_inputs, decoder_inputs, labels = batch_data

                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

                decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

                labels = labels.unsqueeze(-1)  # (B, N, T, 1)

                predict_length = labels.shape[2]  # T

                #torch.cuda.nvtx.range_pop()

                # encode
                #if args.gpu != -1: 
                #torch.cuda.nvtx.range_push("encode")
                encoder_output = net.encode(encoder_inputs)
                #torch.cuda.nvtx.range_pop()
                input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

                # decode
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                decoder_input_list = [decoder_start_inputs]

                # 按着时间步进行预测
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    #if args.gpu != -1: 
                    #torch.cuda.nvtx.range_push("decode")
                    predict_output = net.decode(decoder_inputs, encoder_output)
                    #if args.gpu != -1: 
                    #torch.cuda.nvtx.range_pop()
                    decoder_input_list = [decoder_start_inputs, predict_output]

                prediction.append(predict_output.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))
                
                #profiler
                
                #pytorch profiler
                profiler.step() 

                #nsys
                #if count == 7: torch.cuda.cudart().cudaProfilerStop()
                #count = count + 1
            #torch.cuda.cudart().cudaProfilerStop()

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
