'''
This code evaluates the validation and test performance in an epoch of the model trained in jodie.py.
The task is: interaction prediction, i.e., predicting which item will a user interact with? 

To calculate the performance for one epoch:
$ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49

To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from library_data import *
from library_models import *
import library_models as lib
import time


import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule
import torch.autograd.profiler as profiler
import torch


# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', default='reddit', help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=0, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 

args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print("No interaction prediction for %s" % args.network)
    sys.exit(0)
    
# SET GPU
if args.gpu == -1:
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    device = torch.device("cuda:%d" % args.gpu)
    torch.cuda.set_device(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device_str = 'cuda:' + str(args.gpu)

# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
output_fname = "results/interaction_prediction_%s.txt" % args.network
if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()

# LOAD NETWORK
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
 item2id, item_sequence_id, item_timediffs_sequence, \
 timestamp_sequence, \
 feature_sequence, \
 y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1
true_labels_ratio = len(y_true)/(sum(y_true)+1)
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL PARAMETERS
model = JODIE(args, num_features, num_users, num_items).to(device)
weight = torch.Tensor([1,true_labels_ratio]).to(device)
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch, device)
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()

# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []

''' 
Here we use the trained model to make predictions for the validation and testing interactions.
The model does a forward pass from the start of validation till the end of testing.
For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
This is used to calculate the rank of the true item the user actually interacts with.

After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
'''
tbatch_start_time = None
loss = 0



is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}

reinitialize_tbatches()
total_loss, loss, total_interaction_count = 0, 0, 0

item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

TIME = [0] * 8



# count = 0

# FORWARD PASS
print("*** Making interaction predictions by forward pass (no t-batching) ***")
with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in progress_bar:
        progress_bar.set_description('%dth interaction for validation and testing' % j)

        # if count == 1: torch.cuda.cudart().cudaProfilerStart() ## ---- PROFILE START ---- ##
        torch.cuda.cudart().cudaProfilerStart() ## ---- PROFILE START ---- ##
        # count += 1
        with profiler.record_function("iteration{}".format(j)):

            if is_first_epoch:
                with profiler.record_function("LOAD INTERACTION J"):
                    start = time.time()
                    # READ INTERACTION J
                    userid = user_sequence_id[j]
                    itemid = item_sequence_id[j]
                    feature = feature_sequence[j]
                    user_timediff = user_timediffs_sequence[j]
                    item_timediff = item_timediffs_sequence[j]
                    TIME[0] += time.time() - start
                
                with profiler.record_function("CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH"):
                    start = time.time()
                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                    lib.tbatchid_user[userid] = tbatch_to_insert
                    lib.tbatchid_item[itemid] = tbatch_to_insert

                    lib.current_tbatches_user[tbatch_to_insert].append(userid)
                    lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                    lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
                    TIME[1] += time.time() - start
                    
            

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
            if timestamp - tbatch_start_time > tbatch_timespan:
                tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                # ITERATE OVER ALL T-BATCHES
                if not is_first_epoch:
                    lib.current_tbatches_user = cached_tbatches_user[timestamp]
                    lib.current_tbatches_item = cached_tbatches_item[timestamp]
                    lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                    lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]
                
                with trange(len(lib.current_tbatches_user)) as progress_bar3:

                    with profiler.record_function('Processed %d T-batches ' % (len(lib.current_tbatches_user))):
                    
                        for i in progress_bar3:

                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                            
                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            if is_first_epoch:
                                start = time.time()
                                lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                                lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                                lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                                lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                                lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                                lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                                lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()

                            tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_interactionids = lib.current_tbatches_interactionids[i]
                            feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                            item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                            tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                            item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
                            TIME[2] += time.time() - start
                            

                            with profiler.record_function('PROJECT USER EMBEDDING TO CURRENT TIME'):
                                start = time.time()
                                # PROJECT USER EMBEDDING TO CURRENT TIME
                                user_embedding_input = user_embeddings[tbatch_userids,:]
                                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)
                                TIME[3] += time.time() - start 

                            with profiler.record_function('PREDICT NEXT ITEM EMBEDDING'):
                                start = time.time()
                                # PREDICT NEXT ITEM EMBEDDING                            
                                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)
                                TIME[4] += time.time() - start


                            with profiler.record_function('PREDICT NEXT ITEM EMBEDDING'):
                                start = time.time()
                                # CALCULATE PREDICTION LOSS
                                item_embedding_input = item_embeddings[tbatch_itemids,:]
                                # loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())
                                TIME[5] += time.time() - start

                            with profiler.record_function('UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION'):
                                start = time.time()
                                
                                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                                user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')

                                item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')
                            

                                item_embeddings[tbatch_itemids,:] = item_embedding_output
                                user_embeddings[tbatch_userids,:] = user_embedding_output  

                                user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                                item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output
                                TIME[6] += time.time() - start

                                # torch.cuda.nvtx.range_push("CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS")
                                # # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                                # loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                                # loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                                
                                # # CALCULATE STATE CHANGE LOSS
                                # if args.state_change:
                                #     loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss, device) 
                                
                                # torch.cuda.nvtx.range_pop()
                                        

                # torch.cuda.nvtx.range_push("BACKPROPAGATE ERROR AFTER END OF T-BATCH")
                # # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                # total_loss += loss.item()
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # torch.cuda.nvtx.range_pop()


                with profiler.record_function('RESET LOSS FOR NEXT T-BATCH'):
                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    item_embeddings_timeseries.detach_() 
                    user_embeddings_timeseries.detach_()
                

                # REINITIALIZE
                if is_first_epoch:
                    with profiler.record_function('REINITIALIZE'):
                        start = time.time()

                        cached_tbatches_user[timestamp] = lib.current_tbatches_user
                        cached_tbatches_item[timestamp] = lib.current_tbatches_item
                        cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                        cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                        cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                        cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                        cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                        
                        reinitialize_tbatches()
                        tbatch_to_insert = -1
                        TIME[7] += time.time() - start
                    

            # if j >= train_end_idx+1000:
            #     torch.cuda.cudart().cudaProfilerStop()
            #     exit(0)+
            
torch.cuda.cudart().cudaProfilerStop()
            

print("TIME:", [round(TIME[i]/np.sum(TIME)*100, 2) for i in range(len(TIME))])

