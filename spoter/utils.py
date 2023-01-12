
import logging
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score
from collections import Counter
import pandas as pd
import seaborn as sns
    
# import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
    
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_metrics_epoch_zero(model, train_loader,
                        val_loader,
                        eval_loader,
                        criterion,
                        device,
                        keypoints_model=""):
    model.train(False)
    pred_correct, pred_all = 0, 0
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)
        loss = criterion(outputs[0], labels[0])
        running_loss += loss    
        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    _, _, val_acc = evaluate(model, val_loader, device)    
    _, _, eval_acc = evaluate(model, eval_loader, device)
    _, _, eval_acctop5 = evaluate_top_k(model, eval_loader, device, k=5)

    metrics_log = {
        "train_loss" if keypoints_model=="" else f"train_loss-{keypoints_model}": running_loss,
        "train_acc" if keypoints_model=="" else f"train_acc-{keypoints_model}": (pred_correct / pred_all),
        "val_acc" if keypoints_model=="" else f"val_acc-{keypoints_model}": val_acc,
        "eval_acc" if keypoints_model=="" else f"eval_acc-{keypoints_model}": eval_acc,
        "eval_acctop5" if keypoints_model=="" else f"eval_acctop5-{keypoints_model}": eval_acctop5,
        "max_eval_acc" if keypoints_model=="" else f"max_eval_acc-{keypoints_model}": eval_acc,
        "max_eval_acc_top5" if keypoints_model=="" else f"max_eval_acc_top5-{keypoints_model}": eval_acctop5
    }

    return metrics_log

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)
        '''
        print('len labels',len(labels))
        print('len labels[0]',len(labels[0]))
        print('labels',labels)

        print('len outputs',len(outputs))
        print('len outputs[0]',len(outputs[0]))
        print('outputs',outputs)
        #'''

        loss = criterion(outputs[0], labels[0])
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        scheduler.step(running_loss.item() / len(dataloader))

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all)


def evaluate(model, dataloader, criterion, device, print_stats=False):

    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        loss = criterion(outputs[0], labels[0])
        running_loss += loss    

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all)

def my_evaluate(model,train_set,train_loader,eval_loader,device,experiment_name,print_stats=False):
    
    train_true_values = []
    for i, data in enumerate(train_loader):
        inputs, labels = data
        true_value = int(labels[0][0])
        true_value = train_set.inv_dict_labels_dataset[true_value]
        train_true_values.append(true_value)
        
    dataloader = eval_loader 
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(2000)}

    true_values = []
    predicted_values = []
    true_values_id = []

    predicted_values_proba = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        predicted_value = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))

        predicted_values_proba.append(list(torch.nn.functional.softmax(outputs, dim=2)[0][0].tolist()))

        true_value= int(labels[0][0])
        true_values_id.append(true_value)
        true_value2 = train_set.inv_dict_labels_dataset[true_value]
        predicted_value2 = train_set.inv_dict_labels_dataset[predicted_value]
        true_values.append(true_value2)
        predicted_values.append(predicted_value2)
        # Statistics
        if  true_value == predicted_value:
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    pred_correct, pred_all, (pred_correct / pred_all)  
    
    logging.info("pred_correct : "+str(pred_correct))
    logging.info("pred_all     : "+str(pred_all))
    logging.info("accuracy     : "+str(pred_correct / pred_all))

    acc_total = accuracy_score(true_values, predicted_values)
    acc_total
    logging.info("accuracy     : "+str(acc_total))
    topk = ''
    print('true_values_id uniques : ',len(set(true_values_id)))
    for k in range(1,6):
        value = 'k ='+str(k)+' top_k_accuracy_score = '+str(top_k_accuracy_score(true_values_id, predicted_values_proba, k=k))+'\n'
        topk += value
        logging.info(value)

    print(topk)

    val_true_values_counter = Counter(true_values)
    train_true_values_counter = Counter(train_true_values)
    distribution_val = pd.DataFrame.from_dict(dict(val_true_values_counter), orient='index',columns=['frecuencia']).reset_index()
    distribution_val = distribution_val.rename(columns={'index':'labels'})
    distribution_val['tipo']= 'validation'
    distribution_train = pd.DataFrame.from_dict(dict(train_true_values_counter), orient='index',columns=['frecuencia']).reset_index()
    distribution_train = distribution_train.rename(columns={'index':'labels'})

    distribution_train['tipo']= 'training'

    distribution_total = pd.concat([distribution_train,distribution_val])
    
    

    # set plot style: grey grid in the background:
    sns.set(style="darkgrid")

    if 'WLASL' in experiment_name:
        # set the figure size
        plt.figure(figsize=(50, 40))
    else:
        plt.figure(figsize=(15, 8))

    # top bar -> sum all values(smoker=No and smoker=Yes) to find y position of the bars
    total = distribution_total.groupby('labels')['frecuencia'].sum().reset_index()
    total = total.sort_values('labels').reset_index()

    # bottom bar ->  take only smoker=Yes values from the data
    tipo_training = distribution_total[distribution_total.tipo=='training'].reset_index()
    tipo_training = tipo_training.sort_values('labels').reset_index()

    # bar chart 1 -> top bars (group of 'smoker=No')
    bar1 = sns.barplot(x="labels",  y="frecuencia", data=total, color='darkblue')
    for index, row in total.iterrows():
        bar1.text(index, row.frecuencia+2, round(row.frecuencia, 2), color='black', ha='center')
        bar1.text(index, row.frecuencia-2, round(row.frecuencia-tipo_training['frecuencia'].iloc[index], 2), color='white', ha='center')  # type: ignore


    # bar chart 2 -> bottom bars (group of 'smoker=Yes')
    bar2 = sns.barplot(x="labels", y="frecuencia", data=tipo_training,  color='lightblue')

    for index, row in tipo_training.iterrows():
        bar2.text(index, row.frecuencia-3, round(row.frecuencia, 2), color='black', ha='center')
    plt.xticks(rotation=70)

    # add legend
    top_bar = mpatches.Patch(color='darkblue', label='tipo = Validation')
    bottom_bar = mpatches.Patch(color='lightblue', label='tipo = Training')
    plt.legend(handles=[top_bar, bottom_bar])

    create_folder('out-img')
    create_folder('out-img/'+experiment_name)
    plt.savefig('out-img/'+experiment_name+'/'+experiment_name.split('/')[-1]+'_distribution_labels.png', dpi=100)

    # show the graph
    plt.show()
    train_true_values_unique = list(set(train_true_values))

    result_classification_report = classification_report(true_values, predicted_values,zero_division=0)

    dict_result_classification_report = classification_report(true_values, predicted_values,zero_division=0, output_dict=True)
    df = pd.DataFrame(dict_result_classification_report).transpose()

    df.to_csv('out-img/'+experiment_name+'/'+experiment_name.split('/')[-1]+'_classification_report_dataframe.csv')


    with open('out-img/'+experiment_name+'/'+experiment_name.split('/')[-1]+'_classification_report.txt', 'w') as f:

        f.write('*'*20)
        f.write('\n')    
        f.write('unique values :')
        f.write('\n')    
        f.write(str(train_true_values_unique))
        f.write('\n')    
        f.write('*'*20)
        f.write('\n')
        f.write('Train dataset counter :')
        f.write('\n')
        f.write(str(train_true_values_counter))
        f.write('\n')
        f.write('*'*20)
        f.write('\n')
        f.write('Validation dataset counter :')
        f.write('\n')
        f.write(str(val_true_values_counter))
        f.write('\n')
        f.write('*'*20)
        f.write('\n')
        f.write('accuracy score total = '+str(acc_total))
        f.write('\n')
        f.write('top_k_accuracy_score')
        f.write('\n')
        f.write(topk)
        f.write('\n')    

        f.write(result_classification_report)


    data = confusion_matrix(true_values, predicted_values)

    cm_sum = np.sum(data, axis=0, keepdims=True)
    data = data / (cm_sum.astype(float)+0.0001)

    df_cm = pd.DataFrame(data, columns=np.unique(true_values), index = np.unique(true_values))


    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    

    if 'WLASL' in experiment_name:
        # set the figure size
        plt.figure(figsize=(70, 70))
    else:
        plt.figure(figsize = (25,25))

    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 12}, fmt='.0%')# font size
    plt.savefig('out-img/'+experiment_name+'/'+experiment_name.split('/')[-1]+'_confusion_matrix_norm.png', dpi=100)


    data = confusion_matrix(true_values, predicted_values)

    cm_sum = np.sum(data, axis=0, keepdims=True)
    #data = data / (cm_sum.astype(float)+0.0001)

    df_cm = pd.DataFrame(data, columns=np.unique(true_values), index = np.unique(true_values))


    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    if 'WLASL' in experiment_name:
        # set the figure size
        plt.figure(figsize=(70, 70))
    else:
        plt.figure(figsize = (25,25))

    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 12}, fmt='.2g')# font size
    plt.savefig('out-img/'+experiment_name+'/'+experiment_name.split('/')[-1]+'_confusion_matrix.png', dpi=100)

    logging.info('evaluation completed!')
    logging.info('evaluation completed!')

def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)
