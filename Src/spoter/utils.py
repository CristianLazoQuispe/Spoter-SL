
import torch.nn.utils as nn_utils
import logging
import torch
import csv
import wandb
import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device,clip_grad_max_norm=1.0,epoch=0,args=None):

    k = 5
    
    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    batch_size = args.batch_size
    flag_mean = args.batch_mean
    accumulated_loss = 0
    counter = 0

    labels_original = []
    labels_predicted = []

    optimizer.zero_grad()

    for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train Epoch {epoch + 1}',bar_format='{desc:<25.25}{percentage:3.0f}%|{bar:20}{r_bar}'):
        #print("data:",data)
        inputs_total, labels_total, _ = data
        if inputs_total is None:
            break
        if i < 2 and epoch==0:
            print("")
            print("max inputs:", torch.max(inputs_total).item())
            print("min inputs:", torch.min(inputs_total).item())
            print("std inputs:", torch.std(inputs_total).item())


        for j, (inputs, labels) in enumerate(zip(inputs_total,labels_total)):
            labels = labels.unsqueeze(0)
            outputs = model(inputs).expand(1, -1, -1)
            loss = criterion(outputs[0], labels[0])
            running_loss += loss

            if flag_mean:
                # Acumular la pérdida
                accumulated_loss += loss
                counter += 1

                # Realizar el paso de retropropagación y actualización de parámetros cada batch_size iteraciones
                if counter == batch_size:
                    averaged_loss = accumulated_loss / batch_size
                    averaged_loss.backward()
                    nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                    optimizer.step()

                    # Reiniciar contadores y acumulador de pérdida
                    accumulated_loss = 0
                    counter = 0
                    optimizer.zero_grad()
            else:
                loss.backward()
                nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            label_original = int(labels[0][0])
            label_predicted = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))

            labels_predicted.append(label_predicted)
            labels_original.append(label_original)
            # Statistics
            if label_predicted == label_original:
                stats[label_original][0] += 1
                pred_correct += 1
            
            if label_original in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
                pred_top_5 += 1

            stats[label_original][1] += 1
            pred_all += 1
    # Asegurarse de realizar el último paso de retropropagación si es necesario
    if counter > 0:
        averaged_loss = accumulated_loss / counter
        averaged_loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
        optimizer.step()

    return running_loss/pred_all, pred_correct, pred_all, (pred_correct / pred_all),stats,labels_original,labels_predicted


def evaluate(model, dataloader, criterion, device,epoch=0,args=None):

    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    k = 5 # top 5 (acc)
    labels_original = []
    labels_predicted = []

    for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Evaludate Epoch {epoch + 1}:',bar_format='{desc:<25.20}{percentage:3.0f}%|{bar:20}{r_bar}'):

        #print("data:",data)
        inputs_total, labels_total, _ = data
        if inputs_total is None:
            break
        if i < 2 and epoch==0:
            print("")
            print("max inputs:", torch.max(inputs_total).item())
            print("min inputs:", torch.min(inputs_total).item())
            print("std inputs:", torch.std(inputs_total).item())
        
        for j, (inputs, labels) in enumerate(zip(inputs_total,labels_total)):
            labels = labels.unsqueeze(0)
            outputs = model(inputs).expand(1, -1, -1)
            loss = criterion(outputs[0], labels[0])
            running_loss += loss

            label_original = int(labels[0][0])
            label_predicted = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))

            labels_predicted.append(label_predicted)
            labels_original.append(label_original)
            
            # Statistics
            if label_predicted == label_original:
                stats[label_original][0] += 1
                pred_correct += 1
            
            if label_original in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
                pred_top_5 += 1

            stats[label_original][1] += 1
            pred_all += 1

    return running_loss/pred_all, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), stats,labels_original,labels_predicted


def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)

import pandas as pd

def evaluate_with_features(model, dataloader, cel_criterion, device, print_stats=False, save_results=False):

    pred_correct, pred_top_5, pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}
    k = 5 # top 5 (acc)

    # create a list to store the results
    results = []

    for i, data in enumerate(dataloader):
        inputs, labels, video_name, false_seq,percentage_group,max_consec = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate, video name {video_name}, max_consec {max_consec[i]}")
        outputs = model(inputs).expand(1, -1, -1)

        loss = cel_criterion(outputs[0], labels[0])
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

        # calculate the accuracy per instance
        acc = 1 if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]) else 0

        # append the results to the list
        results.append({
            'video_name': video_name,
            'in_range_sequences': false_seq[i].numpy()[0],
            'percentage_group': percentage_group[i].numpy()[0],
            'max_percentage': max_consec[i].numpy()[0],
            'accuracy': acc
        })

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    # convert the list to a DataFrame
    df_results = pd.DataFrame(results)

    # # save the DataFrame to a CSV file if save_results is True
    # if save_results:
    #     save_path = 'results.csv'
    #     df_results.to_csv(save_path, index=False)

    return running_loss/pred_all, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), df_results


def generate_csv_result(run, model, dataloader, folder_path, meaning, device):

    model.train(False)
    
    submission = dict()
    trueLabels = dict()
    meaningLabels = dict()

    for i, data in enumerate(dataloader):
        inputs, labels, video_name = data

        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)

        pred = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
        trueLab = int(labels[0][0])

        submission[video_name] = pred
        trueLabels[video_name] = trueLab
        meaningLabels[video_name] = meaning[trueLab]

    diccionarios = [submission, trueLabels, meaningLabels]

    # Define the row names
    headers = ['videoName', 'prediction', 'trueLabel', 'class']

    full_path = folder_path+'/submission.csv'

    # create the csv and define the headers
    with open(full_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # write the acummulated data
        for key in diccionarios[0].keys():
            row = [key[0]]
            for d in diccionarios:
                row.append(d[key])
            writer.writerow(row)
    
    #artifact = wandb.Artifact(f'predicciones_{run.id}.csv', type='dataset')
    #artifact.add_file(full_path)
    #run.log_artifact(artifact)
    wandb.save(full_path)


def generate_csv_accuracy(df_stats, folder_path,name='/accuracy.csv'):

    full_path = folder_path+name
    df_stats.to_csv(full_path, index=False, encoding='utf-8')
    wandb.save(full_path)