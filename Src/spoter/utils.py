import torch.nn.utils as nn_utils
import logging
import torch
import csv
import wandb
import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device,clip_grad_max_norm=1.0,epoch=0,args=None):

    k = 5
    
    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0
    
    stats = {i: [0, 0] for i in range(302)}

    batch_size = args.batch_size
    batch_name = args.batch_name
    accumulated_loss = 0
    counter = 0

    labels_original = []
    labels_predicted = []

    optimizer.zero_grad()

    list_depth_map_original = []
    list_label_name_original = []

    with tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train Epoch {epoch + 1}:',bar_format='{desc:<18.23}{percentage:3.0f}%|{bar:20}{r_bar}') as tepoch:
        for i, data in tepoch:
            #print("data:",data)
            inputs_total, labels_total, videos_name_total = data
            if inputs_total is None:
                break
            if i < 2 and epoch==0:
                print("")
                print("max inputs:", torch.max(inputs_total[0]).item())
                print("min inputs:", torch.min(inputs_total[0]).item())
                print("std inputs:", torch.std(inputs_total[0]).item())

            if i==0:
                list_depth_map_original = inputs_total
                list_label_name_original = videos_name_total

            #for j, (inputs, labels,videos_name) in enumerate(zip(inputs_total,labels_total,videos_name_total)):
            for j in range(len(inputs_total)):
                inputs = inputs_total[j]
                labels = labels_total[j]
                videos_name = videos_name_total[j]

                labels = labels.unsqueeze(0)
                outputs = model(inputs).expand(1, -1, -1)
                loss = criterion(outputs[0], labels[0])
                label_original  = int(labels[0][0])
                if torch.isnan(loss):
                    #print(f"NaN loss detected at iteration {i+1}, {j+1}. Skipping this iteration.")
                    tepoch.set_postfix(id_aug=j+1,m_loss=None)
                    labels_predicted.append(-1)
                    labels_original.append(label_original)
                    continue  # Otra opción podría ser detener el bucle o el entrenamiento aquí

                if batch_name =='mean_2':
                    loss.backward()

                running_loss += loss.item()
                if batch_name!='':
                    # Acumular la pérdida
                    accumulated_loss += loss
                    counter += 1

                    # Realizar el paso de retropropagación y actualización de parámetros cada batch_size iteraciones
                    if counter == batch_size:
                        if batch_name =='mean_1':
                            averaged_loss = accumulated_loss / batch_size
                            averaged_loss.backward()
                        nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                        # Reiniciar contadores y acumulador de pérdida
                        accumulated_loss = 0
                        counter = 0
                else:
                    loss.backward()
                    nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
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

                #tqdm.tqdm.write(f"ID:{i+1} IDaug:{j+1} | Loss: {running_loss/pred_all} Acc: {pred_correct / pred_all}")
                tepoch.set_postfix(id_aug=j+1,m_loss=running_loss/pred_all,m_acc=pred_correct / pred_all)

    # Asegurarse de realizar el último paso de retropropagación si es necesario
    if batch_name =='mean_1':
        if counter > 0 and accumulated_loss.item()>0:
            averaged_loss = accumulated_loss / counter
            averaged_loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
            optimizer.step()
            optimizer.zero_grad()
    if batch_name =='mean_2':
        if running_loss>0:
            nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
            optimizer.step()
            optimizer.zero_grad()

    pred_all= 1 if pred_all == 0 else pred_all
    train_loss = None if running_loss == 0 else running_loss/pred_all

    # clear cache
    torch.cuda.empty_cache()
    
    return train_loss,stats,labels_original,labels_predicted,list_depth_map_original,list_label_name_original


def evaluate(model, dataloader, criterion, device,epoch=0,args=None):

    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    k = 5 # top 5 (acc)
    labels_original = []
    labels_predicted = []

    list_depth_map_original = []
    list_label_name_original = []
    
    with tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Val   Epoch {epoch + 1}:',bar_format='{desc:<18.23}{percentage:3.0f}%|{bar:20}{r_bar}') as tepoch:
        for i, data in tepoch:

            #print("data:",data)
            inputs_total, labels_total, videos_name_total = data
            if inputs_total is None:
                break
            if i < 2 and epoch==0:
                print("")
                print("max inputs:", torch.max(inputs_total[0]).item())
                print("min inputs:", torch.min(inputs_total[0]).item())
                print("std inputs:", torch.std(inputs_total[0]).item())
            if i==0:
                list_depth_map_original = inputs_total
                list_label_name_original= videos_name_total

            #for j, (inputs, labels,videos_name) in enumerate(zip(inputs_total,labels_total,videos_name_total)):
            for j in range(len(inputs_total)):
                inputs = inputs_total[j]
                labels = labels_total[j]
                videos_name = videos_name_total[j]

                labels = labels.unsqueeze(0)
                with torch.no_grad():
                    outputs = model(inputs).expand(1, -1, -1)
                loss = criterion(outputs[0], labels[0])
                label_original = int(labels[0][0])
                if torch.isnan(loss):
                    #print(f"NaN loss detected at iteration {i+1}, {j+1}. Skipping this iteration.")
                    tepoch.set_postfix(id_aug=j+1,m_loss=None)
                    labels_predicted.append(-1)
                    labels_original.append(label_original)
                    continue  # Otra opción podría ser detener el bucle o el entrenamiento aquí

                running_loss += loss.item()

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
                #tqdm.tqdm.write(f"ID:{i+1} IDaug:{j+1} | Loss: {running_loss/pred_all} Acc: {pred_correct / pred_all}")
                tepoch.set_postfix(id_aug=j+1,m_loss=running_loss/pred_all,m_acc=pred_correct / pred_all)

    pred_all= 1 if pred_all == 0 else pred_all
    val_loss = None if running_loss == 0 else running_loss/pred_all
    return val_loss, (pred_top_5 / pred_all), stats,labels_original,labels_predicted,list_depth_map_original,list_label_name_original


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