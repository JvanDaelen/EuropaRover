import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Select the desired configurations here. (between 91 and 108)
ids = [92, 93, 108]

if len(ids) == 1:
    save_name = f'results/csvplots/id-{ids[0]}.png'
else:
    id_string = ''
    for id in ids:
        id_string = id_string + f"-{id}"
    save_name = f'results/csvplots/id{id_string}.png'



datas = []
nr_runs = 5
if len(ids) == 1:
    for id in ids:
        for run_nr in range(nr_runs):
            file_path = f'results/csvs/id{id}-training_run{run_nr+1}.csv'
            with open(file_path, 'r', newline='') as csvfile:
                data = {}
                data['name'] = f"{id}-{run_nr+1}"
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'colors':
                        data[row[0]] = row[1:]
                        continue
                    data[row[0]] = np.array(row[1:], dtype=float)
            datas.append(data)
                
    ### Generate the reward, loss, action and accuracy plots
    plt.figure(figsize=(6,8), layout="constrained")
    ### Plot the loss
    plt.subplot(411)
    plt.ylabel(r'Mean Losses $(r_t - q_t)^2$')
    for data in datas:
        # plt.scatter(data["ep_nrs"], data["losses"], 2, alpha=0.5, c=data["colors"])
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(data["losses"]) - window + 1):
            average_x.append(np.mean(data["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(data["losses"][ind:ind+window]))
        plt.plot(average_data, label = data['name'])
    plt.legend()
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)

    ### Plot the rewards (mean only)
    plt.subplot(412)
    plt.ylabel('Mean Rewards')
    # plt.scatter(data["ep_nrs"], data["rewards"], 2, alpha=0.5, c=data["colors"])
    for data in datas:
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(data["rewards"]) - window + 1):
            average_x.append(np.mean(data["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(data["rewards"][ind:ind+window]))
        plt.plot(average_data)
    plt.xlim(0, 20e+3)
    plt.grid(visible=True)

    ### Plot the mean global score
    plt.subplot(413)
    plt.ylabel('Mean Global Score')
    for data in datas:
        plt.plot(data["test_mean_eps"], data["test_mean_rewards"])
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)

    ### Plot the accuracy
    plt.subplot(414)
    plt.ylabel('Global Accuracy [%]')
    for data in datas:
        accuracies = []
        for eps in data["test_mean_eps"]:
            eps_scores = data["test_rewards"][data["test_eps_number"] == eps]
            accuracies.append(len(eps_scores[eps_scores >= 100]) / len(eps_scores) * 100)
        plt.scatter(data["test_mean_eps"], accuracies)
    plt.xlabel('Training Episode')
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)
    plt.ylim(0, 100)
    # plt.show()
    plt.savefig(save_name)

else:
    for id in ids:
        mean_data = {'name': f'{id}-mean'}
        max_data = {'name': f'{id}-max'}
        min_data = {'name': f'{id}-min'}
        temp_data = []
        for run_nr in range(nr_runs):
            file_path = f'results/csvs/id{id}-training_run{run_nr+1}.csv'
            with open(file_path, 'r', newline='') as csvfile:
                data = {}
                data['name'] = f"{id}-{run_nr+1}"
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'colors':
                        data[row[0]] = row[1:]
                        continue
                    data[row[0]] = np.array(row[1:], dtype=float)
            temp_data.append(data)
        
        accuracies = []
        mean_data['accuracy'] = []
        min_data['accuracy'] = []
        max_data['accuracy'] = []
        for a in range(len(temp_data)):
            accuracies_a = []
            for eps in  temp_data[a]["test_mean_eps"]:
                eps_scores = np.array(temp_data[a]["test_rewards"])[ temp_data[a]["test_eps_number"].astype(int) == eps]
                accuracies_a.append(len(eps_scores[eps_scores >= 100]) / len(eps_scores) * 100)
            accuracies.append(accuracies_a)
        
        accuracies_processed = []
        for i in range(len(accuracies[0])):
            accuracies_t = [accuracies[a][i] for a in range(len(accuracies))]
            mean_data['accuracy'].append(np.mean(accuracies_t))
            min_data['accuracy'].append(min(accuracies_t))
            max_data['accuracy'].append(max(accuracies_t))
        # raise ValueError(accuracies_processed)

        for key in temp_data[0].keys():
            if key not in ['rewards', 'losses', 'test_mean_rewards', 'test_rewards']:
                continue
            mean_data[key] = []
            min_data[key] = []
            max_data[key] = []
            
            for i in range(len(temp_data[0][key])):
                temp_vals = [temp_data[a][key][i] for a in range(len(temp_data))]
                mean_data[key].append(np.mean(temp_vals))
                min_data[key].append(min(temp_vals))
                max_data[key].append(max(temp_vals))
        
        mean_data['ep_nrs'] = temp_data[0]['ep_nrs']
        min_data['ep_nrs'] = temp_data[0]['ep_nrs']
        max_data['ep_nrs'] = temp_data[0]['ep_nrs']

        mean_data['test_mean_eps'] = temp_data[0]['test_mean_eps']
        min_data['test_mean_eps'] = temp_data[0]['test_mean_eps']
        max_data['test_mean_eps'] = temp_data[0]['test_mean_eps']

        mean_data['test_eps_number'] = temp_data[0]['test_eps_number']
        min_data['test_eps_number'] = temp_data[0]['test_eps_number']
        max_data['test_eps_number'] = temp_data[0]['test_eps_number']

        datas.append(mean_data)
        datas.append(max_data)
        datas.append(min_data)
    

    lines = {'max': '--', 'mean':  '-', 'min':  '--'}
    ### Generate the reward, loss, action and accuracy plots
    plt.figure(figsize=(6,8), layout="constrained")
    ### Plot the loss
    plt.subplot(411)
    plt.ylabel(r'Mean Losses $(r_t - q_t)^2$')
    for data in datas:
        # plt.scatter(data["ep_nrs"], data["losses"], 2, alpha=0.5, c=data["colors"])
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(data["losses"]) - window + 1):
            average_x.append(np.mean(data["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(data["losses"][ind:ind+window]))
        plt.plot(average_data, linestyle=lines[data['name'].split('-')[1]], label = data['name'], c=list(mcolors.BASE_COLORS.items())[int(data['name'].split('-')[0])%7][1])
    plt.legend(loc='upper right', ncols=len(datas)/3)
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)

    ### Plot the rewards (mean only)
    plt.subplot(412)
    plt.ylabel('Mean Rewards')
    # plt.scatter(data["ep_nrs"], data["rewards"], 2, alpha=0.5, c=data["colors"])
    for data in datas:
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(data["rewards"]) - window + 1):
            average_x.append(np.mean(data["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(data["rewards"][ind:ind+window]))
        plt.plot(average_data, linestyle=lines[data['name'].split('-')[1]], label = data['name'], c=list(mcolors.BASE_COLORS.items())[int(data['name'].split('-')[0])%7][1])
    plt.xlim(0, 20e+3)
    plt.grid(visible=True)

    ### Plot the mean global score
    plt.subplot(413)
    plt.ylabel('Mean Global Score')
    for data in datas:
        plt.plot(data["test_mean_eps"], data["test_mean_rewards"], linestyle=lines[data['name'].split('-')[1]], label = data['name'], c=list(mcolors.BASE_COLORS.items())[int(data['name'].split('-')[0])%7][1])
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)

    ### Plot the accuracy
    plt.subplot(414)
    plt.ylabel('Global Accuracy [%]')
    for data in datas:
        plt.plot(data["test_mean_eps"], data["accuracy"], linestyle=lines[data['name'].split('-')[1]], c=list(mcolors.BASE_COLORS.items())[int(data['name'].split('-')[0])%7][1], label = data['name'])
    plt.xlabel('Training Episode')
    plt.grid(visible=True)
    plt.xlim(0, 20e+3)
    plt.ylim(0, 100)
    # plt.show()
    plt.savefig(save_name)
