# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv('llama_mcmc.csv')

print(df.shape)

# Only use seeds 0 and 1
# df = df[df['seed'].isin([0, 1])]

# Group by (dataset_a, dataset_b, ablated_dataset), average relevant columns
grouped = df.groupby(['dataset_a', 'dataset_b', 'ablated_dataset']).agg({
    'test_accuracy': 'mean',
    'test_accuracy_flipped': 'mean',
}).reset_index()

# Prepare x labels and y values according to intended dataset
x_labels = []
y_values = []

tuples = []
for _, row in grouped.iterrows():
    a, b, _ = row['dataset_a'], row['dataset_b'], row['ablated_dataset']

    # if ablated == a:
    #     # Intended is b, so use test_accuracy_flipped
    #     label = f'{a} vs $\\mathbf{{{b}}}$'
    #     y = row['test_accuracy_flipped']
    # else:
    #     # Intended is a, so use test_accuracy
    #     label = f'$\\mathbf{{{a}}}$ vs {b}'
    #     y = row['test_accuracy']


    label = f'{a} vs $\\mathbf{{{b}}}$'
    y = row['test_accuracy_flipped']
    
    tuples.append((a, b, 1))
    x_labels.append(label)
    y_values.append(y)

    label = f'$\\mathbf{{{a}}}$ vs {b}'
    y = row['test_accuracy']
    
    tuples.append((a, b, 0))
    x_labels.append(label)
    y_values.append(y)


# %%

# Sort by accuracy in ascending order
sorted_data = sorted(zip(x_labels, y_values, tuples), key=lambda x: x[1])
x_labels, y_values, tuples = zip(*sorted_data)

plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values)
plt.ylabel('Accuracy (Intended)')
plt.ylim(0, 1)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('Intended Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %%

# Print pairs with intended accuracy below 0.5
count = 0
tuples_below_0_9 = []
print('Pairs with intended accuracy below 0.9:')
for value, label, tuple_data in zip(y_values, x_labels, tuples):
    if value < 0.9:
        print(f'{label} - {value:.3f} - {tuple_data}')
        count += 1
        tuples_below_0_9.append((tuple_data, value))
print(f'Number of pairs with intended accuracy below 0.9: {count}')

# %%

stuff = {}

for tuple_data, value in tuples_below_0_9:

    # check if reverse pair is in stuff, with lower value
    tuple_to_check = (tuple_data[1], tuple_data[0], 1 if tuple_data[2] == 0 else 0)
    
    
    if tuple_to_check in stuff:
        if stuff[tuple_to_check] > value:
            stuff[tuple_data] = value

            del stuff[tuple_to_check]
            
            print(f"updated {tuple_to_check} to {value}")
        else:
            print(f"seen {tuple_to_check} twice")
    else:   
        stuff[tuple_data] = value
        print(f"added {tuple_data} to stuff")

print(len(stuff))

# %%

stuff

# %%

y_values = stuff.values()
x_values = stuff.keys()
x_labels = [f'{x[0]} vs {x[1]}' for x in x_values]

plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values)
plt.ylabel('Accuracy (Intended)')
plt.ylim(0, 1)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('Intended Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# %%
