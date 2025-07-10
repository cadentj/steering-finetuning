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
for _, row in grouped.iterrows():
    a, b, ablated = row['dataset_a'], row['dataset_b'], row['ablated_dataset']

    if a == "pronouns":
        print(a, b)
    if ablated == a:
        # Intended is b, so use test_accuracy_flipped
        label = f'{a} vs $\\mathbf{{{b}}}$'
        y = row['test_accuracy_flipped']
    else:
        # Intended is a, so use test_accuracy
        label = f'$\\mathbf{{{a}}}$ vs {b}'
        y = row['test_accuracy']
    x_labels.append(label)
    y_values.append(y)

# %%

stuff = [(row['dataset_a'], row['dataset_b']) for _, row in grouped.iterrows()]
print(len(stuff), len(set(stuff)))

# %%

# Sort by accuracy in ascending order
sorted_data = sorted(zip(x_labels, y_values), key=lambda x: x[1])
x_labels, y_values = zip(*sorted_data)

plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values)
plt.ylabel('Accuracy (Intended)')
plt.ylim(0, 1)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('Intended Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print pairs with intended accuracy below 0.5
print('Pairs with intended accuracy below 0.5:')
for i, row in grouped.iterrows():
    a, b, ablated = row['dataset_a'], row['dataset_b'], row['ablated_dataset']
    if ablated == a:
        intended = b
        acc = row['test_accuracy_flipped']
    else:
        intended = a
        acc = row['test_accuracy']
    if acc < 0.5:
        print(f'({a}, {b}) - intended: {intended}, accuracy: {acc:.3f}')