import os

import utilities
from graph_create import create_graph_snapshot


def main():
    lags = 20
    pred = 1

    root_dir = r"F:\New Training\New Data Preparation\Merged_CSV_dir"

    file_path = os.path.join(root_dir, 'merged_csv.csv')

    graph_path = r"F:\New Training\Pearson-Correlation-Graph-Updated\Graph"

    edge_index_path = os.path.join(graph_path, 'mst_graph_pearson_edge_index.pt')
    edge_weight_path = os.path.join(graph_path, 'mst_graph_pearson_edge_weight.pt')

    datasets = create_graph_snapshot(file_path, edge_index_path, edge_weight_path, lags=lags, pred=pred)

    print(f"Shape of data: {datasets[0]}")
    print(f"Type of data: {type(datasets[0])}")

    # Split to create train, val, and test datasets
    train_dataset, val_dataset, test_dataset = utilities.split_list_of_data(dataset=datasets,
                                                                            train_ratio=0.8,
                                                                            val_ratio=0.1)

    print(f"train_datasets: {len(train_dataset)}")
    print(f"val_datasets: {len(val_dataset)}")
    print(f"test_datasets: {len(test_dataset)}")

    train_data_path = './Train'
    valid_data_path = './Validation'
    test_data_path = './Test'

    utilities.create_empty_dir(train_data_path)
    utilities.create_empty_dir(valid_data_path)
    utilities.create_empty_dir((test_data_path))

    # create_dir("./Dataset")
    utilities.create_dir(train_data_path)
    utilities.create_dir(valid_data_path)
    utilities.create_dir(test_data_path)

    # saving train, val, and test dataset
    utilities.save_as_pt_file(dataset=train_dataset, path=train_data_path)
    utilities.save_as_pt_file(dataset=val_dataset, path=valid_data_path)
    utilities.save_as_pt_file(dataset=test_dataset, path=test_data_path)

    print(f"Done!!! creating data.")


if __name__ == '__main__':
    main()
