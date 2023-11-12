import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATASET_PATH = './iris.data'

df = pd.read_csv(DATASET_PATH, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


# part1
def count_true_in_column(data_frame, column_name):
    result = data_frame[column_name].value_counts()
    if True in result:
        return result[True]
    return 0


def count_NaN(data_frame):
    data_frame_isna = data_frame.isna()
    return [count_true_in_column(data_frame_isna, 'sepal_length'), count_true_in_column(data_frame_isna, 'sepal_width'),
            count_true_in_column(data_frame_isna, 'petal_length'),
            count_true_in_column(data_frame_isna, 'petal_width'),
            count_true_in_column(data_frame_isna, 'target')]


def remove_row_have_missing_data(data_frame):
    new_data_frame = data_frame.dropna()
    new_data_frame = new_data_frame.reset_index(drop=True)
    return new_data_frame


# part2
def label_encoder(data_frame):
    le = LabelEncoder()
    copy_data_frame = data_frame.copy()
    copy_data_frame['target'] = le.fit_transform(data_frame['target'])
    return copy_data_frame


# part3
def calculate_mean(data_frame, column_name):
    return data_frame[column_name].mean()


def calculate_variance(data_frame, column_name):
    return data_frame[column_name].var()


def calculate_mean_and_variance_columns(data_frame):
    return {'sepal_length': {'mean': calculate_mean(data_frame, 'sepal_length'),
                             'var': calculate_variance(data_frame, 'sepal_length')},
            'sepal_width': {'mean': calculate_mean(data_frame, 'sepal_width'),
                            'var': calculate_variance(data_frame, 'sepal_width')},
            'petal_length': {'mean': calculate_mean(data_frame, 'petal_length'),
                             'var': calculate_variance(data_frame, 'petal_length')},
            'petal_width': {'mean': calculate_mean(data_frame, 'petal_width'),
                            'var': calculate_variance(data_frame, 'petal_width')}}


def data_frame_normalization(data_frame):
    scaler = StandardScaler()
    normalized_data_frame = scaler.fit_transform(data_frame.iloc[:, 0:-1])
    normalized_data_frame = pd.DataFrame(normalized_data_frame,
                                         columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    target_column = data_frame['target']
    normalized_data_frame['target'] = target_column
    return normalized_data_frame


def my_print(info):
    print('******************************')
    print(f'Column sepal_length => {info["sepal_length"]}')
    print(f'Column sepal_width => {info["sepal_width"]}')
    print(f'Column petal_length => {info["petal_length"]}')
    print(f'Column petal_width => {info["petal_width"]}')
    print('******************************')


def compare(raw_data_frame, normalized_data_frame):
    raw_data_frame_info = calculate_mean_and_variance_columns(raw_data_frame)
    normalized_data_frame_info = calculate_mean_and_variance_columns(normalized_data_frame)
    print("*** Raw data frame info ***")
    my_print(raw_data_frame_info)
    print("*** normalized data frame info ***")
    my_print(normalized_data_frame_info)


# part4
def principal_component_analysis(normalized_data_frame, dimensions):
    normalized_data_frame_without_target = normalized_data_frame.iloc[:, 0: -1]
    pca = PCA(n_components=dimensions)
    new_data_frame = pca.fit_transform(normalized_data_frame_without_target)
    new_data_frame = pd.DataFrame(new_data_frame, columns=['Feature1', 'Feature2'])
    new_data_frame['target'] = normalized_data_frame['target']
    return new_data_frame


# part5
def plot_data_frame(data_frame):
    target_color = {0: 'red', 1: 'blue', 2: 'green'}
    for index, row in data_frame.iterrows():
        plt.scatter(row['Feature1'], row['Feature2'], color=target_color[row['target']])

    plt.title('IRIS Data set with PCA(2 features)\n(Red: Iris-setosa, Blue: Iris-versicolor, Green: Iris-virginica)')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.show()


def box_plot(data_frame):
    columns = [data_frame['sepal_length'], data_frame['sepal_width'], data_frame['petal_length'],
               data_frame['petal_width']]
    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.xticks([1, 2, 3, 4], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    plt.show()


if __name__ == '__main__':
    new_df = remove_row_have_missing_data(df)
    new_df = label_encoder(new_df)
    normalized_df = data_frame_normalization(new_df)
    # compare(new_df, normalized_df)
    pca_data_frame = principal_component_analysis(normalized_df, 2)
    # print(pca_data_frame)
    # plot_data_frame(pca_data_frame)

    box_plot(new_df)
    box_plot(normalized_df)


    # plot_data_frame(pca_data_frame)
