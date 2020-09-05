import os
import pandas as pd


def convert_dataset():
    root = '/mnt/projects/CXR_Pneumonia/Stage2'
    dicom_folder = 'stage_2_train_images'
    out_pic_folder = 'state_2_train_images_pic'




def convert_label_file():
    root = '/mnt/projects/CXR_Pneumonia/Stage2'
    train_label_file = 'stage_2_train_labels.csv'

    data_df = pd.read_csv(os.path.join(root, train_label_file))
    data = {}
    for idx, row in data_df.iterrows():
        pid, x, y, w, h, t = row['patientId'], row['x'], row['y'], row['width'], row['height'], row['Target']
        if not t:
            data[pid] = []
        else:
            if pid in data:
                data[pid].extend([x, y, x + w, y + h])
            else:
                data[pid] = [x, y, x + w, y + h]

    data_content = []
    for key in data.keys():
        name = '{}.dcm'.format(key)
        content = ''
        if data[key]:
            loc = data[key]
            for i in range(len(loc) // 4):
                content += '0 {} {} {} {};'.format(int(loc[4 * i + 0]), int(loc[4 * i + 1]),
                                                   int(loc[4 * i + 2]), int(loc[4 * i + 3]))
        data_content.append([name, content, int(len(content) != 0)])

    train_label_file_out = os.path.join(root, 'all_label.csv')
    df = pd.DataFrame(data_content, columns=['image_name', 'annotation', 'label'])
    df.to_csv(train_label_file_out, index=False)


def dataset_split():
    root = '/mnt/projects/CXR_Pneumonia/Stage2/dataset'
    df = pd.read_csv(os.path.join(root, 'all_label.csv'))
    pos_cases = df[df['label'] == 1]
    neg_cases = df[df['label'] == 0]

    val_test_pos_cases = pos_cases.sample(1000, random_state=0)
    val_test_neg_cases = neg_cases.sample(1000, random_state=1)

    train_pos_cases = pos_cases.drop(val_test_pos_cases.index)
    train_neg_cases = neg_cases.drop(val_test_neg_cases.index)

    val_pos_cases = val_test_pos_cases.sample(500, random_state=2)
    val_neg_cases = val_test_neg_cases.sample(500, random_state=3)

    test_pos_cases = val_test_pos_cases.drop(val_pos_cases.index)
    test_neg_cases = val_test_neg_cases.drop(val_neg_cases.index)

    train_cases = pd.concat([train_pos_cases, train_neg_cases])
    val_cases = pd.concat([val_pos_cases, val_neg_cases])
    test_cases = pd.concat([test_pos_cases, test_neg_cases])

    train_cases.to_csv(os.path.join(root, 'train_label.csv'))
    val_cases.to_csv(os.path.join(root, 'dev_label.csv'))
    test_cases.to_csv(os.path.join(root, 'test_label.csv'))

    print(len(train_pos_cases), len(train_neg_cases))
    print(len(val_pos_cases), len(val_neg_cases))
    print(len(test_pos_cases), len(test_neg_cases))


if __name__ == '__main__':

    # convert_label_file()

    dataset_split()