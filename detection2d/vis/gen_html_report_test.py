import argparse
import glob
import os
import pandas as pd

from detection2d.vis.gen_html_report import gen_html_report


def parse_and_check_arguments():
    """
    Parse input arguments and raise error if invalid.
    """
    default_image_folder = '/mnt/projects/CT_Dental/data'
    default_label_folder = '/mnt/projects/CT_Dental/landmark'
    default_detection_folder = '/mnt/projects/CT_Dental/results/model_0514_2020//batch_3/epoch_1000/test_set'
    default_resolution = [1.5, 1.5, 1.5]
    default_contrast_range = None
    default_output_folder = '/mnt/projects/CT_Dental/results/model_0514_2020//batch_3/epoch_1000/test_set/html_report'
    default_generate_pictures = False

    parser = argparse.ArgumentParser(
        description='Snapshot three planes centered around landmarks.')
    parser.add_argument('--image_folder', type=str,
                        default=default_image_folder,
                        help='Folder containing the source data.')
    parser.add_argument('--label_folder', type=str,
                        default=default_label_folder,
                        help='A folder where CSV files containing labelled landmark coordinates are stored.')
    parser.add_argument('--detection_folder', type=str,
                        default=default_detection_folder,
                        help='A folder where CSV files containing detected or baseline landmark coordinates are stored.')
    parser.add_argument('--resolution', type=list,
                        default=default_resolution,
                        help="Resolution of the snap shot images.")
    parser.add_argument('--contrast_range', type=list,
                        default=default_contrast_range,
                        help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--output_folder', type=str,
                        default=default_output_folder,
                        help='Folder containing the generated snapshot images.')
    parser.add_argument('--generate_pictures', type=bool,
                        default=default_generate_pictures,
                        help='Folder containing the generated snapshot images.')

    return parser.parse_args()


if __name__ == '__main__':

    data_folder = '/mnt/projects/CXR_Object/train'
    labels_file = '/mnt/projects/CXR_Object/train.csv'
    labels_df = pd.read_csv(labels_file, na_filter=False)
    print(f'{len(os.listdir(data_folder))} pictures in {data_folder}')

    labels = labels_df.loc[labels_df['annotation'].astype(bool)].reset_index(drop=True)
    img_class_dict_tr = dict(zip(labels.image_name, labels.annotation))

    output_folder = '/mnt/projects/CXR_Object/vis'
    gen_html_report([img_class_dict_tr], 1, output_folder)