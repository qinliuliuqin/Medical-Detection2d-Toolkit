import argparse
import os
import pandas as pd

from detection2d.vis.gen_html_report import gen_html_report
from detection2d.vis.gen_images import gen_plane_images


def parse_and_check_arguments():
    """
    Parse input arguments and raise error if invalid.
    """
    default_image_folder = '/mnt/projects/CXR_Object/data/dev'
    default_label_file = '/mnt/projects/CXR_Object/dataset/dev.csv'
    default_detection_file = '/mnt/projects/CXR_Object/results/model_0622_2020/contrast_flip/dev/localization.csv'
    default_output_folder = '/mnt/projects/CXR_Object/results/model_0622_2020/html_report'
    default_generate_pictures = True

    parser = argparse.ArgumentParser(
        description='Snapshot three planes centered around landmarks.')
    parser.add_argument('--image_folder', type=str,
                        default=default_image_folder,
                        help='Folder containing the source data.')
    parser.add_argument('--label_file', type=str,
                        default=default_label_file,
                        help='A folder where CSV files containing labelled landmark coordinates are stored.')
    parser.add_argument('--detection_file', type=str,
                        default=default_detection_file,
                        help='A folder where CSV files containing detected or baseline landmark coordinates are stored.')
    parser.add_argument('--output_folder', type=str,
                        default=default_output_folder,
                        help='Folder containing the generated snapshot images.')
    parser.add_argument('--generate_pictures', type=bool,
                        default=default_generate_pictures,
                        help='Folder containing the generated snapshot images.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_and_check_arguments()

    labels_df = pd.read_csv(args.label_file, na_filter=False)
    labels = labels_df.loc[labels_df['annotation'].astype(bool)].reset_index(drop=True)
    labels_dict = dict(zip(labels.image_name, labels.annotation))

    image_list = labels_df['image_name'].to_list()

    usage_flag = 1
    preds_dict = None
    if os.path.isfile(args.detection_file):
        usage_flag = 2
        preds_df = pd.read_csv(args.detection_file, na_filter=False)
        preds = preds_df.loc[preds_df['prediction'].astype(bool)].reset_index(drop=True)
        preds_dict = dict(zip(preds.image_name, preds.prediction))

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    error_summary = gen_html_report(image_list, [labels_dict, preds_dict], usage_flag, args.output_folder)

    if args.generate_pictures:
        print('Start generating planes for the labelled landmarks.')
        gen_plane_images(error_summary, args.image_folder, labels_dict, preds_dict, 0, args.output_folder)
