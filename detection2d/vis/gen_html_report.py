import os
import pandas as pd

from detection2d.vis.error_analysis import error_analysis


def add_document_text(original_text, new_text_to_add):
    """
    Add document text file
    """
    return original_text + r'+"{0}"'.format(new_text_to_add)


def write_html_report(document_text, analysis_text, html_report_path, width):
    """
    Write the html report for a landmark.
    """
    f = open(html_report_path, 'w')
    message = """
    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <title>result analysis</title>
      <style type="text/css">
          *{
              padding:0;
              margin:0;
          }
          .content {
              width: %spx;
              z-index:2;
          }
          .content img {
              width: %spx;
              transition-duration:0.2s;
              z-index:1;
          }
          .content img:active {
              transform: scale(2);
              -webkit-transform: scale(2); /*Safari 和 Chrome*/
              -moz-transform: scale(2); /*Firefox*/
              -ms-transform: scale(2); /*IE9*/
              -o-transform: scale(2); /*Opera*/
          }
      </style>
    </head>
    <body>
      <h1> Summary:</h1>
      %s
      <script type="text/javascript">
        document.write(%s)               
      </script>
    </body>
    </html>""" % (width, width, analysis_text, document_text)

    f.write(message)
    f.close()


def add_image(document_text, image_link_template, image_folder, image_name, width):
    """
    Add a image to the document text file
    :param document_text:
    :param image_link_template:
    :param image_folder:
    :param image_name:
    :param width:
    :return:
    """
    document_text += "\n"
    image_info = r'<td>{0}</td>'.format(image_link_template.format(os.path.join(image_folder, image_name), width))
    document_text = add_document_text(document_text, image_info)
    return document_text


def gen_row_for_html(usage_flag, image_link_template, error_info_template, document_text, image_name,
                     image_idx, error_summary, picture_folder='./pictures', width=200):
    """
    Generate a line of html text contents for labelled cases, in the usage of label checking.
    """
    case_info = r'<b>Case nunmber</b>:{0}: {1};'.format(image_idx, image_name)
    image_name_prefix = image_name.split('.')[0]
    image_name_postfix = image_name.split('.')[1]
    if image_name_postfix == 'dcm':
        image_name_postfix = 'jpg'

    if usage_flag == 1:
        labeled_image_name = '{}_labelled.{}'.format(image_name_postfix, image_name_postfix)
        error_info = error_info_template.format(0)

    elif usage_flag == 2:
        labeled_image_name = '{}_labelled.{}'.format(image_name_prefix, image_name_postfix)
        detected_image_name = '{}_detected.{}'.format(image_name_prefix, image_name_postfix)
        error_info = error_info_template.format(
            error_summary.label[image_idx], error_summary.pred[image_idx], error_summary.error[image_idx])

    else:
        raise ValueError('Unsupported usage flag.')

    document_text = add_document_text(document_text, case_info)
    document_text = add_document_text(document_text, error_info)
    document_text += "\n"
    document_text = add_document_text(document_text, "<table border=1><tr>")

    if usage_flag == 1 or usage_flag == 2:
        document_text = add_image(document_text, image_link_template, picture_folder, labeled_image_name, width)

    else:
        raise ValueError('Unsupported usage flag.')

    document_text += "\n"
    document_text = add_document_text(document_text, r'</tr></table>')

    return document_text


def gen_html_report(image_list, objects_dict, usage_flag, output_folder, decending=True):
    """
    Generate HTML report for object detection.
    :param image_list: the list containing all images
    :param objects_dict:
    :param usage_flag: 1 for label checking, 2 for error evaluation
    :param output_folder:
    :return:
    """

    labelled_objects_dict = objects_dict[0]

    if usage_flag == 2:
        detected_objects_dict = objects_dict[1]
        error_summary = error_analysis(image_list, labelled_objects_dict, detected_objects_dict, decending)
        image_list = error_summary.image_list

    document_text = r'"<h1>check annotations:</h1>"'
    for image_idx, image_name in enumerate(image_list):
        print("Processing image {}.".format(image_name))
        image_link_template = r"<div class='content'><img border=0  src= '{0}'  hspace=1  width={1} class='pic'></div>"
        error_info_template = r'<b>Labelled</b>: {};'
        document_text += "\n"

        if usage_flag == 1:
            document_text = gen_row_for_html(usage_flag, image_link_template,
                error_info_template, document_text, image_name, image_idx, None)

        elif usage_flag == 2:
            error_info_template += r'<b>Detected</b>: {};'
            error_info_template += r'<b>Error</b>: {};'

            document_text = gen_row_for_html(usage_flag, image_link_template,
                error_info_template, document_text, image_name, image_idx, error_summary)
        else:
            raise ValueError('Undefined usage flag!')

    html_report_name = 'result_analysis.html'
    html_report_path = os.path.join(output_folder, html_report_name)
    write_html_report(document_text, '', html_report_path, width=200)

    return error_summary