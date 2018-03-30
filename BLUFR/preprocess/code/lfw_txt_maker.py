import os


def caffe_input_txt_maker(data_folder, outfile_name, phase='train'):
    file_cnt = 0
    class_cnt = 0
    with open(outfile_name, 'w') as fobj:

        for folder_name in os.listdir(data_folder):
            label = folder_name.split('__')[0]
            folder_path = os.path.join(data_folder, folder_name)
            class_cnt += 1
            for file_name in os.listdir(folder_path):
                file_cnt += 1

                file_path = folder_name + '/' + file_name

                if phase == 'val':
                    file_path = 'val/' + folder_name + '/' + file_name

                fobj.writelines(file_path + " " + str(class_cnt - 1) + '\n')

    file_dir, base_name = os.path.split(outfile_name)
    file_name, ext = os.path.splitext(base_name)

    new_outfile_name = file_dir + '/' + file_name + '_%d_%d' % (class_cnt, file_cnt) + ext
    if os.path.exists(new_outfile_name):
        os.remove(new_outfile_name)
    os.rename(outfile_name, new_outfile_name)
    print('Done')


if __name__ == "__main__":
    path = '/home/zhangkun/Desktop/Face Recognition/BLUFR/BLUFR/preprocess/result/'
    caffe_input_txt_maker(data_folder=path + 'lfw-112X96',
                          outfile_name=path + "lfw-112X96.txt", phase='train')

