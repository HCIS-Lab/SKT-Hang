import sys, os , glob
import numpy as np
import matplotlib.pyplot as plt

def main(root_name : str):

    fnames = [root_name]
    if os.path.isdir(root_name):
        fnames = glob.glob(f'{root_name}/*formal_3/*.txt')

    for fname in fnames:

        model_type = fname.split('/')[1]

        print(f'processing {model_type} ...')
        f = open(fname, 'r')
        lines = f.readlines()

        training_model_type = None
        if 'deform' in model_type:
            training_model_type = 'deform'
        elif 'traj' in model_type:
            training_model_type = 'cvae'
        elif 'af' in model_type or 'fusion' in model_type:
            training_model_type = 'affordance'

        mutual = True if 'mutual' in model_type else False

        training_epoch = []
        training_res = None
        validation_epoch = []
        validation_res = None

        if mutual:
            if training_model_type == 'cvae':
                training_res = {
                    'afford_loss': [],
                    'dir_loss': [],
                    'nn_loss': [],
                    'dist_loss': [],
                    'kl_loss': [],
                    'recon_loss': [],
                    'total_loss': []
                }
                validation_res = {
                    'afford_loss': [],
                    'dir_loss': [],
                    'nn_loss': [],
                    'dist_loss': [],
                    'kl_loss': [],
                    'recon_loss': [],
                    'total_loss': []
                }
            if training_model_type == 'deform':
                training_res = {
                    'cls_loss': [],
                    'afford_loss': [],
                    # 'dist_loss': [],
                    # 'dir_loss': [],
                    'deform_loss': [],
                    'total_loss': [],
                }
                validation_res = {
                    'cls_loss': [],
                    'afford_loss': [],
                    # 'dist_loss': [],
                    # 'dir_loss': [],
                    'deform_loss': [],
                    'total_loss': [],
                }
        else :
            if training_model_type == 'cvae':
                training_res = {
                    'dist_loss': [],
                    'nn_loss': [],
                    'dir_loss': [],
                    'kl_loss': [],
                    'recon_loss': [],
                    'total_loss': []
                }
                validation_res = {
                    'dist_loss': [],
                    'nn_loss': [],
                    'dir_loss': [],
                    'kl_loss': [],
                    'recon_loss': [],
                    'total_loss': []
                }


        if training_model_type == 'affordance':
            training_res = {
                'total_loss': []
            }
            validation_res = {
                'total_loss': []
            }

        i = 0
        for i, line in enumerate(lines):
            if 'training stage' in line:

                # if training_model_type == 'ae' or training_model_type == 'cae':
                #     time_line = lines[i + 1]
                #     epoch_line = lines[i + 2]
                #     lr_line = lines[i + 3]
                #     total_liss_line = lines[i + 4]

                #     training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                #     training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                
                if mutual:
                    if training_model_type == 'cvae':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        afford_loss_line = lines[i + 4]
                        dist_loss_line = lines[i + 5]
                        nn_loss_line = lines[i + 6]
                        dir_loss_line = lines[i + 7]
                        kl_loss_line = lines[i + 8]
                        recon_loss_line = lines[i + 9]
                        total_liss_line = lines[i + 10]

                        training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        training_res['afford_loss'].append(float(afford_loss_line.split(':')[-1].strip()))
                        training_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        training_res['nn_loss'].append(float(nn_loss_line.split(':')[-1].strip()))
                        training_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        training_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                        training_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                        training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                    
                    if training_model_type == 'deform':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        cls_loss_line = lines[i + 4]
                        afford_loss_line = lines[i + 5]
                        dist_loss_line = lines[i + 6]
                        dir_loss_line = lines[i + 7]
                        deform_loss_line = lines[i + 8]
                        total_loss_line = lines[i + 9]

                        training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        training_res['cls_loss'].append(float(cls_loss_line.split(':')[-1].strip()))
                        training_res['afford_loss'].append(float(afford_loss_line.split(':')[-1].strip()))
                        # training_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        # training_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        training_res['deform_loss'].append(float(deform_loss_line.split(':')[-1].strip()))
                        training_res['total_loss'].append(float(deform_loss_line.split(':')[-1].strip()))
                        # training_res['total_loss'].append(float(total_loss_line.split(':')[-1].strip()))
                
                else :
                    if training_model_type == 'cvae':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        dist_loss_line = lines[i + 4]
                        nn_loss_line = lines[i + 5]
                        dir_loss_line = lines[i + 6]
                        kl_loss_line = lines[i + 7]
                        recon_loss_line = lines[i + 8]
                        total_liss_line = lines[i + 9]

                        training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        training_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        training_res['nn_loss'].append(float(nn_loss_line.split(':')[-1].strip()))
                        training_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        training_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                        training_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                        training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))


                if training_model_type == 'affordance':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))

            if 'validation stage' in line:
                # if training_model_type == 'ae' or training_model_type == 'cae':
                #     time_line = lines[i + 1]
                #     epoch_line = lines[i + 2]
                #     lr_line = lines[i + 3]
                #     total_liss_line = lines[i + 4]

                #     validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                #     validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                
                if mutual:
                    if training_model_type == 'cvae':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        afford_loss_line = lines[i + 4]
                        dist_loss_line = lines[i + 5]
                        nn_loss_line = lines[i + 6]
                        dir_loss_line = lines[i + 7]
                        kl_loss_line = lines[i + 8]
                        recon_loss_line = lines[i + 9]
                        total_liss_line = lines[i + 10]

                        validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        validation_res['afford_loss'].append(float(afford_loss_line.split(':')[-1].strip()))
                        validation_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        validation_res['nn_loss'].append(float(nn_loss_line.split(':')[-1].strip()))
                        validation_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        validation_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                        validation_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                        validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                    
                    if training_model_type == 'deform':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        cls_loss_line = lines[i + 4]
                        afford_loss_line = lines[i + 5]
                        dist_loss_line = lines[i + 6]
                        dir_loss_line = lines[i + 7]
                        deform_loss_line = lines[i + 8]
                        total_loss_line = lines[i + 9]

                        validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        validation_res['cls_loss'].append(float(cls_loss_line.split(':')[-1].strip()))
                        validation_res['afford_loss'].append(float(afford_loss_line.split(':')[-1].strip()))
                        # validation_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        # validation_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        validation_res['deform_loss'].append(float(deform_loss_line.split(':')[-1].strip()))
                        validation_res['total_loss'].append(float(deform_loss_line.split(':')[-1].strip()))
                        # validation_res['total_loss'].append(float(total_loss_line.split(':')[-1].strip()))
                
                else :
                    if training_model_type == 'cvae':
                        time_line = lines[i + 1]
                        epoch_line = lines[i + 2]
                        lr_line = lines[i + 3]
                        dist_loss_line = lines[i + 4]
                        nn_loss_line = lines[i + 5]
                        dir_loss_line = lines[i + 6]
                        kl_loss_line = lines[i + 7]
                        recon_loss_line = lines[i + 8]
                        total_liss_line = lines[i + 9]

                        validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                        validation_res['dist_loss'].append(float(dist_loss_line.split(':')[-1].strip()))
                        validation_res['nn_loss'].append(float(nn_loss_line.split(':')[-1].strip()))
                        validation_res['dir_loss'].append(float(dir_loss_line.split(':')[-1].strip()))
                        validation_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                        validation_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                        validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))

                if training_model_type == 'affordance':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
        
        fname_1 = fname.split('/')[1]
        fname_2 = fname.split('/')[2][:-4]

        output_dir = f'figures/{fname_1}/{fname_2}'
        os.makedirs(output_dir, exist_ok=True)

        # total only
        plt.figure(figsize=(10, 5))
        plt.plot(training_res['total_loss'], label=f'train_total_loss', zorder=2)
        plt.plot(validation_res['total_loss'], label=f'val_total_loss', zorder=2)
        plt.title('Training History (Total Loss)')
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        # plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/total.png')
        plt.close()
        # print(f'{output_dir}/total.png saved')
        
        if training_model_type != 'affordance':
            # all
            plt.figure(figsize=(10, 5))
            for key in training_res.keys():
                if key != 'total_loss':
                    plt.plot(training_res[key], label=f'train_{key}', zorder=2)
            for key in training_res.keys():
                if key != 'total_loss':
                    plt.plot(validation_res[key], label=f'val_{key}', zorder=2)
            plt.title('Training History (All Loss)')
            plt.xlabel('epoch', fontsize=16)
            plt.ylabel('loss', fontsize=16)
            # plt.yscale("log")
            plt.legend()
            plt.grid()
            plt.savefig(f'{output_dir}/all.png')
            plt.close()
            # print(f'{output_dir}/all.png saved')

if __name__=="__main__":
    if len(sys.argv) < 2:
        print('no input file specified')
        exit(-1)
    root_name = sys.argv[1]
    main(root_name)