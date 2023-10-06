# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, inference'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(
        "../config/sctdn_nc/sctdn_3dof_40wpts_1c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_2c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_3c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_4c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_5c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_10c.yaml"
        "../config/sctdn_nc/sctdn_3dof_40wpts_20c.yaml"
    )

    traj_recon_affordance_datasets=(
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
    )

    category_files=(
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_1c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_2c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_3c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_4c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_5c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_10c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_20c.txt"
    )

    training_tag='' # $1
    log='save' # $2
    time_stamp=$(date +%m.%d.%H.%M)
    training_tag=''

    if [ $# -ge 2 ]; then 
        
        training_tag="${time_stamp}-${2}"

    elif [ $# -ge 3 ]; then 

        training_tag="${time_stamp}-${2}"
        log=$3

    elif [[ $training_tag = "" ]]; then 

        training_tag="${time_stamp}"

    fi 

    echo "training_tag : ${training_tag}"
    echo "log : ${log}"

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        model_config=${model_configs[$i]}
        traj_recon_affordance_dataset=${traj_recon_affordance_datasets[$i]}
        dataset_name=($(echo $traj_recon_affordance_dataset | tr "/" "\n"))
        
        echo "=============================================="
        echo "model_config=${model_config}" 
        echo "dataset=${dataset_name[-1]}"
        echo "=============================================="
        
        mkdir "training_logs/${model_config}-${training_tag}"

        if [ $log = 'save' ]; then 

            output_log="training_logs/${model_config}-${training_tag}/${dataset_name[-2]}-${dataset_name[-1]}.txt"
            python3 run_sctdn_nc.py --dataset_dir $traj_recon_affordance_dataset --category_file ${category_files[$i]} --training_tag $training_tag --config "${model_config}" > $output_log

        else 

            python3 run_sctdn_nc.py --dataset_dir $traj_recon_affordance_dataset --category_file ${category_files[$i]} --training_tag $training_tag --config "${model_config}"

        fi 

            
    done

elif [ $1 = 'inference' ]; then

    # obj_shape_root="../shapes/inference_objs_5" # validation
    obj_shape_root="../shapes/inference_objs_50" # testing
    # hook_shape_root="../shapes/inference_hooks" # validation
    hook_shape_root="../shapes/hook_all_new" # testing

    model_configs=(
        "../config/sctdn/sctdn_3dof_40wpts_1c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_2c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_3c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_4c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_5c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_10c.yaml"
        "../config/sctdn/sctdn_3dof_40wpts_20c.yaml"
    )

    inference_dirs=(
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/test" # /test or /val
    )

    category_files=(
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_1c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_2c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_3c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_4c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_5c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_10c.txt"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/labels_20c.txt"
    )

    traj_recon_shape_checkpoints=(
        # Todo
        "checkpoints/sctdn_3dof_40wpts/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview-5c"
    )

    iters=(
        "6000" "7000" "8000" "9000" "10000"
    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 
            python3 run_sctdn.py --training_mode 'inference' \
                                                --inference_dir ${inference_dirs[$i]} \
                                                --category_file ${category_files[$i]} \
                                                --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                --config ${model_configs[$i]} \
                                                --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                                --obj_shape_root ${obj_shape_root} \
                                                --hook_shape_root ${hook_shape_root} \
                                                --evaluate 
                                                # --visualize
        done
    done 

else 

    echo '[Error] wrong runing type (should be train, inference)'
    exit 

fi

