# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, inference'
    exit 1
fi

if [ $1 = 'train' ]; then 

    # model_configs=(
    #     "sctdn_3dof_40wpts"
    # )

    # traj_recon_affordance_datasets=(

    #     "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
    # )

    model_configs=(
        "sctdn_3dof_40wpts_1.0_0.1_0.1"  
        "sctdn_3dof_40wpts_1.0_0.1_1.0"  
        "sctdn_3dof_40wpts_1.0_1.0_0.1"
        "sctdn_3dof_40wpts_0.1_0.1_1.0"  
        "sctdn_3dof_40wpts_0.1_1.0_0.1"  
        "sctdn_3dof_40wpts_0.1_1.0_1.0"  

        # "sctdn_3dof_40wpts_1.0_0.5_0.5"
        # "sctdn_3dof_40wpts_1.0_0.1_0.5"  
        # "sctdn_3dof_40wpts_1.0_0.5_1.0"
        # "sctdn_3dof_40wpts_1.0_0.5_0.1"  
        # "sctdn_3dof_40wpts_1.0_1.0_0.5"
        
        # "sctdn_3dof_40wpts_0.1_0.1_0.5"  
        # "sctdn_3dof_40wpts_0.1_0.5_1.0"  
        # "sctdn_3dof_40wpts_0.1_0.5_0.1"  
        # "sctdn_3dof_40wpts_0.1_1.0_0.5"  
        # "sctdn_3dof_40wpts_0.1_0.5_0.5"  
        
        # "sctdn_3dof_40wpts_0.5_0.1_0.1"  
        # "sctdn_3dof_40wpts_0.5_0.5_1.0"  
        # "sctdn_3dof_40wpts_0.5_0.1_0.5"  
        # "sctdn_3dof_40wpts_0.5_1.0_0.1"  
        # "sctdn_3dof_40wpts_0.5_0.1_1.0"  
        # "sctdn_3dof_40wpts_0.5_1.0_0.5"  
        # "sctdn_3dof_40wpts_0.5_0.5_0.1"  
        # "sctdn_3dof_40wpts_0.5_1.0_1.0"  
    )

    traj_recon_affordance_datasets=(

        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"

        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"

        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
        # "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c"
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
            python3 run_sctdn.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/sctdn_loss_ab/${model_config}.yaml" > $output_log

        else 

            python3 run_sctdn.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/sctdn_loss_ab/${model_config}.yaml"

        fi 

            
    done

elif [ $1 = 'inference' ]; then

    obj_shape_root="../shapes/inference_objs_50" # inference
    hook_shape_root="../shapes/hook_all_new" # inference

    model_configs=(
        "sctdn_3dof_10wpts"
        "sctdn_3dof_20wpts"
        "sctdn_3dof_40wpts"

        "sctdn_6dof_10wpts"
        "sctdn_6dof_20wpts"
        "sctdn_6dof_40wpts"
    )

    inference_dirs=(

        "../dataset/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview-5c/test"
        "../dataset/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview-5c/test"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c/test"

        "../dataset/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview-5c/test"
        "../dataset/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview-5c/test"
        "../dataset/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c/test"

    )

    traj_recon_shape_checkpoints=(
        
        "checkpoints/sctdn_3dof_10wpts/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview-5c"
        "checkpoints/sctdn_3dof_20wpts/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview-5c"
        "checkpoints/sctdn_3dof_40wpts/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview-5c"

        "checkpoints/sctdn_6dof_10wpts/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview-5c"
        "checkpoints/sctdn_6dof_20wpts/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview-5c"
        "checkpoints/sctdn_6dof_40wpts/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview-5c"

    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        # for iter in "${iters[@]}"
        # do 
            python3 run_sctdn.py --training_mode 'inference' \
                                                --inference_dir ${inference_dirs[$i]} \
                                                --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                --config "../config/sctdn/${model_configs[$i]}.yaml" \
                                                --weight_subpath "1000_points-best.pth" \
                                                --obj_shape_root ${obj_shape_root} \
                                                --hook_shape_root ${hook_shape_root} 
                                                # --evaluate 
                                                # --visualize
        # done
    done 

else 

    echo '[Error] wrong runing type (should be train, inference)'
    exit 

fi

