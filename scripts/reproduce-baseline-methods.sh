for cur_dataset in Aircraft Cars CIFAR10 CIFAR100 DTD Pet SUN397
do
    python feature_extractor.py \
        --gpu 1 \
        --dataset $cur_dataset \
        --batch_size 128 \
        --model_hub googlenet inception_v3 resnet50 resnet101 resnet152 densenet121 densenet169 densenet201 mobilenet_v2 mnasnet1_0 \
        --downstream RK \
        --rk_methods H_Score LEEP LogME NCE NLEEP OTCE PACTranDirichlet GBC LFC \
        --seed 0 \
        --pretrained $PATH_TO_PRETRAINED_MODEL/.cache/torch/hub/checkpoints \
        --save_url $PATH_TO_FEATURE
done

        # --rk_methods H_Score LEEP LogME NCE NLEEP OTCE PACTran GBC LFC \
