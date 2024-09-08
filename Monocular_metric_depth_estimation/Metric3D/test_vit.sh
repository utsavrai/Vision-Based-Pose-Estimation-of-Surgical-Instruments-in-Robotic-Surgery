python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \
    --load-from ./weight/metric_depth_vit_giant2_800k.pth \
    --test_data_path /home/utsav/rgb \
    --launcher None