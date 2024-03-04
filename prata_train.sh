# CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
#     --model_name_or_path facebook/bart-large \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 10 \
#     --tst_lambda 0.1.0 \
#     --gradient_accumulation_steps 4 \
#      --test_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
#     --train_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/train/style_transfer_unsup.json \
#     --validation_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
#     --output_dir trained_models/adapter-tst-tense-style-lambda-adapter_0.1.0/ \
#     --overwrite_output_dir \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --text_column sentence \
#     --summary_column style_label \
#     --evaluation_strategy epoch \
#     --predict_with_generate \
#     --save_strategy no \
#     --num_label_cls1 3 \
#     --tst_task_name tense_adjadv_removal


# CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
#     --model_name_or_path facebook/bart-large \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 10 \
#     --tst_lambda 0.9 \
#     --tst_lambda2 0.9 \
#     --gradient_accumulation_steps 4 \
#     --test_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
#     --train_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/train/style_transfer_unsup.json \
#     --validation_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
#     --output_dir trained_models/adapter-tst-tense-removal-style-lambda-adapter_0.9/ \
#     --overwrite_output_dir \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=16 \
#     --text_column sentence \
#     --summary_column style_label \
#     --evaluation_strategy epoch \
#     --predict_with_generate \
#     --save_strategy no \
#     --num_label_cls1 3 \
#     --num_label_cls2 0 \
#     --train_adapter \
#     --compositional_edits 1 \
#     --tst_task_name tense_adjadv_removal

CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 3 \
    --tst_lambda 0.9 \
    --gradient_accumulation_steps 4 \
    --test_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
    --train_file data/datasets/yelpbaseline/train/sentiment_transfer_unsup.json \
    --validation_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
    --output_dir trained_models/adapter-tst-yelpbaseline-style-lambda-adapter_0.9/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=16 \
    --text_column sentence \
    --summary_column style_label \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --save_strategy no \
    --num_label_cls1 2 \
    --train_adapter \
    --tst_task_name sentiment


CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 3 \
    --tst_lambda 0.95 \
    --gradient_accumulation_steps 4 \
    --test_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
    --train_file data/datasets/yelpbaseline/train/sentiment_transfer_unsup.json \
    --validation_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
    --output_dir trained_models/adapter-tst-yelpbaseline-style-lambda-adapter_0.95/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=16 \
    --text_column sentence \
    --summary_column style_label \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --save_strategy no \
    --num_label_cls1 2 \
    --train_adapter \
    --tst_task_name sentiment



