# Adapter-TST
The code implementation of our EMNLP'23 Findings "Adapter-TST: A Parameter Efficient Method for Multiple-Attribute Text Style Transfer"

### Steps to run
1. Download GloVe embedding for style classifers

    `cd checkpoints_cls`

    `wget https://nlp.stanford.edu/data/glove.840B.300d.zip`

    `unzip glove.840B.300d.zip`

2. Install the dependencies

    Huggingface `transformers` shouldn't be installed. 

    Remember to change to the path in following file:

        1. examples/pytorch/summarization.py L25, L27
        2. PPL_score.py L2
        3. classifer/textcnn.py L9, L18

3. Train the classifier

        # all the classifier weights have beed uploaded, if you want to train a new classifier, you can use the following command

        CUDA_VISIBLE_DEVICES=0 python classifier/textcnn.py -dataset yelp -num_label 2 -batch_size 8

4. Train Adapter-TST model

    Use T5-large as the base model and train Adapter-TST model for sentiment transfer.

    ```python

    CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path t5-large \
        --do_train \
        --do_eval \
        --do_predict \
        --train_adapter \
        --num_train_epochs 1 \
        --tst_lambda 0.99 \
        --gradient_accumulation_steps 4 \
        --test_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
        --train_file data/datasets/yelpbaseline/train/sentiment_transfer_unsup.json \
        --validation_file data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json \
        --output_dir trained_models/adapter-tst-yelp-t5/ \
        --overwrite_output_dir \
        --per_device_train_batch_size=32 \
        --per_device_eval_batch_size=32 \
        --text_column sentence \
        --summary_column style_label \
        --evaluation_strategy epoch \
        --predict_with_generate \
        --save_strategy no \
        --tst_task_name sentiment
    
    #For Tense-Voice multi-attribute transfer

    CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path t5-large \
        --do_train \
        --do_eval \
        --do_predict \
        --train_adapter \
        --num_train_epochs 1 \
        --tst_lambda 0.9 \
        --tst_lambda2 0.97 \
        --gradient_accumulation_steps 4 \
        --test_file data/datasets/StylePTB/adapterTST/tense_voice/test/style_transfer_unsup.json \
        --train_file data/datasets/StylePTB/adapterTST/tense_voice/train/style_transfer_unsup.json \
        --validation_file data/datasets/StylePTB/adapterTST/tense_voice/test/style_transfer_unsup.json \
        --output_dir trained_models/adapter-tst-tense-voice-t5/ \
        --overwrite_output_dir \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --text_column sentence \
        --summary_column style_label \
        --evaluation_strategy epoch \
        --predict_with_generate \
        --save_strategy no \
        --num_label_cls1 3 \
        --num_label_cls2 2 \
        --tst_task_name tense_voice

    #For tense_adjadv_removal training

    CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path t5-large \
        --do_train \
        --do_eval \
        --do_predict \
        --train_adapter \
        --num_train_epochs 1 \
        --tst_lambda 0.9 \
        --tst_lambda2 0.98 \
        --gradient_accumulation_steps 4 \
        --test_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
        --train_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/train/style_transfer_unsup.json \
        --validation_file data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json \
        --output_dir trained_models/adapter-tst-tense-adjadv-removal-t5/ \
        --overwrite_output_dir \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --text_column sentence \
        --summary_column style_label \
        --evaluation_strategy epoch \
        --predict_with_generate \
        --save_strategy no \
        --num_label_cls1 3 \
        --num_label_cls2 2 \
        --tst_task_name tense_adjadv_removal

    #For tense_pp_removal training

    CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path t5-large \
        --do_train \
        --do_eval \
        --do_predict \
        --train_adapter \
        --num_train_epochs 1 \
        --tst_lambda 0.9 \
        --tst_lambda2 0.99 \
        --gradient_accumulation_steps 4 \
        --test_file data/datasets/StylePTB/adapterTST/tense_pp_removal/test/style_transfer_unsup.json \
        --train_file data/datasets/StylePTB/adapterTST/tense_pp_removal/train/style_transfer_unsup.json \
        --validation_file data/datasets/StylePTB/adapterTST/tense_pp_removal/test/style_transfer_unsup.json \
        --output_dir trained_models/adapter-tst-tense-pp-removal-t5/ \
        --overwrite_output_dir \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --text_column sentence \
        --summary_column style_label \
        --evaluation_strategy epoch \
        --predict_with_generate \
        --save_strategy no \
        --num_label_cls1 3 \
        --num_label_cls2 2 \
        --tst_task_name tense_pp_removal

    #For tense_pp_front_back training

    CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path t5-large \
        --do_train \
        --do_eval \
        --do_predict \
        --train_adapter \
        --num_train_epochs 1 \
        --tst_lambda 0.9 \
        --tst_lambda2 0.99 \
        --gradient_accumulation_steps 4 \
        --test_file data/datasets/StylePTB/adapterTST/tense_pp_front_back/test/style_transfer_unsup.json \
        --train_file data/datasets/StylePTB/adapterTST/tense_pp_front_back/train/style_transfer_unsup.json \
        --validation_file data/datasets/StylePTB/adapterTST/tense_pp_front_back/test/style_transfer_unsup.json \
        --output_dir trained_models/adapter-tst-tense-pp-front-back-t5/ \
        --overwrite_output_dir \
        --per_device_train_batch_size=32 \
        --per_device_eval_batch_size=32 \
        --text_column sentence \
        --summary_column style_label \
        --evaluation_strategy epoch \
        --predict_with_generate \
        --save_strategy no \
        --num_label_cls1 3 \
        --num_label_cls2 2 \
        --tst_task_name tense_pp_front_back
    ```

    Tips: You can adjust --tst_lambda and --tst_lambda2 to balance the transfer accuracy and the content preservation.

    
5. Evaluate the performance

    Accuracy
    ```python
    #yelp
    CUDA_VISIBLE_DEVICES=1 python classifier/textcnn.py -dataset yelp -num_label 2 -test_only True -gen_path  trained_models/adapter-tst-yelp-t5/generated_predictions.json

    # tense_voice
    python classifier/textcnn.py -dataset tense_voice -style tense -num_label 3 -test_only True -gen_path  trained_models/adapter-tst-tense-voice-t5/generated_predictions_comp_1.json 

    python classifier/textcnn.py -dataset tense_voice -style voice -num_label 2 -test_only True -gen_path  trained_models/adapter-tst-tense-voice-t5/generated_predictions2_comp_1.json

    #tense_adjadv_removal
    python classifier/textcnn.py -dataset tense_adjadv_removal -style tense -num_label 3 -test_only True -gen_path  trained_models/adapter-tst-tense-adjadv-removal-t5/generated_predictions_comp_1.json

    python classifier/textcnn.py -dataset tense_adjadv_removal -style adjadv_removal -num_label 2 -test_only True -gen_path  trained_models/adapter-tst-tense-adjadv-removal-t5/generated_predictions2_comp_1.json

    #tense_pp_front_back
    python classifier/textcnn.py -dataset tense_pp_front_back -style tense -num_label 3 -test_only True -gen_path  trained_models/adapter-tst-tense-pp-front-back-t5/generated_predictions_comp_1.json

    python classifier/textcnn.py -dataset tense_pp_front_back -style pp -num_label 2 -test_only True -gen_path  trained_models/adapter-tst-tense-pp-front-back-t5/generated_predictions2_comp_1.json

    #tense_pp_removal
    python classifier/textcnn.py -dataset tense_pp_removal -style tense -num_label 3 -test_only True -gen_path  trained_models/adapter-tst-tense-pp-removal-t5/generated_predictions.json

    python classifier/textcnn.py -dataset tense_pp_removal -style pp -num_label 2 -test_only True -gen_path  trained_models/adapter-tst-tense-pp-removal-t5/generated_predictions2.json
    ```

    BertScore
    ```python
    python BERTscore_evaluator.py \
        --ref-file-path trained_models/yelp_outputs/StyleTransformer/reference.txt \
        --gen-file-path trained_models/yelp_outputs/StyleTransformer/generated_predictions.txt
    ```

    PPL

    ```python
    python PPL_scorepy
    ```

    Remember to change the file path in the script.

## Citing 

If you use Adapter-TST in your publication, please cite it by using the following BibTeX entry.

```bibtex
@article{hu2023adapter,
  title={Adapter-TST: A Parameter Efficient Method for Multiple-Attribute Text Style Transfer},
  author={Hu, Zhiqiang and Lee, Roy Ka-Wei and Chen, Nancy F},
  journal={arXiv preprint arXiv:2305.05945},
  year={2023}
}
```
## Acknowledgement

This repo benefits from [Adapter-Transformer](https://github.com/adapter-hub/adapter-transformers). Thanks for their wonderful works. 