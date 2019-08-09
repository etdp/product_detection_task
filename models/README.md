# Detection Models

This folder contains the models' library for object detection task.
For now, it has libraries 
- Tensorflow Object Detection API

## Tensorflow Object Detection API
This is from Tensorflow official [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection "Object Detection API").

### Running the Training Job
A local training job can be run with the following command:
#### Add Libraries to PYTHONPATH
``` bash
LIB_DIR=${path to code base models}
export PYTHONPATH=$PYTHONPATH:${LIB_DIR}:${LIB_DIR}/slim
```
where `${LIB_DIR}` points to the path of this **models** folder

#### Start to train
```bash
LIB_DIR=${path to code base models}
PIPELINE_CONFIG_PATH=${path to pipeline config file}
MODEL_DIR=${path to model directory}
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 ${LIB_DIR}/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```
where `${PIPELINE_CONFIG_PATH}` points to the pipeline config and
`${MODEL_DIR}` points to the directory in which training checkpoints
and events will be written to. Note that this binary will interleave both
training and evaluation.
