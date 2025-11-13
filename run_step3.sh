export HF_TOKEN="<>"
export WANDB_API_KEY="<>"

export WANDB_PROJECT="Coliee 2025 Task 3"
export WANDB_LOG_MODEL="false"


############# Step 3.1: Create data #############
export N_NEGATIVES=50
export N_POSITIVE_REPLICATES=3
python src/create_data_for_step3.py
#################################################


# ############### Step 3.2: Training ##############
# MODEL_NAMES=(
#     "gemma_2_27b_it"
#     "e5_mistral_7b_instruct"
#     "phi_3_medium_4k_instruct"
#     "gemma_2_9b_it"
# )

# for model_name in ${MODEL_NAMES[@]}
# do
#     accelerate launch --num_processes=2 src/run_step3.py\
#         --config configs/train/$model_name.yaml
# done
# #################################################


############## Step 3.3: Inference ##############
MODEL_NAMES=(
    "gemma_2_27b_it"
    "e5_mistral_7b_instruct"
    "phi_3_medium_4k_instruct"
    "gemma_2_9b_it"
)

for model_name in ${MODEL_NAMES[@]}
do
    accelerate launch --num_processes=1 src/run_step3.py\
        --config configs/infer/$model_name.yaml
done
#################################################


clearenv () {
    unset HF_TOKEN
    unset WANDB_PROJECT
    unset WANDB_LOG_MODEL

    unset N_NEGATIVES
    unset N_POSITIVE_REPLICATES
}
