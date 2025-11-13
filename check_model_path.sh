python - <<'PY'
import os
for p in ["/data/vijay/Desktop/SmitMaurya/HybridModel/checkpoints/step3_final_retrieval/models/e5_mistral_7b_instruct",
"/data/vijay/Desktop/SmitMaurya/HybridModel/checkpoints/step3_final_retrieval/models/gemma_2_9b_it",
"/data/vijay/Desktop/SmitMaurya/HybridModel/checkpoints/step3_final_retrieval/models/gemma_2_27b_it",
"/data/vijay/Desktop/SmitMaurya/HybridModel/checkpoints/step3_final_retrieval/models/phi_3_medium_4k_instruct"]:
    print(p)
    print(os.listdir(p))
PY
