PYTHON=python3

install:
	$(PYTHON) -m pip install -r requirements.txt

download_data:
	$(PYTHON) src/download_celeba_subset.py --config configs/final_submission.yaml

baseline:
	sh scripts/run_baseline.sh

prompt1:
	sh scripts/run_prompt1_ablation.sh

multi:
	sh scripts/run_multi_prompt.sh

noise:
	sh scripts/run_noise_edit_full.sh

final:
	sh scripts/run_final_submission.sh

clean:
	rm -rf outputs/final_submission/*
