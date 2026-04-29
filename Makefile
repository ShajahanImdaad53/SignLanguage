.PHONY: help install data ssl train test clean

help:
	@echo "SLT Hybrid SSL — Sign Language Translation"
	@echo ""
	@echo "Commands:"
	@echo "  make install       Install dependencies"
	@echo "  make data          Run full data pipeline"
	@echo "  make ssl           Run SSL pretraining"
	@echo "  make train         Fine-tune on labelled data"
	@echo "  make test          Evaluate on test set"
	@echo "  make clean         Clean logs and models"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm 2>/dev/null || true

data:
	python pipelines/run_data_pipeline.py \
		--manifest data/manifest.csv \
		--config configs/data_config.yaml \
		--base configs/base_config.yaml \
		--workers 4

ssl:
	python src/ssl_pretrain.py \
		--config configs/ssl_config.yaml \
		--base configs/base_config.yaml \
		--seed 42

train:
	python src/trainer.py \
		--config configs/train_config.yaml \
		--base configs/base_config.yaml \
		--ssl_config configs/ssl_config.yaml \
		--seed 42

test:
	python src/evaluate_test.py \
		--checkpoint models/best_model.pt \
		--config configs/train_config.yaml \
		--base configs/base_config.yaml

clean:
	rm -rf logs/* models/*.pt *.pyc __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

format:
	black src/ pipelines/ agents/

lint:
	pylint src/ pipelines/ agents/ --disable=C0111,C0103,R0913
