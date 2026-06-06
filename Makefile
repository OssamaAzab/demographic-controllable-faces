# Pipeline orchestration. All paths resolved from `src/dcfaces/paths.py` —
# no edits required.

.PHONY: help install setup caption train bench demo all clean clean-cache test

help:
	@echo "Targets:"
	@echo "  install   Install package + dependencies (run once)"
	@echo "  setup     Download FFHQ, build splits + identity gallery"
	@echo "  caption   BLIP-2 + FairFace captioning of FFHQ subset (~10h)"
	@echo "  train     Train demographic LoRA + per-identity LoRAs (~9h)"
	@echo "  bench     Run benchmark generations + compute metrics (~40h)"
	@echo "  demo      Launch Gradio demo locally"
	@echo "  all       Run setup + caption + train + bench"
	@echo "  test      Run unit tests"
	@echo "  clean     Remove models/, results/, .cache/, wandb/"
	@echo "  clean-cache   Remove only .cache/ (HF downloads)"

install:
	pip install -e .
	pip install -r requirements.txt

setup:
	python scripts/01_download_ffhq.py
	python scripts/02_split_ffhq.py
	python scripts/03_build_identity_gallery.py

caption:
	python scripts/04_caption_ffhq.py

train:
	python scripts/05_train_demo_lora.py
	python scripts/06_eval_checkpoints.py
	python scripts/07_train_dreambooth_lora.py

bench:
	python scripts/08_run_benchmark.py
	python scripts/09_compute_metrics.py

demo:
	python demo/app.py

all: setup caption train bench

test:
	pytest tests/ -v

clean:
	rm -rf models/ results/ .cache/ wandb/
	mkdir -p models results .cache
	touch models/.gitkeep results/.gitkeep .cache/.gitkeep

clean-cache:
	rm -rf .cache/
	mkdir -p .cache
	touch .cache/.gitkeep
