from pathlib import Path
import sys
from rfdetr import RFDETRNano, RFDETRSmall

# usage: python train.py [nano|small] [epochs] [dataset_dir] [output_dir]
ROOT = str(Path(__file__).resolve().parent.parent)  # repo root
MODELS = {"nano": RFDETRNano, "small": RFDETRSmall}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nano"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    model = MODELS[name]()
    data = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/train_dataset"
    out = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/runs/{name}"
    model.train(dataset_dir=data, epochs=epochs, batch_size=2, grad_accum_steps=4, output_dir=out)


if __name__ == "__main__":  # required on macOS: dataloader workers re-import this file
    main()
