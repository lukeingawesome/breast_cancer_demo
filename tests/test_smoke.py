"""CI smoke test: import & one training step."""
from src import train
import yaml, tempfile, shutil

def test_training_runs():
    cfg = yaml.safe_load(open("config.yml"))
    tmp = tempfile.mkdtemp()
    cfg["out_dir"] = tmp          # write nowhere permanent
    train.main(cfg)               # should finish without error
    shutil.rmtree(tmp)
