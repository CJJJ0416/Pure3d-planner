"""Generate direct local component batches, without a full WFC building."""
import argparse
from pathlib import Path
from Filter.data.local import generate
from utils.pointcloud import read_json,workspace_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('Filter/configs/dataset.json'))
    p.add_argument('--batch',nargs='+')
    a=p.parse_args();config=read_json(workspace_path(a.config))
    if a.batch: config['batches']=[b for b in config['batches'] if b['id'] in a.batch]
    generate(config)
if __name__=='__main__': main()
