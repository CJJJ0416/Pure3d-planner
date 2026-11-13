"""汇总全类型增强结果，并单独回归原地图全部8处楼梯。"""
import argparse
import hashlib
from pathlib import Path

from Traversability.experiments.wide_stair_report import main as general_report
from Traversability.experiments.all_stair_audit import main as stair_audit
from utils.pointcloud import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    general_report()
    stair_audit()
    audit = read_json(run / 'all_stairs_audit/summary.json')
    summary = read_json(run / 'summary.json')
    summary['all_stairs_regression'] = audit['stairs']
    summary['all_stairs_by_kind'] = audit['by_kind']
    protocol = read_json(run / 'protocol.json')
    summary['baseline_checkpoint_preserved'] = hashlib.sha256(Path(protocol['initial_checkpoint']).read_bytes()).hexdigest() == protocol['initial_sha256']
    write_json(run / 'summary.json', summary)
    before, after = [summary['retention_' + name]['cost_mae_raw'] for name in ['before', 'after']]
    with (run / 'REPORT.md').open('a') as stream:
        stream.write(f'\n旧建筑固定验证Cost MAE /50：{before:.4f} → {after:.4f}。\n')
        stream.write('\n原大地图用于几何模板提取和回归。其原始PCD/GT未加入训练或选优，独立seed的验证和测试属于相同模板分布。\n')
        stream.write('\n全部8处楼梯的几何、MAE、误阻断和条件响应见[全楼梯审查](all_stairs_audit/REPORT.md)。\n')
        stream.write('\n完整PCD位于predictions/new_test、new_test_switchback、new_test_spiral和large_terrain。\n')
    print('ALL STAIRS REPORT COMPLETE', run, flush=True)


if __name__ == '__main__':
    main()
