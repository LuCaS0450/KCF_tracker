"""
使用 VOT Toolkit 在 VOT2018 数据集上测试 KCF 跟踪器
支持完整测试和快速验证两种模式
"""
import subprocess
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_vot_test(quick_mode=False):
    """运行 VOT2018 测试
    
    Args:
        quick_mode: 如果为 True，仅在第一个序列上测试（用于验证）
    """
    
    # 切换到项目根目录下的 vot2018_workspace 目录
    workspace_dir = PROJECT_ROOT / 'vot2018_workspace'
    sequences_list = workspace_dir / 'sequences' / 'list.txt'
    
    print("=" * 70)
    if quick_mode:
        print("VOT2018 快速测试模式（单序列验证）")
    else:
        print("VOT2018 完整测试模式（所有 60 个序列）")
    print("=" * 70)
    print(f"工作目录: {workspace_dir}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查 trackers.ini 配置
    tracker_config = workspace_dir / 'trackers.ini'
    if not os.path.exists(tracker_config):
        print(f"[ERROR] 找不到配置文件 {tracker_config}")
        return False
    
    print("[OK] 配置文件检查通过")
    
    # 如果是快速模式，临时修改序列列表
    backup_needed = False
    if quick_mode:
        backup_list = Path(str(sequences_list) + '.backup')
        if not os.path.exists(backup_list):
            with open(sequences_list, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_list, 'w', encoding='utf-8') as f:
                f.write(original_content)
            backup_needed = True
            print("[OK] 已备份序列列表")
        
        with open(sequences_list, 'r', encoding='utf-8') as f:
            all_sequences = f.readlines()
        
        if not all_sequences:
            print("[ERROR] 错误：序列列表为空")
            return False
        
        first_seq = all_sequences[0].strip()
        print(f"\n测试序列: {first_seq}")
        
        with open(sequences_list, 'w', encoding='utf-8') as f:
            f.write(first_seq + '\n')
    
    print()
    print("-" * 70)
    if quick_mode:
        print("正在执行: vot test")
    else:
        print("正在执行: vot run --force KCF_Tracker")
    print("-" * 70)
    print("提示: 测试过程可能需要较长时间，请耐心等待...")
    if not quick_mode:
        print("      按 Ctrl+C 可以中断测试（结果会保留）")
    print()
    
    try:
        # 使用 vot toolkit 进行测试
        # 快速模式使用 'vot test'（dummy sequence），完整模式使用 'vot run'（真实序列）
        if quick_mode:
            command = ['vot', 'test', 'KCF_Tracker']
        else:
            command = ['vot', 'run', '--force', 'KCF_Tracker']
        
        result = subprocess.run(
            command,
            cwd=workspace_dir,
            capture_output=False,
            text=True
        )
        
        # 恢复原始列表（如果是快速模式）
        if quick_mode and backup_needed:
            backup_list = Path(str(sequences_list) + '.backup')
            if os.path.exists(backup_list):
                with open(backup_list, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(sequences_list, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print("\n[OK] 已恢复原始序列列表")
        
        print("\n" + "=" * 70)
        if result.returncode == 0:
            results_dir = workspace_dir / 'results'
            baseline_dir = results_dir / 'KCF_Tracker' / 'baseline'
            bin_files = list(baseline_dir.rglob('*.bin')) if baseline_dir.exists() else []

            # 某些 tracker 异常下 vot 可能仍返回 0，这里需要用结果文件二次校验。
            if not quick_mode and len(bin_files) == 0:
                print("[ERROR] 测试过程异常：未生成任何 VOT 结果文件（.bin）")
                print("=" * 70)
                print("\n请检查:")
                print("  1. vot_wrapper.py 是否能正常运行")
                print("  2. trackers.ini 中的 command 路径是否正确")
                print(f"  3. 查看日志: {workspace_dir / 'logs'}")
                return False

            print("[OK] 测试完成！")
            print("=" * 70)
            print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            if os.path.exists(results_dir):
                if quick_mode:
                    print(f"结果目录: {results_dir}")
                else:
                    print(f"生成的 VOT 结果文件数(.bin): {len(bin_files)}")
                    print(f"结果目录: {results_dir}")
            
            print()
            if quick_mode:
                print("下一步:")
                print("  1. 如果测试成功，运行完整测试: python vot2018/run_vot2018.py")
                print("  2. 评估结果: python vot2018/evaluate_vot2018.py")
            else:
                print("下一步:")
                print("  评估测试结果: python vot2018/evaluate_vot2018.py")
            
            return True
        else:
            print(f"[ERROR] 测试失败，返回码: {result.returncode}")
            print("=" * 70)
            print("\n请检查:")
            print("  1. vot_wrapper.py 是否能正常运行")
            print("  2. trackers.ini 中的 command 路径是否正确")
            print(f"  3. 查看日志: {workspace_dir / 'logs'}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("[WARNING] 测试被用户中断")
        print("=" * 70)
        
        # 恢复原始列表（如果是快速模式）
        if quick_mode and backup_needed:
            backup_list = Path(str(sequences_list) + '.backup')
            if os.path.exists(backup_list):
                with open(backup_list, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(sequences_list, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print("[OK] 已恢复原始序列列表")
        
        print("\n已完成的测试结果已保存。")
        print("可以继续运行此命令以完成剩余序列的测试。")
        return False
        
    except FileNotFoundError:
        print("[ERROR] 错误: 找不到 'vot' 命令")
        print("\n请确保已正确安装 vot-toolkit:")
        print("  pip install vot-toolkit")
        
        # 恢复原始列表（如果是快速模式）
        if quick_mode and backup_needed:
            backup_list = Path(str(sequences_list) + '.backup')
            if os.path.exists(backup_list):
                with open(backup_list, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(sequences_list, 'w', encoding='utf-8') as f:
                    f.write(original_content)
        
        return False
        
    except Exception as e:
        print(f"\n[ERROR] 运行时错误: {e}")
        
        # 恢复原始列表（如果是快速模式）
        if quick_mode and backup_needed:
            backup_list = Path(str(sequences_list) + '.backup')
            if os.path.exists(backup_list):
                with open(backup_list, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(sequences_list, 'w', encoding='utf-8') as f:
                    f.write(original_content)
        
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在 VOT2018 数据集上测试 KCF 跟踪器')
    parser.add_argument('--quick', action='store_true', help='快速模式：仅在第一个序列上测试（用于验证配置）')
    args = parser.parse_args()
    
    success = run_vot_test(quick_mode=args.quick)
    sys.exit(0 if success else 1)
