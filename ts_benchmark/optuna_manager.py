#!/usr/bin/env python3
"""
Optuna 数据库管理工具
用于查看、管理和清理 Optuna 调优进度数据库
"""

import argparse
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple

import optuna


def list_studies(db_path: str) -> List[Tuple[str, int, str, float]]:
    """列出数据库中的所有studies"""
    if not os.path.exists(db_path):
        return []

    try:
        # 使用正确的方法获取所有study名称
        study_names = optuna.get_all_study_names(storage=f"sqlite:///{db_path}")

        results = []
        for study_name in study_names:
            try:
                study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
                n_trials = len(study.trials)

                # 获取最后更新时间
                if study.trials:
                    last_trial = max(
                        study.trials, key=lambda t: t.datetime_complete or datetime.min
                    )
                    last_update = (
                        last_trial.datetime_complete.strftime("%Y-%m-%d %H:%M:%S")
                        if last_trial.datetime_complete
                        else "进行中"
                    )
                else:
                    last_update = "无试验"

                best_value = study.best_value if study.best_trial else 0.0
                results.append((study_name, n_trials, last_update, best_value))
            except Exception as e:
                print(f"⚠️ 读取study {study_name} 失败: {e}")
                results.append((study_name, 0, "错误", 0.0))

        return results
    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")
        return []


def show_study_details(db_path: str, study_name: str):
    """显示特定study的详细信息"""
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return

    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

        print(f"\n📊 Study: {study_name}")
        print(
            f"🎯 方向: {'最大化' if study.direction == optuna.study.StudyDirection.MAXIMIZE else '最小化'}"
        )
        print(f"📈 总试验数: {len(study.trials)}")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print(f"✅ 完成试验: {len(completed_trials)}")
        print(f"✂️ 剪枝试验: {len(pruned_trials)}")
        print(f"❌ 失败试验: {len(failed_trials)}")

        if study.best_trial:
            print(f"\n🏆 最佳试验 (#{study.best_trial.number}):")
            print(f"   分数: {study.best_trial.value:.6f}")
            print(f"   参数:")
            for key, value in study.best_trial.params.items():
                print(f"     {key}: {value}")

        # 显示最近的几个试验
        if completed_trials:
            print(f"\n📋 最近完成的试验:")
            recent_trials = sorted(completed_trials, key=lambda t: t.number, reverse=True)[:5]
            for trial in recent_trials:
                print(f"   Trial #{trial.number}: {trial.value:.6f}")

    except Exception as e:
        print(f"❌ 读取study失败: {e}")


def delete_study(db_path: str, study_name: str):
    """删除特定的study"""
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return

    try:
        # 使用正确的方法删除study
        optuna.delete_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        print(f"🗑️ 成功删除 study: {study_name}")
    except Exception as e:
        print(f"❌ 删除study失败: {e}")


def clean_all_databases():
    """清理所有Optuna数据库"""
    db_dir = "optuna_studies"
    if not os.path.exists(db_dir):
        print("📁 没有找到 optuna_studies 目录")
        return

    db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
    if not db_files:
        print("📁 没有找到任何数据库文件")
        return

    print(f"🗑️ 找到 {len(db_files)} 个数据库文件:")
    for db_file in db_files:
        print(f"   - {db_file}")

    confirm = input("确认删除所有数据库文件? (y/N): ")
    if confirm.lower() == "y":
        for db_file in db_files:
            try:
                os.remove(os.path.join(db_dir, db_file))
                print(f"✅ 删除: {db_file}")
            except Exception as e:
                print(f"❌ 删除失败 {db_file}: {e}")
    else:
        print("⏹️ 取消操作")


def main():
    parser = argparse.ArgumentParser(description="Optuna 数据库管理工具")
    parser.add_argument("--list", action="store_true", help="列出所有数据库中的studies")
    parser.add_argument("--show", type=str, help="显示特定study的详细信息")
    parser.add_argument("--delete", type=str, help="删除特定的study")
    parser.add_argument("--clean-all", action="store_true", help="清理所有数据库")
    parser.add_argument("--db-path", type=str, help="指定数据库路径")

    args = parser.parse_args()

    if args.list:
        if args.db_path:
            studies = list_studies(args.db_path)
            if studies:
                print(f"\n📊 数据库中的 Studies ({args.db_path}):")
                print(f"{'Study名称':<40} {'试验数':<8} {'最后更新':<20} {'最佳分数':<12}")
                print("-" * 80)
                for name, n_trials, last_update, best_value in studies:
                    print(f"{name:<40} {n_trials:<8} {last_update:<20} {best_value:<12.6f}")
            else:
                print("📁 没有找到任何studies")
        else:
            # 列出所有数据库中的studies
            db_dir = "optuna_studies"
            if os.path.exists(db_dir):
                db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
                if db_files:
                    print(f"\n📊 所有数据库中的 Studies:")
                    print(f"{'数据库':<30} {'Study名称':<40} {'试验数':<8} {'最佳分数':<12}")
                    print("-" * 90)
                    for db_file in db_files:
                        db_path = os.path.join(db_dir, db_file)
                        studies = list_studies(db_path)
                        for name, n_trials, _, best_value in studies:
                            print(f"{db_file:<30} {name:<40} {n_trials:<8} {best_value:<12.6f}")
                else:
                    print("📁 没有找到任何数据库文件")
            else:
                print("📁 没有找到 optuna_studies 目录")

    elif args.show:
        if not args.db_path:
            print("❌ 请使用 --db-path 指定数据库路径")
            return
        show_study_details(args.db_path, args.show)

    elif args.delete:
        if not args.db_path:
            print("❌ 请使用 --db-path 指定数据库路径")
            return
        delete_study(args.db_path, args.delete)

    elif args.clean_all:
        clean_all_databases()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
