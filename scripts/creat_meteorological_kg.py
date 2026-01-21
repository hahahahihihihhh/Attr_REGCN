# -*- coding: utf-8 -*-
"""
creat_meteorological_kg.py

输入：
- adjx_kg.csv     (origin, rel=adj0..adj7, destination)  —— 空间八邻接（有向）
- weather_kg.csv  (grid_id, relation=气象变量名, value=离散标签, time=时间戳字符串)

输出（用于时间知识图谱嵌入 / RE-GCN / KGEvolve 类）：
- entity2id.txt      : N<node_id>, L{m}{k:02d}
- relation2id.txt    : A<dir>, M<m>
- stat.txt           : num_entities  num_relations  num_times
- all.txt/train.txt/valid.txt/test.txt : h_id  r_id  t_id  time_id  0

关键要求（你提出的）：
1) train/valid/test 以时间片为单位 7:1:2 划分，并按时间片排序
2) 同一时间片内，adjx 关系（空间边）优先于 weather（气象边）
3) L 实体三位编码：L{m}{k:02d}，其中 k 可到两位数（例如风力等级 0-12）
4) 气象分级严格参考 METEOROLOGICAL_VARS_DIVIDE（upper bound 升序决定 k 的顺序）
"""

import os
import json
from typing import Dict, List, Tuple, Iterable
import pandas as pd


# =========================
# 1) 气象变量与分级表（按你给的版本）
# =========================
METEOROLOGICAL_VARS = [
    "temperature_2m",         # 气温(°C)
    "precipitation",          # 降雨量(mm)
    "cloud_cover",            # 总云量(%)
    "wind_speed_10m",         # 风速(km/h)
    "relative_humidity_2m",   # 相对湿度(%)
    "surface_pressure",       # 气压(hPa)
]

METEOROLOGICAL_VARS_DIVIDE = {
    "temperature_2m": {  # 单位: °C
        -10: "extreme_cold",
        -5: "very_cold",
        0: "cold",
        5: "cool",
        10: "mild",
        15: "comfortable",
        20: "warm",
        25: "hot",
        30: "very_hot",
        35: "extreme_hot",
        40: "scorching",
    },
    "precipitation": {  # 单位: mm
        0: "no_rain",
        0.1: "very_light_rain",
        2.5: "light_rain",
        7.6: "moderate_rain",
        50: "heavy_rain",
        100: "very_heavy_rain",
    },
    "cloud_cover": {  # 单位: %
        0: "clear_sky",
        25: "few_clouds",
        50: "partly_cloudy",
        75: "mostly_cloudy",
        100: "overcast",
    },
    "wind_speed_10m": {  # 单位: km/h
        0.9: "calm",
        5.9: "light_air",
        11.9: "light_breeze",
        19.9: "gentle_breeze",
        28.9: "moderate_breeze",
        38.9: "fresh_breeze",
        49.9: "strong_breeze",
        61.9: "near_gale",
        74.9: "gale",
        88.9: "strong_gale",
        102.9: "violent_storm",
        117.9: "storm",
        133.0: "hurricane_force",
    },
    "relative_humidity_2m": {  # 单位: %
        20: "very_low",
        40: "low",
        60: "medium",
        80: "high",
        100: "very_high",
    },
    "surface_pressure": {  # 单位: hPa
        950: "very_low_pressure",
        980: "low_pressure",
        1010: "normal_pressure",
        1030: "high_pressure",
        1060: "very_high_pressure",
        1100: "extremely_high_pressure",
    },
}


# =========================
# 2) 工具函数
# =========================
def parse_adj_dir(rel_str: str) -> int:
    """adj7 -> 7"""
    return int(str(rel_str).replace("adj", ""))


def build_time_index(weather_csv: str, chunksize: int = 1_000_000) -> Tuple[Dict[str, int], List[str]]:
    """扫描 weather_kg.csv 的 time 列，生成 time2id（按时间字符串排序）"""
    times = set()
    usecols = ["time"]
    for chunk in pd.read_csv(weather_csv, usecols=usecols, chunksize=chunksize):
        times.update(chunk["time"].astype(str).tolist())
    times_sorted = sorted(times)
    time2id = {t: i for i, t in enumerate(times_sorted)}
    return time2id, times_sorted


def build_label_index_from_divide(
    met_vars: List[str],
    met_divide: Dict[str, Dict[float, str]],
) -> List[Tuple[str, str, str, int, int]]:
    """
    严格按 METEOROLOGICAL_VARS_DIVIDE（upper bound 升序）生成标签实体：
    返回 [(ename, var, label, m, k), ...]

    ename = L{m}{k:02d} —— L 后三位：m(1位) + k(2位)
    """
    out = []
    for m, var in enumerate(met_vars):
        if var not in met_divide:
            raise KeyError(f"{var} not in METEOROLOGICAL_VARS_DIVIDE")
        items = sorted(met_divide[var].items(), key=lambda x: float(x[0]))  # upper bound 升序
        for k, (_, label) in enumerate(items):
            ename = f"L{m}{k:02d}"
            out.append((ename, var, str(label), m, k))
    return out


def write_relation2id(out_dir: str, adj_dirs: int, num_met: int) -> Dict[str, int]:
    """
    relation2id:
      A0..A(adj_dirs-1)  -> 0..adj_dirs-1
      M0..M(num_met-1)   -> adj_dirs..adj_dirs+num_met-1
    """
    rel2id = {}
    idx = 0
    for d in range(adj_dirs):
        rel2id[f"A{d}"] = idx
        idx += 1
    for m in range(num_met):
        rel2id[f"M{m}"] = idx
        idx += 1

    path = os.path.join(out_dir, "relation2id.txt")
    with open(path, "w", encoding="utf-8") as f:
        for name, rid in rel2id.items():
            f.write(f"{name}\t{rid}\n")
    return rel2id


def write_entity2id(
    out_dir: str,
    node_ids: List[int],
    label_entity_names: List[Tuple[str, str, str, int, int]],
) -> Dict[str, int]:
    """
    entity2id:
      节点实体：N<node_id>（按 node_id 升序）
      标签实体：L{m}{k:02d}（按 met_vars 顺序、再按 k 顺序）
    """
    ent2id = {}
    eid = 0

    for nid in sorted(node_ids):
        name = f"N{nid}"
        ent2id[name] = eid
        eid += 1

    for (ename, _, _, _, _) in label_entity_names:
        ent2id[ename] = eid
        eid += 1

    path = os.path.join(out_dir, "entity2id.txt")
    with open(path, "w", encoding="utf-8") as f:
        for name, idx in ent2id.items():
            f.write(f"{name}\t{idx}\n")
    return ent2id


def append_quads(fp, h: int, r: int, t: int, time_id: int):
    """输出格式：h  r  t  time_id  0"""
    fp.write(f"{h}\t{r}\t{t}\t{time_id}\t0\n")


def split_time_ids_7_1_2(num_times: int) -> Tuple[set, set, set]:
    """
    以时间片为单位，按 7:1:2 连续切分（0..T-1）：
      train: [0, ..., n_train-1]
      valid: [n_train, ..., n_train+n_valid-1]
      test : [n_train+n_valid, ..., T-1]
    """
    if num_times < 3:
        raise ValueError(f"时间片太少({num_times})，无法切分 train/valid/test。")

    n_train = int(num_times * 0.7)
    n_valid = int(num_times * 0.1)
    # 保底
    n_train = max(1, n_train)
    n_valid = max(1, n_valid)
    n_test = num_times - n_train - n_valid
    if n_test < 1:
        n_test = 1
        n_train = max(1, num_times - n_valid - n_test)

    train_ids = set(range(0, n_train))
    valid_ids = set(range(n_train, n_train + n_valid))
    test_ids = set(range(n_train + n_valid, num_times))
    return train_ids, valid_ids, test_ids


# =========================
# 3) 主流程：保证“时间片排序 + adjx优先”
# =========================
def main_build(
    adjx_csv: str,
    weather_csv: str,
    out_dir: str,
    met_vars: List[str],
    chunksize: int = 1_000_000,
):
    os.makedirs(out_dir, exist_ok=True)

    # 0) 读空间邻接
    adjx = pd.read_csv(adjx_csv)
    if not {"origin", "rel", "destination"}.issubset(adjx.columns):
        raise ValueError(f"{adjx_csv} 必须包含列：origin, rel, destination")

    origins = adjx["origin"].astype(int).tolist()
    rels = adjx["rel"].astype(str).tolist()
    dests = adjx["destination"].astype(int).tolist()
    node_ids = set(origins) | set(dests)

    # 1) time2id（按时间字符串排序）
    time2id, times_sorted = build_time_index(weather_csv, chunksize=chunksize)
    num_times = len(times_sorted)

    # 2) 标签实体（严格按分级表）
    label_entity_names = build_label_index_from_divide(met_vars, METEOROLOGICAL_VARS_DIVIDE)

    # 3) 写 relation2id / entity2id / stat
    rel2id = write_relation2id(out_dir, adj_dirs=8, num_met=len(met_vars))
    ent2id = write_entity2id(out_dir, node_ids=list(node_ids), label_entity_names=label_entity_names)

    # (var, label_str) -> tail_entity_id
    label_lookup = {}
    for (ename, var, label, _, _) in label_entity_names:
        label_lookup[(var, str(label))] = ent2id[ename]

    with open(os.path.join(out_dir, "stat.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(ent2id)}\t{len(rel2id)}\t{num_times}\n")

    # 4) 时间切分 7:1:2（按 time_id 连续切分）
    train_time_ids, valid_time_ids, test_time_ids = split_time_ids_7_1_2(num_times)

    # 5) 先把 weather 边分桶到磁盘：tmp_weather/time_{time_id}.txt
    tmp_dir = os.path.join(out_dir, "_tmp_weather")
    os.makedirs(tmp_dir, exist_ok=True)

    var2m = {v: i for i, v in enumerate(met_vars)}
    usecols = ["grid_id", "relation", "value", "time"]

    # 为了减少同时打开的文件句柄，用“按 time_id 打开-写入-关闭”的策略
    # 这里直接 append 写到每个 time 文件里（顺序无所谓，后面会按 time_id 合并）
    for chunk in pd.read_csv(weather_csv, usecols=usecols, chunksize=chunksize):
        grid_ids = chunk["grid_id"].astype(int).tolist()
        rel_vars = chunk["relation"].astype(str).tolist()
        vals = chunk["value"].astype(str).tolist()
        times = chunk["time"].astype(str).tolist()

        # 按 time_id 分组，减少反复 open/close
        bucket: Dict[int, List[Tuple[int, int, int]]] = {}
        for gid, var, val, tstr in zip(grid_ids, rel_vars, vals, times):
            m = var2m.get(var, None)
            if m is None:
                continue
            time_id = time2id.get(tstr, None)
            if time_id is None:
                continue

            h = ent2id.get(f"N{gid}", None)
            if h is None:
                continue

            tail = label_lookup.get((var, val), None)
            if tail is None:
                # 若发生，说明 weather_kg.csv 的 value 不在分级表标签集合里
                continue

            r = rel2id[f"M{m}"]
            bucket.setdefault(time_id, []).append((h, r, tail))

        for time_id, triples in bucket.items():
            tmp_path = os.path.join(tmp_dir, f"time_{time_id}.txt")
            with open(tmp_path, "a", encoding="utf-8") as fp:
                for (h, r, t) in triples:
                    fp.write(f"{h}\t{r}\t{t}\n")

    # 6) 最终按 time_id=0..T-1 输出：先 adjx 后 weather（同时间片内满足你要求）
    fp_all = open(os.path.join(out_dir, "all.txt"), "w", encoding="utf-8")
    fp_train = open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8")
    fp_valid = open(os.path.join(out_dir, "valid.txt"), "w", encoding="utf-8")
    fp_test = open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8")

    def write_split(h, r, t, time_id):
        append_quads(fp_all, h, r, t, time_id)
        if time_id in train_time_ids:
            append_quads(fp_train, h, r, t, time_id)
        elif time_id in valid_time_ids:
            append_quads(fp_valid, h, r, t, time_id)
        elif time_id in test_time_ids:
            append_quads(fp_test, h, r, t, time_id)

    try:
        for time_id in range(num_times):
            # 6.1 空间边（adjx）—— 同一时间片内优先输出
            for o, rel, d in zip(origins, rels, dests):
                dir_id = parse_adj_dir(rel)  # adj7 -> 7
                h = ent2id[f"N{o}"]
                t = ent2id[f"N{d}"]
                r = rel2id[f"A{dir_id}"]
                write_split(h, r, t, time_id)

            # 6.2 气象边（weather）—— 同一时间片内后输出
            tmp_path = os.path.join(tmp_dir, f"time_{time_id}.txt")
            if os.path.exists(tmp_path):
                with open(tmp_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        h, r, t = line.strip().split("\t")
                        write_split(int(h), int(r), int(t), time_id)

    finally:
        fp_all.close()
        fp_train.close()
        fp_valid.close()
        fp_test.close()

    # 7) 打印信息
    n_train = len(train_time_ids)
    n_valid = len(valid_time_ids)
    n_test = len(test_time_ids)
    print(f"[OK] wrote files to: {out_dir}")
    print(f"     num_entities={len(ent2id)}, num_relations={len(rel2id)}, num_times={num_times}")
    print(f"     split(time_slices) train/valid/test = {n_train}/{n_valid}/{n_test}  (7:1:2)")
    print(f"     time_id ranges: train=[0,{n_train-1}], valid=[{n_train},{n_train+n_valid-1}], test=[{n_train+n_valid},{num_times-1}]")


# =========================
# 4) 入口：从 settings.json 读取路径
# =========================
if __name__ == "__main__":
    dataset = "TDRIVE20150406"  # 你可改成自己的 key
    with open("settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
    cfg = settings[dataset]

    # settings.json 中要求包含：
    # cfg["relation_kg_path"]  -> adjx_kg.csv 路径
    # cfg["attribute_kg_path"] -> weather_kg.csv 路径
    # cfg["data_dir"]          -> 输出目录根
    adjx_kg = cfg["relation_kg_path"]
    weather_kg = cfg["attribute_kg_path"]
    data_dir = cfg["data_dir"]

    main_build(
        adjx_csv=adjx_kg,
        weather_csv=weather_kg,
        out_dir=data_dir,
        met_vars=METEOROLOGICAL_VARS,
        chunksize=1_000_000,
    )
