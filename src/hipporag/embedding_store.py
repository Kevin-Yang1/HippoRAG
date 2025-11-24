import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        初始化类，配置必要的参数并设置工作目录。

        参数:
        embedding_model: 用于生成嵌入的模型。
        db_filename: 存储或检索数据的目录路径。
        batch_size: 处理时使用的批次大小。
        namespace: 用于数据隔离的唯一标识符（命名空间）。

        功能:
        - 将提供的参数分配给实例变量。
        - 检查 `db_filename` 指定的目录是否存在。
          - 如果不存在，则创建该目录并记录操作日志。
        - 构建用于存储数据的 parquet 文件名。
        - 调用 `_load_data()` 方法初始化数据加载过程。
        """
        self.embedding_model = embedding_model  # 用于生成文本嵌入向量的模型实例
        self.batch_size = batch_size            # 调用嵌入模型时使用的批次大小
        self.namespace = namespace              # 数据隔离的命名空间（例如 'chunk', 'entity', 'fact'）

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )  # 存储数据的 Parquet 文件的完整路径
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        """
        识别并返回尚未在存储中存在的文本及其哈希 ID。

        参数:
        texts: 待检查的文本列表。

        返回:
        一个字典，键为缺失文本的哈希 ID，值为对应的文本内容。
        """
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 获取输入字典中的所有哈希 ID。
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的哈希 ID。
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        将新文本插入存储。计算哈希 ID，检查是否存在，对缺失文本进行编码并保存。

        参数:
        texts: 待插入的文本列表。
        """
        nodes_dict = {}
        # 构建哈希 ID 到文本内容的映射字典。
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 获取输入字典中的所有哈希 ID。
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # 没有需要插入的内容。

        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的哈希 ID。
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# 所有记录已存在。

        # 准备从 "content" 字段编码的文本。
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        """
        从 parquet 文件加载数据到内存。
        如果文件不存在，则初始化为空列表和字典。
        """
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            # 内存中维护的数据存储属性
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            # self.hash_ids: 存储所有记录的哈希 ID 列表
            # self.texts: 存储所有记录的文本内容列表
            # self.embeddings: 存储所有记录的嵌入向量列表

            # 辅助索引属性（用于快速查找）
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            # self.hash_id_to_idx: 将哈希 ID 映射到其在列表中的索引位置

            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            # self.hash_id_to_row: 将哈希 ID 映射到包含完整行数据（hash_id, content）的字典

            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            # self.hash_id_to_text: 将哈希 ID 映射到对应的文本内容

            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            # self.text_to_hash_id: 将文本内容映射回其哈希 ID（用于反向查找）

            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        """
        将当前内存中的数据保存到 parquet 文件。
        同时更新内存中的辅助映射字典。
        """
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        """
        更新或插入记录。将新数据追加到内存列表并保存到磁盘。

        参数:
        hash_ids: 哈希 ID 列表。
        texts: 文本内容列表。
        embeddings: 嵌入向量列表。
        """
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        """
        根据哈希 ID 删除记录。

        参数:
        hash_ids: 要删除的哈希 ID 列表。
        """
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        """根据哈希 ID 获取单行数据。"""
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        """根据文本内容获取哈希 ID。"""
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        """根据哈希 ID 列表获取多行数据。"""
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        """获取所有哈希 ID 的深拷贝。"""
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        """获取所有哈希 ID 到行数据的映射的深拷贝。"""
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        """获取所有文本内容的集合。"""
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        """根据哈希 ID 获取单个嵌入向量。"""
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        """根据哈希 ID 列表获取多个嵌入向量。"""
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings